"""Training loop for GSN temporal link prediction.

Implements Algorithm 1 from the paper:
  For each time bucket:
    1. Build G_end snapshot (events up to bucket end)
    2. Score links with GradientTape → ranking loss + write penalty
    3. Accumulate / apply gradients
    4. Forward G_end with commit=True to update state table
"""

from __future__ import annotations
import json
from contextlib import contextmanager
from dataclasses import dataclass
from termcolor import cprint
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from rich.console import Console
from rich.progress import (
                            BarColumn,
                            MofNCompleteColumn,
                            Progress,
                            TextColumn,
                            TimeElapsedColumn,
                            TimeRemainingColumn
                          )


try:
    from ..snapshot import Snapshot
    from ..state.table import DenseStateTable
    from ..layers.gsn_block import GSNBlock, PersistentGSNBlock
    from ..layers.link_predictor import LinkPredictor
    from ..datasets.tgb_loader import TGBSplit
    from ..datasets.negative_sampling import (
                                                    TGBStyleTrainNegativeSampler,
                                                    TrainNegativeSampler,
                                                    OfficialNegativeSampler,
                                                )
    from ..train.loss import ranking_loss, write_penalty_loss
    from ..train.metrics import compute_mrr, compute_ap, compute_auc
except Exception as e:
    cprint("[loop.py] Failed with relative import. Trying with absolute import.", "yellow")
    from gsn.snapshot import Snapshot
    from gsn.state.table import DenseStateTable
    from gsn.layers.gsn_block import GSNBlock, PersistentGSNBlock
    from gsn.layers.link_predictor import LinkPredictor
    from gsn.datasets.tgb_loader import TGBSplit
    from gsn.datasets.negative_sampling import (
                                                    TGBStyleTrainNegativeSampler,
                                                    TrainNegativeSampler,
                                                    OfficialNegativeSampler,
                                                )
    from gsn.train.loss import ranking_loss, write_penalty_loss
    from gsn.train.metrics import compute_mrr, compute_ap, compute_auc


# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------

@contextmanager
def _progress_ctx(console: Console, label: str, total: int):
    p = Progress(
                    TextColumn(f"[cyan]{label:<6}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    TextColumn("[dim]{task.fields[suffix]}"),
                    console   = console,
                    transient = False,
                )
    with p:
        task = p.add_task("", total = total, suffix = "")
        yield p, task


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GSNLinkPredictor(keras.Model):
    """Stack of PersistentGSNBlocks + link-prediction scorer."""

    def __init__(
                    self,
                    num_nodes:        int,
                    hidden:           int   = 128,
                    num_heads:        int   = 4,
                    head_dim:         int   = 32,
                    state_dim:        int   = 16,
                    num_layers:       int   = 1,
                    embed_dim:        int   = 128,
                    scorer:           str   = "mlp",
                    commit_alpha:     float = 0.2,
                    time_feat_dim:    int   = 8,
                    time_scale:       float = 86400.0,
                    edge_gate_hidden: int   = 32,
                    dropout:          float = 0.0,
                    self_loops:       bool  = True,
                    noise_scale:      float = 0.005,
                    id_dim:           int   = 0,
                    name:             str   = None,
                    **kwargs
                ):
        super().__init__(name = name, **kwargs)
        assert hidden == num_heads * head_dim

        # Store all constructor params for serialization
        self._num_heads        = int(num_heads)
        self._head_dim         = int(head_dim)
        self._state_dim        = int(state_dim)
        self._num_layers       = int(num_layers)
        self._embed_dim        = int(embed_dim)
        self._scorer           = str(scorer)
        self._commit_alpha     = float(commit_alpha)
        self._time_feat_dim    = int(time_feat_dim)
        self._time_scale       = float(time_scale)
        self._edge_gate_hidden = int(edge_gate_hidden)
        self._dropout          = float(dropout)
        self._self_loops       = bool(self_loops)
        self._noise_scale      = float(noise_scale)
        self._id_dim           = int(id_dim)

        self.num_nodes       = int(num_nodes)
        self.hidden          = int(hidden)
        self.state_dim_total = self._num_heads * self._head_dim * self._state_dim

        self.state_tables: List[DenseStateTable]    = []
        self.blocks:       List[PersistentGSNBlock] = []

        for i in range(self._num_layers):
            table = DenseStateTable(
                                        num_entities = self.num_nodes,
                                        state_dim    = self.state_dim_total,
                                        name         = f"state_table_{i}"
                                    )
            block = GSNBlock(
                                hidden           = hidden,
                                num_heads        = self._num_heads,
                                head_dim         = self._head_dim,
                                state_dim        = self._state_dim,
                                time_feat_dim    = self._time_feat_dim,
                                time_scale       = self._time_scale,
                                edge_gate_hidden = self._edge_gate_hidden,
                                dropout          = self._dropout,
                                self_loops       = self._self_loops,
                                name             = f"gsn_block_{i}"
                            )
            pblock = PersistentGSNBlock(
                                            block        = block,
                                            state_table  = table,
                                            commit_alpha = self._commit_alpha,
                                            noise_scale  = self._noise_scale,
                                            name         = f"persistent_block_{i}"
                                        )
            self.state_tables.append(table)
            self.blocks.append(pblock)
            setattr(self, f"_pblock_{i}", pblock)
            setattr(self, f"_table_{i}",  table)

        self.num_layers = len(self.blocks)

        self.id_dim = self._id_dim
        if self.id_dim > 0:
            self.node_id_emb = layers.Embedding(
                                                    self.num_nodes,
                                                    self.id_dim,
                                                    embeddings_initializer = keras.initializers.TruncatedNormal(stddev = 0.02),
                                                    name                   = "node_id_emb"
                                               )
            self.id_emb_drop = layers.Dropout(self._dropout, name = "node_emb_dropout")
            self.node_id_emb.build((None, self.num_nodes))
            self.id_emb_drop.build((None, self.num_nodes, self.id_dim))

        total_state     = self.num_layers * self.state_dim_total
        self.state_proj = layers.Dense(self._embed_dim, use_bias = True,  name = "state_proj")
        self.item_proj  = layers.Dense(self._embed_dim, use_bias = False, name = "item_proj")
        self.state_proj.build((None, total_state))
        self.item_proj.build((None, hidden + self.id_dim))

        self.scorer_head = LinkPredictor(
                                            embed_dim      = self._embed_dim,
                                            hidden         = self._embed_dim,
                                            scorer         = self._scorer,
                                        )
        self.ln_out = layers.LayerNormalization(epsilon = 1e-5)

        self.temp = self.add_weight(
                                        name        = "temp",
                                        shape       = (),
                                        trainable   = True,
                                        initializer = keras.initializers.Ones()
                                    )
        
        self.build()

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
                    "num_nodes":        self.num_nodes,
                    "hidden":           self.hidden,
                    "num_heads":        self._num_heads,
                    "head_dim":         self._head_dim,
                    "state_dim":        self._state_dim,
                    "num_layers":       self._num_layers,
                    "embed_dim":        self._embed_dim,
                    "scorer":           self._scorer,
                    "commit_alpha":     self._commit_alpha,
                    "time_feat_dim":    self._time_feat_dim,
                    "time_scale":       self._time_scale,
                    "edge_gate_hidden": self._edge_gate_hidden,
                    "dropout":          self._dropout,
                    "self_loops":       self._self_loops,
                    "noise_scale":      self._noise_scale,
                    "id_dim":           self._id_dim
                }

    def _build_from_dummy(self, edge_feat_dim: int = None) -> None:
        """Trigger Keras weight creation so load_weights works."""
        dummy = Snapshot.from_events(
                                        src_global = np.zeros(1, dtype = np.int64),
                                        dst_global = np.zeros(1, dtype = np.int64),
                                        timestamps = np.zeros(1, dtype = np.int64),
                                        t_ref      = 0,
                                        dt         = 1.0,
                                        edge_feat  = np.zeros((1, edge_feat_dim), dtype = np.int64)
                                    )
        self.forward(dummy, commit = False, training = False)

    @classmethod
    def from_pretrained(cls, path, edge_feat_dim: int = None) -> "GSNLinkPredictor":
        """Load model config and weights from a saved directory."""
        path = Path(path)
        with open(path / "config.json", encoding = "utf-8") as f:
            config = json.load(f)
        model = cls(**config)
        model._build_from_dummy(edge_feat_dim = edge_feat_dim)
        best = path / "best.weights.h5"
        if best.exists():
            weights_path = best
        else:
            candidates = sorted(path.glob("epoch_*.weights.h5"))
            if not candidates:
                raise FileNotFoundError(f"No weights found in {path}")
            weights_path = candidates[-1]
        
        print(f"\nLoaded from {weights_path}")
        model.load_weights(str(weights_path))
        return model

    # ------------------------------------------------------------------

    def reset_states_all(self) -> None:
        for t in self.state_tables:
            t.reset_state()
            
    def build(self, input_shape = None):
        self.built = True
        super().build(input_shape)
            
    def call(
                self, 
                *, 
                snap: Snapshot = None,
                commit: bool = False,
                training: Optional[bool] = None
            ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        
        return self.forward(snap, commit = commit, training = training)

    def forward(
                    self,
                    snap:     Snapshot,
                    commit:   bool           = False,
                    training: Optional[bool] = None
                ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Run all persistent blocks on a snapshot.

        Returns
        -------
        H      : [N, hidden]           — final node embeddings
        states : list of [N, S_total]  — updated states from each block
        """
        out        = None
        all_states = []
        for pblock in self.blocks:
            h, s_next = pblock(snap = snap, commit = commit, training = training)
            out = h
            all_states.append(s_next)
        return out, all_states

    def score_pairs(
                        self,
                        H:        tf.Tensor,
                        node_ids: np.ndarray,
                        cand_src: np.ndarray,
                        cand_dst: np.ndarray,
                        training: Optional[bool] = None
                    ) -> tf.Tensor:
        """Score (u, v) candidate pairs.

        Source : persistent-state projection from each state table.
        Dest   : snapshot embedding H (zeroed for out-of-snapshot nodes)
                 + optional learned ID embedding.

        The original code used id_to_local.get(v, 0) as a fallback, which
        silently gave every out-of-snapshot negative the embedding of whichever
        node happened to sit at local index 0.  With batch_events=128 and
        999 negatives per positive, ~99 % of negative destinations fell into
        this path and received *identical* corrupt embeddings.

        The fix: build an in-snapshot boolean mask and multiply H by it, so
        out-of-snapshot nodes get h_dst=0.  The id_emb path (when id_dim>0)
        still supplies a meaningful per-node signal for those positions via
        item_proj's last id_dim columns, keeping them distinguishable.
        Gradient flow through H (and therefore through the SSM) is preserved.
        """
        id_to_local  = {int(gid): i for i, gid in enumerate(node_ids)}

        # 1-for-in-snapshot, 0-for-out-of-snapshot  [B, 1]
        in_snap      = np.array([int(v) in id_to_local for v in cand_dst],
                                dtype = np.float32)
        in_snap_tf   = tf.constant(in_snap, dtype = tf.float32)[:, None]

        dst_local    = np.array([id_to_local.get(int(v), 0) for v in cand_dst],
                                dtype = np.int32)
        dst_local_tf = tf.cast(dst_local, tf.int32)

        u_ids_tf = tf.cast(cand_src, tf.int32)
        u_states = tf.concat(
            [t.get(u_ids_tf) for t in self.state_tables], axis = -1
        )
        u_z = self.state_proj(u_states)

        h_dst = tf.gather(H, dst_local_tf) * in_snap_tf   # zero out invalid
        if self.id_dim > 0:
            v_ids_tf = tf.cast(cand_dst, tf.int32)
            id_emb   = self.node_id_emb(v_ids_tf, training = training)
            id_emb   = self.id_emb_drop(id_emb,   training = training)
            h_dst    = tf.concat([h_dst, tf.cast(id_emb, h_dst.dtype)], axis = -1)
        v_z = self.item_proj(h_dst)

        logits = self.scorer_head(u_z, v_z, training = training)
        denom  = tf.cast(tf.nn.softplus(self.temp) + 1e-6, tf.float32)
        return tf.cast(logits, tf.float32) / denom


# ---------------------------------------------------------------------------
# Bucket iterator helpers
# ---------------------------------------------------------------------------

def _iter_buckets(
                    split:       TGBSplit,
                    bucket_size: int
                 ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                     Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Yield chronological event buckets from a split.
    """
    E     = len(split.src)
    start = 0
    while start < E:
        end = min(start + bucket_size, E)
        sl  = slice(start, end)
        yield (
                split.src[sl],
                split.dst[sl],
                split.ts[sl],
                split.edge_feat[sl] if split.edge_feat is not None else None,
                split.neg_dst[sl]   if split.neg_dst   is not None else None
              )
        start = end


def _build_snapshot(
                        src:       np.ndarray,
                        dst:       np.ndarray,
                        ts:        np.ndarray,
                        edge_feat: Optional[np.ndarray],
                        t_ref:     int,
                        last_t:    int
                   ) -> Optional[Snapshot]:
    if src.shape[0] == 0:
        return None
    dt = max(float(t_ref - last_t), 1.0)
    return Snapshot.from_events(
                                    src_global = src,
                                    dst_global = dst,
                                    timestamps = ts,
                                    t_ref      = t_ref,
                                    dt         = dt,
                                    edge_feat  = edge_feat
                                )


# ---------------------------------------------------------------------------
# Trainer config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    lr:               float          = 3e-4
    beta_1:           float          = 0.9
    beta_2:           float          = 0.999
    weight_decay:     float          = 0.0
    clip_norm:        Optional[float] = 1.0
    loss_fn:          str            = "ce"
    lambda_wr:        float          = 1e-4
    epochs:           int            = 5
    batch_events:     int            = 20_000
    accumulate_every: int            = 1
    train_neg_per_pos:    int            = 1
    val_test_neg_per_pos: int            = 49   # -1 = all negatives (requires precomputed neg_dst)
    seed:             int            = 1337
    weights_dir:      Optional[str]  = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer(keras.Model):
    """GSN trainer for temporal link prediction.

    Usage
    -----
        trainer = Trainer(model, cfg, train, val, test, meta)
        history = trainer.fit()
    """

    def __init__(
                        self,
                        model: GSNLinkPredictor,
                        cfg:   TrainerConfig,
                        train: TGBSplit,
                        val:   TGBSplit,
                        test:  TGBSplit,
                        meta:  Dict[str, Any],
                        **kwargs
                ):
        super().__init__(**kwargs)
        self.model       = model
        self.cfg         = cfg
        self.train_split = train
        self.val_split   = val
        self.test_split  = test
        self.meta        = meta

        self._gsn_optimizer = keras.optimizers.AdamW(
                                                        learning_rate = cfg.lr,
                                                        beta_1        = cfg.beta_1,
                                                        beta_2        = cfg.beta_2,
                                                        weight_decay  = cfg.weight_decay
                                                    )

        dst_cands = np.unique(train.dst) if meta.get("bipartite", False) else None
        self.neg_sampler = TGBStyleTrainNegativeSampler(
                                                            train_src      = train.src,
                                                            train_dst      = train.dst,
                                                            train_ts       = train.ts,
                                                            num_nodes      = model.num_nodes,
                                                            num_neg_e      = cfg.train_neg_per_pos,
                                                            seed           = cfg.seed,
                                                            dst_candidates = dst_cands
                                                        )

        def _make_eval_sampler(split_data, seed_offset: int):
            if split_data.neg_dst is not None:
                return OfficialNegativeSampler(split_data.neg_dst)
            if cfg.val_test_neg_per_pos == -1:
                raise ValueError(
                    "val_test_neg_per_pos=-1 (all negatives) requires pre-computed "
                    "negatives attached to the split.  Call load_dataset() with "
                    "precompute_negatives=True and num_neg_e=-1 before constructing "
                    "the Trainer, or set val_test_neg_per_pos to a positive integer "
                    "to use on-the-fly random sampling."
                )
            return TrainNegativeSampler(
                model.num_nodes,
                neg_per_pos = cfg.val_test_neg_per_pos,
                seed        = cfg.seed + seed_offset,
            )

        self.val_sampler  = _make_eval_sampler(val,  seed_offset = 1)
        self.test_sampler = _make_eval_sampler(test, seed_offset = 2)

        self._accum_grads: Optional[List[tf.Tensor]] = None
        self._accum_count  = 0

    # ------------------------------------------------------------------

    def call(self, inputs, training = None):
        raise NotImplementedError("Use Trainer.fit() — this model is not callable directly.")

    # ------------------------------------------------------------------

    def train_step(self, data: Dict) -> Dict[str, float]:
        """Score links for one temporal bucket and accumulate gradients."""
        src_b  = data["src"]
        dst_b  = data["dst"]
        ts_b   = data["ts"]
        ef_b   = data.get("edge_feat")
        last_t = int(data["last_t"])
        b_idx  = int(data["b_idx"])
        n_bkts = int(data["num_buckets"])

        if src_b.shape[0] == 0:
            return {}

        t_end    = int(ts_b.max())
        snap_end = _build_snapshot(src_b, dst_b, ts_b, ef_b, t_end, last_t)
        if snap_end is None:
            return {}

        cand_src, cand_dst, _, sizes = self.neg_sampler.build_candidates(src_b, dst_b, ts = ts_b)
        sizes_tf = tf.cast(sizes, tf.int32)

        with tf.GradientTape() as tape:
            H, states   = self.model.forward(snap_end, commit = False, training = True)
            logits      = self.model.score_pairs(H, snap_end.node_ids, cand_src, cand_dst, training = True)
            logits_flat = tf.reshape(logits, [-1])
            rank_loss   = ranking_loss(logits_flat, sizes_tf, mode = self.cfg.loss_fn)

            wr_loss = tf.constant(0.0)
            if self.cfg.lambda_wr > 0.0:
                ids_tf = tf.cast(snap_end.node_ids, tf.int32)
                for table, s_next in zip(self.model.state_tables, states):
                    s_prev   = table.get(ids_tf)
                    wr_loss += write_penalty_loss(s_prev, s_next, logits_flat)

            loss = rank_loss + self.cfg.lambda_wr * wr_loss

        grads = tape.gradient(loss, self.model.trainable_weights)
        self._accumulate(grads)
        if (b_idx + 1) % self.cfg.accumulate_every == 0 or b_idx == n_bkts - 1:
            self._apply_gradients()

        # Commit state after gradient update
        self.model.forward(snap_end, commit = True, training = False)

        return {
                    "loss":      float(loss),
                    "rank_loss": float(rank_loss),
                    "wr_loss":   float(wr_loss),
               }

    # ------------------------------------------------------------------

    def test_step(self, data: Dict) -> Dict[str, Any]:
        """Score links for one temporal bucket during evaluation."""
        src_b        = data["src"]
        dst_b        = data["dst"]
        ts_b         = data["ts"]
        ef_b         = data.get("edge_feat")
        last_t       = int(data["last_t"])
        sampler      = data["sampler"]
        event_offset = int(data.get("event_offset", 0))

        if src_b.shape[0] == 0:
            return {"logits": None, "sizes": None}

        t_end = int(ts_b.max())
        snap  = _build_snapshot(src_b, dst_b, ts_b, ef_b, t_end, last_t)
        if snap is None:
            return {"logits": None, "sizes": None}

        # Build candidates first, then score with pre-commit H (consistent
        # with train_step which also scores before committing), then commit.
        if isinstance(sampler, OfficialNegativeSampler):
            cand_src, cand_dst, _, sizes = sampler.build_candidates(src_b, dst_b, offset = event_offset)
        else:
            cand_src, cand_dst, _, sizes = sampler.build_candidates(src_b, dst_b)

        H, _ = self.model.forward(snap, commit = False, training = False)
        logits = self.model.score_pairs(H, snap.node_ids, cand_src, cand_dst, training = False)
        self.model.forward(snap, commit = True, training = False)
        return {"logits": tf.reshape(logits, [-1]), "sizes":  sizes}

    # ------------------------------------------------------------------

    def fit(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:  # type: ignore[override]
        n_epochs = epochs if epochs is not None else self.cfg.epochs
        history: Dict[str, List[float]] = {
                                            "train_loss": [], "train_rank": [], "train_wr":  [],
                                            "val_mrr":    [], "val_ap":    [], "val_auc":    [],
                                            "test_mrr":   [], "test_ap":   [], "test_auc":   []
                                          }
        console  = Console()
        best_mrr = 0.0

        # Single reset before the first training epoch.  Resetting at the top
        # of every epoch discards the persistent memory the model accumulated
        # over the previous epoch's event stream, which (a) defeats the purpose
        # of persistent state and (b) causes the loss to explode in epoch 2
        # because the trained weights now receive zero states instead of the
        # non-zero states they adapted to.
        
        self.model.reset_states_all()

        for epoch in range(n_epochs):
            console.rule(f"[bold white]Epoch {epoch + 1} / {n_epochs}")

            train_m = self._train_epoch(console)
            history["train_loss"].append(train_m["loss"])
            history["train_rank"].append(train_m["rank_loss"])
            history["train_wr"].append(train_m["wr_loss"])

            # Save training state, evaluate from a clean slate, then restore
            # so training can continue from where it left off next epoch.
            train_states = [t.clone() for t in self.model.state_tables]

            self.model.reset_states_all()
            vmrr, vap, vauc = self._eval_epoch("Val",  self.val_split,  self.val_sampler,  console)
            history["val_mrr"].append(vmrr)
            history["val_ap"].append(vap)
            history["val_auc"].append(vauc)

            self.model.reset_states_all()
            tmrr, tap, tauc = self._eval_epoch("Test", self.test_split, self.test_sampler, console)

            for table, saved in zip(self.model.state_tables, train_states):
                table.copy_from(saved)
            history["test_mrr"].append(tmrr)
            history["test_ap"].append(tap)
            history["test_auc"].append(tauc)

            is_best = vmrr > best_mrr
            if is_best:
                best_mrr = vmrr
            if self.cfg.weights_dir is not None:
                self._save(epoch + 1, val_mrr = vmrr, is_best = is_best)

        return history

    # ------------------------------------------------------------------

    def _train_epoch(self, console: Console) -> Dict[str, float]:
        buckets     = list(_iter_buckets(self.train_split, self.cfg.batch_events))
        num_buckets = len(buckets)
        self._accum_grads = None
        self._accum_count = 0

        total_loss = total_rank = total_wr = 0.0
        n_done = 0
        last_t = 0

        with _progress_ctx(console, "Train", num_buckets) as (progress, task):
            for b_idx, (src_b, dst_b, ts_b, ef_b, _) in enumerate(buckets):
                data = {
                            "src":         src_b,
                            "dst":         dst_b,
                            "ts":          ts_b,
                            "edge_feat":   ef_b,
                            "last_t":      last_t,
                            "b_idx":       b_idx,
                            "num_buckets": num_buckets,
                        }
                metrics = self.train_step(data)
                if src_b.shape[0] > 0:
                    last_t = int(ts_b.max())
                if metrics:
                    total_loss += metrics["loss"]
                    total_rank += metrics["rank_loss"]
                    total_wr   += metrics["wr_loss"]
                    n_done     += 1
                safe   = max(n_done, 1)
                suffix = (
                            f"loss: {total_loss/safe:.4f} - "
                            f"rank: {total_rank/safe:.4f} - "
                            f"wr: {total_wr/safe:.5f}"
                         )
                progress.update(task, advance = 1, suffix = suffix)

        return {
                    "loss":      total_loss / max(n_done, 1),
                    "rank_loss": total_rank / max(n_done, 1),
                    "wr_loss":   total_wr   / max(n_done, 1),
               }

    def _eval_epoch(
                        self,
                        name:    str,
                        split:   TGBSplit,
                        sampler: Any,
                        console: Console
                    ) -> Tuple[float, float, float]:
        buckets      = list(_iter_buckets(split, self.cfg.batch_events))
        num_buckets  = len(buckets)
        all_logits:  List[tf.Tensor]  = []
        all_sizes:   List[np.ndarray] = []
        last_t       = 0
        event_offset = 0

        # Metrics computed on the final bucket and stored for return
        _mrr = _ap = _auc = 0.0

        with _progress_ctx(console, name, num_buckets) as (progress, task):
            for b_idx, (src_b, dst_b, ts_b, ef_b, _) in enumerate(buckets):
                data = {
                            "src":          src_b,
                            "dst":          dst_b,
                            "ts":           ts_b,
                            "edge_feat":    ef_b,
                            "last_t":       last_t,
                            "sampler":      sampler,
                            "event_offset": event_offset,
                        }
                result = self.test_step(data)
                if src_b.shape[0] > 0:
                    last_t        = int(ts_b.max())
                    event_offset += src_b.shape[0]
                if result.get("logits") is not None:
                    all_logits.append(result["logits"])
                    all_sizes.append(result["sizes"])

                if b_idx == num_buckets - 1 and all_logits:
                    lg   = tf.concat(all_logits, axis = 0)
                    sz   = tf.cast(np.concatenate(all_sizes), tf.int32)
                    _mrr = float(compute_mrr(lg, sz))
                    _ap  = float(compute_ap( lg, sz))
                    _auc = float(compute_auc(lg, sz))
                    suffix = f"MRR: {_mrr:.4f}  AP: {_ap:.4f}  AUC: {_auc:.4f}"
                else:
                    suffix = "..."
                progress.update(task, advance = 1, suffix = suffix)

        if not all_logits:
            return 0.0, 0.0, 0.0
        return _mrr, _ap, _auc

    # ------------------------------------------------------------------

    def _accumulate(self, grads: List[Optional[tf.Tensor]]) -> None:
        if self._accum_grads is None:
            self._accum_grads = [tf.zeros_like(g) if g is not None else None for g in grads]
        for i, g in enumerate(grads):
            if g is not None and self._accum_grads[i] is not None:
                self._accum_grads[i] = self._accum_grads[i] + g
        self._accum_count += 1

    def _apply_gradients(self) -> None:
        if self._accum_grads is None or self._accum_count == 0:
            return
        n         = float(self._accum_count)
        avg_grads = [(g / n if g is not None else None) for g in self._accum_grads]

        if self.cfg.clip_norm is not None:
            valid_g = [g for g in avg_grads if g is not None]
            valid_w = [w for g, w in zip(avg_grads, self.model.trainable_weights) if g is not None]
            clipped, _ = tf.clip_by_global_norm(valid_g, self.cfg.clip_norm)
            pairs = list(zip(clipped, valid_w))
        else:
            pairs = [(g, w) for g, w in zip(avg_grads, self.model.trainable_weights) if g is not None]

        self._gsn_optimizer.apply_gradients(pairs)
        self._accum_grads = None
        self._accum_count = 0

    def _save(self, epoch: int, val_mrr: float = 0.0, is_best: bool = False) -> None:
        d = Path(self.cfg.weights_dir)
        d.mkdir(parents = True, exist_ok = True)

        # self.model.save_weights(str(d / f"epoch_{epoch:03d}.weights.h5"))
        if is_best:
            self.model.save_weights(str(d / "best.weights.h5"))

        with open(d / "config.json", "w", encoding = "utf-8") as f:
            json.dump(self.model.get_config(), f, indent = 4)
