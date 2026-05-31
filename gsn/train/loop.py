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

import math
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
    from ..state.conv_cache import ConvCacheTable
    from ..state.activity_buffers import NodeActivityBuffers
    from ..state.pair_recurrence import PairRecurrenceBuffers
    from ..state.query_history import QueryHistoryBuffers
    from ..layers.gsn_block import GSNBlock, PersistentGSNBlock
    from ..layers.adaptive_commit_gate import (
                                                AdaptiveCommitGate,
                                                alpha_prior_loss,
                                                alpha_saturation_loss,
                                               )
    from ..layers.link_predictor import LinkPredictor
    from ..datasets.tgb_loader import TGBSplit
    from ..datasets.negative_sampling import (
                                                    TGBStyleTrainNegativeSampler,
                                                    TrainNegativeSampler,
                                                )
    from ..train.loss import ranking_loss, write_penalty_loss
    from ..train.metrics import compute_mrr, compute_mrr_1v1_sum_count, compute_ap, compute_auc
except Exception as e:
    cprint("[loop.py] Failed with relative import. Trying with absolute import.", "yellow")
    from gsn.snapshot import Snapshot
    from gsn.state.table import DenseStateTable
    from gsn.state.conv_cache import ConvCacheTable
    from gsn.state.activity_buffers import NodeActivityBuffers
    from gsn.state.pair_recurrence import PairRecurrenceBuffers
    from gsn.state.query_history import QueryHistoryBuffers
    from gsn.layers.gsn_block import GSNBlock, PersistentGSNBlock
    from gsn.layers.adaptive_commit_gate import (
                                                  AdaptiveCommitGate,
                                                  alpha_prior_loss,
                                                  alpha_saturation_loss,
                                                 )
    from gsn.layers.link_predictor import LinkPredictor
    from gsn.datasets.tgb_loader import TGBSplit
    from gsn.datasets.negative_sampling import (
                                                    TGBStyleTrainNegativeSampler,
                                                    TrainNegativeSampler,
                                                )
    from gsn.train.loss import ranking_loss, write_penalty_loss
    from gsn.train.metrics import compute_mrr, compute_mrr_1v1_sum_count, compute_ap, compute_auc


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
                    num_nodes:            int,
                    hidden:               int   = 128,
                    num_heads:            int   = 4,
                    head_dim:             int   = 32,
                    state_dim:            int   = 16,
                    sequence_length:      int   = 1,
                    num_chunks:           int   = 1,
                    num_layers:           int   = 1,
                    embed_dim:            int   = 128,
                    scorer:               str   = "mlp",
                    commit_alpha:         float = 0.2,
                    time_feat_dim:        int   = 8,
                    time_scale:           float = 86400.0,
                    edge_gate_hidden:     int   = 32,
                    dropout:              float = 0.0,
                    self_loops:           bool  = True,
                    pre_message:          bool  = False,
                    run_ssm_in_step_mode: bool = True,      # whether to do the forward pass in one shot over the entire sequence
                                                            # or to to it one-step at a time for L steps; L is seq. length.
                                                            
                    conv_cache:           bool  = False,    # if using step mode, then setting this flag dictates whether the conv
                                                            # cache is propagated or initialized afresh every time.
                    conv_cache_dt_decay: Optional[float] = None,
                    intra_bucket_seq: bool  = False,
                    conv1d_kernel_size: int = 4,
                    noise_scale:      float = 0.005,
                    id_dim:           int   = 0,
                    temp:             float = -2.624,
                    pair_recurrence:  bool  = False,
                    pair_recurrence_dim: int = 16,
                    pair_recurrence_tau: Optional[float] = None,
                    pair_recurrence_undirected: bool = False,
                    pair_recurrence_reset_per_epoch: bool = True,
                    query_history: bool = False,
                    query_history_k: int = 16,
                    query_history_dim: int = 16,
                    query_history_tau: Optional[float] = None,
                    query_history_undirected: bool = True,
                    query_history_reset_per_epoch: bool = True,
                    # ---- Adaptive commit parameters ----
                    commit_mode:      str   = "uniform",
                    gate_hidden:      int   = 64,
                    gate_layers:      int   = 2,
                    alpha_min:        float = 1e-4,
                    alpha_max:        float = 0.999,
                    lambda_min:       float = 1e-5,
                    exposure_delta0:  float = 0.05,
                    exposure_cn:      float = 0.25,
                    # ------------------------------------
                    name:             str   = None,
                    **kwargs
                ):
        super().__init__(name = name, **kwargs)
        assert hidden == num_heads * head_dim

        # Store all constructor params for serialization
        self._num_heads        = int(num_heads)
        self._head_dim         = int(head_dim)
        self._state_dim        = int(state_dim)
        self._sequence_length  = int(sequence_length)
        self._num_chunks       = int(num_chunks)
        self._num_layers       = int(num_layers)
        self._embed_dim        = int(embed_dim)
        self._scorer           = str(scorer)
        self._commit_alpha     = float(commit_alpha)
        self._time_feat_dim    = int(time_feat_dim)
        self._time_scale       = float(time_scale)
        self._edge_gate_hidden = int(edge_gate_hidden)
        self._dropout          = float(dropout)
        self._self_loops       = bool(self_loops)
        self._pre_message      = bool(pre_message)
        self._step_mode        = bool(run_ssm_in_step_mode)
        self._conv_cache       = bool(conv_cache)
        self._conv_cache_dt_decay = (
            float(conv_cache_dt_decay) if conv_cache_dt_decay else None
        )
        self._intra_bucket_seq = bool(intra_bucket_seq)
        self._conv1d_kernel_size = int(conv1d_kernel_size)
        self._noise_scale      = float(noise_scale)
        self._id_dim           = int(id_dim)
        self._pair_recurrence  = bool(pair_recurrence)
        self._pair_recurrence_dim = int(pair_recurrence_dim)
        self._pair_recurrence_tau = (
            float(pair_recurrence_tau) if pair_recurrence_tau is not None else None
        )
        self._pair_recurrence_undirected = bool(pair_recurrence_undirected)
        self._pair_recurrence_reset_per_epoch = bool(pair_recurrence_reset_per_epoch)
        self._query_history = bool(query_history)
        self._query_history_k = int(query_history_k)
        self._query_history_dim = int(query_history_dim)
        self._query_history_tau = (
            float(query_history_tau) if query_history_tau is not None else None
        )
        self._query_history_undirected = bool(query_history_undirected)
        self._query_history_reset_per_epoch = bool(query_history_reset_per_epoch)

        if self._step_mode:
            self._sequence_length = 1
            self._num_chunks = 1
        if self._sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if self._num_chunks <= 0:
            raise ValueError("num_chunks must be positive.")
        if self._sequence_length % self._num_chunks != 0:
            raise ValueError(
                f"sequence_length = {self._sequence_length} must be divisible "
                f"by num_chunks = {self._num_chunks}."
            )
        if not self._step_mode:
            if self._conv_cache:
                raise ValueError("conv_cache is only supported when run_ssm_in_step_mode = true.")
            if self._intra_bucket_seq:
                raise ValueError("intra_bucket_seq is only supported when run_ssm_in_step_mode = true.")

        # Adaptive commit
        self._commit_mode      = str(commit_mode)
        self._gate_hidden      = int(gate_hidden)
        self._gate_layers      = int(gate_layers)
        self._alpha_min        = float(alpha_min)
        self._alpha_max        = float(alpha_max)
        self._lambda_min       = float(lambda_min)
        self._exposure_delta0  = float(exposure_delta0)
        self._exposure_cn      = float(exposure_cn)

        self.num_nodes       = int(num_nodes)
        self.hidden          = int(hidden)
        self.state_dim_total = self._num_heads * self._head_dim * self._state_dim

        self.state_tables:      List[DenseStateTable]    = []
        self.conv_cache_tables: List[Optional[ConvCacheTable]] = []
        self.blocks:            List[PersistentGSNBlock] = []
        self.activity_buffers_list: List[Optional[NodeActivityBuffers]] = []
        self.pair_recurrence_buffers: Optional[PairRecurrenceBuffers] = None
        if self._pair_recurrence:
            self.pair_recurrence_buffers = PairRecurrenceBuffers(
                num_nodes = self.num_nodes,
                tau_data = (
                    self._pair_recurrence_tau
                    if self._pair_recurrence_tau is not None else self._time_scale
                ),
                undirected = self._pair_recurrence_undirected,
            )
        self.query_history_buffers: Optional[QueryHistoryBuffers] = None
        if self._query_history:
            self.query_history_buffers = QueryHistoryBuffers(
                num_nodes = self.num_nodes,
                history_k = self._query_history_k,
                tau_data = (
                    self._query_history_tau
                    if self._query_history_tau is not None else self._time_scale
                ),
                undirected = self._query_history_undirected,
            )

        # One shared AdaptiveCommitGate (all layers share the same gate architecture;
        # each could in principle have its own, but one gate is sufficient for now).
        self.commit_gate: Optional[AdaptiveCommitGate] = None
        if self._commit_mode == "adaptive_hazard":
            self.commit_gate = AdaptiveCommitGate(
                                    hidden     = self._gate_hidden,
                                    num_layers = self._gate_layers,
                                    alpha_min  = self._alpha_min,
                                    alpha_max  = self._alpha_max,
                                    lambda_min = self._lambda_min,
                                    delta0     = self._exposure_delta0,
                                    cn         = self._exposure_cn,
                                    name       = "commit_gate",
                               )

        for i in range(self._num_layers):
            table = DenseStateTable(
                                        num_entities = self.num_nodes,
                                        state_dim    = self.state_dim_total,
                                        name         = f"state_table_{i}"
                                    )

            # Activity buffers are created with placeholder tau_data=1.0; the
            # Trainer calls setup_adaptive_commit() before training to set the
            # real tau_data and beta computed from the training data.
            act_buf: Optional[NodeActivityBuffers] = None
            if self._commit_mode == "adaptive_hazard":
                act_buf = NodeActivityBuffers(
                                num_nodes = self.num_nodes,
                                tau_data  = 1.0,     # overwritten by Trainer
                                beta      = 0.05,    # overwritten by Trainer
                          )

            block = GSNBlock(
                                hidden              = hidden,
                                num_heads           = self._num_heads,
                                head_dim            = self._head_dim,
                                state_dim           = self._state_dim,
                                sequence_length     = self._sequence_length,
                                num_chunks          = self._num_chunks,
                                time_feat_dim       = self._time_feat_dim,
                                time_scale          = self._time_scale,
                                edge_gate_hidden    = self._edge_gate_hidden,
                                dropout             = self._dropout,
                                self_loops          = self._self_loops,
                                pre_message         = self._pre_message,
                                conv_cache          = self._conv_cache,
                                conv_cache_dt_decay = self._conv_cache_dt_decay,
                                intra_bucket_seq    = self._intra_bucket_seq,
                                conv1d_kernel_size  = self._conv1d_kernel_size,
                                name                = f"gsn_block_{i}"
                            )

            # Persistent conv cache: one ConvCacheTable per layer when enabled.
            # K is taken from the freshly-built Mamba2SSD inside the block so
            # we never duplicate the kernel-size constant.
            conv_cache_table: Optional[ConvCacheTable] = None
            if self._conv_cache:
                conv_cache_table = ConvCacheTable(
                                                    num_entities = self.num_nodes,
                                                    kernel_size  = block.mamba2.conv1d_kernel_size,
                                                    channels     = block.mamba2.xbc_channels,
                                                    name         = f"conv_cache_table_{i}",
                                                 )

            pblock = PersistentGSNBlock(
                                            block             = block,
                                            state_table       = table,
                                            commit_alpha      = self._commit_alpha,
                                            noise_scale       = self._noise_scale,
                                            commit_gate       = self.commit_gate,
                                            activity_buffers  = act_buf,
                                            conv_cache_table  = conv_cache_table,
                                            name              = f"persistent_block_{i}"
                                        )
            self.state_tables.append(table)
            self.conv_cache_tables.append(conv_cache_table)
            self.blocks.append(pblock)
            self.activity_buffers_list.append(act_buf)
            setattr(self, f"_pblock_{i}", pblock)
            setattr(self, f"_table_{i}",  table)
            if conv_cache_table is not None:
                setattr(self, f"_conv_cache_table_{i}", conv_cache_table)

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
                                            pair_feature_dim = (
                                                PairRecurrenceBuffers.feature_dim
                                                if self._pair_recurrence else 0
                                            ),
                                            pair_hidden    = self._pair_recurrence_dim,
                                            query_history_feature_dim = (
                                                QueryHistoryBuffers.feature_dim
                                                if self._query_history else 0
                                            ),
                                            query_history_hidden = self._query_history_dim,
                                        )
        self.ln_out = layers.LayerNormalization(epsilon = 1e-6)

        self.temp = self.add_weight(
                                        name        = "temp",
                                        shape       = (),
                                        trainable   = True,
                                        initializer = keras.initializers.Constant(temp)
                                    )
        
        self.build()

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
                    "num_nodes"             : self.num_nodes,
                    "hidden"                : self.hidden,
                    "num_heads"             : self._num_heads,
                    "head_dim"              : self._head_dim,
                    "state_dim"             : self._state_dim,
                    "sequence_length"       : self._sequence_length,
                    "num_chunks"            : self._num_chunks,
                    "num_layers"            : self._num_layers,
                    "embed_dim"             : self._embed_dim,
                    "scorer"                : self._scorer,
                    "commit_alpha"          : self._commit_alpha,
                    "time_feat_dim"         : self._time_feat_dim,
                    "time_scale"            : self._time_scale,
                    "edge_gate_hidden"      : self._edge_gate_hidden,
                    "dropout"               : self._dropout,
                    "self_loops"            : self._self_loops,
                    "pre_message"           : self._pre_message,
                    "run_ssm_in_step_mode"  : self._step_mode,
                    "conv_cache"            : self._conv_cache,
                    "conv_cache_dt_decay"   : self._conv_cache_dt_decay,
                    "intra_bucket_seq"      : self._intra_bucket_seq,
                    "conv1d_kernel_size"    : self._conv1d_kernel_size,
                    "noise_scale"           : self._noise_scale,
                    "id_dim"                : self._id_dim,
                    "temp"                  : float(self.temp.numpy()),
                    "pair_recurrence"       : self._pair_recurrence,
                    "pair_recurrence_dim"   : self._pair_recurrence_dim,
                    "pair_recurrence_tau"   : self._pair_recurrence_tau,
                    "pair_recurrence_undirected"       : self._pair_recurrence_undirected,
                    "pair_recurrence_reset_per_epoch"  : self._pair_recurrence_reset_per_epoch,
                    "query_history"                    : self._query_history,
                    "query_history_k"                  : self._query_history_k,
                    "query_history_dim"                : self._query_history_dim,
                    "query_history_tau"                : self._query_history_tau,
                    "query_history_undirected"         : self._query_history_undirected,
                    "query_history_reset_per_epoch"    : self._query_history_reset_per_epoch,
                    # Adaptive commit
                    "commit_mode"                      : self._commit_mode,
                    "gate_hidden"                      : self._gate_hidden,
                    "gate_layers"                      : self._gate_layers,
                    "alpha_min"                        : self._alpha_min,
                    "alpha_max"                        : self._alpha_max,
                    "lambda_min"                       : self._lambda_min,
                    "exposure_delta0"                  : self._exposure_delta0,
                    "exposure_cn"                      : self._exposure_cn
                }

    def _build_from_dummy(self, edge_feat_dim: int = None) -> None:
        """Trigger Keras weight creation so load_weights works.

        Important: ``forward()`` alone builds the GSN/state path, but it does
        NOT build the link-prediction scorer.  For MLP / BatchNorm scorers this
        means ``from_pretrained()`` can load the checkpoint before
        ``scorer_head`` has created its variables, so the scorer is left at a
        fresh random initialization when ``score_pairs()`` is first called
        during standalone evaluation.

        Therefore this dummy build must exercise all three paths:

          1. ``forward``        -> state table / GSN blocks / H
          2. ``score_pairs``    -> state_proj / item_proj / LinkPredictor scorer
          3. ``commit_gate``    -> AdaptiveCommitGate MLP (adaptive mode only)
                                   The gate is only called inside _adaptive_commit
                                   (commit=True) and compute_alpha_for_reg, neither
                                   of which fires during a plain forward pass.
                                   A direct dummy call here ensures the Dense weights
                                   are created before load_weights() runs.

        Keep the dummy candidates inside ``dummy.node_ids`` so the
        ``states=...`` local-gather path is also built exactly like training
        and evaluation.
        """
        # Two non-self-loop edges → the intra-bucket per-event branch (used
        # when ``intra_bucket_seq=True``) fires with K_max=1, building the
        # ``event_msg_lin`` kernel under its proper parent scope so the
        # optimizer's variable-known check passes on the first training step.
        # With ``self_loops=False`` (the default), a single src==dst edge gets
        # stripped by ``Snapshot``, leaving E=0 and the per-event branch
        # unbuilt.  Two distinct (src, dst) pairs keep this path active across
        # all configurations.
        dummy = Snapshot.from_events(
                                        src_global = np.asarray([0, 1], dtype = np.int64),
                                        dst_global = np.asarray([1, 0], dtype = np.int64),
                                        timestamps = np.asarray([0, 1], dtype = np.int64),
                                        t_ref      = 2,
                                        dt         = 1.0,
                                        edge_feat  = (
                                                        np.zeros((2, int(edge_feat_dim)), dtype = np.float32)
                                                        if edge_feat_dim is not None else None
                                                     ),
                                    )
        if not self._step_mode:
            dummy = Snapshot.concatenate(
                                        [dummy],
                                        seq_len = self._sequence_length,
                                      )
        H, states = self.forward(dummy, commit = False, training = False)

        dummy_src = np.asarray([0, 0], dtype = np.int64)
        dummy_dst = np.asarray([0, 0], dtype = np.int64)
        _ = self.score_pairs(
                                H,
                                dummy.node_ids,
                                dummy_src,
                                dummy_dst,
                                states = states,
                                pair_current_ts = np.asarray([2, 2], dtype = np.float32),
                                query_history_current_ts = np.asarray([2, 2], dtype = np.float32),
                                training = False,
                            )

        if len(self.scorer_head.weights) == 0:
            raise RuntimeError(
                                "Dummy build failed to create scorer_head weights; checkpoint "
                                "loading would leave the scorer randomly initialized."
                              )

        # Path 3: build the adaptive commit gate MLP.
        # Neither forward(commit=False) nor score_pairs touches the gate, so
        # its Dense layers remain unbuilt and load_weights() would silently
        # skip them, leaving the gate randomly initialized.
        if self.has_adaptive_commit:
            _D = self.state_dim_total
            _ = self.commit_gate(
                    s_old            = tf.zeros([1, _D], dtype = tf.float32),
                    s_new            = tf.zeros([1, _D], dtype = tf.float32),
                    delta_t          = tf.zeros([1],    dtype = tf.float32),
                    event_count      = tf.ones( [1],    dtype = tf.float32),
                    update_count     = tf.ones( [1],    dtype = tf.float32),
                    ema_interarrival = tf.ones( [1],    dtype = tf.float32),
                    tau_data         = 1.0,
                    warmup_eta       = 1.0,
                    alpha0_uniform   = float(self._commit_alpha),
                    training         = False,
                )
            if len(self.commit_gate.weights) == 0:
                raise RuntimeError(
                    "Dummy build failed to create commit_gate weights; checkpoint "
                    "loading would leave the gate randomly initialized."
                )

    @classmethod
    def from_pretrained(
                            cls,
                            path,
                            edge_feat_dim:   int  = None,
                            config_override: dict = None,
                            epoch:           Optional[int] = None,
                        ) -> "GSNLinkPredictor":
        """Load model config and weights from a saved directory.

        Parameters
        ----------
        path            : checkpoint directory containing config.json and
                          best.weights.h5 (or epoch_NNN.weights.h5).
        edge_feat_dim   : edge feature dimension for the dummy build.
        config_override : optional dict merged into the saved config BEFORE
                          constructing the model.  Use this to upgrade a
                          ``commit_mode: uniform`` checkpoint to
                          ``adaptive_hazard`` by passing the adaptive-commit
                          fields from the YAML — the YAML always takes
                          precedence over what was frozen in config.json.
        epoch           : optional 1-indexed epoch selector.  When provided,
                          loads ``epoch_{NNN}.weights.h5`` and the matching
                          ``epoch_{NNN}_activity_buffers.npz`` instead of the
                          best-val-MRR checkpoint.  Requires the run to have
                          been trained with ``save_every_epoch: true``.

        Notes
        -----
        * Activity buffers (last_update_time, update_count, ema_interarrival)
          are plain NumPy arrays, invisible to save_weights().  They are
          restored from ``activity_buffers.npz`` if present; otherwise they
          start from zero with a printed warning.
        * Pair-recurrence buffers are also plain NumPy/Python state and are
          restored from ``pair_recurrence.npz`` (or epoch-specific sibling)
          when ``pair_recurrence=True``.
        * Query-history buffers follow the same pattern via
          ``query_history.npz`` when ``query_history=True``.
        * When upgrading from uniform → adaptive, the gate weights will not
          be present in the checkpoint file; load_weights skips them and they
          stay at their initialised values (correct behaviour: gate starts
          fresh on top of a warm backbone).
        * After load, the function asserts every trainable model variable was
          written to (or was already at the saved value).  This catches
          shape-mismatch silent skips that ``skip_mismatch=True`` would
          otherwise hide.
        """
        path = Path(path)
        with open(path / "config.json", encoding = "utf-8") as f:
            config = json.load(f)

        # YAML overrides win over whatever was frozen in config.json.
        # This is the primary mechanism for upgrading a uniform checkpoint to
        # adaptive_hazard without re-training from scratch.
        if config_override:
            config.update(config_override)

        model = cls(**config)
        # _build_from_dummy creates ALL weight variables (including the gate
        # MLP) before load_weights() runs; without this the gate is skipped.
        model._build_from_dummy(edge_feat_dim = edge_feat_dim)

        if epoch is not None:
            weights_path = path / f"epoch_{int(epoch):03d}.weights.h5"
            buf_path     = path / f"epoch_{int(epoch):03d}_activity_buffers.npz"
            pair_buf_path = path / f"epoch_{int(epoch):03d}_pair_recurrence.npz"
            query_buf_path = path / f"epoch_{int(epoch):03d}_query_history.npz"
            if not weights_path.exists():
                available = sorted(p.name for p in path.glob("epoch_*.weights.h5"))
                raise FileNotFoundError(
                    f"Epoch checkpoint {weights_path.name} not found in {path}. "
                    f"Available epoch checkpoints: {available}. "
                    "Was this run trained with save_every_epoch: true?"
                )
        else:
            best = path / "best.weights.h5"
            buf_path = path / "activity_buffers.npz"
            pair_buf_path = path / "pair_recurrence.npz"
            query_buf_path = path / "query_history.npz"
            if best.exists():
                weights_path = best
            else:
                candidates = sorted(path.glob("epoch_*.weights.h5"))
                if not candidates:
                    raise FileNotFoundError(f"No weights found in {path}")
                weights_path = candidates[-1]
                buf_path = path / weights_path.name.replace(
                    ".weights.h5", "_activity_buffers.npz"
                )
                pair_buf_path = path / weights_path.name.replace(
                    ".weights.h5", "_pair_recurrence.npz"
                )
                query_buf_path = path / weights_path.name.replace(
                    ".weights.h5", "_query_history.npz"
                )

        print(f"\nLoaded from {weights_path}")

        # Snapshot all variables BEFORE load so we can verify every model
        # variable was written (catches silent skips from skip_mismatch=True).
        pre_snapshot = {v.path: v.numpy().copy() for v in model.variables}

        # skip_mismatch=True: if the checkpoint pre-dates the gate (uniform
        # mode save), those variables simply don't exist in the file and are
        # left at their initialised values rather than raising an error.
        model.load_weights(str(weights_path), skip_mismatch = True)

        # Verification: every model variable should either have been written
        # by load_weights (value changed), OR be a known-shared scalar whose
        # saved value happens to match the init.  We report any UNCHANGED
        # TRAINABLE variable whose pre-load value was zero — that pattern is
        # diagnostic of a silently-skipped weight.
        skipped_zeros = []
        pair_unchanged = []
        pair_changed = 0
        query_unchanged = []
        query_changed = 0
        unchanged_total = 0
        for v in model.variables:
            post = v.numpy()
            is_pair_var = "pair_recurrence" in v.path and v.trainable
            is_query_var = "query_history" in v.path and v.trainable
            if np.array_equal(pre_snapshot[v.path], post):
                unchanged_total += 1
                if is_pair_var:
                    pair_unchanged.append(v.path)
                    continue
                if is_query_var:
                    query_unchanged.append(v.path)
                    continue
                if v.trainable and np.abs(post).sum() == 0.0:
                    skipped_zeros.append(v.path)
            elif is_pair_var:
                pair_changed += 1
            elif is_query_var:
                query_changed += 1
        if skipped_zeros:
            print(
                f"  [warning] {len(skipped_zeros)} trainable variable(s) appear "
                "silently skipped by load_weights (unchanged & all-zero after "
                "load).  This usually means a shape mismatch between the model "
                "and the checkpoint.  Affected:"
            )
            for p in skipped_zeros:
                print(f"    - {p}")
        if model.has_pair_recurrence and pair_unchanged and pair_changed == 0:
            print(
                f"  [warning] {len(pair_unchanged)} pair-recurrence scorer "
                "variable(s) were unchanged by load_weights. If this checkpoint "
                "predates pair_recurrence, the auxiliary pair logit starts from "
                "its zero-initialised baseline and must be trained before it can "
                "contribute. Affected:"
            )
            for p in pair_unchanged:
                print(f"    - {p}")
        if model.has_query_history and query_unchanged and query_changed == 0:
            print(
                f"  [warning] {len(query_unchanged)} query-history scorer "
                "variable(s) were unchanged by load_weights. If this checkpoint "
                "predates query_history, the auxiliary history logit starts from "
                "its zero-initialised baseline and must be trained before it can "
                "contribute. Affected:"
            )
            for p in query_unchanged:
                print(f"    - {p}")

        # Restore activity buffers (adaptive mode only)
        if model.has_adaptive_commit:
            if buf_path.exists():
                data = np.load(str(buf_path))
                for i, buf in enumerate(model.activity_buffers_list):
                    if buf is None:
                        continue
                    key_lut = f"last_update_time_{i}"
                    key_uc  = f"update_count_{i}"
                    key_ema = f"ema_interarrival_{i}"
                    key_la  = f"last_alpha_{i}"
                    if key_lut in data:
                        buf.last_update_time[:] = data[key_lut]
                        buf.update_count[:]     = data[key_uc]
                        buf.ema_interarrival[:] = data[key_ema]
                        buf.last_alpha[:]       = data[key_la]
                print(f"  Activity buffers restored from {buf_path.name}")
            else:
                print(
                    f"  [warning] {buf_path.name} not found — "
                    "activity buffers initialised to zero.  "
                    "Gate weights are loaded correctly; only the temporal "
                    "context (delta_t, ema) is missing."
                )

        # Restore pair recurrence buffers (pair-recurrence mode only)
        if model.has_pair_recurrence:
            if pair_buf_path.exists():
                model.pair_recurrence_buffers.load_npz(pair_buf_path)
                print(f"  Pair recurrence buffers restored from {pair_buf_path.name}")
            else:
                print(
                    f"  [warning] {pair_buf_path.name} not found — pair recurrence "
                    "history initialised empty. Scorer weights may be loaded, but "
                    "count/recency features will start from zero history."
                )

        # Restore query history buffers (query-history mode only)
        if model.has_query_history:
            if query_buf_path.exists():
                model.query_history_buffers.load_npz(query_buf_path)
                print(f"  Query history buffers restored from {query_buf_path.name}")
            else:
                print(
                    f"  [warning] {query_buf_path.name} not found — query history "
                    "initialised empty. Scorer weights may be loaded, but "
                    "recent-neighbor features will start from zero history."
                )

        return model

    # ------------------------------------------------------------------

    @property
    def has_pair_recurrence(self) -> bool:
        return self._pair_recurrence and self.pair_recurrence_buffers is not None

    @property
    def pair_recurrence_reset_per_epoch(self) -> bool:
        return self._pair_recurrence_reset_per_epoch

    @property
    def has_query_history(self) -> bool:
        return self._query_history and self.query_history_buffers is not None

    @property
    def query_history_reset_per_epoch(self) -> bool:
        return self._query_history_reset_per_epoch

    def reset_states_all(
        self,
        reset_pair_recurrence: bool = True,
        reset_query_history: bool = True,
    ) -> None:
        for t in self.state_tables:
            t.reset_state()
        for c in self.conv_cache_tables:
            if c is not None:
                c.reset_state()
        for buf in self.activity_buffers_list:
            if buf is not None:
                buf.reset()
        if reset_pair_recurrence and self.pair_recurrence_buffers is not None:
            self.pair_recurrence_buffers.reset()
        if reset_query_history and self.query_history_buffers is not None:
            self.query_history_buffers.reset()

    def update_pair_recurrence(
                                self,
                                src: np.ndarray,
                                dst: np.ndarray,
                                timestamps: np.ndarray,
                              ) -> None:
        if self.pair_recurrence_buffers is not None:
            self.pair_recurrence_buffers.update(src, dst, timestamps)

    def update_query_history(
                            self,
                            src: np.ndarray,
                            dst: np.ndarray,
                            timestamps: np.ndarray,
                          ) -> None:
        if self.query_history_buffers is not None:
            self.query_history_buffers.update(src, dst, timestamps)

    # ------------------------------------------------------------------
    # Adaptive commit setup (called by Trainer before first epoch)
    # ------------------------------------------------------------------

    @property
    def has_adaptive_commit(self) -> bool:
        return self._commit_mode == "adaptive_hazard" and self.commit_gate is not None

    def setup_adaptive_commit(
                                self,
                                tau_data:  float,
                                beta:      float,
                                mean_n:    float = 2.0,
                              ) -> None:
        """Calibrate the adaptive commit gate from training-data statistics.

        Must be called AFTER the dummy build (so Dense weights exist) and
        BEFORE the first training epoch.

        Parameters
        ----------
        tau_data : median inter-event time from training data.
        beta     : EMA decay for the interarrival buffer.
        mean_n   : expected events per node per bucket (used for b_0 init).
        """
        if not self.has_adaptive_commit:
            return

        for buf in self.activity_buffers_list:
            if buf is not None:
                buf.tau_data = float(tau_data)
                buf.beta     = float(beta)
                buf.ema_interarrival[:] = float(tau_data)

        # Initialise gate bias to reproduce commit_alpha at median exposure
        self.commit_gate.initialize_bias(
                             alpha0   = self._commit_alpha,
                             tau_data = tau_data,
                             mean_n   = mean_n,
                         )

        # Pre-compute λ_0 so PersistentGSNBlock can compute prior_alpha
        g = self.commit_gate
        phi_bar = (
            g.delta0
            + float(np.log(2.0))
            + g.cn * float(np.log(1.0 + max(mean_n, 0.0)))
        )
        span = g.alpha_max - g.alpha_min
        alpha0_scaled = max(
            1e-6, min(1.0 - 1e-6, (self._commit_alpha - g.alpha_min) / span)
        )
        lambda0 = -float(np.log(1.0 - alpha0_scaled)) / max(phi_bar, 1e-8)

        for pblock in self.blocks:
            pblock._lambda0 = lambda0

    # ------------------------------------------------------------------
    # Adaptive regularisation losses (called inside GradientTape)
    # ------------------------------------------------------------------

    def compute_alpha_reg_loss(
                                    self,
                                    snap:              "Snapshot",
                                    states:            List[tf.Tensor],
                                    lambda_prior:      float,
                                    lambda_saturation: float,
                                    training:          Optional[bool] = None,
                               ) -> tf.Tensor:
        """Compute prior + saturation regularisation losses for the commit gate.

        Called INSIDE the GradientTape during the snap_pre scoring forward so
        the gate's MLP parameters receive gradients via the novelty features
        (which depend on s_next, which is in the tape).

        Returns scalar tf.Tensor (0.0 if not in adaptive mode).
        """
        if not self.has_adaptive_commit:
            return tf.constant(0.0, dtype = tf.float32)

        total = tf.constant(0.0, dtype = tf.float32)
        for pblock, s_next in zip(self.blocks, states):
            if not pblock.adaptive_mode:
                continue
            alpha, alpha_prior = pblock.compute_alpha_for_reg(snap, s_next, training = training)
            if lambda_prior > 0.0:
                total += tf.cast(lambda_prior, tf.float32) * alpha_prior_loss(alpha, alpha_prior)
            if lambda_saturation > 0.0:
                total += tf.cast(lambda_saturation, tf.float32) * alpha_saturation_loss(alpha)
        return total
    
    def compute_shadow_committed_states_for_score(
                                                    self,
                                                    snap:              "Snapshot",
                                                    states:            List[tf.Tensor],
                                                    lambda_prior:      float,
                                                    lambda_saturation: float,
                                                    training:          Optional[bool] = None
                                                 ) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """Build differentiable committed states used only for scoring.

        This makes rank_loss depend on alpha without mutating the persistent
        DenseStateTable inside the GradientTape.
        """

        if not self.has_adaptive_commit:
            return states, tf.constant(0.0, dtype = tf.float32)

        states_for_score = []
        alpha_reg_loss = tf.constant(0.0, dtype = tf.float32)

        for pblock, s_next in zip(self.blocks, states):
            if not pblock.adaptive_mode:
                states_for_score.append(s_next)
                continue

            s_score, alpha, alpha_prior = pblock.compute_shadow_committed_state(
                                                                                    snap = snap,
                                                                                    s_next = s_next,
                                                                                    training = training,
                                                                                )

            states_for_score.append(s_score)

            if lambda_prior > 0.0:
                alpha_reg_loss += tf.cast(lambda_prior, tf.float32) * alpha_prior_loss(alpha, alpha_prior)

            if lambda_saturation > 0.0:
                alpha_reg_loss += tf.cast(lambda_saturation, tf.float32) * alpha_saturation_loss(alpha)

        return states_for_score, alpha_reg_loss
            
    def build(self, input_shape = None):
        self.built = True
        if input_shape is None:
            return
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
            h, s_next = pblock(snap = snap, commit = commit, training = training, run_step_mode = self._step_mode)
            out = h
            all_states.append(s_next)
        return out, all_states

    # def score_pairs(
    #                     self,
    #                     H:        tf.Tensor,
    #                     node_ids: np.ndarray,
    #                     cand_src: np.ndarray,
    #                     cand_dst: np.ndarray,
    #                     training: Optional[bool] = None
    #                 ) -> tf.Tensor:
    #     """Score (u, v) candidate pairs.

    #     Source : persistent-state projection from each state table.
    #     Dest   : snapshot embedding H (zeroed for out-of-snapshot nodes)
    #              + optional learned ID embedding.

    #     The original code used id_to_local.get(v, 0) as a fallback, which
    #     silently gave every out-of-snapshot negative the embedding of whichever
    #     node happened to sit at local index 0.  With batch_events=128 and
    #     999 negatives per positive, ~99 % of negative destinations fell into
    #     this path and received *identical* corrupt embeddings.

    #     The fix: build an in-snapshot boolean mask and multiply H by it, so
    #     out-of-snapshot nodes get h_dst=0.  The id_emb path (when id_dim>0)
    #     still supplies a meaningful per-node signal for those positions via
    #     item_proj's last id_dim columns, keeping them distinguishable.
    #     Gradient flow through H (and therefore through the SSM) is preserved.
    #     """
    #     id_to_local  = {int(gid): i for i, gid in enumerate(node_ids)}

    #     # 1-for-in-snapshot, 0-for-out-of-snapshot  [B, 1]
    #     in_snap      = np.array([int(v) in id_to_local for v in cand_dst],
    #                             dtype = np.float32)
    #     in_snap_tf   = tf.constant(in_snap, dtype = tf.float32)[:, None]

    #     dst_local    = np.array([id_to_local.get(int(v), 0) for v in cand_dst],
    #                             dtype = np.int32)
    #     dst_local_tf = tf.cast(dst_local, tf.int32)

    #     u_ids_tf = tf.cast(cand_src, tf.int32)
    #     u_states = tf.concat(
    #         [t.get(u_ids_tf) for t in self.state_tables], axis = -1
    #     )
    #     u_z = self.state_proj(u_states)

    #     h_dst = tf.gather(H, dst_local_tf) * in_snap_tf   # zero out invalid
    #     if self.id_dim > 0:
    #         v_ids_tf = tf.cast(cand_dst, tf.int32)
    #         id_emb   = self.node_id_emb(v_ids_tf, training = training)
    #         id_emb   = self.id_emb_drop(id_emb,   training = training)
    #         h_dst    = tf.concat([h_dst, tf.cast(id_emb, h_dst.dtype)], axis = -1)
    #     v_z = self.item_proj(h_dst)

    #     logits = self.scorer_head(u_z, v_z, training = training)
    #     denom  = tf.cast(tf.nn.softplus(self.temp) + 1e-6, tf.float32)
    #     return tf.cast(logits, tf.float32) / denom
    
    
    def score_pairs(
                        self,
                        H,
                        node_ids,
                        cand_src,
                        cand_dst,
                        states = None,
                        pair_features = None,
                        pair_current_ts = None,
                        query_history_features = None,
                        query_history_current_ts = None,
                        training = None,
                    ):
        id_to_local = {int(gid): i for i, gid in enumerate(node_ids)}

        src_local = np.array([id_to_local[int(u)] for u in cand_src], dtype = np.int32)
        dst_local = np.array([id_to_local.get(int(v), 0) for v in cand_dst], dtype = np.int32)

        src_local_tf = tf.cast(src_local, tf.int32)
        dst_local_tf = tf.cast(dst_local, tf.int32)

        if states is None:
            u_ids_tf = tf.cast(cand_src, tf.int32)
            u_states = tf.concat([t.get(u_ids_tf) for t in self.state_tables], axis = -1)
        else:
            u_states = tf.concat(
                                    [tf.gather(s, src_local_tf) for s in states],
                                    axis = -1,
                                )

        u_z = self.state_proj(u_states)

        h_dst = tf.gather(H, dst_local_tf)

        if self.id_dim > 0:
            v_ids_tf = tf.cast(cand_dst, tf.int32)
            id_emb = self.node_id_emb(v_ids_tf, training = training)
            id_emb = self.id_emb_drop(id_emb, training = training)
            h_dst = tf.concat([h_dst, tf.cast(id_emb, h_dst.dtype)], axis = -1)

        v_z = self.item_proj(h_dst)

        pair_features_tf = None
        if self.has_pair_recurrence:
            if pair_features is None:
                if pair_current_ts is None:
                    raise ValueError(
                        "pair_current_ts or pair_features must be provided when "
                        "pair_recurrence is enabled"
                    )
                pair_features = self.pair_recurrence_buffers.get_features(
                    cand_src, cand_dst, pair_current_ts
                )
            pair_features_tf = tf.cast(pair_features, tf.float32)
        query_history_features_tf = None
        if self.has_query_history:
            if query_history_features is None:
                if query_history_current_ts is None:
                    query_history_current_ts = pair_current_ts
                if query_history_current_ts is None:
                    raise ValueError(
                        "query_history_current_ts or query_history_features must "
                        "be provided when query_history is enabled"
                    )
                query_history_features = self.query_history_buffers.get_features(
                    cand_src, cand_dst, query_history_current_ts
                )
            query_history_features_tf = tf.cast(query_history_features, tf.float32)

        logits = self.scorer_head(
            u_z,
            v_z,
            pair_features = pair_features_tf,
            query_history_features = query_history_features_tf,
            training = training,
        )
        denom = tf.cast(tf.nn.softplus(self.temp) + 1e-6, tf.float32)
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

    Two modes
    ---------
      • ``bucket_size > 0``  — fixed-count buckets.  Events are sliced in
        contiguous chunks of size ``bucket_size`` (the final chunk may be
        shorter).  This is the original behaviour.

      • ``bucket_size == -1`` — **timestamp-grouped** buckets.  Every
        contiguous run of events sharing the same timestamp is yielded as
        one bucket.  This makes one SSM step correspond to one instant in
        time and, combined with the two-snapshot scoring protocol,
        structurally eliminates the within-timestamp leak channel: events
        at time `t` never appear in the `snap_pre` used to score other
        events at time `t`.  Highly recommended on datasets with heavy
        timestamp clustering (e.g. tgbl-contacts).

        Caveat: bucket sizes become variable (Contacts: 1..1746 events per
        bucket).  When using this mode, prefer ``accumulate_every: 1`` so
        the optimiser sees one timestamp at a time rather than aggregating
        an unpredictable number of events per step.

    The ``-1`` path asserts that ``split.ts`` is sorted in non-decreasing
    order; this is the TGB / DyGLib loader convention but is checked
    explicitly so a future loader change cannot silently corrupt buckets.
    """
    E = len(split.src)
    if E == 0:
        return

    if bucket_size == -1:
        ts = np.asarray(split.ts)
        # Loader convention: chronological order.  Assert loudly rather
        # than silently producing wrong buckets if it ever changes.
        if E > 1:
            assert bool(np.all(np.diff(ts) >= 0)), (
                "_iter_buckets(bucket_size=-1) requires split.ts to be "
                "sorted in non-decreasing order; got an out-of-order ts "
                "array.  Re-sort the split before iterating."
            )

        # Boundaries between distinct timestamps.  `np.diff(ts) != 0` is
        # the indicator at each adjacent pair; +1 turns it into the start
        # index of each new group.  Bracket with 0 and E.
        changes    = np.flatnonzero(np.diff(ts) != 0) + 1
        boundaries = np.concatenate([[0], changes, [E]]) if changes.size > 0 \
                     else np.array([0, E], dtype = np.int64)

        for i in range(boundaries.size - 1):
            sl = slice(int(boundaries[i]), int(boundaries[i + 1]))
            yield (
                    split.src[sl],
                    split.dst[sl],
                    split.ts[sl],
                    split.edge_feat[sl] if split.edge_feat is not None else None,
                    split.neg_dst[sl]   if split.neg_dst   is not None else None
                  )
        return

    # Fixed-count path (original behaviour).
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


def _build_pre_snapshot(
                            prev_src:           np.ndarray,
                            prev_dst:           np.ndarray,
                            prev_ts:            np.ndarray,
                            prev_edge_feat:     Optional[np.ndarray],
                            extra_node_ids:     np.ndarray,
                            t_ref:              int,
                            last_t:             int,
                            edge_feat_template: Optional[np.ndarray] = None,
                       ) -> Optional[Snapshot]:
    """Build a leakage-free pre-bucket snapshot for the scoring forward pass.

    Two design choices, both essential for correctness:

      1. **Edges come from the PREVIOUS bucket only** (or empty for the very
         first bucket of a phase).  The current bucket's positive events are
         deliberately excluded so that `H[v_positive]` is *not* informed by
         a freshly-added edge `(u, v)` — that is the same-bucket leak that
         drove `MRR/AP/AUC = 1.0000` on tgbl-contacts.

      2. **`node_ids` is the union of previous-bucket nodes AND all
         candidate nodes** (cand_src ∪ cand_dst ∪ current bucket src/dst).
         This makes `score_pairs`'s `in_snap` mask universally 1 over
         candidates, removing the binary "positive is in-snap, negatives
         are out-of-snap" signal that Bug 2's fix had unintentionally
         turned into a perfect separator at `batch_events = 1`.

    For the first bucket of a phase `prev_src/prev_dst` are empty and the
    returned snapshot has no edges — message-passing parameters get no
    gradient that bucket, but the persistent state path still trains.
    For every subsequent bucket the snapshot has 1+ edges (the previous
    bucket's events), so message-passing weights receive a real gradient
    signal from a leakage-free graph context.

    Build-shape invariance
    ----------------------
    To keep Keras's first-call Dense build dimensionally consistent with
    every subsequent forward pass, the empty-prev branch must produce a
    snapshot whose `edge_ts` and `edge_feat` *schemas* match what the
    populated-prev branch (and the post-bucket `snap_end`) will later
    feed into `GSNBlock`'s edge_gate.  Concretely:

      • `edge_ts` is ALWAYS a (possibly empty) `np.int64` array, never
        `None`.  Without this, the first-bucket forward built
        `edge_gate.d1` with input dim `2*hidden`, and the very next
        commit forward (which carries `edge_ts`) hit a shape mismatch
        of `2*hidden + time_feat_dim`.

      • `edge_feat`, when the dataset has edge features, is a `(0, F_e)`
        empty array of the template's dtype rather than `None`.  Same
        reasoning — keeps F_gate stable across the first commit.

    `edge_feat_template` should be the current bucket's `ef_b` (or any
    other sample edge_feat from the dataset).  Pass `None` for
    feature-less datasets like tgbl-contacts.
    """
    prev_src = np.asarray(prev_src, dtype = np.int64).reshape(-1)
    prev_dst = np.asarray(prev_dst, dtype = np.int64).reshape(-1)
    extra    = np.asarray(extra_node_ids, dtype = np.int64).reshape(-1)

    all_ids = np.unique(np.concatenate([prev_src, prev_dst, extra])) if (
                            prev_src.size + prev_dst.size + extra.size > 0
                        ) else np.empty(0, dtype = np.int64)
    if all_ids.size == 0:
        return None

    id_to_local = {int(gid): i for i, gid in enumerate(all_ids)}
    N           = int(all_ids.size)

    if prev_src.size > 0:
        edge_src = np.fromiter(
                                  (id_to_local[int(u)] for u in prev_src),
                                  dtype = np.int32, count = prev_src.size,
                              )
        edge_dst = np.fromiter(
                                  (id_to_local[int(v)] for v in prev_dst),
                                  dtype = np.int32, count = prev_dst.size,
                              )
        edge_ts  = np.asarray(prev_ts, dtype = np.int64)
        edge_ef  = prev_edge_feat
    else:
        edge_src = np.empty(0, dtype = np.int32)
        edge_dst = np.empty(0, dtype = np.int32)
        # Always non-None so the first forward builds edge_gate with the
        # same F_gate as every subsequent forward (which carries edge_ts).
        edge_ts  = np.empty(0, dtype = np.int64)
        if edge_feat_template is None:
            edge_ef = None
        else:
            tpl = np.asarray(edge_feat_template)
            if tpl.ndim == 2 and tpl.shape[1] > 0:
                # Empty (0, F_e) with the dataset's edge-feature dtype.
                edge_ef = np.empty((0, int(tpl.shape[1])), dtype = tpl.dtype)
            else:
                # 1-D template, or zero-width — caller treats this as
                # "no edge features", same as `None`.
                edge_ef = None

    dt = max(float(t_ref - last_t), 1.0)
    return Snapshot(
                       node_ids  = all_ids,
                       edge_src  = edge_src,
                       edge_dst  = edge_dst,
                       num_nodes = N,
                       t_ref     = int(t_ref),
                       dt        = float(dt),
                       edge_feat = edge_ef,
                       edge_ts   = edge_ts,
                       x         = None,
                  )


def _empty_snapshot(
                       node_ids:               np.ndarray,
                       t_ref:                  int,
                       last_t:                 int,
                       edge_feat_template:     Optional[np.ndarray] = None,
                   ) -> Snapshot:
    node_ids = np.asarray(node_ids, dtype = np.int64).reshape(-1)
    edge_feat = None
    if edge_feat_template is not None:
        tpl = np.asarray(edge_feat_template)
        if tpl.ndim == 2 and tpl.shape[1] > 0:
            edge_feat = np.empty((0, int(tpl.shape[1])), dtype = tpl.dtype)

    return Snapshot(
                   node_ids  = np.unique(node_ids).astype(np.int64),
                   edge_src  = np.empty(0, dtype = np.int32),
                   edge_dst  = np.empty(0, dtype = np.int32),
                   num_nodes = int(np.unique(node_ids).shape[0]),
                   t_ref     = int(t_ref),
                   dt        = max(float(t_ref - last_t), 1.0),
                   edge_feat = edge_feat,
                   edge_ts   = np.empty(0, dtype = np.int64),
                   x         = None,
                  )


def _build_sequence_snapshot(
                               src:                np.ndarray,
                               dst:                np.ndarray,
                               ts:                 np.ndarray,
                               edge_feat:          Optional[np.ndarray],
                               t_ref:              int,
                               last_t:             int,
                               seq_len:            int,
                               extra_node_ids:     Optional[np.ndarray] = None,
                               edge_feat_template: Optional[np.ndarray] = None,
                           ) -> Optional[Snapshot]:
    src = np.asarray(src, dtype = np.int64).reshape(-1)
    dst = np.asarray(dst, dtype = np.int64).reshape(-1)
    ts  = np.asarray(ts,  dtype = np.int64).reshape(-1)
    if src.shape[0] > seq_len:
        raise ValueError(
            f"Cannot build a sequence snapshot with {src.shape[0]} events "
            f"and seq_len = {seq_len}."
        )

    extra = (
                np.asarray(extra_node_ids, dtype = np.int64).reshape(-1)
                if extra_node_ids is not None else np.empty(0, dtype = np.int64)
            )

    if src.shape[0] == 0:
        if extra.size == 0:
            return None
        empty = _empty_snapshot(
                               node_ids           = extra,
                               t_ref              = t_ref,
                               last_t             = last_t,
                               edge_feat_template = edge_feat_template,
                             )
        return Snapshot.concatenate([empty], seq_len = seq_len)

    snapshots = []
    prev_t = int(last_t)
    for i in range(src.shape[0]):
        ti = int(ts[i])
        ef_i = None if edge_feat is None else np.asarray(edge_feat)[i : i + 1]
        snapshots.append(
            Snapshot.from_events(
                               src_global = src[i : i + 1],
                               dst_global = dst[i : i + 1],
                               timestamps = ts[i : i + 1],
                               t_ref      = ti,
                               dt         = max(float(ti - prev_t), 1.0),
                               edge_feat  = ef_i,
                             )
        )
        prev_t = ti

    snap = Snapshot.concatenate(
                               snapshots,
                               seq_len        = seq_len,
                               extra_node_ids = extra,
                             )
    snap.t_ref = int(t_ref)
    snap.dt = max(float(t_ref - last_t), 1.0)
    return snap


def _build_pre_sequence_snapshot(
                                   prev_src:           np.ndarray,
                                   prev_dst:           np.ndarray,
                                   prev_ts:            np.ndarray,
                                   prev_edge_feat:     Optional[np.ndarray],
                                   extra_node_ids:     np.ndarray,
                                   t_ref:              int,
                                   last_t:             int,
                                   seq_len:            int,
                                   edge_feat_template: Optional[np.ndarray] = None,
                               ) -> Optional[Snapshot]:
    return _build_sequence_snapshot(
                                   src                = prev_src,
                                   dst                = prev_dst,
                                   ts                 = prev_ts,
                                   edge_feat          = prev_edge_feat,
                                   t_ref              = t_ref,
                                   last_t             = last_t,
                                   seq_len            = seq_len,
                                   extra_node_ids     = extra_node_ids,
                                   edge_feat_template = edge_feat_template,
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
    initial_epoch:    int            = 0
    batch_events:     int            = 20_000
    accumulate_every: int            = 1
    train_neg_sampler: str           = "base"
    train_neg_per_pos:    int            = 1
    val_test_neg_per_pos: int            = 49   # -1 = all negatives (requires precomputed neg_dst)
    seed:             int            = 1337
    weights_dir:      Optional[str]  = None
    save_every_epoch: bool           = False  # write epoch_NNN.weights.h5 + buffers every epoch (besides best-by-val-MRR)
    train_on_val:     bool           = False  # If True, val events are concatenated into training data (regime-3 retrain-on-train+val with frozen hyperparameters). The flag is informational here — the actual merge is performed upstream in examples/train.py. Val MRR in subsequent logs becomes a memorization check, NOT a model-selection signal.
    # ---- Adaptive commit trainer settings ----
    # Active only when model.commit_mode = "adaptive_hazard".
    lambda_alpha_prior:      float = 1e-3   # weight of prior regularisation loss
    lambda_alpha_saturation: float = 1e-4   # weight of saturation penalty
    alpha_warmup_epochs:     int   = 2      # epochs before gate is fully active


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
                        eval_train: Optional[TGBSplit] = None,
                        **kwargs
                ):
        super().__init__(**kwargs)
        self.model       = model
        self.cfg         = cfg
        self.train_split = train
        self.val_split   = val
        self.test_split  = test
        self.meta        = meta

        # `eval_train` is the ORIGINAL (un-merged) train split, supplied by the
        # CLI launcher when `train_on_val=True` re-binds `train` to a merged
        # stream.  Falls back to `train` for normal (un-merged) runs.  Used to
        # define inductive-NSS cutoffs so they match the standalone evaluator.
        self._eval_train = eval_train if eval_train is not None else train

        if not self.model._step_mode:
            if cfg.batch_events <= 0:
                raise ValueError(
                    "Sequence mode requires trainer.batch_events to be a positive "
                    "fixed sequence length."
                )
            if int(cfg.batch_events) != int(self.model._sequence_length):
                raise ValueError(
                    f"trainer.batch_events = {cfg.batch_events} must match "
                    f"model.sequence_length = {self.model._sequence_length} "
                    "when run_ssm_in_step_mode = false."
                )

        AdamW = getattr(keras.optimizers, "AdamW", None)
        if AdamW is None and hasattr(keras.optimizers, "experimental"):
            AdamW = getattr(keras.optimizers.experimental, "AdamW", None)
        if AdamW is not None:
            self._gsn_optimizer = AdamW(
                                            learning_rate = cfg.lr,
                                            beta_1        = cfg.beta_1,
                                            beta_2        = cfg.beta_2,
                                            weight_decay  = cfg.weight_decay,
                                        )
        else:
            if float(cfg.weight_decay) != 0.0:
                raise RuntimeError(
                    "This TensorFlow/Keras build does not provide AdamW; "
                    "set trainer.weight_decay = 0.0 or install TensorFlow >= 2.15."
                )
            self._gsn_optimizer = keras.optimizers.Adam(
                                                            learning_rate = cfg.lr,
                                                            beta_1        = cfg.beta_1,
                                                            beta_2        = cfg.beta_2,
                                                        )

        dst_cands = np.unique(train.dst) if meta.get("bipartite", False) else None
        
        if cfg.train_neg_sampler == "tgb-style":
            self.neg_sampler = TGBStyleTrainNegativeSampler(
                                                                train_src      = train.src,
                                                                train_dst      = train.dst,
                                                                train_ts       = train.ts,
                                                                num_nodes      = model.num_nodes,
                                                                num_neg_e      = cfg.train_neg_per_pos,
                                                                seed           = cfg.seed,
                                                                dst_candidates = dst_cands
                                                            )
        elif cfg.train_neg_sampler == "base":
            self.neg_sampler = TrainNegativeSampler(
                                                        num_nodes = model.num_nodes,
                                                        neg_per_pos = cfg.train_neg_per_pos,
                                                        seed = cfg.seed,
                                                        dst_pool = dst_cands
                                                    )
        else:
            raise AttributeError(f"Invalid Train Negative Sampler: {cfg.train_neg_sampler}. "
                                 "Valid samplers are: 'tgb-style' or 'base'.")

        # ------------------------------------------------------------------
        # DyGLib / DyGMamba-aligned eval samplers (1 neg / pos).
        #
        # Trainer evaluation must use the same sampler instances and the same
        # snap_pre packing as the standalone CLI evaluator (`evaluate.py`).
        # Otherwise the per-epoch logged numbers drift from `python
        # examples/evaluate.py --checkpoint ...`.
        #
        # Pool: unique destinations seen anywhere in (train ∪ val ∪ test) —
        # the "dyglib" pool used by the CLI by default.
        # Seeds: 0/2 (DyGLib val/test) offset by cfg.seed.
        # Inductive: DyGLib edge/time sampler; required so snap_pre.node_ids
        # contains the same `extras` set as the standalone (otherwise scores
        # diverge by a few thousandths).
        # ------------------------------------------------------------------
        from gsn.train.eval import DyGLibRandomNegativeSampler, DyGLibInductiveNegativeSampler
        eval_train = self._eval_train
        full_src = np.concatenate([
                                    eval_train.src.astype(np.int64),
                                    val.src.astype(np.int64),
                                    test.src.astype(np.int64)
                                  ])
        full_dst = np.concatenate([
                                    eval_train.dst.astype(np.int64),
                                    val.dst.astype(np.int64),
                                    test.dst.astype(np.int64)
                                  ])
        full_ts  = np.concatenate([eval_train.ts, val.ts, test.ts])
        random_dst_pool = np.unique(full_dst).astype(np.int64)

        val_seed  = int(cfg.seed)
        test_seed = int(cfg.seed) + 2
        self.val_random_sampler  = DyGLibRandomNegativeSampler(random_dst_pool, seed = val_seed)
        self.test_random_sampler = DyGLibRandomNegativeSampler(random_dst_pool, seed = test_seed)
        self.val_inductive_sampler = DyGLibInductiveNegativeSampler(
                                                                        full_src           = full_src,
                                                                        full_dst           = full_dst,
                                                                        full_ts            = full_ts,
                                                                        last_observed_time = float(eval_train.ts[-1]),
                                                                        seed               = val_seed,
                                                                    )
        self.test_inductive_sampler = DyGLibInductiveNegativeSampler(
                                                                        full_src           = full_src,
                                                                        full_dst           = full_dst,
                                                                        full_ts            = full_ts,
                                                                        last_observed_time = float(val.ts[-1]),
                                                                        seed               = test_seed,
                                                                    )

        self._accum_grads: Optional[List[tf.Tensor]] = None
        self._accum_count  = 0

        # Cache of the previous bucket's events used to build the leakage-free
        # pre-snapshot for the NEXT bucket's scoring forward pass.  Reset to
        # `None` at the top of every phase (train epoch, val pass, test pass)
        # so the first bucket of each phase has an edge-less pre-snapshot.
        # See `_build_pre_snapshot` for the protocol.
        self._prev_bucket: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = None

        # ------------------------------------------------------------------
        # Adaptive commit initialisation
        # ------------------------------------------------------------------
        if model.has_adaptive_commit:
            tau_data, beta = self._compute_tau_and_beta(train)
            cprint(
                    f"[Adaptive commit] tau_data={tau_data:.4g}  beta={beta:.4g}",
                    "cyan",
                  )
            # setup_adaptive_commit is deferred to fit() so the dummy build
            # (which creates the gate's Dense weights) has already run by then.
            self._pending_tau_data = float(tau_data)
            self._pending_beta     = float(beta)
        else:
            self._pending_tau_data = None
            self._pending_beta     = None

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_tau_and_beta(train: TGBSplit) -> tuple:
        """Compute dataset-level tau_data and EMA beta from the training split.

        tau_data  = median of **per-node** inter-arrival times — the gap
                    between a node's own consecutive appearances.  This is the
                    correct reference scale for the hazard gate: φ should be
                    O(1) for a "typical" gap, which requires τ to match the
                    node's own event clock.  The old global-timestamp-gap
                    estimate underestimates τ, inflates φ, and drives the gate
                    into α-saturation from the first epoch.

        beta      = 1 / (1 + median events per active node), clamped to
                    [0.01, 0.10].  Median is used instead of mean to avoid
                    extreme skew from hub nodes inflating the estimate.
        """
        # Per-node inter-arrival gaps
        last: dict = {}
        gaps: list = []
        for u, v, t in zip(train.src, train.dst, train.ts):
            t_f = float(t)
            for node in (int(u), int(v)):
                if node in last:
                    gap = t_f - last[node]
                    if gap > 0.0:
                        gaps.append(gap)
                last[node] = t_f
        tau_data = float(np.median(gaps)) if gaps else 1.0

        # Per-node event counts (using median to avoid hub-node skew)
        node_counts: dict = {}
        for u, v in zip(train.src, train.dst):
            node_counts[int(u)] = node_counts.get(int(u), 0) + 1
            node_counts[int(v)] = node_counts.get(int(v), 0) + 1
        counts = list(node_counts.values())
        median_events = float(np.median(counts)) if counts else 1.0
        beta = float(np.clip(1.0 / (1.0 + median_events), 0.01, 0.10))

        return tau_data, beta

    # ------------------------------------------------------------------

    def call(self, inputs, training = None):
        raise NotImplementedError("Use Trainer.fit() — this model is not callable directly.")

    # ------------------------------------------------------------------
    def train_step(self, data: Dict) -> Dict[str, float]:
        """Score links for one temporal bucket and accumulate gradients.

        Two-snapshot protocol (leakage-free):

          • `snap_pre`  — built from the *previous* bucket's events plus the
                          union of candidate node IDs for the current bucket.
                          Used inside the gradient tape to compute `H` for
                          scoring.  Crucially, the current bucket's positive
                          edges are NOT in this snapshot, so `H[v_positive]`
                          is not informed by a freshly-added `(u, v)` edge.

          • `snap_end`  — built from the current bucket's events.  Used
                          *only* for the post-scoring `commit = True` forward
                          pass that writes the new SSM state into the
                          persistent state table.
        """
        src_b  = data["src"]
        dst_b  = data["dst"]
        ts_b   = data["ts"]
        ef_b   = data.get("edge_feat")
        last_t = int(data["last_t"])
        b_idx  = int(data["b_idx"])
        n_bkts = int(data["num_buckets"])

        if src_b.shape[0] == 0:
            return {}

        t_end = int(ts_b.max())

        # 1. Candidates first (their node set drives `snap_pre.node_ids`).
        cand_src, cand_dst, _, sizes = self.neg_sampler.build_candidates(src_b, dst_b, ts = ts_b)
        sizes_tf = tf.cast(sizes, tf.int32)
        cand_ts = np.repeat(
            np.asarray(ts_b, dtype = np.float64).reshape(-1),
            np.asarray(sizes, dtype = np.int64).reshape(-1),
        )

        # 2. Pre-bucket snapshot: edges from the previous bucket only.
        prev = self._prev_bucket
        if prev is not None:
            prev_src, prev_dst, prev_ts, prev_ef = prev
        else:
            prev_src = np.empty(0, dtype = np.int64)
            prev_dst = np.empty(0, dtype = np.int64)
            prev_ts  = np.empty(0, dtype = np.int64)
            prev_ef  = None

        extra_ids = np.concatenate([
                                       np.asarray(src_b,    dtype = np.int64).reshape(-1),
                                       np.asarray(dst_b,    dtype = np.int64).reshape(-1),
                                       np.asarray(cand_src, dtype = np.int64).reshape(-1),
                                       np.asarray(cand_dst, dtype = np.int64).reshape(-1),
                                  ])

        if self.model._step_mode:
            snap_pre = _build_pre_snapshot(
                                                prev_src, prev_dst, prev_ts, prev_ef,
                                                extra_ids, t_end, last_t,
                                                edge_feat_template = ef_b,
                                          )
        else:
            snap_pre = _build_pre_sequence_snapshot(
                                                        prev_src, prev_dst, prev_ts, prev_ef,
                                                        extra_ids, t_end, last_t,
                                                        seq_len = self.model._sequence_length,
                                                        edge_feat_template = ef_b,
                                                  )
        if snap_pre is None:
            return {}

        # 3. End-bucket snapshot: used ONLY for the commit forward pass.
        if self.model._step_mode:
            snap_end = _build_snapshot(src_b, dst_b, ts_b, ef_b, t_end, last_t)
        else:
            snap_end = _build_sequence_snapshot(
                                                    src_b, dst_b, ts_b, ef_b,
                                                    t_end, last_t,
                                                    seq_len = self.model._sequence_length,
                                                 )
        if snap_end is None:
            return {}

        with tf.GradientTape() as tape:
            H, states_prop = self.model.forward(
                                                    snap_pre,
                                                    commit = False,
                                                    training = True,
                                                )

            states_for_score, alpha_reg_loss = self.model.compute_shadow_committed_states_for_score(
                                                                                                        snap_pre,
                                                                                                        states_prop,
                                                                                                        lambda_prior      = self.cfg.lambda_alpha_prior,
                                                                                                        lambda_saturation = self.cfg.lambda_alpha_saturation,
                                                                                                        training          = True
                                                                                                    )

            logits = self.model.score_pairs(
                                                H,
                                                snap_pre.node_ids,
                                                cand_src,
                                                cand_dst,
                                                states   = states_for_score,
                                                pair_current_ts = cand_ts,
                                                query_history_current_ts = cand_ts,
                                                training = True
                                            )

            logits_flat = tf.reshape(logits, [-1])
            rank_loss   = ranking_loss(logits_flat, sizes_tf, mode = self.cfg.loss_fn)

            wr_loss = tf.constant(0.0)
            if self.cfg.lambda_wr > 0.0:
                ids_tf = tf.cast(snap_pre.node_ids, tf.int32)
                for table, s_next in zip(self.model.state_tables, states_prop):
                    s_prev = table.get(ids_tf)
                    wr_loss += write_penalty_loss(s_prev, s_next, logits_flat)

            loss = rank_loss + self.cfg.lambda_wr * wr_loss + alpha_reg_loss

        grads = tape.gradient(loss, self.model.trainable_weights)
        self._accumulate(grads)
        if (b_idx + 1) % self.cfg.accumulate_every == 0 or b_idx == n_bkts - 1:
            self._apply_gradients()

        # 1-vs-1 train MRR components (positive vs first negative per block).
        # Decoupled from `train_neg_per_pos` so curves remain comparable across
        # runs with different K (see `compute_mrr_1v1_sum_count` for bias note).
        mrr_1v1_sum, mrr_1v1_n = compute_mrr_1v1_sum_count(logits_flat, sizes_tf)

        # 4. Commit state using the END snapshot (the bucket's actual events).
        self.model.forward(snap_end, commit = True, training = False)
        self.model.update_pair_recurrence(src_b, dst_b, ts_b)
        self.model.update_query_history(src_b, dst_b, ts_b)

        # 5. Save this bucket as G_prev for the next bucket's pre-snapshot.
        self._prev_bucket = (
                                np.asarray(src_b, dtype = np.int64).copy(),
                                np.asarray(dst_b, dtype = np.int64).copy(),
                                np.asarray(ts_b,  dtype = np.int64).copy(),
                                None if ef_b is None else np.asarray(ef_b).copy(),
                            )

        return {
                    "loss":           float(loss),
                    "rank_loss":      float(rank_loss),
                    "wr_loss":        float(wr_loss),
                    "alpha_reg_loss": float(alpha_reg_loss),
                    "mrr_1v1_sum":    float(mrr_1v1_sum),
                    "mrr_1v1_n":      float(mrr_1v1_n),
               }

    # ------------------------------------------------------------------
    def fit(self, epochs: Optional[int] = None, initial_epoch: Optional[int] = None) -> Dict[str, List[float]]:  # type: ignore[override]
        n_epochs = epochs if epochs is not None else self.cfg.epochs
        n_initial_epoch = initial_epoch if initial_epoch is not None else self.cfg.initial_epoch
        
        if n_epochs < n_initial_epoch:
            raise ValueError(f"`initial_epoch` must be less than the `epochs`. Found "
                             f"initial_epoch = {n_initial_epoch} and epochs = {n_epochs}.")

        # Ensure ALL trainable variables exist before the optimizer's lazy
        # build at first apply_gradients(). Without this warm-up, branches
        # that are conditional on snapshot content (e.g. the per-event SSM
        # head ``event_msg_lin`` introduced for intra_bucket_seq=True) only
        # build their weights on a later training step, after the optimizer
        # has already bound itself to a smaller variable set, which raises
        # ``Unknown variable`` on the second call.  The persistent state
        # tables touched here are cleared again by ``reset_states_all`` below.
        train_edge_feat_dim = (
            int(self.train_split.edge_feat.shape[1])
            if self.train_split.edge_feat is not None else None
        )
        self.model._build_from_dummy(edge_feat_dim = train_edge_feat_dim)

        # Eagerly bind the optimizer to the full trainable-variable set.  The
        # default lazy build runs at first ``apply_gradients`` and only sees
        # the subset of variables whose grads were non-None on that batch.
        # When subsequent batches contribute gradients to newly-active
        # variables (e.g. ``event_msg_lin`` only fires when a bucket has
        # destination-receiving events), Keras then raises ``Unknown
        # variable`` because the optimizer was already built with the
        # smaller set.  Forcing the build here pins the optimizer's known
        # set to ``model.trainable_weights`` exactly once.
        if not self._gsn_optimizer.built:
            self._gsn_optimizer.build(self.model.trainable_weights)
            
        history: Dict[str, List[float]] = {
                                            "train_loss": [], "train_rank": [], "train_wr":    [],
                                            "train_alpha_reg": [], "train_mrr_1v1": [],
                                            "val_mrr":    [], "val_ap":    [], "val_auc":    [],
                                            "test_mrr":   [], "test_ap":   [], "test_auc":   []
                                          }
        console  = Console()
        best_mrr = 0.0

        # One-time adaptive commit calibration (after dummy build creates weights)
        if self.model.has_adaptive_commit and self._pending_tau_data is not None:
            # mean_n: median of per-active-node event counts across all buckets.
            # The old estimate (mean bucket size / global unique sources) produces
            # an artificially small value, mis-calibrating the gate bias b_0.
            buckets = list(_iter_buckets(self.train_split, self.cfg.batch_events))
            per_node_counts: list = []
            for src_b, dst_b, *_ in buckets:
                if len(src_b) == 0:
                    continue
                _, cnts = np.unique(
                    np.concatenate([
                        np.asarray(src_b, dtype=np.int64),
                        np.asarray(dst_b, dtype=np.int64),
                    ]),
                    return_counts=True,
                )
                per_node_counts.extend(cnts.tolist())
            mean_n = float(np.median(per_node_counts)) if per_node_counts else 1.0
            mean_n = max(mean_n, 0.5)

            self.model.setup_adaptive_commit(
                tau_data = self._pending_tau_data,
                beta     = self._pending_beta,
                mean_n   = mean_n,
            )
            self._pending_tau_data = None
            self._pending_beta     = None

        self.model.reset_states_all(
                                        reset_pair_recurrence = self.model.pair_recurrence_reset_per_epoch,
                                        reset_query_history = self.model.query_history_reset_per_epoch
                                    )
        
        for epoch in range(n_initial_epoch, n_epochs):
            console.rule(f"[bold white]Epoch {epoch + 1} / {n_epochs}")

            # Warmup: linearly ramp gate influence from 0 → 1 over warmup_epochs
            if self.model.has_adaptive_commit:
                warmup_epochs = max(self.cfg.alpha_warmup_epochs, 1)
                eta = min(1.0, (epoch + 1) / warmup_epochs)
                for pblock in self.model.blocks:
                    pblock.warmup_eta = eta
                if epoch < warmup_epochs:
                    cprint(
                            f"  [adaptive α] warmup eta: {eta:.3f} "
                            f"(epoch {epoch+1}/{warmup_epochs})",
                            "cyan"
                          )
                # Reset per-epoch accumulators before each training epoch so
                # _print_alpha_diagnostics sees only THIS epoch's commits.
                for buf in self.model.activity_buffers_list:
                    if buf is not None:
                        buf.clear_epoch_stats()

            train_m = self._train_epoch(console)
            history["train_loss"].append(train_m["loss"])
            history["train_rank"].append(train_m["rank_loss"])
            history["train_wr"].append(train_m["wr_loss"])
            history["train_alpha_reg"].append(train_m.get("alpha_reg_loss", 0.0))
            history["train_mrr_1v1"].append(train_m.get("mrr_1v1", 0.0))

            # Save training state (DenseStateTable, ConvCacheTable, activity
            # buffers, pair-recurrence buffers), evaluate, then restore.
            # Without restoring these, per-epoch eval mutation leaks into the
            # next training epoch AND contaminates the saved checkpoint,
            # breaking trainer-vs-standalone evaluator parity.
            train_states = [t.clone() for t in self.model.state_tables]
            train_caches = [
                c.clone() if c is not None else None
                for c in self.model.conv_cache_tables
            ]
            train_act_buf = [
                b.clone() if b is not None else None
                for b in self.model.activity_buffers_list
            ]
            train_pair_buf = (
                self.model.pair_recurrence_buffers.clone()
                if self.model.pair_recurrence_buffers is not None else None
            )
            train_query_buf = (
                self.model.query_history_buffers.clone()
                if self.model.query_history_buffers is not None else None
            )

            vmrr, vap, vauc = self._eval_epoch(
                                                "Val",
                                                self.val_split,
                                                self.val_random_sampler,
                                                self.val_inductive_sampler,
                                                console,
                                            )
            history["val_mrr"].append(vmrr)
            history["val_ap"].append(vap)
            history["val_auc"].append(vauc)

            tmrr, tap, tauc = self._eval_epoch(
                                                "Test",
                                                self.test_split,
                                                self.test_random_sampler,
                                                self.test_inductive_sampler,
                                                console,
                                            )

            # Restore training state, conv cache, and activity buffers
            for table, saved in zip(self.model.state_tables, train_states):
                table.copy_from(saved)
            for cache, saved_cache in zip(self.model.conv_cache_tables, train_caches):
                if cache is not None and saved_cache is not None:
                    cache.copy_from(saved_cache)
            for buf, saved_buf in zip(self.model.activity_buffers_list, train_act_buf):
                if buf is not None and saved_buf is not None:
                    buf.copy_from(saved_buf)
            if (
                self.model.pair_recurrence_buffers is not None
                and train_pair_buf is not None
            ):
                self.model.pair_recurrence_buffers.copy_from(train_pair_buf)
            if (
                self.model.query_history_buffers is not None
                and train_query_buf is not None
            ):
                self.model.query_history_buffers.copy_from(train_query_buf)

            history["test_mrr"].append(tmrr)
            history["test_ap"].append(tap)
            history["test_auc"].append(tauc)

            # Alpha diagnostics (adaptive mode only)
            if self.model.has_adaptive_commit:
                self._print_alpha_diagnostics(console)

            is_best = vmrr > best_mrr
            if is_best:
                best_mrr = vmrr
            if self.cfg.weights_dir is not None:
                self._save(epoch + 1, val_mrr = vmrr, is_best = is_best)

        return history

    # ------------------------------------------------------------------

    def _print_alpha_diagnostics(self, console: Console) -> None:
        """Print per-epoch α statistics using ALL per-commit values.

        Replaces the old last_alpha-only view with epoch accumulators so that
        hub nodes with many commits per epoch are not unfairly represented by
        only their final commit.

        Quartile axis: ema_interarrival (short ema = high-frequency node).
        This directly tests whether the gate is doing the right thing:
          high-freq nodes (low ema_ia) → should have low α (protect memory)
          low-freq  nodes (high ema_ia) → should have high α (overwrite stale)
        """
        all_mu:  List[np.ndarray] = []
        all_ema: List[np.ndarray] = []

        for buf in self.model.activity_buffers_list:
            if buf is None:
                continue
            stats = buf.get_epoch_alpha_stats()
            if not stats:
                continue
            all_mu.append(stats["per_node_mean_alpha"])
            all_ema.append(stats["ema_ia"])

        if not all_mu:
            return

        mu  = np.concatenate(all_mu)
        ema = np.concatenate(all_ema)
        pct = np.percentile(mu, [1, 5, 50, 95, 99])

        console.print(
            f"  [cyan][α stats (per-commit mean)][/cyan] "
            f"mean={float(np.mean(mu)):.4f} std={float(np.std(mu)):.4f} "
            f"p01={pct[0]:.4f} p05={pct[1]:.4f} p50={pct[2]:.4f} "
            f"p95={pct[3]:.4f} p99={pct[4]:.4f}"
        )

        # Quartile by ema_interarrival: tests frequency-aware gate behaviour.
        # Short ema_ia = high-frequency node (should → low α).
        # Long  ema_ia = low-frequency node  (should → high α).
        q25_ema, q75_ema = np.percentile(ema, [25, 75])
        for label, sel in [
            ("Q1 high-freq (short ema_ia)", ema <= q25_ema),
            ("Q2-Q3",                       (ema > q25_ema) & (ema <= q75_ema)),
            ("Q4 low-freq  (long  ema_ia)", ema > q75_ema),
        ]:
            if np.any(sel):
                console.print(
                    f"    {label:<36}: "
                    f"mean_α={float(np.mean(mu[sel])):.4f}  "
                    f"median_α={float(np.median(mu[sel])):.4f}"
                )

    # ------------------------------------------------------------------

    def _train_epoch(self, console: Console) -> Dict[str, float]:
        buckets     = list(_iter_buckets(self.train_split, self.cfg.batch_events))
        num_buckets = len(buckets)
        # Progress is event-weighted, not bucket-weighted: under
        # ``batch_events: -1`` bucket sizes range from 1 to ~1746 on
        # Contacts, so a bucket-count bar would jitter wildly and the ETA
        # would be meaningless.  Tracking events processed gives a smooth
        # percentage and a stable events-per-second ETA regardless of
        # bucket-size policy.
        total_events = int(self.train_split.src.shape[0])
        self._accum_grads = None
        self._accum_count = 0

        # Each phase starts with no previous bucket, so the first bucket's
        # pre-snapshot is edge-less (message-passing receives no gradient
        # that single bucket).  This also prevents the cache from carrying
        # an unrelated bucket's edges across phase boundaries.
        self._prev_bucket = None

        total_loss = total_rank = total_wr = total_alpha_reg = 0.0
        total_mrr_1v1_sum = 0.0
        total_mrr_1v1_n   = 0.0
        n_done = 0
        last_t = 0

        with _progress_ctx(console, "Train", total_events) as (progress, task):
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
                metrics  = self.train_step(data)
                n_events = int(src_b.shape[0])
                if n_events > 0:
                    last_t = int(ts_b.max())
                if metrics:
                    total_loss        += metrics["loss"]
                    total_rank        += metrics["rank_loss"]
                    total_wr          += metrics["wr_loss"]
                    total_alpha_reg   += metrics.get("alpha_reg_loss", 0.0)
                    total_mrr_1v1_sum += metrics.get("mrr_1v1_sum", 0.0)
                    total_mrr_1v1_n   += metrics.get("mrr_1v1_n",   0.0)
                    n_done            += 1
                safe       = max(n_done, 1)
                mrr_1v1_d  = total_mrr_1v1_sum / max(total_mrr_1v1_n, 1.0)
                suffix = (
                            f"bucket {b_idx + 1}/{num_buckets} - "
                            f"loss: {total_loss/safe:.4f} - "
                            f"rank: {total_rank/safe:.4f} - "
                            f"MRR_1v1: {mrr_1v1_d:.4f} - "
                            f"wr: {total_wr/safe:.5f}"
                         )
                if self.model.has_adaptive_commit:
                    suffix += f" - α_reg: {total_alpha_reg/safe:.5f}"
                progress.update(task, advance = n_events, suffix = suffix)

        return {
                    "loss":           total_loss      / max(n_done, 1),
                    "rank_loss":      total_rank      / max(n_done, 1),
                    "wr_loss":        total_wr        / max(n_done, 1),
                    "alpha_reg_loss": total_alpha_reg / max(n_done, 1),
                    "mrr_1v1":        total_mrr_1v1_sum / max(total_mrr_1v1_n, 1.0),
               }

    def _eval_epoch(
                        self,
                        name:              str,
                        split:             TGBSplit,
                        random_sampler:    Any,
                        inductive_sampler: Optional[Any],
                        console:           Console,
                    ) -> Tuple[float, float, float]:
        """DyGLib/DyGMamba-aligned 1-negative-per-positive evaluation.

        Delegates to ``gsn.train.eval.evaluate_split`` so the trainer's per-epoch
        logged numbers match ``examples/evaluate.py`` exactly (same negative
        sampler, same snap_pre packing, same chunked sklearn AP/AUC, same MRR@1).

        Returns
        -------
        ``(mrr_1neg, ap, auc)`` from the random NSS strategy. If
        ``inductive_sampler`` is provided, the per-bucket inductive line is also
        printed to the console for diagnostic purposes, but only random metrics
        are returned (and stored in the trainer's history dict).
        """
        # Lazy import to avoid circular dependency (eval.py imports loop helpers).
        from gsn.train.eval import evaluate_split

        results = evaluate_split(
                                    self.model, split,
                                    random_sampler    = random_sampler,
                                    inductive_sampler = inductive_sampler,
                                    batch_events      = self.cfg.batch_events,
                                    metric_batch_size = 200,
                                    label             = name,
                                    report_global     = False,
                                    console           = console,
                                    compute_loss      = True,
                                )

        if "random" not in results:
            return 0.0, 0.0, 0.0
        rnd = results["random"]

        if "inductive" in results:
            ind = results["inductive"]
            console.print(
                f"  [dim]{name} \\[inductive]  "
                f"loss: {rnd.get('loss', float('nan')):.4f}  "
                f"MRR: {ind['mrr_1neg']:.4f}  AP: {ind['ap']:.4f}  AUC: {ind['auc']:.4f}[/dim]"
            )

        return float(rnd["mrr_1neg"]), float(rnd["ap"]), float(rnd["auc"])

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

        def _activity_buffers_dict() -> dict:
            arrays: dict = {}
            for i, buf in enumerate(self.model.activity_buffers_list):
                if buf is not None:
                    arrays[f"last_update_time_{i}"] = buf.last_update_time
                    arrays[f"update_count_{i}"]     = buf.update_count
                    arrays[f"ema_interarrival_{i}"] = buf.ema_interarrival
                    arrays[f"last_alpha_{i}"]       = buf.last_alpha
            return arrays

        if is_best:
            self.model.save_weights(str(d / "best.weights.h5"))

            # Save activity buffers alongside the weights so from_pretrained()
            # can restore the gate's temporal context for evaluation or
            # continued training.  These are plain NumPy arrays (not TF
            # variables) and are therefore invisible to save_weights().
            if self.model.has_adaptive_commit:
                np.savez(str(d / "activity_buffers.npz"), **_activity_buffers_dict())
            if self.model.has_pair_recurrence:
                self.model.pair_recurrence_buffers.save_npz(
                    d / "pair_recurrence.npz"
                )
            if self.model.has_query_history:
                self.model.query_history_buffers.save_npz(
                    d / "query_history.npz"
                )

        if self.cfg.save_every_epoch:
            self.model.save_weights(str(d / f"epoch_{epoch:03d}.weights.h5"))
            if self.model.has_adaptive_commit:
                np.savez(
                    str(d / f"epoch_{epoch:03d}_activity_buffers.npz"),
                    **_activity_buffers_dict(),
                )
            if self.model.has_pair_recurrence:
                self.model.pair_recurrence_buffers.save_npz(
                    d / f"epoch_{epoch:03d}_pair_recurrence.npz"
                )
            if self.model.has_query_history:
                self.model.query_history_buffers.save_npz(
                    d / f"epoch_{epoch:03d}_query_history.npz"
                )

        with open(d / "config.json", "w", encoding = "utf-8") as f:
            json.dump(self.model.get_config(), f, indent = 4)
