"""Shared DyGLib/DyGMamba-aligned link-prediction evaluation.

Single source of truth used by both:
  - the trainer's per-epoch eval (`gsn.train.loop.Trainer._eval_epoch`)
  - the standalone CLI evaluator (`examples/evaluate.py`)

This eliminates the prior trainer/evaluator drift that made trainer-logged
MRR/AP/AUC disagree with the standalone numbers (mismatched negative samplers,
per-query vs sklearn AP, pooled-vs-chunked metric aggregation).

Protocol:
  - 1 negative per positive (DyGLib/DyGMamba protocol).
  - Random NSS: draw one negative destination per positive from a fixed
    destination pool (default = unique full-data dst), keep positive source.
    No collision repair (matches DyGLib `random_sample()`).
  - Inductive NSS (optional): DyGLib edge/time sampler — draws negative edges
    from historical edges that have not been observed up to the
    last-observed-time and are not in the current batch.
  - Metrics: sklearn AP/AUC on sigmoid(logits) within each metric_batch_size
    chunk, then averaged. Global diagnostic AP/AUC over the whole split
    optionally reported.
  - Loss: mean BCE over [pos, neg] across all events of the split (reported
    so the trainer's per-epoch summary keeps its `loss:` field).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# We avoid importing from gsn.train.loop at module top to keep import-time
# circular-dependency risk low; the trainer imports `evaluate_split` lazily
# from inside `_eval_epoch`, while the CLI evaluator imports it eagerly here.
# The helpers below are stable, module-level functions defined in loop.py.
from gsn.train.loop import (  # noqa: E402  (after rich imports, intentional)
    _iter_buckets,
    _build_pre_snapshot,
    _build_snapshot,
)


# ---------------------------------------------------------------------------
# DyGLib-style negative samplers
# ---------------------------------------------------------------------------

class DyGLibRandomNegativeSampler:
    """DyGLib random NSS.

    Draws one negative destination per positive from a fixed pool. The
    evaluator-side convention is to keep the positive source as the negative
    source (so this class only returns destinations). No collision repair —
    matches DyGLib's `random_sample()`.
    """

    def __init__(self, dst_pool: np.ndarray, seed: int = 0) -> None:
        self.dst_pool = np.asarray(dst_pool, dtype=np.int64)
        if self.dst_pool.size == 0:
            raise ValueError("random NSS destination pool is empty")
        self.seed = int(seed)
        self.random_state = np.random.RandomState(self.seed)

    def reset_random_state(self) -> None:
        self.random_state = np.random.RandomState(self.seed)

    def sample_batch(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        del src
        batch_size = int(len(dst))
        return self.random_state.choice(
            self.dst_pool, size=batch_size, replace=True
        ).astype(np.int64)


class DyGLibInductiveNegativeSampler:
    """DyGLib inductive NSS (edge/time based, NOT unseen-node based).

    For a current evaluation batch [t_start, t_end]:
        historical_edges = edges with earliest_time <= t <= t_start
        observed_edges   = edges with earliest_time <= t <= last_observed_time
        current_edges    = edges with t_start <= t <= t_end
        inductive_edges  = historical_edges - observed_edges - current_edges

    Negative edges are sampled from `inductive_edges`. If there are fewer than
    requested, the remaining slots are filled with random (src, dst) pairs
    drawn from the cartesian product of (unique src, unique dst) minus the
    current batch's edges.
    """

    def __init__(
        self,
        full_src: np.ndarray,
        full_dst: np.ndarray,
        full_ts: np.ndarray,
        last_observed_time: float,
        seed: int = 0,
    ) -> None:
        self.src_node_ids   = np.asarray(full_src, dtype=np.int64)
        self.dst_node_ids   = np.asarray(full_dst, dtype=np.int64)
        self.interact_times = np.asarray(full_ts)

        if not (len(self.src_node_ids) == len(self.dst_node_ids) == len(self.interact_times)):
            raise ValueError("full_src, full_dst, and full_ts must have equal length")

        self.seed         = int(seed)
        self.random_state = np.random.RandomState(self.seed)

        self.unique_src_node_ids   = np.unique(self.src_node_ids).astype(np.int64)
        self.unique_dst_node_ids   = np.unique(self.dst_node_ids).astype(np.int64)
        self.unique_interact_times = np.unique(self.interact_times)
        self.earliest_time         = float(np.min(self.unique_interact_times))
        self.last_observed_time    = float(last_observed_time)

        self.possible_edges: Set[Tuple[int, int]] = set(
            (int(s), int(d))
            for s in self.unique_src_node_ids
            for d in self.unique_dst_node_ids
        )
        self.observed_edges = self._edges_between(self.earliest_time, self.last_observed_time)

    def reset_random_state(self) -> None:
        self.random_state = np.random.RandomState(self.seed)

    def _edges_between(self, start_time: float, end_time: float) -> Set[Tuple[int, int]]:
        mask = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        return set(
            (int(s), int(d))
            for s, d in zip(self.src_node_ids[mask], self.dst_node_ids[mask])
        )

    @staticmethod
    def _batch_edge_set(src: np.ndarray, dst: np.ndarray) -> Set[Tuple[int, int]]:
        return set(
            (int(s), int(d))
            for s, d in zip(
                np.asarray(src, dtype=np.int64),
                np.asarray(dst, dtype=np.int64),
            )
        )

    def _sample_edges_from_list(
        self,
        edge_list: List[Tuple[int, int]],
        size: int,
        replace: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if size <= 0 or len(edge_list) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        idx = self.random_state.choice(len(edge_list), size=int(size), replace=bool(replace))
        src = np.array([edge_list[int(i)][0] for i in idx], dtype=np.int64)
        dst = np.array([edge_list[int(i)][1] for i in idx], dtype=np.int64)
        return src, dst

    def _random_fill(
        self,
        size: int,
        batch_src_node_ids: np.ndarray,
        batch_dst_node_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_edges = self._batch_edge_set(batch_src_node_ids, batch_dst_node_ids)
        candidates = list(self.possible_edges - batch_edges)
        if len(candidates) == 0:
            raise ValueError("No valid random fill-in edges remain after collision check")
        replace = len(candidates) < int(size)
        return self._sample_edges_from_list(candidates, size=int(size), replace=replace)

    def sample_batch(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        current_batch_start_time: float,
        current_batch_end_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        size = int(len(src))

        historical = self._edges_between(self.earliest_time, float(current_batch_start_time))
        current    = self._edges_between(float(current_batch_start_time), float(current_batch_end_time))
        inductive  = list(historical - self.observed_edges - current)

        if size > len(inductive):
            n_random = size - len(inductive)
            rnd_src, rnd_dst = self._random_fill(n_random, src, dst)
            ind_src = np.array([e[0] for e in inductive], dtype=np.int64)
            ind_dst = np.array([e[1] for e in inductive], dtype=np.int64)
            neg_src = np.concatenate([rnd_src, ind_src])
            neg_dst = np.concatenate([rnd_dst, ind_dst])
        else:
            neg_src, neg_dst = self._sample_edges_from_list(inductive, size=size, replace=False)

        return neg_src.astype(np.int64), neg_dst.astype(np.int64)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _binary_metrics_from_probs(
    pos_probs: np.ndarray, neg_probs: np.ndarray
) -> Dict[str, float]:
    E = int(len(pos_probs))
    if E == 0:
        return {"ap": float("nan"), "auc": float("nan"),
                "acc": float("nan"), "mrr_1neg": float("nan")}
    preds  = np.concatenate([pos_probs, neg_probs])
    labels = np.concatenate([np.ones(E, dtype=np.float32),
                             np.zeros(E, dtype=np.float32)])
    ap  = float(average_precision_score(labels, preds))
    auc = float(roc_auc_score(labels, preds))
    acc = float(np.mean(pos_probs > neg_probs))
    return {"ap": ap, "auc": auc, "acc": acc, "mrr_1neg": 0.5 + 0.5 * acc}


def _binary_metrics_global(
    pos_logits_batches: Sequence[np.ndarray],
    neg_logits_batches: Sequence[np.ndarray],
) -> Dict[str, float]:
    pos_probs = _sigmoid(np.concatenate(pos_logits_batches).reshape(-1))
    neg_probs = _sigmoid(np.concatenate(neg_logits_batches).reshape(-1))
    return _binary_metrics_from_probs(pos_probs, neg_probs)


def _binary_metrics_dyglib_chunks(
    pos_logits_batches: Sequence[np.ndarray],
    neg_logits_batches: Sequence[np.ndarray],
    metric_batch_size: int = 200,
) -> Dict[str, float]:
    """Mean per-batch sklearn metrics, matching DyGLib reporting."""
    pos_logits = np.concatenate(pos_logits_batches).reshape(-1)
    neg_logits = np.concatenate(neg_logits_batches).reshape(-1)
    if len(pos_logits) != len(neg_logits):
        raise ValueError("positive and negative logits must have equal length")
    if metric_batch_size < 0 and metric_batch_size != -1:
        raise ValueError(
            f"metric_batch_size can be either positive or -1 (indicating "
            f"batch_events). Found metric_batch_size={metric_batch_size}"
        )

    aps:  List[float] = []
    aucs: List[float] = []
    accs: List[float] = []

    if metric_batch_size != -1:
        for start in range(0, len(pos_logits), int(metric_batch_size)):
            end       = min(start + int(metric_batch_size), len(pos_logits))
            pos_probs = _sigmoid(pos_logits[start:end])
            neg_probs = _sigmoid(neg_logits[start:end])
            m         = _binary_metrics_from_probs(pos_probs, neg_probs)
            aps.append(m["ap"])
            aucs.append(m["auc"])
            accs.append(m["acc"])
    else:
        pos_probs = _sigmoid(pos_logits)
        neg_probs = _sigmoid(neg_logits)
        m         = _binary_metrics_from_probs(pos_probs, neg_probs)
        aps.append(m["ap"])
        aucs.append(m["auc"])
        accs.append(m["acc"])

    acc = float(np.mean(accs))
    return {
        "ap":       float(np.mean(aps)),
        "auc":      float(np.mean(aucs)),
        "acc":      acc,
        "mrr_1neg": 0.5 + 0.5 * acc,
    }


def _bce_from_logits(pos_logits: np.ndarray, neg_logits: np.ndarray) -> float:
    """Mean BCE over [pos, neg], in log-domain, matching DyGMamba's `nn.BCELoss`
    applied to sigmoid(logits). Equivalent to `binary_cross_entropy_with_logits`.
    """
    pos = pos_logits.astype(np.float64).reshape(-1)
    neg = neg_logits.astype(np.float64).reshape(-1)
    # softplus(-x) for label=1; softplus(x) for label=0
    def _softplus(z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
    loss = np.mean(np.concatenate([_softplus(-pos), _softplus(neg)]))
    return float(loss)


# ---------------------------------------------------------------------------
# Rich progress helper
# ---------------------------------------------------------------------------

@contextmanager
def _progress_ctx(console: Optional[Console], label: str, total: int):
    p = Progress(
        TextColumn(f"[cyan]{label:<6}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[suffix]}"),
        console=console,
        transient=False,
    )
    with p:
        task = p.add_task("", total=max(int(total), 1), suffix="")
        yield p, task


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_split(
    model,
    split,
    *,
    random_sampler:    DyGLibRandomNegativeSampler,
    inductive_sampler: Optional[DyGLibInductiveNegativeSampler] = None,
    batch_events:      int  = 256,
    metric_batch_size: int  = 200,
    label:             str  = "Eval",
    report_global:     bool = True,
    console:           Optional[Console] = None,
    compute_loss:      bool = True,
) -> Dict[str, Dict[str, float]]:
    """DyGLib/DyGMamba-aligned 1-negative-per-positive evaluation.

    Returns
    -------
    A dict ``{"random": {...}, "inductive": {...}}`` (the latter only if
    ``inductive_sampler`` is given). Each inner dict contains:
        ``ap``, ``auc``, ``acc``, ``mrr_1neg``, ``n_events``
    and, when ``report_global`` is True, the corresponding ``global_*`` keys.
    For the ``random`` strategy, a ``loss`` key holds the mean BCE over the
    whole split (used by the trainer's per-epoch summary).

    State side-effects: this function commits node states and, when enabled,
    pair-recurrence history bucket-by-bucket (matching the trainer's eval
    semantics). It does NOT reset the model's persistent state before or after.
    """
    random_sampler.reset_random_state()
    if inductive_sampler is not None:
        inductive_sampler.reset_random_state()

    buckets      = list(_iter_buckets(split, batch_events))
    n_buckets    = len(buckets)
    total_events = int(split.src.shape[0])
    last_t       = 0
    prev_bucket: Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
    ] = None

    rnd_pos: List[np.ndarray] = []
    rnd_neg: List[np.ndarray] = []
    ind_pos: List[np.ndarray] = []
    ind_neg: List[np.ndarray] = []

    with _progress_ctx(console, label, total_events) as (progress, task):
        for b_idx, (src_b, dst_b, ts_b, ef_b, _) in enumerate(buckets):
            if src_b.shape[0] == 0:
                continue

            t_start   = float(ts_b[0])
            t_end     = float(ts_b[-1])
            t_end_int = int(np.max(ts_b))

            rnd_neg_dst = random_sampler.sample_batch(src_b, dst_b)
            rnd_neg_src = src_b

            ind_neg_src: Optional[np.ndarray] = None
            ind_neg_dst: Optional[np.ndarray] = None
            if inductive_sampler is not None:
                ind_neg_src, ind_neg_dst = inductive_sampler.sample_batch(
                    src_b, dst_b,
                    current_batch_start_time=t_start,
                    current_batch_end_time=t_end,
                )

            if prev_bucket is not None:
                prev_src, prev_dst, prev_ts, prev_ef = prev_bucket
            else:
                prev_src = np.empty(0, dtype=np.int64)
                prev_dst = np.empty(0, dtype=np.int64)
                prev_ts  = np.empty(0, dtype=np.int64)
                prev_ef  = None

            extras = [
                np.asarray(src_b,        dtype=np.int64).reshape(-1),
                np.asarray(dst_b,        dtype=np.int64).reshape(-1),
                np.asarray(rnd_neg_src,  dtype=np.int64).reshape(-1),
                np.asarray(rnd_neg_dst,  dtype=np.int64).reshape(-1),
            ]
            if ind_neg_src is not None and ind_neg_dst is not None:
                extras.extend([
                    np.asarray(ind_neg_src, dtype=np.int64).reshape(-1),
                    np.asarray(ind_neg_dst, dtype=np.int64).reshape(-1),
                ])
            extra_ids = np.concatenate(extras)

            snap_pre = _build_pre_snapshot(
                prev_src, prev_dst, prev_ts, prev_ef,
                extra_ids, t_end_int, last_t,
                edge_feat_template=ef_b,
            )
            snap_end = _build_snapshot(src_b, dst_b, ts_b, ef_b, t_end_int, last_t)
            if snap_pre is None or snap_end is None:
                continue
            last_t = t_end_int

            H, states_prop = model.forward(snap_pre, commit=False, training=False)
            states_for_score, _ = model.compute_shadow_committed_states_for_score(
                snap_pre, states_prop,
                lambda_prior=0.0, lambda_saturation=0.0, training=False,
            )

            rnd_cand_src = np.concatenate([src_b, rnd_neg_src]).astype(np.int64)
            rnd_cand_dst = np.concatenate([dst_b, rnd_neg_dst]).astype(np.int64)
            rnd_cand_ts  = np.concatenate([ts_b, ts_b]).astype(np.float64)
            rnd_logits = model.score_pairs(
                H, snap_pre.node_ids,
                rnd_cand_src, rnd_cand_dst,
                states=states_for_score,
                pair_current_ts=rnd_cand_ts,
                query_history_current_ts=rnd_cand_ts,
                training=False,
            )
            rnd_logits_np = tf.reshape(rnd_logits, [-1]).numpy()
            B = int(len(src_b))
            rnd_pos.append(rnd_logits_np[:B])
            rnd_neg.append(rnd_logits_np[B:])

            if ind_neg_src is not None and ind_neg_dst is not None:
                ind_cand_src = np.concatenate([src_b, ind_neg_src]).astype(np.int64)
                ind_cand_dst = np.concatenate([dst_b, ind_neg_dst]).astype(np.int64)
                ind_cand_ts  = np.concatenate([ts_b, ts_b]).astype(np.float64)
                ind_logits = model.score_pairs(
                    H, snap_pre.node_ids,
                    ind_cand_src, ind_cand_dst,
                    states=states_for_score,
                    pair_current_ts=ind_cand_ts,
                    query_history_current_ts=ind_cand_ts,
                    training=False,
                )
                ind_logits_np = tf.reshape(ind_logits, [-1]).numpy()
                ind_pos.append(ind_logits_np[:B])
                ind_neg.append(ind_logits_np[B:])

            model.forward(snap_end, commit=True, training=False)
            model.update_pair_recurrence(src_b, dst_b, ts_b)
            model.update_query_history(src_b, dst_b, ts_b)
            prev_bucket = (
                np.asarray(src_b, dtype=np.int64).copy(),
                np.asarray(dst_b, dtype=np.int64).copy(),
                np.asarray(ts_b,  dtype=np.int64).copy(),
                None if ef_b is None else np.asarray(ef_b).copy(),
            )

            # Running suffix for the Rich progress bar (cheap to compute).
            if rnd_pos:
                m = _binary_metrics_dyglib_chunks(rnd_pos, rnd_neg, metric_batch_size)
                if compute_loss:
                    suffix = (
                        f"bucket {b_idx + 1}/{n_buckets} - "
                        f"loss: {_bce_from_logits(np.concatenate(rnd_pos), np.concatenate(rnd_neg)):.4f}  "
                        f"MRR: {m['mrr_1neg']:.4f}  AP: {m['ap']:.4f}  AUC: {m['auc']:.4f}"
                    )
                else:
                    suffix = (
                        f"bucket {b_idx + 1}/{n_buckets} - "
                        f"MRR: {m['mrr_1neg']:.4f}  AP: {m['ap']:.4f}  AUC: {m['auc']:.4f}"
                    )
            else:
                suffix = f"bucket {b_idx + 1}/{n_buckets} - ..."
            progress.update(task, advance=int(src_b.shape[0]), suffix=suffix)

    results: Dict[str, Dict[str, float]] = {}

    if rnd_pos:
        m = _binary_metrics_dyglib_chunks(rnd_pos, rnd_neg, metric_batch_size)
        m["n_events"] = float(sum(len(b) for b in rnd_pos))
        if report_global:
            g = _binary_metrics_global(rnd_pos, rnd_neg)
            m.update({f"global_{k}": v for k, v in g.items()})
        if compute_loss:
            m["loss"] = _bce_from_logits(np.concatenate(rnd_pos), np.concatenate(rnd_neg))
        results["random"] = m

    if ind_pos:
        m = _binary_metrics_dyglib_chunks(ind_pos, ind_neg, metric_batch_size)
        m["n_events"] = float(sum(len(b) for b in ind_pos))
        if report_global:
            g = _binary_metrics_global(ind_pos, ind_neg)
            m.update({f"global_{k}": v for k, v in g.items()})
        results["inductive"] = m

    return results
