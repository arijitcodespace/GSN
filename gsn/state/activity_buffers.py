"""Per-node activity statistics buffers for the adaptive commit gate.

These are plain NumPy arrays (not TF Variables) because they are read as
static features into the gate at commit time (outside the GradientTape) and
updated after each committed write.  They live alongside the DenseStateTable
and are cloned / restored alongside it during the val→test eval protocol.
"""

from __future__ import annotations
from typing import Tuple, Optional

import numpy as np


class NodeActivityBuffers:
    """Per-node activity statistics for the adaptive commit gate.

    Tracks four quantities per node:

      last_update_time[i]  — wall-clock timestamp of the last committed update.
      update_count[i]      — total number of committed updates (c_i in the design).
      ema_interarrival[i]  — exponential moving average of inter-update intervals.
      last_alpha[i]        — α used at the most recent commit (diagnostics only).

    Parameters
    ----------
    num_nodes : int
        Total number of nodes (= state-table rows).
    tau_data  : float
        Dataset-level median inter-event time (seconds or the dataset's raw
        time unit).  Used to initialise ``ema_interarrival`` so the first
        event of every node is treated as a median-frequency arrival.
    beta      : float
        EMA decay for ``ema_interarrival``.  Typical range: 0.01 – 0.10.
    """

    def __init__(
                    self,
                    num_nodes: int,
                    tau_data:  float,
                    beta:      float = 0.05,
                ):
        self.num_nodes = int(num_nodes)
        self.tau_data  = float(tau_data)
        self.beta      = float(beta)

        # All buffers as float64 for timestamp precision; last_alpha as float32.
        self.last_update_time = np.zeros(self.num_nodes, dtype=np.float64)
        self.update_count     = np.zeros(self.num_nodes, dtype=np.int64)
        self.ema_interarrival = np.full(self.num_nodes, tau_data, dtype=np.float64)
        self.last_alpha       = np.zeros(self.num_nodes, dtype=np.float32)

        # Per-epoch accumulators for diagnostics — reset each training epoch,
        # NOT cloned/restored across val/test phases (they are scratch-only).
        # Tracks ALL per-commit α values, not just the last one per node.
        self.epoch_alpha_sum  = np.zeros(self.num_nodes, dtype=np.float64)
        self.epoch_alpha_sq   = np.zeros(self.num_nodes, dtype=np.float64)
        self.epoch_commit_cnt = np.zeros(self.num_nodes, dtype=np.int64)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_features(
                        self,
                        node_ids:    np.ndarray,
                        current_ts:  float,
                        event_count: np.ndarray,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        ids = np.asarray(node_ids, dtype = np.int64)

        seen = self.update_count[ids] > 0

        raw_delta = np.maximum(
                                current_ts - self.last_update_time[ids],
                                0.0
                              ).astype(np.float64)

        delta_t = np.where(
                            seen,
                            raw_delta,
                            float(self.tau_data)
                          ).astype(np.float64)

        c   = self.update_count[ids].astype(np.float64)
        ema = self.ema_interarrival[ids].astype(np.float64)
        n   = np.asarray(event_count, dtype = np.float64)

        return delta_t, n, c, ema

    # ------------------------------------------------------------------
    # Update after commit
    # ------------------------------------------------------------------

    def update(
                self,
                node_ids:   np.ndarray,
                current_ts: float,
                alpha:      np.ndarray
              ) -> None:

        ids = np.asarray(node_ids, dtype = np.int64)

        seen = self.update_count[ids] > 0

        raw_delta = np.maximum(
                                current_ts - self.last_update_time[ids],
                                0.0
                              )

        delta_t = np.where(
                            seen,
                            raw_delta,
                            float(self.tau_data)
                          )

        self.ema_interarrival[ids] = (
                                        (1.0 - self.beta) * self.ema_interarrival[ids]
                                        + self.beta * delta_t
                                     )

        self.last_update_time[ids] = float(current_ts)
        self.update_count[ids] += 1

        alpha_arr = np.asarray(alpha, dtype = np.float32).reshape(-1)
        self.last_alpha[ids] = alpha_arr

        a64 = alpha_arr.astype(np.float64)
        self.epoch_alpha_sum[ids] += a64
        self.epoch_alpha_sq[ids] += a64 ** 2
        self.epoch_commit_cnt[ids] += 1

    # ------------------------------------------------------------------
    # Lifecycle helpers (mirror DenseStateTable API)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero all buffers (mirrors DenseStateTable.reset_state)."""
        self.last_update_time[:] = 0.0
        self.update_count[:]     = 0
        self.ema_interarrival[:] = self.tau_data
        self.last_alpha[:]       = 0.0
        self.epoch_alpha_sum[:]  = 0.0
        self.epoch_alpha_sq[:]   = 0.0
        self.epoch_commit_cnt[:] = 0

    def clear_epoch_stats(self) -> None:
        """Reset per-epoch accumulators.  Call at the start of each training epoch."""
        self.epoch_alpha_sum[:]  = 0.0
        self.epoch_alpha_sq[:]   = 0.0
        self.epoch_commit_cnt[:] = 0

    def clone(self) -> "NodeActivityBuffers":
        """Return a deep copy of the temporal state buffers.

        Epoch-level accumulators (epoch_alpha_sum/sq/cnt) are NOT cloned —
        they are diagnostic scratch that does not need to survive the
        val/test eval round-trip.
        """
        copy = NodeActivityBuffers(self.num_nodes, self.tau_data, self.beta)
        copy.last_update_time[:] = self.last_update_time
        copy.update_count[:]     = self.update_count
        copy.ema_interarrival[:] = self.ema_interarrival
        copy.last_alpha[:]       = self.last_alpha
        return copy

    def copy_from(self, src: "NodeActivityBuffers") -> None:
        """Overwrite temporal state from *src* (mirrors DenseStateTable.copy_from).

        Epoch-level accumulators are NOT restored so that the per-epoch stats
        accumulated during training are not clobbered when restoring after eval.
        """
        self.last_update_time[:] = src.last_update_time
        self.update_count[:]     = src.update_count
        self.ema_interarrival[:] = src.ema_interarrival
        self.last_alpha[:]       = src.last_alpha

    # ------------------------------------------------------------------
    # Diagnostic summary
    # ------------------------------------------------------------------

    def get_epoch_alpha_stats(self) -> dict:
        """Summary stats over ALL per-commit α values this epoch.

        Returns a dict with:
          per_node_mean_alpha : [M] float64 — mean α across all commits for
                                each active node this epoch (M = #active nodes).
          per_node_std_alpha  : [M] float64 — std of α across commits.
          ema_ia              : [M] float64 — ema_interarrival for each node
                                (proxy for node event frequency).
          update_count        : [M] int64   — commits this epoch.
          + scalar summary fields: mean, std, p01, p05, p50, p95, p99.

        Returns empty dict if no nodes were committed this epoch.
        """
        mask = self.epoch_commit_cnt > 0
        if not np.any(mask):
            return {}
        n   = self.epoch_commit_cnt[mask].astype(np.float64)
        mu  = self.epoch_alpha_sum[mask] / n
        sq  = self.epoch_alpha_sq[mask]  / n
        std = np.sqrt(np.maximum(sq - mu ** 2, 0.0))
        pct = np.percentile(mu, [1, 5, 50, 95, 99])
        return {
            # Per-node arrays (for quartile breakdowns)
            "per_node_mean_alpha": mu,
            "per_node_std_alpha":  std,
            "ema_ia":              self.ema_interarrival[mask],
            "update_count":        self.epoch_commit_cnt[mask],
            # Scalar summaries
            "mean":  float(np.mean(mu)),
            "std":   float(np.std(mu)),
            "p01":   float(pct[0]),
            "p05":   float(pct[1]),
            "p50":   float(pct[2]),
            "p95":   float(pct[3]),
            "p99":   float(pct[4]),
        }

    def alpha_diagnostics(
                            self,
                            node_ids:    Optional[np.ndarray] = None,
                            percentiles: Tuple[int, ...] = (1, 5, 25, 50, 75, 95, 99),
                          ) -> dict:
        """Return a dict of summary statistics over last_alpha for the given nodes
        (or all nodes if node_ids is None).
        """
        alpha = self.last_alpha if node_ids is None else self.last_alpha[node_ids]
        if alpha.size == 0:
            return {}
        pcts = np.percentile(alpha, percentiles)
        result = {
                     "mean":  float(np.mean(alpha)),
                     "std":   float(np.std(alpha)),
                     "min":   float(np.min(alpha)),
                     "max":   float(np.max(alpha)),
                 }
        for p, v in zip(percentiles, pcts):
            result[f"p{p:02d}"] = float(v)
        return result
