"""Pair-recurrence buffers for scoring-side temporal link features.

The buffer tracks only historical positive edges.  It is intentionally not a
Keras layer: pair histories are non-trainable bookkeeping, analogous to
``NodeActivityBuffers`` for adaptive commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


class PairRecurrenceBuffers:
    """Sparse pair-history table used to build candidate-pair features.

    Features returned by :meth:`get_features` are, in order:

    1. ``seen_before`` in ``{0, 1}``
    2. ``log1p(pair_count)``
    3. ``log1p(delta_t_since_last_seen)`` (0 for unseen pairs)
    4. ``exp(-delta_t_since_last_seen / tau_data)`` (0 for unseen pairs)

    Updates are one-per-positive-event.  Repeated pairs inside the same bucket
    increment the count once per event and keep the maximum timestamp.
    """

    feature_dim: int = 4

    def __init__(
        self,
        num_nodes: int,
        tau_data: float = 1.0,
        undirected: bool = False,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.tau_data = float(max(tau_data, 1e-8))
        self.undirected = bool(undirected)
        self.counts: dict[int, int] = {}
        self.last_seen_time: dict[int, float] = {}

    def _keys(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        src_arr = np.asarray(src, dtype=np.int64).reshape(-1)
        dst_arr = np.asarray(dst, dtype=np.int64).reshape(-1)
        if src_arr.shape[0] != dst_arr.shape[0]:
            raise ValueError(
                f"src and dst must have the same length; got "
                f"{src_arr.shape[0]} and {dst_arr.shape[0]}"
            )
        if self.undirected:
            a = np.minimum(src_arr, dst_arr)
            b = np.maximum(src_arr, dst_arr)
        else:
            a, b = src_arr, dst_arr
        return a * np.int64(self.num_nodes) + b

    @staticmethod
    def _broadcast_time(current_ts: Union[float, np.ndarray], size: int) -> np.ndarray:
        ts = np.asarray(current_ts, dtype=np.float64)
        if ts.ndim == 0:
            return np.full(size, float(ts), dtype=np.float64)
        ts = ts.reshape(-1)
        if ts.shape[0] != size:
            raise ValueError(
                f"current_ts must be scalar or length {size}; got length {ts.shape[0]}"
            )
        return ts.astype(np.float64)

    def get_features(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        current_ts: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Return recurrence features for candidate pairs before any update."""
        keys = self._keys(src, dst)
        now = self._broadcast_time(current_ts, keys.shape[0])

        counts = np.fromiter(
            (self.counts.get(int(k), 0) for k in keys),
            dtype=np.float64,
            count=keys.shape[0],
        )
        last = np.fromiter(
            (self.last_seen_time.get(int(k), 0.0) for k in keys),
            dtype=np.float64,
            count=keys.shape[0],
        )

        seen = counts > 0.0
        delta = np.where(seen, np.maximum(now - last, 0.0), 0.0)
        recency = np.where(seen, np.exp(-delta / self.tau_data), 0.0)

        return np.stack(
            [
                seen.astype(np.float32),
                np.log1p(counts).astype(np.float32),
                np.log1p(delta).astype(np.float32),
                recency.astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

    def update(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        timestamps: Union[float, np.ndarray],
    ) -> None:
        """Update the table from positive edges only."""
        keys = self._keys(src, dst)
        ts = self._broadcast_time(timestamps, keys.shape[0])
        for key, t in zip(keys, ts):
            k = int(key)
            self.counts[k] = self.counts.get(k, 0) + 1
            old_t = self.last_seen_time.get(k, None)
            self.last_seen_time[k] = float(t) if old_t is None else max(float(old_t), float(t))

    def reset(self) -> None:
        self.counts.clear()
        self.last_seen_time.clear()

    def clone(self) -> "PairRecurrenceBuffers":
        copy = PairRecurrenceBuffers(
            num_nodes=self.num_nodes,
            tau_data=self.tau_data,
            undirected=self.undirected,
        )
        copy.counts = dict(self.counts)
        copy.last_seen_time = dict(self.last_seen_time)
        return copy

    def copy_from(self, src: "PairRecurrenceBuffers") -> None:
        if self.num_nodes != src.num_nodes:
            raise ValueError(
                f"num_nodes mismatch: self={self.num_nodes}, src={src.num_nodes}"
            )
        self.tau_data = float(src.tau_data)
        self.undirected = bool(src.undirected)
        self.counts = dict(src.counts)
        self.last_seen_time = dict(src.last_seen_time)

    def to_npz_dict(self) -> dict:
        keys = np.asarray(
            sorted(set(self.counts.keys()) | set(self.last_seen_time.keys())),
            dtype=np.int64,
        )
        counts = np.asarray([self.counts.get(int(k), 0) for k in keys], dtype=np.int64)
        last = np.asarray(
            [self.last_seen_time.get(int(k), 0.0) for k in keys],
            dtype=np.float64,
        )
        return {
            "pair_keys": keys,
            "pair_counts": counts,
            "pair_last_seen_time": last,
            "pair_tau_data": np.asarray(self.tau_data, dtype=np.float64),
            "pair_undirected": np.asarray(int(self.undirected), dtype=np.int8),
            "pair_num_nodes": np.asarray(self.num_nodes, dtype=np.int64),
        }

    def save_npz(self, path: Union[str, Path]) -> None:
        np.savez(str(path), **self.to_npz_dict())

    def load_npz(self, path: Union[str, Path]) -> None:
        data = np.load(str(path), allow_pickle=False)
        try:
            self.load_npz_data(data)
        finally:
            data.close()

    def load_npz_data(self, data) -> None:
        keys = np.asarray(data["pair_keys"], dtype=np.int64).reshape(-1)
        counts = np.asarray(data["pair_counts"], dtype=np.int64).reshape(-1)
        last = np.asarray(data["pair_last_seen_time"], dtype=np.float64).reshape(-1)
        if not (keys.shape[0] == counts.shape[0] == last.shape[0]):
            raise ValueError("pair recurrence npz arrays have inconsistent lengths")
        saved_num_nodes = (
            int(np.asarray(data["pair_num_nodes"]).item())
            if "pair_num_nodes" in data else self.num_nodes
        )
        if saved_num_nodes != self.num_nodes:
            raise ValueError(
                f"pair recurrence num_nodes mismatch: checkpoint={saved_num_nodes}, "
                f"model={self.num_nodes}"
            )
        if "pair_tau_data" in data:
            self.tau_data = float(np.asarray(data["pair_tau_data"]).item())
        if "pair_undirected" in data:
            self.undirected = bool(int(np.asarray(data["pair_undirected"]).item()))
        self.counts = {int(k): int(c) for k, c in zip(keys, counts) if int(c) > 0}
        self.last_seen_time = {
            int(k): float(t) for k, t in zip(keys, last) if int(k) in self.counts
        }
