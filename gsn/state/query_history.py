"""Bounded recent-neighbor history buffers for scorer-side query features."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


class QueryHistoryBuffers:
    """Sparse per-node recent interaction history.

    The buffer stores only historical positive edges and exposes bounded,
    query-conditional features for candidate pairs.  It is intentionally
    scorer-side: features are computed before the current positive bucket is
    written, then the buffer is updated after scoring/commit.
    """

    feature_dim: int = 14

    def __init__(
        self,
        num_nodes: int,
        history_k: int = 16,
        tau_data: float = 1.0,
        undirected: bool = True,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.history_k = max(int(history_k), 1)
        self.tau_data = float(max(tau_data, 1e-8))
        self.undirected = bool(undirected)
        self.histories: dict[int, list[tuple[int, float]]] = {}

    @staticmethod
    def _as_int_array(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.int64).reshape(-1)

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

    def _append_one(self, node: int, neighbor: int, timestamp: float) -> None:
        hist = self.histories.setdefault(int(node), [])
        hist.append((int(neighbor), float(timestamp)))
        if len(hist) > self.history_k:
            del hist[: len(hist) - self.history_k]

    def update(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        timestamps: Union[float, np.ndarray],
    ) -> None:
        """Update from positive edges only, once per event."""
        src_arr = self._as_int_array(src)
        dst_arr = self._as_int_array(dst)
        if src_arr.shape[0] != dst_arr.shape[0]:
            raise ValueError("src and dst must have the same length")
        ts = self._broadcast_time(timestamps, src_arr.shape[0])
        for u, v, t in zip(src_arr, dst_arr, ts):
            self._append_one(int(u), int(v), float(t))
            if self.undirected:
                self._append_one(int(v), int(u), float(t))

    def _node_stats(self, hist: list[tuple[int, float]], other: int, now: float):
        if not hist:
            return 0.0, 0.0, 0.0, 0.0, set(), 0.0

        neigh = np.asarray([n for n, _ in hist], dtype=np.int64)
        ts = np.asarray([t for _, t in hist], dtype=np.float64)
        delta = np.maximum(float(now) - ts, 0.0)
        recency_all = np.exp(-delta / self.tau_data)

        matches = neigh == int(other)
        if np.any(matches):
            last_ts = float(np.max(ts[matches]))
            pair_delta = max(float(now) - last_ts, 0.0)
            pair_count = float(np.sum(matches))
            pair_recency = float(np.exp(-pair_delta / self.tau_data))
        else:
            pair_delta = 0.0
            pair_count = 0.0
            pair_recency = 0.0

        return (
            float(len(hist)),
            pair_count,
            float(np.log1p(pair_delta)) if pair_count > 0.0 else 0.0,
            pair_recency,
            set(int(x) for x in neigh.tolist()),
            float(np.mean(recency_all)) if recency_all.size else 0.0,
        )

    def get_features(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        current_ts: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Return bounded recent-history features for candidate pairs."""
        src_arr = self._as_int_array(src)
        dst_arr = self._as_int_array(dst)
        if src_arr.shape[0] != dst_arr.shape[0]:
            raise ValueError("src and dst must have the same length")
        now_arr = self._broadcast_time(current_ts, src_arr.shape[0])

        out = np.zeros((src_arr.shape[0], self.feature_dim), dtype=np.float32)
        for i, (u, v, now) in enumerate(zip(src_arr, dst_arr, now_arr)):
            src_hist = self.histories.get(int(u), [])
            dst_hist = self.histories.get(int(v), [])

            (
                src_len,
                src_pair_count,
                src_pair_logdt,
                src_pair_recency,
                src_neigh,
                src_mean_recency,
            ) = self._node_stats(src_hist, int(v), float(now))

            (
                dst_len,
                dst_pair_count,
                dst_pair_logdt,
                dst_pair_recency,
                dst_neigh,
                dst_mean_recency,
            ) = self._node_stats(dst_hist, int(u), float(now))

            common = src_neigh.intersection(dst_neigh)
            union = src_neigh.union(dst_neigh)
            out[i] = np.asarray(
                [
                    np.log1p(src_len),
                    np.log1p(dst_len),
                    1.0 if src_pair_count > 0.0 else 0.0,
                    1.0 if dst_pair_count > 0.0 else 0.0,
                    np.log1p(src_pair_count),
                    np.log1p(dst_pair_count),
                    src_pair_logdt,
                    dst_pair_logdt,
                    src_pair_recency,
                    dst_pair_recency,
                    src_mean_recency,
                    dst_mean_recency,
                    np.log1p(len(common)),
                    len(common) / max(len(union), 1) if union else 0.0,
                ],
                dtype=np.float32,
            )

        return out

    def reset(self) -> None:
        self.histories.clear()

    def clone(self) -> "QueryHistoryBuffers":
        copy = QueryHistoryBuffers(
            num_nodes=self.num_nodes,
            history_k=self.history_k,
            tau_data=self.tau_data,
            undirected=self.undirected,
        )
        copy.histories = {k: list(v) for k, v in self.histories.items()}
        return copy

    def copy_from(self, src: "QueryHistoryBuffers") -> None:
        if self.num_nodes != src.num_nodes:
            raise ValueError(
                f"num_nodes mismatch: self={self.num_nodes}, src={src.num_nodes}"
            )
        self.history_k = int(src.history_k)
        self.tau_data = float(src.tau_data)
        self.undirected = bool(src.undirected)
        self.histories = {k: list(v) for k, v in src.histories.items()}

    def to_npz_dict(self) -> dict:
        node_ids = np.asarray(sorted(self.histories.keys()), dtype=np.int64)
        offsets = [0]
        neighbors = []
        timestamps = []
        for node in node_ids:
            hist = self.histories.get(int(node), [])
            neighbors.extend([n for n, _ in hist])
            timestamps.extend([t for _, t in hist])
            offsets.append(len(neighbors))
        return {
            "query_history_node_ids": node_ids,
            "query_history_offsets": np.asarray(offsets, dtype=np.int64),
            "query_history_neighbors": np.asarray(neighbors, dtype=np.int64),
            "query_history_timestamps": np.asarray(timestamps, dtype=np.float64),
            "query_history_k": np.asarray(self.history_k, dtype=np.int64),
            "query_history_tau_data": np.asarray(self.tau_data, dtype=np.float64),
            "query_history_undirected": np.asarray(int(self.undirected), dtype=np.int8),
            "query_history_num_nodes": np.asarray(self.num_nodes, dtype=np.int64),
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
        saved_num_nodes = (
            int(np.asarray(data["query_history_num_nodes"]).item())
            if "query_history_num_nodes" in data else self.num_nodes
        )
        if saved_num_nodes != self.num_nodes:
            raise ValueError(
                f"query history num_nodes mismatch: checkpoint={saved_num_nodes}, "
                f"model={self.num_nodes}"
            )
        self.history_k = int(np.asarray(data["query_history_k"]).item())
        self.tau_data = float(np.asarray(data["query_history_tau_data"]).item())
        self.undirected = bool(
            int(np.asarray(data["query_history_undirected"]).item())
        )
        node_ids = np.asarray(data["query_history_node_ids"], dtype=np.int64)
        offsets = np.asarray(data["query_history_offsets"], dtype=np.int64)
        neighbors = np.asarray(data["query_history_neighbors"], dtype=np.int64)
        timestamps = np.asarray(data["query_history_timestamps"], dtype=np.float64)
        if offsets.shape[0] != node_ids.shape[0] + 1:
            raise ValueError("query history offsets length mismatch")
        self.histories = {}
        for i, node in enumerate(node_ids):
            start, end = int(offsets[i]), int(offsets[i + 1])
            hist = [
                (int(n), float(t))
                for n, t in zip(neighbors[start:end], timestamps[start:end])
            ]
            self.histories[int(node)] = hist[-self.history_k :]
