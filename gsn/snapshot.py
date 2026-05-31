from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import numpy as np


@dataclass
class Snapshot:
    """A mini-graph snapshot for one time bucket.

    node_ids  : [N] int64   — global node IDs present in this snapshot
    edge_src  : [E] or [L, E] int32  — local source indices into node_ids
    edge_dst  : [E] or [L, E] int32  — local destination indices into node_ids
    num_nodes : int        — N (len of node_ids)
    t_ref     : int        — reference timestamp (end of this bucket)
    dt        : float      — time since last commit (bucket width)
    edge_feat : [E, F_e] or [L, E, F_e] float32 or None
    edge_ts   : [E] or [L, E] int64 or None  — per-edge timestamps
    x         : [N, F_n] or [N, L, F_n] float32 or None  — optional node features
    """

    node_ids:  np.ndarray
    edge_src:  np.ndarray
    edge_dst:  np.ndarray
    num_nodes: int
    t_ref:     int
    dt:        float
    edge_feat: Optional[np.ndarray] = None
    edge_ts:   Optional[np.ndarray] = None
    x:         Optional[np.ndarray] = None
    seq_t_ref: Optional[np.ndarray] = None
    seq_dt:    Optional[np.ndarray] = None
    actual_seq_len: int = 1

    def __len__(self) -> int:
        return self.num_nodes

    @property
    def num_edges(self) -> int:
        edge_src = np.asarray(self.edge_src)
        if edge_src.ndim == 1:
            return int(edge_src.shape[0])
        return int(np.count_nonzero(edge_src >= 0))

    @property
    def is_sequence(self) -> bool:
        return np.asarray(self.edge_src).ndim == 2

    @property
    def sequence_length(self) -> int:
        return int(self.edge_src.shape[0]) if self.is_sequence else 1
    
    @property
    def shape(self) -> Tuple[int]:
        return (int(self.num_nodes), )

    @classmethod
    def from_events(
                        cls,
                        src_global: np.ndarray,
                        dst_global: np.ndarray,
                        timestamps:  np.ndarray,
                        t_ref:       int,
                        dt:          float,
                        edge_feat:   Optional[np.ndarray] = None,
                        x:           Optional[np.ndarray] = None,
                    ) -> "Snapshot":
        """Build a Snapshot from raw (global-id) event arrays.

        Global node IDs are compacted to a contiguous local index space.
        Self-loops are not added here; add them before calling if desired.
        """
        src_g = np.asarray(src_global, dtype = np.int64)
        dst_g = np.asarray(dst_global, dtype = np.int64)

        all_ids = np.unique(np.concatenate([src_g, dst_g]))
        id_to_local = {int(gid): i for i, gid in enumerate(all_ids)}
        N = len(all_ids)

        edge_src = np.array([id_to_local[int(u)] for u in src_g], dtype = np.int32)
        edge_dst = np.array([id_to_local[int(v)] for v in dst_g], dtype = np.int32)

        return cls(
                    node_ids  = all_ids,
                    edge_src  = edge_src,
                    edge_dst  = edge_dst,
                    num_nodes = N,
                    t_ref     = int(t_ref),
                    dt        = float(dt),
                    edge_feat = edge_feat,
                    edge_ts   = np.asarray(timestamps, dtype = np.int64),
                    x         = x,
                  )
    
    @classmethod
    def concatenate(
                        cls,
                        snapshots: Sequence[Snapshot],
                        seq_len: Optional[int] = None,
                        extra_node_ids: Optional[np.ndarray] = None,
                    ) -> "Snapshot":
        """Pack scalar snapshots into one padded sequence snapshot.

        The returned snapshot uses one shared local node-index space over the
        union of all input ``node_ids`` plus any ``extra_node_ids``.  Each input
        snapshot's local edge endpoints are remapped into that shared space.

        Sequence fields are padded to ``seq_len`` (or ``len(snapshots)`` when
        omitted).  Edge arrays have shape ``[L, E_max]`` where ``E_max`` is the
        largest edge count in any input snapshot; unused edge slots are filled
        with ``-1`` in ``edge_src``, ``edge_dst``, and ``edge_ts``.  If edge
        features are present, ``edge_feat`` has shape ``[L, E_max, F_e]`` and
        padded feature slots are zero.  If node features are present, ``x`` has
        shape ``[N_union, L, F_n]`` and is zero for nodes absent from a step.

        Aggregate metadata describes only the real, unpadded inputs:
        ``t_ref`` is the maximum input ``t_ref``, ``dt`` is the sum of input
        ``dt`` values, and ``actual_seq_len`` records ``len(snapshots)``.
        Per-step timing is stored in ``seq_t_ref`` and ``seq_dt``.

        Args:
            snapshots: A time-contiguous sequence of snapshots (contiguity
                is not checked).  Each input must be a scalar snapshot, not an
                already-concatenated sequence snapshot.
            seq_len: Optional padded sequence length.  Must be at least
                ``len(snapshots)`` when provided.
            extra_node_ids: Optional global node ids to include in the shared
                node set even if they do not appear in the input snapshots.
        
        Returns:
            A sequence ``Snapshot`` with 2-D padded edge arrays and optional
            3-D node/edge feature arrays.
        """
        snaps = list(snapshots)
        if seq_len is None:
            seq_len = len(snaps)
        seq_len = int(seq_len)
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if len(snaps) > seq_len:
            raise ValueError(f"Cannot concatenate {len(snaps)} snapshots into seq_len = {seq_len}.")
        for snap in snaps:
            if snap.is_sequence:
                raise ValueError("Snapshot.concatenate expects scalar snapshots.")

        node_parts = [np.asarray(s.node_ids, dtype = np.int64) for s in snaps]
        if extra_node_ids is not None:
            node_parts.append(np.asarray(extra_node_ids, dtype = np.int64).reshape(-1))
        node_parts = [part for part in node_parts if part.size]
        if node_parts:
            node_ids = np.unique(np.concatenate(node_parts)).astype(np.int64)
        else:
            node_ids = np.zeros((0,), dtype = np.int64)
        id_to_local = {int(gid): i for i, gid in enumerate(node_ids)}

        max_edges = max((int(s.edge_src.shape[0]) for s in snaps), default = 0)
        edge_src = np.full((seq_len, max_edges), -1, dtype = np.int64)
        edge_dst = np.full((seq_len, max_edges), -1, dtype = np.int64)

        have_edge_ts = any(s.edge_ts is not None for s in snaps)
        edge_ts = (
                    np.full((seq_len, max_edges), -1, dtype = np.int64)
                    if have_edge_ts
                    else None
                  )

        edge_feat_dim = _consistent_last_dim(s.edge_feat for s in snaps if s.edge_feat is not None)
        edge_feat = (
                        np.zeros((seq_len, max_edges, edge_feat_dim), dtype = np.float32)
                        if edge_feat_dim is not None
                        else None
                    )

        x_dim = _consistent_last_dim(s.x for s in snaps if s.x is not None)
        x = (
                np.zeros((node_ids.shape[0], seq_len, x_dim), dtype = np.float32)
                if x_dim is not None
                else None
            )

        seq_t_ref = np.zeros((seq_len,), dtype = np.int64)
        seq_dt = np.zeros((seq_len,), dtype = np.float32)

        for step, snap in enumerate(snaps):
            e_count = int(snap.edge_src.shape[0])
            if e_count:
                src_global = snap.node_ids[np.asarray(snap.edge_src, dtype = np.int64)]
                dst_global = snap.node_ids[np.asarray(snap.edge_dst, dtype = np.int64)]
                edge_src[step, :e_count] = [id_to_local[int(gid)] for gid in src_global]
                edge_dst[step, :e_count] = [id_to_local[int(gid)] for gid in dst_global]

            if edge_ts is not None:
                if snap.edge_ts is not None:
                    edge_ts[step, :e_count] = np.asarray(snap.edge_ts, dtype = np.int64)
                elif e_count:
                    edge_ts[step, :e_count] = int(snap.t_ref)

            if edge_feat is not None and snap.edge_feat is not None:
                feat = np.asarray(snap.edge_feat, dtype = np.float32)
                if feat.ndim != 2:
                    raise ValueError("Scalar snapshot edge_feat must be rank 2.")
                edge_feat[step, :e_count, :] = feat

            if x is not None and snap.x is not None:
                x_step = np.asarray(snap.x, dtype = np.float32)
                if x_step.ndim != 2:
                    raise ValueError("Scalar snapshot x must be rank 2.")
                local = [id_to_local[int(gid)] for gid in snap.node_ids]
                x[local, step, :] = x_step

            seq_t_ref[step] = int(snap.t_ref)
            seq_dt[step] = float(snap.dt)

        if snaps and len(snaps) < seq_len:
            seq_t_ref[len(snaps) :] = seq_t_ref[len(snaps) - 1]

        actual_seq_len = len(snaps)
        t_ref = int(np.max(seq_t_ref[:actual_seq_len])) if actual_seq_len else 0
        dt = float(np.sum(seq_dt[:actual_seq_len])) if actual_seq_len else 0.0

        return cls(
                    node_ids       = node_ids,
                    edge_src       = edge_src,
                    edge_dst       = edge_dst,
                    num_nodes      = int(node_ids.shape[0]),
                    t_ref          = t_ref,
                    dt             = dt,
                    edge_feat      = edge_feat,
                    edge_ts        = edge_ts,
                    x              = x,
                    seq_t_ref      = seq_t_ref,
                    seq_dt         = seq_dt,
                    actual_seq_len = actual_seq_len
                  )


def _consistent_last_dim(arrays: Iterable[np.ndarray]) -> Optional[int]:
    dim: Optional[int] = None
    for arr in arrays:
        value = np.asarray(arr)
        if value.ndim < 1:
            raise ValueError("Expected an array with at least one dimension.")
        last_dim = int(value.shape[-1])
        if dim is None:
            dim = last_dim
        elif dim != last_dim:
            raise ValueError(f"Inconsistent feature dimensions: {dim} and {last_dim}.")
    return dim