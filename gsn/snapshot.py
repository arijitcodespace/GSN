from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Snapshot:
    """A mini-graph snapshot for one time bucket.

    node_ids : [N] int64   — global node IDs present in this snapshot
    edge_src  : [E] int32  — local source indices into node_ids
    edge_dst  : [E] int32  — local destination indices into node_ids
    num_nodes : int        — N (len of node_ids)
    t_ref     : int        — reference timestamp (end of this bucket)
    dt        : float      — seconds since last commit (bucket width)
    edge_feat : [E, F_e] float32 or None
    edge_ts   : [E] int64 or None  — per-edge timestamps
    x         : [N, F_n] float32 or None  — optional node features
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

    def __len__(self) -> int:
        return self.num_nodes

    @property
    def num_edges(self) -> int:
        return int(self.edge_src.shape[0])
    
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
                    edge_ts   = np.asarray(timestamps, dtype=np.int64),
                    x         = x,
                  )
