"""Graph message-passing ops (TF-backed) and pure-numpy utilities."""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Pure numpy (no TF required at import time)
# ---------------------------------------------------------------------------

def add_self_loops_local(
    edge_src:  np.ndarray,
    edge_dst:  np.ndarray,
    num_nodes: int,
    edge_feat: Optional[np.ndarray] = None,
    edge_ts:   Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Append self-loops (i→i for i in [0, num_nodes)) to edge arrays."""
    loop = np.arange(num_nodes, dtype=np.int32)
    src_new = np.concatenate([edge_src, loop])
    dst_new = np.concatenate([edge_dst, loop])

    feat_new = None
    if edge_feat is not None:
        loop_feat = np.zeros((num_nodes, edge_feat.shape[1]), dtype=edge_feat.dtype)
        feat_new = np.concatenate([edge_feat, loop_feat], axis=0)

    ts_new = None
    if edge_ts is not None:
        loop_ts = np.zeros(num_nodes, dtype=edge_ts.dtype)
        ts_new = np.concatenate([edge_ts, loop_ts])

    return src_new, dst_new, feat_new, ts_new


# ---------------------------------------------------------------------------
# TF-backed (TF imported lazily so the module loads without TF installed)
# ---------------------------------------------------------------------------

def gather_src(x, edge_src):
    """Gather source-node features for each edge.  x: [N,F], edge_src: [E] → [E,F]."""
    import tensorflow as tf
    return tf.gather(x, tf.cast(edge_src, tf.int32))


def gather_dst(x, edge_dst):
    """Gather destination-node features for each edge."""
    import tensorflow as tf
    return tf.gather(x, tf.cast(edge_dst, tf.int32))


def aggregate(messages, edge_dst, num_nodes: int):
    """Sum-aggregate messages to destination nodes.  messages:[E,F], edge_dst:[E] → [N,F]."""
    import tensorflow as tf
    return tf.math.unsorted_segment_sum(
        messages, tf.cast(edge_dst, tf.int32), num_nodes
    )
