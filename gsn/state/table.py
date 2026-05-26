from __future__ import annotations
from typing import Callable, Union
import tensorflow as tf
from tensorflow.keras import layers

_num_entities: int = None
_state_dim: int    = None

class DenseStateTable(layers.Layer):
    """Dense TF-backed node-id → state table.

    Stores a non-trainable float32 variable of shape [num_entities, state_dim].
    All reads/writes use pure TF ops so the table is compatible with tf.function.

    Parameters
    ----------
    num_entities : int   — total number of nodes (global IDs in [0, N-1])
    state_dim    : int   — flat state dimension per node
    """

    def __init__(
                    self,
                    num_entities: int,
                    state_dim: int,
                    name: str = "state_table",
                    **kwargs
                ):
        super().__init__(name = name, trainable = False, **kwargs)
        self.num_entities = int(num_entities)
        self.state_dim = int(state_dim)
        
        _num_entities = self.num_entities
        _state_dim    = self.state_dim
        
        self._table = self.add_weight(
                                        shape = (self.num_entities, self.state_dim),
                                        trainable = False,
                                        initializer = "zeros",
                                        dtype = tf.float32,
                                        name = "table"
                                     )

    def call(self, inputs = None):
        return self._table

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @tf.function(
                    reduce_retracing = True,
                    input_signature = [tf.TensorSpec(shape = (None, ), dtype = tf.int32)]
                )
    def get(self, ids: tf.Tensor) -> tf.Tensor:
        """Return states for the given (possibly repeated) node IDs.

        Returns [|ids|, state_dim] float32.
        """
        return tf.gather(tf.cast(self._table, tf.float32), tf.cast(ids, tf.int32))

    # ------------------------------------------------------------------
    # Write
    @tf.function(reduce_retracing = True,
                 input_signature = [
                                        tf.TensorSpec(shape = (None, ), dtype = tf.int32),
                                        tf.TensorSpec(shape = (None, _state_dim), dtype = tf.float32),
                                        tf.TensorSpec(shape = (None, 1), dtype = tf.float32)
                                   ])
    def put(
                self,
                ids: tf.Tensor,
                new_states: tf.Tensor,
                alpha: tf.Tensor
            ) -> None:
        """
        EMA commit: S_i ← S_i + α · (s_new − S_i).

        ``alpha`` may be:
          • a Python float / scalar tf.Tensor  — uniform commit (original behaviour)
          • a callable returning a scalar      — uniform commit, evaluated lazily
          • a [N, 1] or [N] tf.Tensor         — per-node adaptive commit

        Duplicate IDs are collapsed by tf.unique before the scatter-update.
        When alpha is per-node, duplicate entries are averaged (they should not
        occur in practice since snap.node_ids is already unique).
        """
        if callable(alpha):
            alpha = tf.cast(alpha(), tf.float32)
        else:
            alpha = tf.cast(alpha, tf.float32)

        ids        = tf.cast(ids,        tf.int32)
        new_states = tf.cast(new_states, tf.float32)

        uniq_ids, idx = tf.unique(ids)
        old     = tf.cast(tf.gather(self._table, uniq_ids), tf.float32)
        new_agg = tf.math.unsorted_segment_mean(
                        new_states, idx, tf.shape(uniq_ids)[0]
                  )

        # Dispatch on alpha rank: scalar → uniform path, tensor → per-node path
        alpha_rank = len(alpha.shape)
        if alpha_rank == 0:
            # Scalar: retain the fast tf.cond short-circuit for α ≥ 1
            committed = tf.cond(
                            alpha >= 1.0,
                            lambda: new_agg,
                            lambda: old + alpha * (new_agg - old),
                        )
        else:
            # Per-node tensor: [N] or [N, 1] → aggregate duplicates, then broadcast
            alpha_agg = tf.math.unsorted_segment_mean(
                            tf.reshape(alpha, [-1, 1]),
                            idx,
                            tf.shape(uniq_ids)[0],
                        )                           # [N_uniq, 1]
            committed = old + alpha_agg * (new_agg - old)

        self._table.assign(
                                tf.tensor_scatter_nd_update(
                                    tf.cast(self._table, tf.float32),
                                    indices = tf.expand_dims(uniq_ids, axis = 1),
                                    updates = committed
                                )
                           )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @tf.function
    def reset_ids(self, ids: tf.Tensor) -> None:
        """Zero out states for the given node IDs."""
        ids = tf.cast(ids, tf.int32)
        zeros = tf.zeros([tf.shape(ids)[0], self.state_dim], dtype=tf.float32)
        self._table.assign(
                                tf.tensor_scatter_nd_update(
                                                                tf.cast(self._table, tf.float32),
                                                                tf.expand_dims(ids, 1),
                                                                zeros
                                                            )
                          )

    @tf.function
    def reset_state(self) -> None:
        """Zero the entire table."""
        self._table.assign(tf.zeros([self.num_entities, self.state_dim], tf.float32))

    @tf.function
    def clone(self) -> tf.Tensor:
        """Return a copy of the table as a plain tensor."""
        return tf.identity(self._table)

    @tf.function
    def copy_from(self, src: tf.Tensor) -> None:
        """Overwrite the table from a tensor of same shape."""
        self._table.assign(tf.cast(src, tf.float32))
