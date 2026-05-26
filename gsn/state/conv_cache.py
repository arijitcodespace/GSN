"""Persistent per-node convolution-cache table for Mamba-2 step mode.

The Mamba-2 block's causal DepthwiseConv1D needs (kernel_size − 1) past
XBC tokens to produce a true streaming output. In step mode that history
is zeroed out by default; ``ConvCacheTable`` makes it persistent across
snapshots, mirroring how ``DenseStateTable`` persists the SSM state.

Differences vs ``DenseStateTable``:

* Native 3-D layout ``[num_entities, kernel_size − 1, channels]`` so the
  caller never has to flatten / reshape.
* Overwrite-only semantics: a node's cache is a sliding window, not an
  EMA, so ``put`` always replaces.
* Duplicate IDs in a single ``put`` are collapsed by averaging (they
  should not occur because ``snap.node_ids`` is already unique).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class ConvCacheTable(layers.Layer):
    """Per-node persistent conv cache for Mamba-2 step mode.

    Parameters
    ----------
    num_entities : int  — total number of nodes (global IDs in [0, num_entities-1])
    kernel_size  : int  — Mamba-2 causal-conv kernel size (K). The cache stores
                          the last ``K - 1`` XBC tokens per node.
    channels     : int  — XBC channel count, i.e. ``H * (2N + P)``.
    """

    def __init__(
                    self,
                    num_entities: int,
                    kernel_size:  int,
                    channels:     int,
                    name:         str = "conv_cache_table",
                    **kwargs,
                ):
        super().__init__(name = name, trainable = False, **kwargs)
        if kernel_size < 2:
            raise ValueError(
                f"kernel_size must be >= 2 for a non-empty conv cache "
                f"(got {kernel_size})"
            )
        self.num_entities = int(num_entities)
        self.kernel_size  = int(kernel_size)
        self.channels     = int(channels)
        self.history_len  = self.kernel_size - 1

        self._table = self.add_weight(
                                        shape       = (self.num_entities,
                                                       self.history_len,
                                                       self.channels),
                                        trainable   = False,
                                        initializer = "zeros",
                                        dtype       = tf.float32,
                                        name        = "table",
                                     )

    def call(self, inputs = None):
        return self._table

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing = True)
    def get(self, ids: tf.Tensor) -> tf.Tensor:
        """Return cached XBC history for the given (possibly repeated) node IDs.

        Returns ``[|ids|, kernel_size - 1, channels]`` float32.
        """
        return tf.gather(tf.cast(self._table, tf.float32),
                         tf.cast(ids, tf.int32))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing = True)
    def put(self, ids: tf.Tensor, new_caches: tf.Tensor) -> None:
        """Overwrite the cache for the given node IDs (sliding-window update).

        Parameters
        ----------
        ids        : [M]                       int32 node IDs
        new_caches : [M, K-1, C]               float32 fresh per-node histories

        Duplicate IDs are collapsed by averaging (defensive — should not occur
        in practice since ``snap.node_ids`` is unique).
        """
        ids        = tf.cast(ids, tf.int32)
        new_caches = tf.cast(new_caches, tf.float32)

        uniq_ids, idx = tf.unique(ids)
        agg = tf.math.unsorted_segment_mean(
                                                new_caches, idx,
                                                tf.shape(uniq_ids)[0],
                                            )                  # [U, K-1, C]

        self._table.assign(
            tf.tensor_scatter_nd_update(
                tf.cast(self._table, tf.float32),
                indices = tf.expand_dims(uniq_ids, axis = 1),
                updates = agg,
            )
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @tf.function
    def reset_ids(self, ids: tf.Tensor) -> None:
        """Zero out the cache for the given node IDs."""
        ids   = tf.cast(ids, tf.int32)
        zeros = tf.zeros(
                            [tf.shape(ids)[0], self.history_len, self.channels],
                            dtype = tf.float32,
                        )
        self._table.assign(
            tf.tensor_scatter_nd_update(
                tf.cast(self._table, tf.float32),
                tf.expand_dims(ids, 1),
                zeros,
            )
        )

    @tf.function
    def reset_state(self) -> None:
        """Zero the entire cache."""
        self._table.assign(
            tf.zeros(
                [self.num_entities, self.history_len, self.channels],
                tf.float32,
            )
        )

    @tf.function
    def clone(self) -> tf.Tensor:
        """Return a copy of the table as a plain tensor.

        Used by the trainer to snapshot conv-cache state before evaluation
        and restore it afterwards, so per-epoch eval mutation does not leak
        into the next training epoch or contaminate the saved checkpoint.
        """
        return tf.identity(self._table)

    @tf.function
    def copy_from(self, src: tf.Tensor) -> None:
        """Overwrite the table from a tensor of same shape."""
        self._table.assign(tf.cast(src, tf.float32))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                num_entities = self.num_entities,
                kernel_size  = self.kernel_size,
                channels     = self.channels,
            )
        )
        return cfg
