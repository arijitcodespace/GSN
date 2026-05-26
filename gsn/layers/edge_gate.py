"""Per-edge, per-head gating MLP."""

from __future__ import annotations
from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers


class EdgeGate(layers.Layer):
    """
    Lightweight MLP producing per-edge gates in \\mathbb{R}^H.

    Input  : concatenated edge features [E, F_in]  (hi, hj, edge_feat, time_enc, …)
    Output : raw logits [E, num_heads]  (caller applies activation if desired)

    The output is left as raw logits so the caller can choose
    sigmoid (0,1) gating or (1+tanh)/2 ∈ (0,2) gating.
    """

    def __init__(
                    self,
                    num_heads: int,
                    hidden: int = 32,
                    name: Optional[str] = None,
                ):
        super().__init__(name = name)
        self.prefix = f"{name}_" if name is not None else ""
        self.num_heads = int(num_heads)
        self.hidden = int(hidden)

    def build(self, input_shape):
        if self.hidden > 0:
            self.d1 = layers.Dense(self.hidden, activation = "relu", name = f"{self.prefix}dense_0")
            self.d1.build(input_shape)
        else:
            self.d1 = None
        self.out = layers.Dense(self.num_heads, activation = None,
                                bias_initializer = "glorot_normal", name = f"{self.prefix}dense_1")
        shape = [*input_shape[:-1], self.hidden] if self.hidden > 0 else input_shape
        self.out.build(shape)
        super().build(input_shape)

    def call(self, edge_feat: tf.Tensor, training = None) -> tf.Tensor:
        z = self.d1(edge_feat, training = training) if self.d1 is not None else edge_feat
        return self.out(z, training = training)   # [E, num_heads]

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(num_heads = self.num_heads, hidden = self.hidden))
        return cfg
