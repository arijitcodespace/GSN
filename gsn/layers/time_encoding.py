"""TGAT-style harmonic time encoding with learnable frequencies."""

from __future__ import annotations
from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TGATTimeEncoding(layers.Layer):
    """Encode a scalar (or per-node/per-edge) Δt into a d-dimensional vector.

    φ(Δt) = [cos(ω₁·Δt), sin(ω₁·Δt), …, cos(ω_{d/2}·Δt), sin(ω_{d/2}·Δt)]

    Frequencies ω are learnable; inputs are optionally clipped and scaled.

    Input  : arbitrary shape [...], int64 or float
    Output : [..., dim]
    """

    def __init__(
                    self,
                    dim: int,
                    time_scale: float = 1.0,
                    max_dt: Optional[float] = None,
                    name: Optional[str] = None,
                    **kwargs
                ):
        super().__init__(name = name)
        self.dim = int(dim)
        self.time_scale = float(time_scale)
        self.max_dt = float(max_dt) if max_dt is not None else None

    def build(self, input_shape):
        half = max(1, self.dim // 2)
        self.freq = self.add_weight(
                                        name = "freq",
                                        shape = (half,),
                                        initializer = keras.initializers.RandomUniform(0.0, 1.0),
                                        trainable = True,
                                   )
        super().build(input_shape)

    def call(self, dt) -> tf.Tensor:
        z = tf.cast(dt, tf.float32)
        if self.max_dt is not None:
            z = tf.clip_by_value(z, 0.0, self.max_dt)
        z = z / self.time_scale
        z = tf.expand_dims(z, axis = -1)        # [..., 1]
        wt = z * self.freq                     # [..., half]
        phi = tf.concat([tf.cos(wt), tf.sin(wt)], axis = -1)   # [..., 2*half]
        if phi.shape[-1] > self.dim:
            phi = phi[..., : self.dim]
        elif phi.shape[-1] < self.dim:
            pad = self.dim - phi.shape[-1]
            phi = tf.pad(phi, [[0, 0]] * (phi.shape.rank - 1) + [[0, pad]])
        return phi

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(dim = self.dim, time_scale = self.time_scale, max_dt = self.max_dt))
        return cfg
