"""Link-prediction scorer: MLP over (u_embed, v_embed) pairs."""

from __future__ import annotations
from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LinkPredictor(layers.Layer):
    """
    Score (source, destination) node embedding pairs.

    For each candidate pair (u, v):
      - source representation  h_u comes from the persistent state projection
      - destination representation h_v comes from the node embedding

    Scorer modes
    ------------
    "dot"      : logit = h_u · h_v  (inner product after proj)
    "bilinear" : logit = h_u W h_v
    "mlp"      : logit = MLP([h_u ; h_v])

    Parameters
    ----------
    embed_dim   : dimensionality of h_u / h_v
    hidden      : hidden units for MLP scorer (ignored for dot/bilinear)
    scorer      : "dot" | "bilinear" | "mlp"
    """

    def __init__(
                    self,
                    embed_dim: int,
                    hidden: int = 256,
                    scorer: str = "mlp",
                    name: Optional[str] = None,
                    **kwargs
                ):
        super().__init__(name = name, **kwargs)
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        self.scorer = str(scorer).lower()
        assert self.scorer in ("dot", "normalized_dot", "bilinear", "mlp"), f"scorer must be 'dot', 'normalized_dot', 'bilinear', or 'mlp'. Got '{scorer}'"

    def build(self, input_shape):
        d = self.embed_dim
        if self.scorer == "bilinear":
            self.W = self.add_weight(name = "W", shape = (d, d),
                                     initializer = "glorot_uniform")
        elif self.scorer == "mlp":
            self.mlp = keras.Sequential(
                                            [
                                                layers.Dense(self.hidden // 2, activation = None, name = "dense_1"),
                                                layers.BatchNormalization(name = "norm"),
                                                layers.Activation("gelu"),
                                                layers.Dropout(0.5, name = "dropout"),
                                                layers.Dense(1, activation = None, name = "score_head"),
                                            ], name = "scorer_mlp"
                                       )
        # Source and destination projection heads (shared input space -> embed_dim)
        self.src_proj = layers.Dense(d, use_bias = True,  name = "src_proj")
        self.dst_proj = layers.Dense(d, use_bias = False, name = "dst_proj")
        super().build(input_shape)

    def call(
                self,
                h_src: tf.Tensor,   # [B, D]  source state / embedding
                h_dst: tf.Tensor,   # [B, D]  destination embedding
                training = None,
            ) -> tf.Tensor:
        """
        Return scalar logits [B, 1].
        """
        u = self.src_proj(h_src)   # [B, embed_dim]
        v = self.dst_proj(h_dst)   # [B, embed_dim]

        if self.scorer == "normalized_dot":
            u = tf.math.l2_normalize(u, axis = -1)
            v = tf.math.l2_normalize(v, axis = -1)
            return tf.reduce_sum(u * v, axis = -1, keepdims = True)    # [B, 1]
        elif self.scorer == "dot":
            return tf.reduce_sum(u * v, axis = -1, keepdims = True)
        elif self.scorer == "bilinear":
            return tf.reduce_sum(tf.matmul(u, self.W) * v, axis = -1, keepdims = True)
        else:  # mlp
            return self.mlp(tf.concat([u, v, v, u], axis = -1), training = training)  # [B, 1]

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(embed_dim = self.embed_dim, hidden = self.hidden, scorer = self.scorer))
        return cfg
