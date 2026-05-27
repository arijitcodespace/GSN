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
                    pair_feature_dim: int = 0,
                    pair_hidden: int = 16,
                    query_history_feature_dim: int = 0,
                    query_history_hidden: int = 16,
                    name: Optional[str] = None,
                    **kwargs
                ):
        super().__init__(name = name, **kwargs)
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        self.scorer = str(scorer).lower()
        self.pair_feature_dim = int(pair_feature_dim)
        self.pair_hidden = int(pair_hidden)
        self.query_history_feature_dim = int(query_history_feature_dim)
        self.query_history_hidden = int(query_history_hidden)
        assert self.scorer in ("dot", "normalized_dot", "bilinear", "mlp"), f"scorer must be 'dot', 'normalized_dot', 'bilinear', or 'mlp'. Got '{scorer}'"
        if self.pair_feature_dim < 0:
            raise ValueError("pair_feature_dim must be non-negative")
        if self.query_history_feature_dim < 0:
            raise ValueError("query_history_feature_dim must be non-negative")

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
        if self.pair_feature_dim > 0:
            self.pair_mlp = keras.Sequential(
                                            [
                                                layers.Dense(
                                                    max(self.pair_hidden, 1),
                                                    activation = "gelu",
                                                    name = "pair_recurrence_dense",
                                                ),
                                                layers.Dense(
                                                    1,
                                                    activation = None,
                                                    kernel_initializer = "zeros",
                                                    bias_initializer = "zeros",
                                                    name = "pair_recurrence_score",
                                                ),
                                            ],
                                            name = "pair_recurrence_mlp",
                                       )
        else:
            self.pair_mlp = None
        if self.query_history_feature_dim > 0:
            self.query_history_mlp = keras.Sequential(
                                            [
                                                layers.Dense(
                                                    max(self.query_history_hidden, 1),
                                                    activation = "gelu",
                                                    name = "query_history_dense",
                                                ),
                                                layers.Dense(
                                                    1,
                                                    activation = None,
                                                    kernel_initializer = "zeros",
                                                    bias_initializer = "zeros",
                                                    name = "query_history_score",
                                                ),
                                            ],
                                            name = "query_history_mlp",
                                       )
        else:
            self.query_history_mlp = None
        super().build(input_shape)

    def call(
                self,
                h_src: tf.Tensor,   # [B, D]  source state / embedding
                h_dst: tf.Tensor,   # [B, D]  destination embedding
                pair_features: Optional[tf.Tensor] = None,
                query_history_features: Optional[tf.Tensor] = None,
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
            logits = tf.reduce_sum(u * v, axis = -1, keepdims = True)    # [B, 1]
        elif self.scorer == "dot":
            logits = tf.reduce_sum(u * v, axis = -1, keepdims = True)
        elif self.scorer == "bilinear":
            logits = tf.reduce_sum(tf.matmul(u, self.W) * v, axis = -1, keepdims = True)
        else:  # mlp
            logits = self.mlp(tf.concat([u, v, v, u], axis = -1), training = training)  # [B, 1]

        if self.pair_mlp is not None:
            if pair_features is None:
                raise ValueError(
                    "pair_features must be provided when pair recurrence is enabled"
                )
            pair_features = tf.cast(pair_features, tf.float32)
            logits = logits + self.pair_mlp(pair_features, training = training)
        if self.query_history_mlp is not None:
            if query_history_features is None:
                raise ValueError(
                    "query_history_features must be provided when query history is enabled"
                )
            query_history_features = tf.cast(query_history_features, tf.float32)
            logits = logits + self.query_history_mlp(
                query_history_features, training = training
            )
        return logits

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                embed_dim = self.embed_dim,
                hidden = self.hidden,
                scorer = self.scorer,
                pair_feature_dim = self.pair_feature_dim,
                pair_hidden = self.pair_hidden,
                query_history_feature_dim = self.query_history_feature_dim,
                query_history_hidden = self.query_history_hidden,
            )
        )
        return cfg
