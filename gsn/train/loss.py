"""Loss functions for temporal link prediction."""

from __future__ import annotations
import tensorflow as tf


def ranking_loss(
    logits: tf.Tensor,
    sizes: tf.Tensor,
    mode: str = "ce",
) -> tf.Tensor:
    """Ranking loss over ragged candidate sets.

    For each query i, candidate block layout: [pos, neg_1, …, neg_K_i].
    We classify the positive (index 0 in each block) against all negatives.

    Parameters
    ----------
    logits : [sum(sizes)] or [sum(sizes), 1]  — raw model scores
    sizes  : [B] int32  — number of candidates per query (1 + K_i)
    mode   : "ce"  (cross-entropy, treats first entry as label=1) |
             "bce" (binary cross-entropy, labels = [1, 0, …, 0])

    Returns
    -------
    scalar loss
    """
    logits = tf.cast(tf.reshape(logits, [-1]), tf.float32)
    sizes  = tf.cast(sizes, tf.int32)
    B      = tf.shape(sizes)[0]
    offsets = tf.concat(
        [tf.zeros([1], tf.int32), tf.cumsum(sizes[:-1])], axis=0
    )  # [B]

    if mode == "bce":
        # Build flat labels: 1 at start of each block
        labels = tf.tensor_scatter_nd_update(
            tf.zeros_like(logits),
            tf.expand_dims(offsets, 1),
            tf.ones([B], tf.float32),
        )
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        )

    # Cross-entropy: for each block, softmax over the block, label index = 0
    def ce_body(i, acc):
        o = offsets[i]
        block = logits[o: o + sizes[i]]         # [1 + K_i]
        lp = tf.nn.log_softmax(block)
        return i + 1, acc + (-lp[0])            # negative log-prob of the positive

    _, total = tf.while_loop(
        lambda i, _: i < B,
        ce_body,
        [tf.constant(0, tf.int32), tf.constant(0.0, tf.float32)],
    )
    return total / tf.cast(B, tf.float32)


def write_penalty_loss(
    s_prev: tf.Tensor,
    s_next: tf.Tensor,
    scores: tf.Tensor,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Penalise large state changes relative to prediction uncertainty.

    write_penalty = mean( ||s_next − s_prev||² / (var(scores) + ε) )

    s_prev, s_next : [N, D]
    scores         : [B_cand] flat logits (used to compute variance as a proxy)

    Returns scalar.
    """
    delta = tf.cast(s_next, tf.float32) - tf.cast(s_prev, tf.float32)
    per_node_norm = tf.reduce_mean(tf.square(delta), axis=-1)   # [N]
    score_var = tf.math.reduce_variance(tf.cast(scores, tf.float32))
    return tf.reduce_mean(per_node_norm) / (score_var + tf.cast(eps, tf.float32))
