"""Evaluation metrics for temporal link prediction."""

from __future__ import annotations
import tensorflow as tf


@tf.function
def compute_mrr(logits: tf.Tensor, sizes: tf.Tensor) -> tf.Tensor:
    """Mean Reciprocal Rank over ragged candidate sets.

    logits : [sum(sizes)] or [sum(sizes), 1]
    sizes  : [B] int32  — 1 + K_i per query; positive is always index 0
    """
    scores = tf.cast(tf.reshape(logits, [-1]), tf.float32)
    sizes  = tf.cast(sizes, tf.int32)
    B      = tf.shape(sizes)[0]
    offsets = tf.concat(
        [tf.zeros([1], tf.int32), tf.cumsum(sizes[:-1])], axis=0
    )

    def body(i, offset, acc):
        block = scores[offset: offset + sizes[i]]   # [1 + K_i]
        pos   = block[0:1]
        negs  = block[1:]
        rank  = tf.reduce_sum(tf.cast(negs >= pos - 1e-6, tf.float32)) + 1.0
        return i + 1, offset + sizes[i], acc + 1.0 / rank

    _, _, sum_rr = tf.while_loop(
        lambda i, *_: i < B,
        body,
        [tf.constant(0, tf.int32),
         tf.constant(0, tf.int32),
         tf.constant(0.0, tf.float32)],
    )
    return sum_rr / tf.cast(B, tf.float32)


@tf.function
def compute_mrr_1v1_sum_count(logits: tf.Tensor, sizes: tf.Tensor):
    """Per-block 1-vs-1 reciprocal-rank components for streaming accumulation.

    Returns
    -------
    (sum_inv_rank, count) : (float32 scalar, float32 scalar)
        sum_inv_rank : sum over valid blocks (size >= 2) of `1.0` if
                       pos > first_neg (+1e-6 tolerance), else `0.5`.
        count        : number of valid blocks (size >= 2).

    Layout assumption: each block is [positive, neg_0, neg_1, ...].
    Blocks of size < 2 (no negatives) are skipped.

    Bias note: "first negative" inherits the sampler's intrinsic ordering.
    For TGBStyleTrainNegativeSampler with hist_ratio > 0 and K large enough
    that max_hist = int(K * hist_ratio) >= 1, position 1 is a *historical*
    (hard) negative. For K == 1, max_hist == 0, so the single negative is
    purely random (easy). Cross-K comparisons therefore reflect both model
    quality AND sampler distribution; interpret accordingly.
    """
    scores  = tf.cast(tf.reshape(logits, [-1]), tf.float32)
    sizes   = tf.cast(sizes, tf.int32)
    B       = tf.shape(sizes)[0]
    N       = tf.shape(scores)[0]

    offsets = tf.concat(
        [tf.zeros([1], tf.int32), tf.cumsum(sizes[:-1])], axis=0
    )

    pos_idx = offsets
    # Clip neg_idx into bounds — invalid (size==1) blocks are masked out
    # before contributing, so the gathered value is harmless.
    neg_idx = tf.minimum(offsets + 1, N - 1)

    pos_scores = tf.gather(scores, pos_idx)
    neg_scores = tf.gather(scores, neg_idx)

    valid    = sizes >= 2
    inv_rank = tf.where(pos_scores > neg_scores + 1e-6, 1.0, 0.5)
    inv_rank = tf.where(valid, inv_rank, tf.zeros_like(inv_rank))

    sum_inv_rank = tf.reduce_sum(inv_rank)
    count        = tf.cast(tf.reduce_sum(tf.cast(valid, tf.int32)), tf.float32)
    return sum_inv_rank, count


@tf.function
def compute_mrr_1v1(logits: tf.Tensor, sizes: tf.Tensor) -> tf.Tensor:
    """1-vs-1 MRR over a single batch (mean over valid blocks).

    Use during evaluation/display of a single window. For streaming
    accumulation across many buckets in one epoch, call
    `compute_mrr_1v1_sum_count` and aggregate the (sum, count) pair.
    """
    sum_inv_rank, count = compute_mrr_1v1_sum_count(logits, sizes)
    return sum_inv_rank / tf.maximum(count, 1.0)


@tf.function
def compute_ap(logits: tf.Tensor, sizes: tf.Tensor) -> tf.Tensor:
    """Per-query Average Precision, averaged over B queries.

    For exactly 1 positive per query this equals MRR.  Computing AP globally
    (pooling all candidates together) requires cross-query score calibration
    that the per-query CE training objective never enforces, so it produces
    misleadingly low numbers.  We compute per-query AP instead, consistent
    with the TGB evaluation protocol.

    logits : [sum(sizes)]
    sizes  : [B] int32  — 1 + K_i per query; positive is always index 0
    """
    scores  = tf.cast(tf.reshape(logits, [-1]), tf.float32)
    sizes   = tf.cast(sizes, tf.int32)
    B       = tf.shape(sizes)[0]
    offsets = tf.concat(
        [tf.zeros([1], tf.int32), tf.cumsum(sizes[:-1])], axis=0
    )

    def body(i, offset, acc):
        block = scores[offset : offset + sizes[i]]   # [1 + K_i]
        pos   = block[0:1]                           # positive score
        # rank = number of candidates (incl. positive itself) scoring >= pos
        rank  = tf.reduce_sum(tf.cast(block >= pos - 1e-6, tf.float32))
        return i + 1, offset + sizes[i], acc + 1.0 / rank

    _, _, total = tf.while_loop(
        lambda i, *_: i < B,
        body,
        [tf.constant(0, tf.int32),
         tf.constant(0, tf.int32),
         tf.constant(0.0, tf.float32)],
    )
    return total / tf.cast(B, tf.float32)


@tf.function
def compute_auc(logits: tf.Tensor, sizes: tf.Tensor) -> tf.Tensor:
    """Micro ROC-AUC over all queries (trapezoidal)."""
    scores = tf.cast(tf.reshape(logits, [-1]), tf.float32)
    sizes  = tf.cast(sizes, tf.int32)
    B      = tf.shape(sizes)[0]
    N      = tf.shape(scores)[0]

    offsets = tf.concat(
        [tf.zeros([1], tf.int32), tf.cumsum(sizes[:-1])], axis=0
    )
    labels = tf.tensor_scatter_nd_update(
        tf.zeros([N], tf.float32),
        tf.expand_dims(offsets, 1),
        tf.ones([B], tf.float32),
    )

    order        = tf.argsort(scores, direction="DESCENDING", stable=True)
    y_sorted     = tf.gather(labels, order)
    one_minus_y  = 1.0 - y_sorted
    tp           = tf.cumsum(y_sorted)
    fp           = tf.cumsum(one_minus_y)
    num_pos      = tf.reduce_sum(y_sorted)
    num_neg      = tf.reduce_sum(one_minus_y)

    def _auc():
        tpr  = tp  / tf.maximum(num_pos, 1.0)
        fpr  = fp  / tf.maximum(num_neg, 1.0)
        z    = tf.zeros([1], tf.float32)
        tpr2 = tf.concat([z, tpr], axis=0)
        fpr2 = tf.concat([z, fpr], axis=0)
        return tf.reduce_sum((fpr2[1:] - fpr2[:-1]) * (tpr2[1:] + tpr2[:-1]) * 0.5)

    return tf.cond(
        tf.logical_and(num_pos > 0.0, num_neg > 0.0),
        _auc,
        lambda: tf.constant(0.0),
    )
