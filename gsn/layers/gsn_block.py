"""GSN block: SSM step + gated message passing + FFN.

GSNBlock          – stateless; caller manages the state tensor
PersistentGSNBlock – wraps GSNBlock with DenseStateTable read/write
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Optional, Tuple
from termcolor import cprint

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .mamba2.layers import Mamba2SSD   # type: ignore

try:
    from ..snapshot import Snapshot
    from ..state.table import DenseStateTable
    from ..state.conv_cache import ConvCacheTable
    from ..state.activity_buffers import NodeActivityBuffers
    from .time_encoding import TGATTimeEncoding
    from .edge_gate import EdgeGate
    from .adaptive_commit_gate import AdaptiveCommitGate
except Exception as e:
    cprint("[gsn_block.py] Failed with relative import. Trying with absolute import.", "yellow")
    from gsn.snapshot import Snapshot
    from gsn.state.table import DenseStateTable
    from gsn.state.conv_cache import ConvCacheTable
    from gsn.state.activity_buffers import NodeActivityBuffers
    from gsn.layers.time_encoding import TGATTimeEncoding
    from gsn.layers.edge_gate import EdgeGate
    from gsn.layers.adaptive_commit_gate import AdaptiveCommitGate


class GSNBlock(layers.Layer):
    """
    Single GSN block: SSM step + gated message passing + FFN.

    Stateless — the caller passes the per-node SSM state tensor in and
    receives the updated state tensor out.

    Parameters
    ----------
    hidden          : model dimension (d_model for Mamba2SSD)
    num_heads       : number of SSM heads (H)
    head_dim        : dimension per SSM head (P); must equal hidden // num_heads
    state_dim       : SSM state dimension per head (N)
    time_feat_dim   : dimension of TGAT time encoding
    time_scale      : Δt normalisation (seconds in one "unit")
    edge_gate_hidden: hidden units in EdgeGate MLP (0 = linear)
    dropout         : dropout rate in FFN
    self_loops      : whether to include self-loops in message passing
    pre_message     : Experiment B0. If True, aggregate a simple per-destination
                      summary of neighbor `xh` and add it (gated by a learnable
                      scalar `g_pre_msg` initialised to 0) to `xh` *before* the
                      Mamba SSM step. This makes the committed SSM state ingest
                      edge/interaction information instead of being driven only
                      by node features + time. Initialising the gate to 0 keeps
                      the model bit-identical to the previous behaviour at init.
    conv_cache      : Experiment B. If True, the GSNBlock expects a per-node
                      conv-cache tensor to be threaded through `call()` so the
                      Mamba-2 causal DepthwiseConv1D sees true streaming history
                      across snapshots instead of zero-padding every step. The
                      cache itself is owned by `PersistentGSNBlock` via a second
                      DenseStateTable; this flag merely tells GSNBlock to forward
                      it to `mamba2.step()`.
    conv_cache_dt_decay :
                      Optional global decay τ (in the same time units as
                      ``snap.dt``). When set, the read conv-cache is multiplied
                      by ``exp(-snap.dt / τ)`` before being fed to Mamba — a
                      lightweight Δt-staleness mitigation for sparse buckets.
                      ``None`` (default) disables decay.
    intra_bucket_seq : Experiment C. If True, instead of running the Mamba-2
                      SSM as a single step over the bucket-aggregated input
                      ``xh``, the block runs:
                        (i) a base step using ``ln1(xh)`` so every node in the
                            snapshot gets at least one SSM update (preserves
                            the existing per-bucket semantics for src-only or
                            isolated nodes), then
                        (ii) for each node that *receives* events in this
                             bucket, additional per-event SSM steps in ts-order.
                             The per-event input token is
                             ``ln1(event_msg_lin(xh[src]) +
                                   lin_time(edge_time_enc(t_ref - edge_ts)))``.
                      Persistent SSM state ``s`` and the conv cache ``conv_s``
                      are threaded across all per-event steps via scatter
                      updates, so the *final* state per node (after all its
                      events in this bucket) is what gets committed. The
                      per-node output ``h_ssm`` used by the downstream
                      message-passing / FFN path is the SSM output of the
                      *last* step processed for that node.
                      Disabled by default → bit-identical to the previous
                      bucket-aggregated behaviour.
    """

    def __init__(
                    self,
                    hidden: int,
                    num_heads: int,
                    head_dim: int,
                    state_dim: int,
                    sequence_length: int = 1,
                    num_chunks: int = 1,
                    time_feat_dim: int = 8,
                    time_scale: float = 86400.0,
                    edge_gate_hidden: int = 32,
                    dropout: float = 0.0,
                    self_loops: bool = True,
                    pre_message: bool = False,
                    conv_cache: bool = False,
                    conv_cache_dt_decay: Optional[float] = None,
                    intra_bucket_seq: bool = False,
                    conv1d_kernel_size: int = 4,
                    name: Optional[str] = None,
                    **kwargs
                ):
        super().__init__(name = name, **kwargs)
        assert hidden == num_heads * head_dim, \
            f"hidden ({hidden}) must equal num_heads*head_dim ({num_heads}*{head_dim})"

        self.hidden = int(hidden)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.state_dim = int(state_dim)
        self.sequence_length = int(sequence_length)
        self.num_chunks = int(num_chunks)
        self.time_feat_dim = int(time_feat_dim)
        self.time_scale = float(time_scale)
        self.self_loops = bool(self_loops)
        self.pre_message = bool(pre_message)
        self.conv_cache  = bool(conv_cache)
        self.conv_cache_dt_decay = (
            float(conv_cache_dt_decay) if conv_cache_dt_decay else None
        )
        self.intra_bucket_seq = bool(intra_bucket_seq)
        self.conv1d_kernel_size = int(conv1d_kernel_size)

        # SSM core (sequence_length=1, num_chunks=1 -> pure step mode)
        self.mamba2 = Mamba2SSD(
                                    num_heads = num_heads,
                                    head_dim = head_dim,
                                    state_dim = state_dim,
                                    d_model = hidden,
                                    sequence_length = sequence_length,
                                    num_chunks = num_chunks,
                                    conv1d_kernel_size = self.conv1d_kernel_size,
                                    name = (f"{name}_mamba2" if name else "mamba2"),
                                )

        # Time encoders
        self.node_time_enc = TGATTimeEncoding(
                                                dim = time_feat_dim,
                                                time_scale = time_scale,
                                                name = (f"{name}_node_te" if name else "node_te")
                                             )
        self.edge_time_enc = TGATTimeEncoding(
                                                dim = time_feat_dim,
                                                time_scale = time_scale,
                                                name = (f"{name}_edge_te" if name else "edge_te"),
                                             )

        # Projections
        # lin_in: project (node_feat | degree_feats) -> hidden
        # set input_dim lazily in build()
        self.lin_in   = layers.Dense(hidden, use_bias = True,  name = "lin_in")
        self.lin_time = layers.Dense(hidden, use_bias = False, name = "lin_time")
        self.msg_lin  = layers.Dense(hidden, use_bias = False, name = "msg_lin")

        self.edge_gate = EdgeGate(
                                    num_heads = num_heads,
                                    hidden = edge_gate_hidden,
                                    name = (f"{name}_edge_gate" if name else "edge_gate")
                                 )

        self.ln1 = layers.LayerNormalization(epsilon = 1e-5, name = "ln1")
        self.ln2 = layers.LayerNormalization(epsilon = 1e-5, name = "ln2")

        self.ffn = keras.Sequential(
                                        [
                                            layers.Dense(4 * hidden, activation = "gelu", name = f"{name}_ffn_dense_0"),
                                            layers.LayerNormalization(epsilon = 1e-6, name = f"{name}_ffn_layernorm"),
                                            layers.Dense(hidden, name = f"{name}_ffn_dense_1"),
                                            layers.Dropout(dropout, name = f"{name}_ffn_dropout")
                                        ], 
                                        name = "ffn"
                                    )

        self.g_ssm = self.add_weight(
                                        name = "g_ssm",
                                        shape = [],
                                        trainable = True,
                                        initializer = keras.initializers.Constant(1.0)
                                    )
        self.g_msg = self.add_weight(
                                        name = "g_msg",
                                        shape = [],
                                        trainable = True,
                                        initializer = keras.initializers.Constant(1.0)
                                    )

        # Experiment B0: optional pre-Mamba neighbour summary that feeds the
        # committed SSM state. Disabled by default so backwards-compatible.
        if self.pre_message:
            self.pre_msg_lin = layers.Dense(
                                                hidden,
                                                use_bias = False,
                                                name = "pre_msg_lin"
                                           )
            self.g_pre_msg = self.add_weight(
                                                name = "g_pre_msg",
                                                shape = [],
                                                trainable = True,
                                                initializer = keras.initializers.Constant(0.0)
                                            )
        else:
            self.pre_msg_lin = None
            self.g_pre_msg = None

        # Experiment C: per-event SSM stepping inside a bucket. The event
        # message head is a dedicated linear projection (kept separate from
        # ``msg_lin`` so the downstream message-passing path is unchanged when
        # C is off). Only created when intra_bucket_seq is True. Its kernel is
        # built lazily on first call — the parent dummy-build path must
        # therefore include at least one non-self-loop edge so the per-event
        # branch fires (see ``GSNLinkPredictor._build_from_dummy``).
        if self.intra_bucket_seq:
            self.event_msg_lin = layers.Dense(
                                                hidden,
                                                use_bias = False,
                                                name = "event_msg_lin",
                                              )
        else:
            self.event_msg_lin = None

    # ------------------------------------------------------------------

    @property
    def state_dim_total(self) -> int:
        return self.num_heads * self.head_dim * self.state_dim

    @property
    def conv_state_dim_total(self) -> int:
        """Flattened per-node conv-cache size = (K-1) * H * (2N + P)."""
        return self.mamba2.conv_cache_dim

    # ------------------------------------------------------------------

    def _node_features(self, snap: Snapshot) -> tf.Tensor:
        """Build input node features: structural degree feats + optional snap.x."""
        N = snap.num_nodes
        E = snap.num_edges
        ones = tf.ones([E], dtype = tf.float32)
        src_i = tf.cast(snap.edge_src, tf.int32)
        dst_i = tf.cast(snap.edge_dst, tf.int32)
        deg_out = tf.math.unsorted_segment_sum(ones, src_i, N)  # [N]
        deg_in  = tf.math.unsorted_segment_sum(ones, dst_i, N)  # [N]
        f_deg = tf.math.log1p(tf.stack([deg_in, deg_out], axis = 1))  # [N, 2]

        if snap.x is not None:
            x_np = tf.cast(snap.x, tf.float32)           # [N, F_n]
            return tf.concat([x_np, f_deg], axis = -1)     # [N, F_n+2]
        return f_deg                                      # [N, 2]

    def _node_features_sequence(self, snap: Snapshot) -> tf.Tensor:
        """Build per-step node features for a padded sequence snapshot."""
        N = snap.num_nodes
        L = snap.sequence_length
        src = tf.cast(snap.edge_src, tf.int32)             # [L, E_max]
        dst = tf.cast(snap.edge_dst, tf.int32)             # [L, E_max]
        valid = tf.logical_and(src >= 0, dst >= 0)

        E_max = int(np.asarray(snap.edge_src).shape[1])
        step_ids = tf.repeat(tf.range(L, dtype = tf.int32), E_max)
        valid_flat = tf.reshape(valid, [-1])
        src_flat = tf.boolean_mask(tf.reshape(src, [-1]), valid_flat)
        dst_flat = tf.boolean_mask(tf.reshape(dst, [-1]), valid_flat)
        step_flat = tf.boolean_mask(step_ids, valid_flat)

        ones = tf.ones(tf.shape(src_flat), dtype = tf.float32)
        num_segments = L * N
        deg_out = tf.math.unsorted_segment_sum(
            ones, step_flat * N + src_flat, num_segments
        )
        deg_in = tf.math.unsorted_segment_sum(
            ones, step_flat * N + dst_flat, num_segments
        )
        deg_out = tf.transpose(tf.reshape(deg_out, [L, N]), [1, 0])
        deg_in = tf.transpose(tf.reshape(deg_in, [L, N]), [1, 0])
        f_deg = tf.math.log1p(tf.stack([deg_in, deg_out], axis = -1))  # [N, L, 2]

        if snap.x is not None:
            x_np = tf.cast(snap.x, tf.float32)
            if x_np.shape.rank == 2:
                x_np = tf.tile(tf.expand_dims(x_np, axis = 1), [1, L, 1])
            return tf.concat([x_np, f_deg], axis = -1)     # [N, L, F_n+2]
        return f_deg                                      # [N, L, 2]

    def _sequence_dt(self, snap: Snapshot, N: int, L: int) -> tf.Tensor:
        if snap.seq_dt is not None:
            dt = tf.cast(snap.seq_dt, tf.float32)
        else:
            dt = tf.fill([L], tf.cast(snap.dt, tf.float32))
        return tf.tile(tf.expand_dims(dt, axis = 0), [N, 1])  # [N, L]

    def _sequence_actual_len(self, snap: Snapshot) -> int:
        actual = int(getattr(snap, "actual_seq_len", snap.sequence_length))
        return max(0, min(actual, snap.sequence_length))

    def _pre_message_sequence(
        self,
        snap: Snapshot,
        xh: tf.Tensor,
    ) -> tf.Tensor:
        """Apply the optional pre-message summary independently per step."""
        N = snap.num_nodes
        L = snap.sequence_length
        pre_messages = []
        src_np_all = np.asarray(snap.edge_src, dtype = np.int64)
        dst_np_all = np.asarray(snap.edge_dst, dtype = np.int64)

        for step in range(L):
            src_np = src_np_all[step]
            dst_np = dst_np_all[step]
            valid = np.logical_and(src_np >= 0, dst_np >= 0)
            src_i = tf.constant(src_np[valid], dtype = tf.int32)
            dst_i = tf.constant(dst_np[valid], dtype = tf.int32)
            xh_src = tf.gather(xh[:, step, :], src_i)
            pre_m = tf.math.unsorted_segment_sum(xh_src, dst_i, N)
            pre_messages.append(pre_m)

        pre_m_seq = tf.stack(pre_messages, axis = 1)       # [N, L, hidden]
        return xh + self.g_pre_msg * self.pre_msg_lin(pre_m_seq)

    def _sequence_ssm_step_loop(
        self,
        u: tf.Tensor,
        s_4d: tf.Tensor,
        actual_len: int,
        training: Optional[bool],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Process a short padded sequence exactly, without consuming padding."""
        h_ssm = None
        s_running = s_4d
        for step in range(actual_len):
            h_ssm, s_running, _ = self.mamba2.step(
                u[:, step, :],
                state = s_running,
                conv_state = None,
                training = training,
            )
        if h_ssm is None:
            raise ValueError("Sequence snapshots must contain at least one timestep.")
        return h_ssm, s_running

    def _call_sequence(
        self,
        *,
        snap: Snapshot,
        s: tf.Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """Forward pass for a padded sequence snapshot."""
        if not snap.is_sequence:
            raise ValueError("Sequence mode requires a sequence Snapshot.")
        if self.intra_bucket_seq:
            raise RuntimeError("`intra_bucket_seq` is only supported in step mode.")

        N = snap.num_nodes
        L = snap.sequence_length
        actual_len = self._sequence_actual_len(snap)
        if actual_len <= 0:
            raise ValueError("Sequence snapshots must contain at least one timestep.")
        if L != self.sequence_length:
            raise ValueError(
                f"Sequence snapshot length {L} does not match model sequence_length "
                f"{self.sequence_length}."
            )

        x_in = self._node_features_sequence(snap)           # [N, L, F]
        xh = self.lin_in(x_in)                              # [N, L, hidden]
        te = self.node_time_enc(self._sequence_dt(snap, N, L))
        xh = xh + self.lin_time(te)

        if self.pre_message:
            xh = self._pre_message_sequence(snap, xh)

        u = self.ln1(xh, training = training)               # [N, L, hidden]
        s_4d = tf.reshape(
                            tf.cast(s, u.dtype),
                            [N, self.num_heads, self.head_dim, self.state_dim]
                         )

        if actual_len == self.sequence_length:
            s_5d = tf.expand_dims(s_4d, axis = 1)
            h_seq, s_next_4d = self.mamba2(
                                            inputs = (u, s_5d),
                                            training = training
                                          )
            h_ssm = h_seq[:, actual_len - 1, :]
        else:
            h_ssm, s_next_4d = self._sequence_ssm_step_loop(
                                                                    u          = u,
                                                                    s_4d       = s_4d,
                                                                    actual_len = actual_len,
                                                                    training   = training,
                                                                )

        s_next = tf.reshape(s_next_4d, [N, self.state_dim_total])
        xh_final = xh[:, actual_len - 1, :]
        out = self._message_passing(
                                    snap       = snap,
                                    xh         = xh_final,
                                    h_ssm      = h_ssm,
                                    training   = training,
                                    step_index = actual_len - 1,
                                  )
        return out, tf.cast(s_next, tf.float32), None

    def _message_passing(
        self,
        snap: Snapshot,
        xh: tf.Tensor,
        h_ssm: tf.Tensor,
        training: Optional[bool],
        step_index: Optional[int] = None,
    ) -> tf.Tensor:
        """Run the gated message-passing/FFN tail on a scalar or sequence step."""
        N = snap.num_nodes
        xw = self.msg_lin(h_ssm)            # [N, hidden]

        if snap.is_sequence:
            if step_index is None:
                step_index = max(self._sequence_actual_len(snap) - 1, 0)
            src_np = np.asarray(snap.edge_src[step_index], dtype = np.int64)
            dst_np = np.asarray(snap.edge_dst[step_index], dtype = np.int64)
            valid = np.logical_and(src_np >= 0, dst_np >= 0)
            src_i = tf.constant(src_np[valid], dtype = tf.int32)
            dst_i = tf.constant(dst_np[valid], dtype = tf.int32)
            edge_feat = (
                np.asarray(snap.edge_feat[step_index][valid], dtype = np.float32)
                if snap.edge_feat is not None
                else None
            )
            edge_ts = (
                np.asarray(snap.edge_ts[step_index][valid], dtype = np.int64)
                if snap.edge_ts is not None
                else None
            )
            if snap.seq_t_ref is not None:
                t_ref_value = float(np.asarray(snap.seq_t_ref)[step_index])
            else:
                t_ref_value = float(snap.t_ref)
        else:
            src_i = tf.cast(snap.edge_src, tf.int32)
            dst_i = tf.cast(snap.edge_dst, tf.int32)
            edge_feat = snap.edge_feat
            edge_ts = snap.edge_ts
            t_ref_value = float(snap.t_ref)

        H, d = self.num_heads, self.head_dim
        xw_hd = tf.reshape(xw, [-1, H, d])          # [N, H, d]
        hi_hd = tf.gather(xw_hd, src_i)             # [E, H, d]

        hi_flat = tf.reshape(hi_hd, [-1, self.hidden])  # [E, hidden]
        hj_flat = tf.reshape(
            tf.gather(xw_hd, dst_i), [-1, self.hidden]
        )
        gate_parts = [hi_flat, hj_flat]

        if edge_feat is not None:
            gate_parts.append(tf.cast(edge_feat, tf.float32))

        if edge_ts is not None:
            t_ref = tf.cast(t_ref_value, tf.float32)
            dt_e = tf.maximum(t_ref - tf.cast(edge_ts, tf.float32), 0.0)
            te_e = self.edge_time_enc(dt_e)
            gate_parts.append(te_e)

        e_in = tf.concat(gate_parts, axis = -1)
        g = tf.sigmoid(self.edge_gate(e_in, training = training))  # [E, H]

        g_exp = tf.expand_dims(g, axis = -1)
        msg_hd = g_exp * hi_hd
        msg_flat = tf.reshape(msg_hd, [-1, self.hidden])

        if self.self_loops:
            loop_idx = tf.range(N, dtype = tf.int32)
            self_msg = tf.reshape(tf.gather(xw_hd, loop_idx), [N, self.hidden])
            msg_flat = tf.concat([msg_flat, self_msg], axis = 0)
            dst_agg = tf.concat([dst_i, loop_idx], axis = 0)
        else:
            dst_agg = dst_i

        m = tf.math.unsorted_segment_sum(msg_flat, dst_agg, N)
        v = xh + self.g_ssm * h_ssm + self.g_msg * m
        return v + self.ffn(self.ln2(v, training = training), training = training)
    
    def __call__(self,
                 *,
                 snap: Snapshot,
                 s: tf.Tensor,
                 conv_s: Optional[tf.Tensor] = None,
                 training: Optional[bool] = None,
                 run_step_mode: Optional[bool] = True,
                ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:

        if not self.built:
            self._N = snap.num_nodes
            self.built = True

        return self.call(
                            snap          = snap,
                            s             = s,
                            conv_s        = conv_s,
                            training      = training,
                            run_step_mode = run_step_mode,
                         )

    def call(
                self,
                *,
                snap: Snapshot,
                s: tf.Tensor,                       # [N, state_dim_total]  float32
                conv_s: Optional[tf.Tensor] = None, # [N, conv_state_dim_total] float32 or None
                training: Optional[bool] = None,
                run_step_mode: Optional[bool] = True
            ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """
        Forward pass for one snapshot.

        Returns
        -------
        out          : [N, hidden]                   - new node embeddings
        s_next       : [N, state_dim_total]          - updated SSM state
        conv_s_next  : [N, conv_state_dim_total] or None
                       updated per-node conv cache, or None when conv_cache
                       is disabled (caller should not commit it).
        """
        
        if run_step_mode and snap.is_sequence:
            raise ValueError("Step mode requires a scalar Snapshot.")

        if run_step_mode and (self.sequence_length != 1 or self.num_chunks != 1):
            raise AttributeError(
                                    "Running in step mode requires `sequence_length=1` and `num_chunks=1`. "
                                    f"Found sequence_length = {self.sequence_length} and num_chunks = {self.num_chunks}."
                                )
        
        if (not run_step_mode) and self.conv_cache:
            raise RuntimeError(
                                "`conv_cache` is meant to be used with step mode in order to "
                                "cache the conv states. If not using step mode then setting "
                                "`conv_cache` to True is redundant and creates confusion. "
                                "Re-run using `conv_cache=False`."
                              )
         
        N = snap.num_nodes

        if not run_step_mode:
            return self._call_sequence(
                                       snap     = snap,
                                       s        = s,
                                       training = training,
                                     )

        # 1. Input projection
        x_in = self._node_features(snap)    # [N, F]
        xh = self.lin_in(x_in)              # [N, hidden]

        # 2. Node-level time encoding
        dt_val = tf.cast(snap.dt, tf.float32)
        dt_node = tf.fill([N], dt_val)       # [N]
        te = self.node_time_enc(dt_node)     # [N, time_feat_dim]
        xh = xh + self.lin_time(te)          # [N, hidden]

        # 2.5 (B0) Pre-Mamba neighbour summary -> ingested into committed state.
        # When `pre_message=False` (default) this branch is skipped entirely so
        # the model is bit-identical to the previous baseline. When enabled,
        # `g_pre_msg` is initialised to 0 so the very first forward pass is
        # still bit-identical at init; the scalar learns how much pre-message
        # signal to inject into the SSM input.
        if self.pre_message:
            src_i_pm = tf.cast(snap.edge_src, tf.int32)
            dst_i_pm = tf.cast(snap.edge_dst, tf.int32)
            xh_src   = tf.gather(xh, src_i_pm)                              # [E, hidden]
            pre_m    = tf.math.unsorted_segment_sum(xh_src, dst_i_pm, N)    # [N, hidden]
            xh       = xh + self.g_pre_msg * self.pre_msg_lin(pre_m)        # [N, hidden]

        # 3. SSM step
        u = self.ln1(xh, training = training)  # [N, hidden]
        s_4d = tf.reshape(
                            tf.cast(s, u.dtype),
                            [N, self.num_heads, self.head_dim, self.state_dim]
                         )

        # (B) Persistent conv cache: when enabled and a cache was supplied, feed
        # it to mamba2.step() so the causal conv sees true streaming history
        # rather than zero-pad. The cache is shaped [N, K-1, C] natively — it
        # comes from a ConvCacheTable so no flatten/reshape is needed.
        # When conv_cache is disabled or no cache was provided, conv_s_in
        # stays None and mamba2.step() falls back to zero-pad (legacy
        # behaviour, bit-identical to before).
        conv_s_in: Optional[tf.Tensor] = None
        if self.conv_cache and conv_s is not None:
            conv_s_in = tf.cast(conv_s, u.dtype)
            if self.conv_cache_dt_decay is not None:
                decay = tf.exp(
                    -tf.cast(snap.dt, conv_s_in.dtype)
                    / tf.cast(self.conv_cache_dt_decay, conv_s_in.dtype)
                )
                conv_s_in = conv_s_in * decay

        if self.intra_bucket_seq:
            # (C) Run a base step on the per-node bucket context, then one
            # additional SSM step per *incoming* event in ts-order for each
            # node. SSM state and conv cache are threaded across all steps;
            # the final per-node SSM output (last-step h_ssm) is what flows
            # into the downstream message-passing / FFN path.
            h_ssm, s_next_4d, conv_s_next_3d = self._intra_bucket_ssm(
                                                                        snap          = snap,
                                                                        xh            = xh,
                                                                        u0            = u,
                                                                        s_4d          = s_4d,
                                                                        conv_s_in     = conv_s_in,
                                                                        run_step_mode = run_step_mode,
                                                                        training      = training,
                                                                     )
        else:
            if run_step_mode:
                h_ssm, s_next_4d, conv_s_next_3d = self.mamba2.step(
                                                                        u,
                                                                        state = s_4d,
                                                                        conv_state = conv_s_in,
                                                                        training = training
                                                                    )
            else:
                s_5d = tf.expand_dims(s_4d, axis = 1)       # (batch, 1, num_heads, head_dim, state_dim)
                h_ssm, s_next_4d = self.mamba2(
                                                inputs = (u, s_5d),
                                                training = training
                                              )
                conv_s_next_3d = None
               
        # h_ssm: [N, hidden],  s_next_4d: [N, H, P, N_state]
        # conv_s_next_3d: [N, K-1, C] (always returned by step)
        s_next = tf.reshape(s_next_4d, [N, self.state_dim_total])

        # Only forward the conv cache to the caller when we intend to persist it.
        conv_s_next: Optional[tf.Tensor] = None
        if self.conv_cache:
            if conv_s_next_3d is None:
                conv_s_next = tf.zeros(
                                        shape = (
                                                    N,                              # num nodes
                                                    self.conv1d_kernel_size - 1,    # K - 1
                                                    self.mamba2.xbc_channels        # H * (2N + P)
                                                ),
                                        dtype = tf.float32
                                      )
            conv_s_next = tf.cast(conv_s_next_3d, tf.float32)   # [N, K-1, C]

        # 4. Message passing with edge gates
        xw = self.msg_lin(h_ssm)            # [N, hidden]

        src_i = tf.cast(snap.edge_src, tf.int32)
        dst_i = tf.cast(snap.edge_dst, tf.int32)

        H, d = self.num_heads, self.head_dim
        xw_hd = tf.reshape(xw, [-1, H, d])          # [N, H, d]
        hi_hd = tf.gather(xw_hd, src_i)              # [E, H, d]

        # Build edge gate input
        hi_flat = tf.reshape(hi_hd, [-1, self.hidden])           # [E, hidden]
        hj_flat = tf.reshape(tf.gather(xw_hd, dst_i), [-1, self.hidden])  # [E, hidden]
        gate_parts = [hi_flat, hj_flat]

        if snap.edge_feat is not None:
            gate_parts.append(tf.cast(snap.edge_feat, tf.float32))

        if snap.edge_ts is not None:
            t_ref = tf.cast(snap.t_ref, tf.float32)
            dt_e  = tf.maximum(t_ref - tf.cast(snap.edge_ts, tf.float32), 0.0)
            te_e  = self.edge_time_enc(dt_e)   # [E, time_feat_dim]
            gate_parts.append(te_e)

        e_in = tf.concat(gate_parts, axis = -1)                    # [E, F_gate]
        g = self.edge_gate(e_in, training = training)              # [E, H]  raw logits
        g = tf.sigmoid(g)                                        # [E, H] ∈ (0,1)

        # Gated per-head message
        g_exp = tf.expand_dims(g, axis = -1)            # [E, H, 1]
        msg_hd = g_exp * hi_hd                        # [E, H, d]
        msg_flat = tf.reshape(msg_hd, [-1, self.hidden])         # [E, hidden]

        # Add self-loops in message passing if requested
        if self.self_loops:
            loop_idx = tf.range(N, dtype = tf.int32)
            self_msg = tf.reshape(
                tf.gather(xw_hd, loop_idx),      # [N, H, d]
                [N, self.hidden],
            )
            msg_flat = tf.concat([msg_flat, self_msg], axis = 0)
            dst_agg  = tf.concat([dst_i, loop_idx],    axis = 0)
        else:
            dst_agg = dst_i

        m = tf.math.unsorted_segment_sum(msg_flat, dst_agg, N)  # [N, hidden]

        # 5. Gated residual + FFN
        v   = xh + self.g_ssm * h_ssm + self.g_msg * m
        out = v + self.ffn(self.ln2(v, training = training), training = training)

        return out, tf.cast(s_next, tf.float32), conv_s_next

    # ------------------------------------------------------------------
    # Experiment C: per-event SSM stepping inside a bucket
    # ------------------------------------------------------------------

    def _intra_bucket_ssm(
                            self,
                            snap: Snapshot,
                            xh:        tf.Tensor,                  # [N, hidden]
                            u0:        tf.Tensor,                  # [N, hidden] = ln1(xh)
                            s_4d:      tf.Tensor,                  # [N, H, P, N_state]
                            conv_s_in: Optional[tf.Tensor],        # [N, K-1, C] or None
                            run_step_mode: Optional[bool] = False,
                            training:  Optional[bool] = None
                         ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """
        Run the Mamba-2 SSM as a *sequence* of per-event steps inside a single
        snapshot/bucket. See the class docstring for the design rationale.

        Returns the same triple as ``mamba2.step`` but where each per-node
        slot reflects the cumulative effect of (i) the bucket-level base step
        plus (ii) all ts-ordered events whose destination is that node.
        """
        N = snap.num_nodes
        H, P, Ns = self.num_heads, self.head_dim, self.state_dim
        C_chan   = self.mamba2.xbc_channels
        K        = self.mamba2.conv1d_kernel_size

        # ---- (i) Base step ------------------------------------------------
        # Every node steps once on the bucket-aggregated context. This ensures
        # src-only / isolated nodes still get a single SSM update — mirroring
        # the original bucket-aggregated semantics.
        # ``mamba2.step`` always returns a sliding-window conv state (even
        # when the input ``conv_state`` is None — zero-pad fallback), so
        # ``conv_s_running`` is always a real tensor we can keep threading
        # through the per-event loop. Whether we *persist* it across buckets
        # is decided by ``self.conv_cache`` at the very end.
        if run_step_mode:
            h_ssm, s_4d, conv_s_running = self.mamba2.step(
                                                            u0,
                                                            state = s_4d,
                                                            conv_state = conv_s_in,
                                                            training = training
                                                          ) # [N, hidden]
        else:
            h_ssm, s_4d = self.mamba2(u0, state = s_4d, training = training)
            conv_s_running = tf.zeros(
                                        shape = (N, K - 1, C_chan),
                                        dtype = tf.float32
                                      )

        # ---- (ii) Per-event steps ----------------------------------------
        E = snap.num_edges
        if E == 0:
            conv_s_next = tf.cast(conv_s_running, tf.float32) if self.conv_cache else None
            return h_ssm, s_4d, conv_s_next

        # Build the per-event ordering with numpy (snap is an eager dataclass).
        # Sort edges by (dst, ts) so per-node sequences are contiguous & ordered.
        dst_np = np.asarray(snap.edge_dst, dtype = np.int64)
        src_np = np.asarray(snap.edge_src, dtype = np.int64)
        ts_np  = np.asarray(snap.edge_ts,  dtype = np.int64) \
                 if snap.edge_ts is not None else np.zeros(E, dtype = np.int64)

        order  = np.lexsort((ts_np, dst_np))      # primary key = dst, then ts
        dst_s  = dst_np[order]
        src_s  = src_np[order]
        ts_s   = ts_np[order]

        # Rank-within-dst-group via cumcount: for each i, how many earlier
        # edges share the same dst? Vectorised with searchsorted.
        new_group     = np.concatenate(([True], dst_s[1:] != dst_s[:-1]))
        group_starts  = np.where(new_group)[0]    # [G]
        group_of_each = np.searchsorted(group_starts, np.arange(E), side = "right") - 1
        rank          = np.arange(E) - group_starts[group_of_each]      # [E]
        K_max         = int(rank.max()) + 1

        t_ref_f = tf.cast(snap.t_ref, tf.float32)

        for k in range(K_max):
            mask_np = (rank == k)
            if not np.any(mask_np):
                continue
            edge_idx_np  = np.where(mask_np)[0]
            active_dst_np = dst_s[edge_idx_np]
            active_src_np = src_s[edge_idx_np]
            active_ts_np  = ts_s[edge_idx_np]

            active_dst = tf.constant(active_dst_np, dtype = tf.int32)
            active_src = tf.constant(active_src_np, dtype = tf.int32)
            active_ts  = tf.constant(active_ts_np,  dtype = tf.float32)

            # Per-event input token: source's bucket-context (xh[src]) projected
            # through the dedicated event head, plus a time encoding of how
            # long ago the event happened relative to the bucket reference.
            h_src   = tf.gather(xh, active_src)                  # [n_active, hidden]
            msg     = self.event_msg_lin(h_src)                  # [n_active, hidden]
            dt_e    = tf.maximum(t_ref_f - active_ts, 0.0)
            te_e    = self.edge_time_enc(dt_e)                   # [n_active, time_feat_dim]
            evt_in  = msg + self.lin_time(te_e)                  # [n_active, hidden]
            evt_u   = self.ln1(evt_in, training = training)      # share LN1 with base path

            # Gather active per-node SSM state and conv cache rows.
            # NOTE: we always thread the conv state across per-event steps so
            # the causal conv has true intra-bucket sliding. ``self.conv_cache``
            # only controls whether the *cross-bucket* persistence happens at
            # the end of this call.
            s_act       = tf.gather(s_4d,            active_dst)              # [n_active, H, P, Ns]
            conv_s_act  = tf.gather(conv_s_running,  active_dst)              # [n_active, K-1, C]

            h_k, s_act_new, conv_s_act_new = self.mamba2.step(
                evt_u, state = s_act, conv_state = conv_s_act, training = training,
            )

            idx2d = tf.expand_dims(active_dst, axis = 1)                # [n_active, 1]
            s_4d  = tf.tensor_scatter_nd_update(s_4d,            idx2d, s_act_new)
            conv_s_running = tf.tensor_scatter_nd_update(
                conv_s_running, idx2d, tf.cast(conv_s_act_new, conv_s_running.dtype),
            )
            h_ssm = tf.tensor_scatter_nd_update(h_ssm,           idx2d, tf.cast(h_k, h_ssm.dtype))

        conv_s_next = (
            tf.cast(conv_s_running, tf.float32) if self.conv_cache else None
        )
        return h_ssm, s_4d, conv_s_next

    def compute_output_shape(self, input_shape):
        return (
            (self._N, self.hidden),
            (self._N, self.state_dim_total),
            (self._N, self.mamba2.conv1d_kernel_size - 1, self.mamba2.xbc_channels)
            if self.conv_cache else None,
        )


    def get_config(self):
        cfg = super().get_config()
        cfg.update(
                    dict(
                            hidden              = self.hidden,
                            num_heads           = self.num_heads,
                            head_dim            = self.head_dim,
                            state_dim           = self.state_dim,
                            sequence_length     = self.sequence_length,
                            num_chunks          = self.num_chunks,
                            time_feat_dim       = self.time_feat_dim,
                            time_scale          = self.time_scale,
                            self_loops          = self.self_loops,
                            pre_message         = self.pre_message,
                            conv_cache          = self.conv_cache,
                            conv_cache_dt_decay = self.conv_cache_dt_decay,
                            intra_bucket_seq    = self.intra_bucket_seq,
                            conv1d_kernel_size  = self.conv1d_kernel_size,
                        )
                   )
        return cfg


# ---------------------------------------------------------------------------
# Persistent wrapper
# ---------------------------------------------------------------------------

class PersistentGSNBlock(layers.Layer):
    """GSNBlock with attached DenseStateTable.

    Reads per-node state before the forward pass and optionally commits
    the updated state back to the table afterwards.

    Two commit modes
    ----------------
    ``commit_mode = 'uniform'`` (default / backward-compatible):
        Uses the fixed scalar ``commit_alpha``.  No gate, no buffers.

    ``commit_mode = 'adaptive_hazard'``:
        Uses an ``AdaptiveCommitGate`` and ``NodeActivityBuffers`` to
        compute a per-node α_{i,k} based on elapsed time, node activity,
        and state novelty.  Set via the ``commit_gate`` and
        ``activity_buffers`` constructor arguments.

    Parameters
    ----------
    block             : a GSNBlock instance
    state_table       : DenseStateTable shared across all calls
    commit_alpha      : EMA coefficient α for uniform commit (0 < α ≤ 1)
    noise_scale       : std of Gaussian noise added to read state during training
    commit_gate       : AdaptiveCommitGate instance (adaptive mode only)
    activity_buffers  : NodeActivityBuffers instance (adaptive mode only)
    conv_cache_table  : Optional ConvCacheTable. When provided AND the wrapped
                        block has ``conv_cache=True``, the per-node Mamba-2
                        causal-conv history is read before the forward pass,
                        threaded into ``mamba2.step()``, and overwritten back
                        to the table after the forward pass (sliding-window
                        update). When ``None`` the block falls back to the
                        legacy zero-pad step behaviour.
    warmup_eta        : blending coefficient – 0 = pure α_0, 1 = pure learned
                        (managed externally by the Trainer during warmup)
    """

    def __init__(
                    self,
                    block:            GSNBlock,
                    state_table:      DenseStateTable,
                    commit_alpha:     float                          = 0.2,
                    noise_scale:      float                          = 0.005,
                    commit_gate:      Optional[AdaptiveCommitGate]   = None,
                    activity_buffers: Optional[NodeActivityBuffers]  = None,
                    conv_cache_table: Optional[ConvCacheTable]       = None,
                    name:             Optional[str]                  = None,
                    **kwargs
                ):
        super().__init__(name = name, **kwargs)
        self.block            = block
        self.state_table      = state_table
        self.commit_alpha     = float(commit_alpha)
        self.noise_scale      = float(noise_scale)
        self.commit_gate      = commit_gate        # None → uniform mode
        self.activity_buffers = activity_buffers   # None → uniform mode
        self.conv_cache_table = conv_cache_table   # None → no persistent conv cache
        self.warmup_eta: float = 1.0               # set by Trainer each epoch

        if self.conv_cache_table is not None and not getattr(block, "conv_cache", False):
            cprint(
                "[PersistentGSNBlock] conv_cache_table provided but block.conv_cache=False; "
                "the table will be ignored. Enable conv_cache on the GSNBlock to use it.",
                "yellow",
            )

    @property
    def adaptive_mode(self) -> bool:
        return self.commit_gate is not None and self.activity_buffers is not None

    def __call__(self,
                 *,
                 snap: Snapshot,
                 commit: bool = False,
                 training: Optional[bool] = None,
                 run_step_mode: Optional[bool] = True
                ) -> Tuple[tf.Tensor, tf.Tensor]:

        if not self.built:
            self._N = snap.num_nodes
            self.built = True

        return self.call(
                            snap          = snap,
                            commit        = commit,
                            training      = training,
                            run_step_mode = run_step_mode,
                         )


    def call(
                self,
                *,
                snap: Snapshot,
                run_step_mode: Optional[bool] = True,
                commit: Optional[bool] = False,
                training: Optional[bool] = None
            ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass.

        Returns
        -------
        out    : [N, hidden]          - new node embeddings
        s_next : [N, state_dim_total] - updated SSM state (not yet committed)
        """
        ids  = tf.cast(snap.node_ids, tf.int32)
        s_in = self.state_table.get(ids)          # [N, state_dim_total]

        # Read persistent conv cache for these nodes (3-D natural layout).
        # ConvCacheTable stores [num_entities, K-1, C]; .get returns [N, K-1, C].
        # Falls back to None when no table is attached, in which case GSNBlock
        # will use zero-pad inside mamba2.step().
        conv_s_in = None
        if self.conv_cache_table is not None and getattr(self.block, "conv_cache", False):
            conv_s_in = self.conv_cache_table.get(ids)        # [N, K-1, C]

        if training and self.noise_scale > 0.0:
            s_in = s_in + tf.random.normal(tf.shape(s_in), stddev = self.noise_scale)

        out, s_next, conv_s_next = self.block(
                                                snap = snap,
                                                s = s_in,
                                                conv_s = conv_s_in,
                                                training = training,
                                                run_step_mode = run_step_mode
                                              )

        if commit:
            if self.adaptive_mode:
                self._adaptive_commit(ids, s_in, s_next, snap)
            else:
                # Broadcast Python-float commit_alpha to [N, 1] so it matches
                # DenseStateTable.put's strict (None, 1) input_signature.
                alpha_vec = tf.fill([tf.shape(ids)[0], 1],
                                    tf.cast(self.commit_alpha, tf.float32))
                self.state_table.put(ids, s_next, alpha = alpha_vec)

            # Conv cache is a sliding window, never an EMA — always overwrite.
            if self.conv_cache_table is not None and conv_s_next is not None:
                self.conv_cache_table.put(ids, conv_s_next)

        return out, s_next

    # ------------------------------------------------------------------
    # Adaptive commit helpers
    # ------------------------------------------------------------------

    def _event_count_per_node(self, snap: Snapshot) -> np.ndarray:
        """Return [N] int array: number of edge appearances per local node index."""
        N    = snap.num_nodes
        src  = np.asarray(snap.edge_src, dtype = np.int64)
        dst  = np.asarray(snap.edge_dst, dtype = np.int64)
        if src.ndim == 2:
            valid = np.logical_and(src >= 0, dst >= 0)
            src = src[valid]
            dst = dst[valid]
        ones = np.ones(src.shape[0], dtype = np.int64)
        counts = np.zeros(N, dtype = np.int64)
        np.add.at(counts, src, ones)
        np.add.at(counts, dst, ones)
        return counts

    def _adaptive_commit(
                            self,
                            ids:    tf.Tensor,
                            s_in:   tf.Tensor,
                            s_next: tf.Tensor,
                            snap:   Snapshot,
                        ) -> None:
        """Compute per-node α via the commit gate and write to the state table."""
        ids_np       = np.asarray(snap.node_ids, dtype = np.int64)
        event_counts = self._event_count_per_node(snap)

        delta_t, n_events, c, ema_ia = self.activity_buffers.get_features(
            ids_np, float(snap.t_ref), event_counts
        )

        tau = self.activity_buffers.tau_data
        alpha = self.commit_gate(
                                    s_old             = s_in,
                                    s_new             = s_next,
                                    delta_t           = tf.constant(delta_t,  dtype = tf.float32),
                                    event_count       = tf.constant(n_events, dtype = tf.float32),
                                    update_count      = tf.constant(c,        dtype = tf.float32),
                                    ema_interarrival  = tf.constant(ema_ia,   dtype = tf.float32),
                                    tau_data          = tau,
                                    warmup_eta        = self.warmup_eta,
                                    alpha0_uniform    = self.commit_alpha,
                                    training          = False,           # commit is outside tape
                                )                                        # [N, 1]

        self.state_table.put(ids, s_next, alpha = alpha)
        self.activity_buffers.update(
                                        ids_np,
                                        float(snap.t_ref),
                                        alpha.numpy().reshape(-1),
                                    )

    # ------------------------------------------------------------------

    def compute_alpha_for_reg(
                                self,
                                snap:    Snapshot,
                                s_next:  tf.Tensor,
                                training: Optional[bool] = None,
                              ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute (alpha, alpha_prior) on snap's node set using s_next (in tape).

        Called inside the GradientTape during the snap_pre scoring forward so
        that the gate's MLP parameters receive gradients from the regularisation
        losses.

        Parameters
        ----------
        snap    : the snap_pre snapshot (scoring forward).
        s_next  : [N, D] — SSM output from the snap_pre forward (in the tape).
        training: bool.

        Returns
        -------
        alpha       : [N, 1] — learned gate output (differentiable).
        alpha_prior : [N, 1] — time-based prior (used in prior_loss).
        """
        ids_np       = np.asarray(snap.node_ids, dtype = np.int64)
        event_counts = self._event_count_per_node(snap)

        delta_t, n_events, c, ema_ia = self.activity_buffers.get_features(
            ids_np, float(snap.t_ref), event_counts
        )

        # s_old is a constant (tf.gather from non-trainable table)
        ids  = tf.cast(snap.node_ids, tf.int32)
        s_old = tf.stop_gradient(self.state_table.get(ids))

        tau = self.activity_buffers.tau_data
        alpha = self.commit_gate(
                                    s_old             = s_old,
                                    s_new             = s_next,
                                    delta_t           = tf.constant(delta_t,  dtype = tf.float32),
                                    event_count       = tf.constant(n_events, dtype = tf.float32),
                                    update_count      = tf.constant(c,        dtype = tf.float32),
                                    ema_interarrival  = tf.constant(ema_ia,   dtype = tf.float32),
                                    tau_data          = tau,
                                    warmup_eta        = self.warmup_eta,
                                    alpha0_uniform    = self.commit_alpha,
                                    training          = training
                                )                                               # [N, 1]

        # Prior alpha for regularisation
        phi         = self.commit_gate.exposure(
                                                    tf.constant(delta_t,  dtype=tf.float32),
                                                    tf.constant(n_events, dtype=tf.float32),
                                                    tau
                                                )
        alpha_prior = self.commit_gate.prior_alpha(phi, self._lambda0)

        return alpha, alpha_prior
    
    def compute_shadow_committed_state(
                                            self,
                                            snap: Snapshot,
                                            s_next: tf.Tensor,
                                            training: Optional[bool] = None,
                                      ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return differentiable shadow-committed state for scoring.

        This does NOT mutate the persistent state table.

        s_score = S_old + alpha * (s_next - S_old)

        The ranking loss can now train alpha because logits depend on s_score.
        """

        if not self.adaptive_mode:
            return s_next, None, None

        ids_np       = np.asarray(snap.node_ids, dtype = np.int64)
        event_counts = self._event_count_per_node(snap)

        delta_t, n_events, c, ema_ia = self.activity_buffers.get_features(
                                                                            ids_np,
                                                                            float(snap.t_ref),
                                                                            event_counts
                                                                         )

        ids = tf.cast(snap.node_ids, tf.int32)

        # Important: old table state is context, not a trainable variable.
        s_old = tf.stop_gradient(self.state_table.get(ids))
        s_new = tf.cast(s_next, tf.float32)

        tau = self.activity_buffers.tau_data

        alpha = self.commit_gate(
                                    s_old             = s_old,
                                    s_new             = s_new,
                                    delta_t           = tf.constant(delta_t,  dtype = tf.float32),
                                    event_count       = tf.constant(n_events, dtype = tf.float32),
                                    update_count      = tf.constant(c,        dtype = tf.float32),
                                    ema_interarrival  = tf.constant(ema_ia,   dtype = tf.float32),
                                    tau_data          = tau,
                                    warmup_eta        = self.warmup_eta,
                                    alpha0_uniform    = self.commit_alpha,
                                    training          = training
                                )

        s_score = s_old + alpha * (s_new - s_old)

        phi = self.commit_gate.exposure(
                                            tf.constant(delta_t,  dtype = tf.float32),
                                            tf.constant(n_events, dtype = tf.float32),
                                            tau
                                        )
        alpha_prior = self.commit_gate.prior_alpha(phi, self._lambda0)

        return s_score, alpha, alpha_prior

    # Internal: lambda0 is set by Trainer after initialize_bias()
    _lambda0: float = 1.0

    def compute_output_shape(self, input_shape):
        return ((self._N, self.block.hidden), (self._N, self.block.state_dim_total))