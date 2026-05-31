from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from einops import einsum, rearrange
from typing import Optional, Tuple

try:
    from src.ssd import SSDLayer
    from src.utils import ShapeError
    from src.utils import get_type_of_arguments
except:
    from .ssd import SSDLayer
    from ..utils import ShapeError
    from ..utils import get_type_of_arguments


class Mamba2SSD(layers.Layer):
    def __init__(self, num_heads: int,
                 head_dim: int,
                 state_dim: int,
                 d_model: int,
                 sequence_length: int,
                 num_chunks: int,
                 conv1d_kernel_size: Optional[int] = 4,
                 dtype: str = "float32",
                 name: Optional[str] = None,
                 **kwargs):
        
        super().__init__(dtype = dtype, name = name, **kwargs)
        self.prefix = f"{name}_" if name is not None else ""
        
        self.num_heads = int(num_heads)     # H
        self.head_dim = int(head_dim)       # P
        self.state_dim = int(state_dim)     # N
        self.d_model = int(d_model)
        self.sequence_length = int(sequence_length)
        self.num_chunks = int(num_chunks)
        self.conv1d_kernel_size = int(conv1d_kernel_size)

    @property
    def conv_cache_dim(self) -> int:
        """Per-node persistent conv-cache size = (kernel_size - 1) * H * (2N + P)."""
        H, P, N = self.num_heads, self.head_dim, self.state_dim
        return (self.conv1d_kernel_size - 1) * H * (2 * N + P)

    @property
    def xbc_channels(self) -> int:
        """Channel count of the XBC stream entering the causal conv."""
        H, P, N = self.num_heads, self.head_dim, self.state_dim
        return H * (2 * N + P)
        
    def build(self, input_shape: Tuple[Tuple[int, int, int], Tuple[int, int, int, int, int]]):
        _, length, dim = input_shape[0]             # input sequence
        _, _one, _H, _P, _N  = input_shape[1]       # initial state
        
        if length != self.sequence_length:
            raise ShapeError(f"[Mamba2SSD] Expected dimension 1 (zero-based) of input to have shape "
                             f"{self.sequence_length}. Found input_shape = {input_shape}.")
        if dim != self.d_model:
            raise ShapeError(f"[Mamba2SSD] Expected dimension 2 (zero-based) of input to have shape "
                             f"{self.d_model}. Found input_shape = {input_shape}.")
            
        H, P, N = self.num_heads, self.head_dim, self.state_dim
        
        if _one != 1:
            raise ShapeError(f"[Mamba2SSD] Expected dimension 1 (zero-based) of input to have shape "
                             f"1. Found {_one}.")
        if _H != H:
            raise ShapeError(f"[Mamba2SSD] Expected dimension 2 (zero-based) of input to have shape "
                             f"{H}. Found {_H}.")
        if _P != P:
            raise ShapeError(f"[Mamba2SSD] Expected dimention 3 (zero-based) if input to have shape "
                             f"{P}. Found {_P}.")
        if _N != N:
            raise ShapeError(f"[Mamba2SSD] Expected dimention 4 (zero-based) if input to have shape "
                             f"{N}. Found {_N}.")
        
        self._ip_proj_mat = self.add_weight(name = f"{self.prefix}ip_projection_matrix",
                                            shape = (self.d_model, H * (2 * N + 2 * P + 1)),
                                            dtype = self.variable_dtype,
                                            trainable = True)
        
        self.ssd = SSDLayer(num_chunks = self.num_chunks,
                            dtype = self.dtype_policy,
                            name = f"{self.prefix}ssd")
        
        self.conv = layers.DepthwiseConv1D(kernel_size = self.conv1d_kernel_size,
                                           padding = "valid",
                                           use_bias = True,
                                           activation = None,
                                           dtype = self.dtype_policy)
        
        self.norm = layers.GroupNormalization(groups = H,
                                              axis = -1,
                                              epsilon = 1e-5,
                                              center = True,
                                              scale = True,
                                              name = f"{self.prefix}group_norm",
                                              dtype = self.dtype_policy)
        
        self._op_proj_mat = self.add_weight(name = f"{self.prefix}_op_projection_matrix",
                                            shape = (H * P, self.d_model),
                                            dtype = self.variable_dtype,
                                            trainable = True)
        
        super().build(input_shape)
        
    def call(
                self,
                inputs,
                training: Optional[bool] = None
             ):
        """
        Args:
            inputs:
                u:          [batch, seq_len, d_model]          – **entire** sequence
                state:      [batch, 1, H, P, N] or None        – previous SSM state (zeros if None)

        Returns:
            y:               [batch, d_model]
            new_state:       [batch, H, P, N]
        """
        u, state = inputs[0], inputs[1]
        H, P, N = self.num_heads, self.head_dim, self.state_dim
        dtype = u.dtype
        u = tf.cast(u, self.variable_dtype)
        projected = einsum(u, self._ip_proj_mat, "b l di, di do-> b l do")
        projected = tf.cast(projected, self.compute_dtype)
        A, XBC, Z = tf.split(projected, num_or_size_splits = [H, H * (2 * N + P), H * P], axis = -1)
        A = -tf.nn.softplus(A)
        z = tf.nn.silu(Z)
        XBC = tf.pad(
                        XBC,
                        [[0, 0], [self.conv1d_kernel_size - 1, 0], [0, 0]],
                        name = f"{self.prefix}zero_pad"
                    )
        XBC = tf.cast(XBC, self.variable_dtype)
        XBC = self.conv(XBC, training = training)
        XBC = tf.nn.silu(XBC)
        XBC = tf.cast(XBC, self.compute_dtype)
        X, B, C = tf.split(XBC, num_or_size_splits = [H * P, H * N, H * N], axis = -1)

        X = rearrange(X, "b l (h p) -> b l h p", h = H)
        B = rearrange(B, "b l (h n) -> b l h n", h = H)
        C = rearrange(C, "b l (h n) -> b l h n", h = H)

        Y_ssm, final_state = self.ssd(X, A, B, C, initial_states = state, training = training)
        Y_ssm = rearrange(Y_ssm, "b l h p -> b l (h p)", h = H)
        Y_ssm = self.norm(Y_ssm, training = training)
        gate = tf.multiply(Y_ssm, z)      # (B, L, H * P)
        y = einsum(gate, self._op_proj_mat, "b l d_inner, d_inner d_model -> b l d_model")
        return (
                    tf.cast(y, dtype),    # (batch, d_model)
                    final_state           # (batch, H, P, N)
                )

    def step(self, u: tf.Tensor, state: Optional[tf.Tensor] = None,
             conv_state: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Single-step streaming SSM update.

        Mathematically equivalent to call() with L=1 *when conv_state is None*
        (zero-pad fallback — i.e. the previous behaviour). When a persistent
        per-node conv_state is supplied, the causal DepthwiseConv1D sees the
        true streaming history rather than zero context, making this step a
        true streaming Mamba-2 update instead of an L=1 sequence-mode call.

        Args:
            u:          [batch, d_model]                  – one input token per item
            state:      [batch, H, P, N] or None          – previous SSM state (zeros if None)
            conv_state: [batch, K-1, H*(2N+P)] or None    – previous conv cache.
                        When None, falls back to zero-pad (legacy step behaviour).

        Returns:
            y:               [batch, d_model]
            new_state:       [batch, H, P, N]
            new_conv_state:  [batch, K-1, H*(2N+P)]
                             — sliding-window update of conv_state. Always returned
                             (even when conv_state was None) so callers can decide
                             whether to persist it.
        """
        if not self.built:
            # Trigger weight creation via a dummy sequence-of-1 forward pass
            dummy = (
                        tf.zeros(
                                    [1, self.sequence_length, self.d_model],
                                    dtype = self.variable_dtype
                                ),
                        tf.zeros(
                                    [1, 1, self.num_heads, self.head_dim, self.state_dim],
                                    dtype = self.variable_dtype
                                )
                    )
            self(dummy, training = False)

        H, P, N = self.num_heads, self.head_dim, self.state_dim
        K       = self.conv1d_kernel_size
        dtype = u.dtype
        batch = tf.shape(u)[0]

        # ---- input projection ------------------------------------------------
        u_cast = tf.cast(u, self.variable_dtype)
        # [batch, d_model] × [d_model, H*(2N+2P+1)] → [batch, H*(2N+2P+1)]
        projected = tf.matmul(u_cast, self._ip_proj_mat)
        projected = tf.cast(projected, self.compute_dtype)

        # Split: A [batch,H], XBC [batch, H*(P+2N)], Z [batch, H*P]
        A_raw, XBC_flat, Z_flat = tf.split(projected, num_or_size_splits = [H, H * (2 * N + P), H * P], axis = -1)
        z = tf.nn.silu(Z_flat)  # [batch, H*P]

        # ---- causal depthwise conv (same kernel as call()) -------------------
        # XBC_seq is the single new token; we either:
        #   - prepend the persistent cache (true streaming), or
        #   - prepend zeros (legacy step / L=1 sequence-mode equivalence)
        XBC_seq      = tf.expand_dims(XBC_flat, axis = 1)              # [batch, 1, C]
        XBC_seq_cast = tf.cast(XBC_seq, self.variable_dtype)
        if conv_state is not None:
            cs             = tf.cast(conv_state, self.variable_dtype)  # [batch, K-1, C]
            XBC_with_cache = tf.concat([cs, XBC_seq_cast], axis = 1)   # [batch, K, C]
        else:
            XBC_with_cache = tf.pad(XBC_seq_cast,
                                    [[0, 0], [K - 1, 0], [0, 0]])      # [batch, K, C]

        XBC_conv = self.conv(XBC_with_cache, training = training)      # [batch, 1, C]
        XBC_conv = tf.nn.silu(XBC_conv)
        XBC_conv = tf.cast(XBC_conv[:, 0, :], self.compute_dtype)      # [batch, C]

        # Sliding-window update: drop oldest token, keep latest K-1
        new_conv_state = XBC_with_cache[:, 1:, :]                      # [batch, K-1, C]
        new_conv_state = tf.cast(new_conv_state, dtype)

        X_flat, B_flat, C_flat = tf.split(XBC_conv, num_or_size_splits = [H * P, H * N, H * N], axis = -1)
        X = tf.reshape(X_flat, [batch, H, P])        # [batch, H, P]
        B = tf.reshape(B_flat, [batch, H, N])        # [batch, H, N]
        C = tf.reshape(C_flat, [batch, H, N])        # [batch, H, N]

        # ---- SSM step: s_next = decay * s + X ⊗ B -----------------------------------
        decay = tf.exp(-tf.nn.softplus(A_raw))        # [batch, H] — guaranteed ∈ (0, 1)

        if state is not None:
            s = tf.cast(tf.reshape(state, [batch, H, P, N]), self.compute_dtype)
        else:
            s = tf.zeros([batch, H, P, N], dtype = self.compute_dtype)

        # Outer product: X[h,p] * B[h,n] → [batch, H, P, N]
        XB = einsum(X, B, "b h p, b h n -> b h p n")
        s_next = decay[:, :, None, None] * s + XB    # [batch, H, P, N]

        # ---- output: y = C * s_next (contract over state dim) ---------------
        y_ssm = einsum(C, s_next, "b h n, b h p n -> b h p")   # [batch, H, P]

        # Norm + gate (same layers as call())
        y_flat = tf.reshape(y_ssm, [batch, 1, H * P])
        y_norm = self.norm(y_flat, training = training)
        y_norm = y_norm[:, 0, :]                         # [batch, H*P]
        gate_out = y_norm * z                            # [batch, H*P]

        # Output projection
        y = tf.matmul(gate_out, self._op_proj_mat)       # [batch, d_model]

        return tf.cast(y, dtype), tf.cast(s_next, dtype), new_conv_state