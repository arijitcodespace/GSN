from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from einops import einsum, rearrange

from typing import Optional, Tuple

try:
    from src.utils import ShapeError
    from src.utils import get_type_of_arguments
except:
    from ..utils import ShapeError
    from ..utils import get_type_of_arguments

def segsum(x, inf: float = -1e9):
    T = tf.shape(x)[-1]
    x_cs = tf.cumsum(x, axis = -1)
    x_segsum = x_cs[..., :, None] - x_cs[..., None, :]
    mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
    x_segsum = tf.where(mask == 0, inf, x_segsum)
    return x_segsum
    

class SSDLayer(layers.Layer):
    def __init__(self, num_chunks: int, dtype : str= "float32", name: Optional[str] = None, **kwargs):
        super().__init__(dtype = dtype, name = name, **kwargs)
        self.num_chunks = int(num_chunks)
        prefix = f"{name}_" if name is not None else ""
    
    def build(self, input_shape: Tuple[Tuple[int, int, int, int],
                                       Tuple[int, int, int], 
                                       Tuple[int, int, int, int],
                                       Tuple[int, int, int, int]]):
        X_shape, A_shape, B_shape, C_shape = input_shape
        if len(X_shape) != 4:
            raise ShapeError(f"[SSDLayer] Expected X to be a rank-4 tensor. "
                             f"Found X.shape = {X_shape}.")
        if len(A_shape) != 3:
            raise ShapeError(f"[SSDLayer] Expected X to be a rank-3 tensor. "
                             f"Found A.shape = {A_shape}.")
        if len(B_shape) != 4:
            raise ShapeError(f"[SSDLayer] Expected B to be a rank-4 tensor. "
                             f"Found B.shape = {B_shape}.")
        if len(C_shape) != 4:
            raise ShapeError(f"[SSDLayer] Expected C to be a rank-4 tensor. "
                             f"Found C.shape = {C_shape}.")
        
        if not (B_shape == C_shape):
            raise ShapeError(f"[SSDLayer] Expected B and C to have same shape. Found "
                             f"B.shape = {B_shape}, "
                             f"C.shape = {C_shape}.")
            
        if not (X_shape[:3] == A_shape == B_shape[:3] == C_shape[:3]):
            raise ShapeError(f"[SSDLayer] Expected the first three dimensions of "
                             f"shapes of X, A, B, C to match. Found: "
                             f"X.shape = {X_shape} "
                             f"A.shape = {A_shape} "
                             f"B.shape = {B_shape} "
                             f"C.shape = {C_shape}.")
        
        L, H, P = X_shape[1], X_shape[2], X_shape[3]
        N = B_shape[3]
        
        if L % self.num_chunks != 0:
            raise ValueError(f"`L` must be divisible by `num_chunks`. "
                             f"Found L = {L}, num_chunks = {self.num_chunks} .")
        
        self.L = L      # Sequence Length
        self.H = H      # Number of Heads
        self.P = P      # Head/Projection dim
        self.N = N      # State dim
        
        super().build(input_shape)
    
    def __call__(self, *args, **kwargs):
        args = list(args)
        if len(args) == 3:
            C = kwargs.pop('C', None)
            if C is None:
                pos_args, kw_args = get_type_of_arguments(*args, **kwargs)
                raise ValueError(f"[SSDLayer] Expected at least 4 arguments (X, A, B, C) to be passed. "
                                 f"Found positional arguments\n"
                                 f"{pos_args}\n"
                                 f"and keyword arguments\n"
                                 f"{kw_args}")
            else:
                args.append(C)
        elif len(args) == 2:
            B = kwargs.pop('B', None)
            C = kwargs.pop('C', None)
            if (C is None) or (B is None):
                pos_args, kw_args = get_type_of_arguments(*args, **kwargs)
                raise ValueError(f"[SSDLayer] Expected at least 4 arguments (X, A, B, C) to be passed. "
                                 f"Found positional arguments\n"
                                 f"{pos_args}\n"
                                 f"and keyword arguments\n"
                                 f"{kw_args}")
            else:
                args.append(B)
                args.append(C)
        elif len(args) == 1:
            A = kwargs.pop('A', None)
            B = kwargs.pop('B', None)
            C = kwargs.pop('C', None)
            if (C is None) or (B is None) or (A is None):
                pos_args, kw_args = get_type_of_arguments(*args, **kwargs)
                raise ValueError(f"[SSDLayer] Expected at least 4 arguments (X, A, B, C) to be passed. "
                                 f"Found positional arguments\n"
                                 f"{pos_args}\n"
                                 f"and keyword arguments\n"
                                 f"{kw_args}")
            else:
                args.append(A)
                args.append(B)
                args.append(C)
                
        elif len(args) == 0:
            X = kwargs.pop('X', None)
            A = kwargs.pop('A', None)
            B = kwargs.pop('B', None)
            C = kwargs.pop('C', None)
            if (C is None) or (B is None) or (A is None) or (X is None):
                pos_args, kw_args = get_type_of_arguments(*args, **kwargs)
                raise ValueError(f"[SSDLayer] Expected at least 4 arguments (X, A, B, C) to be passed. "
                                 f"Found positional arguments\n"
                                 f"{pos_args}\n"
                                 f"and keyword arguments\n"
                                 f"{kw_args}")
            else:
                args.append(X)
                args.append(A)
                args.append(B)
                args.append(C)
        elif len(args) != 4:
            raise ValueError(f"[SSDLayer] Expected at most 4 positional arguments (X, A, B, C). "
                             f"Found {len(args)}")
        
        if not self.built:
            super().build([arg.shape for arg in args])
        
        args = tuple(args)
        return super().__call__(*args, **kwargs)
            
    
    def call(self, X: tf.Tensor,
             A: tf.Tensor,
             B: tf.Tensor,
             C: tf.Tensor,
             initial_states: Optional[tf.Tensor] = None,
             training: Optional[bool] = None):        
        
        dtype = X.dtype
        X = tf.cast(X, self.compute_dtype)
        A = tf.cast(A, self.compute_dtype)
        B = tf.cast(B, self.compute_dtype)
        C = tf.cast(C, self.compute_dtype)
        
        batch, L, H, P = X.shape
        N = B.shape[-1]
        
        num_chunks = self.num_chunks
        Q = L // num_chunks
        
        # X: (batch, L, H, P)
        # A: (batch, L, H)
        # B: (batch, L, H, N)
        # C: (batch, L, H, N)
        
        # reshape to (batch, num_chunks, Q, ...)
        X = tf.reshape(X, [batch, num_chunks, Q, H, P])
        A = tf.reshape(A, [batch, num_chunks, Q, H])
        B = tf.reshape(B, [batch, num_chunks, Q, H, N])
        C = tf.reshape(C, [batch, num_chunks, Q, H, N])
        
        A = rearrange(A, "b c q h -> b h c q")
        A_cumsum = tf.cumsum(A, axis = -1)
        
        # 1-SS mask
        L = tf.exp(segsum(A))
        
        # intra-chunk output
        Y_diag = einsum(C, B, L, X, "b c q h n, b c s h n, b h c q s, b c s h p -> b c q h p")
        decay_states = tf.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = einsum(B, decay_states, X, "b c q h n, b h c q, b c q h p -> b c h p n")
        
        # inter-chunk output
        if initial_states is None:
            initial_states = tf.zeros_like(states[:, :1])
        
        states = tf.concat([initial_states, states], axis = 1)
        decay_chunk = tf.exp(segsum(tf.pad(A_cumsum[:, :, :, -1], [[0, 0], [0, 0], [1, 0]])))
        new_states = einsum(decay_chunk, states, "b h z c, b c h p n -> b z h p n")
        states, final_state = new_states[:, :-1], new_states[:, -1]
        
        # state -> output conversion
        state_decay_out = tf.exp(A_cumsum)
        Y_off = einsum(C, states, state_decay_out, "b c q h n, b c h p n, b h c q -> b c q h p")
        Y = rearrange(Y_diag + Y_off, "b c q h p -> b (c q) h p")
        
        return tf.cast(Y, dtype), tf.cast(final_state, dtype)
        