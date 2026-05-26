"""Adaptive node-wise commit gate for GSN.

Replaces the uniform scalar α in the EMA commit rule

    S_i ← (1 − α) S_i + α s′_i

with a learned, bounded, time- and content-aware per-node gate

    S_i ← S_i + α_{i,k} (s′_i − S_i),   α_{i,k} ∈ [α_min, α_max]

parameterised as a continuous-time hazard:

    α_{i,k} = α_min + (α_max − α_min) · [1 − exp(−λ_{i,k} · φ_{i,k})]

    λ_{i,k} = softplus(b_0 + MLP_ψ(z_{i,k})) + λ_min

    φ_{i,k} = δ_0 + log(1 + Δt / τ_data) + c_n · log(1 + n_{i,k})

Feature vector z_{i,k} (7 scalars):
    [log(1+Δt), log(1+n), log(1+c), log(1+ema_ia),
     log(1 + Δt/ema_ia),            ← normalised time gap
     ‖s′−S‖ / (‖S‖ + ε),           ← relative novelty
     cos(S, s′)]                     ← directional change

Gradient note
-------------
In the current GSN training loop the actual commit (DenseStateTable.put)
happens OUTSIDE the GradientTape.  The gate is therefore trained primarily
via the regularisation losses (prior + saturation) that are computed INSIDE
the tape on the snap_pre forward pass.  Both losses are differentiable
through s_new (which IS in the tape) via the novelty features.

A warmup schedule blends the gate output toward a fixed α_0 during the
first ``warmup_epochs`` training epochs so that early-training instability
is suppressed.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AdaptiveCommitGate(layers.Layer):
    """Learned node-wise commit gate using a hazard parameterisation.

    Parameters
    ----------
    hidden      : MLP hidden dimension.
    num_layers  : number of MLP layers (excluding final projection).
    alpha_min   : lower bound on α (prevents exact 0).
    alpha_max   : upper bound on α (prevents exact 1).
    lambda_min  : minimum hazard rate (prevents λ collapsing to 0).
    delta0      : exposure floor (prevents φ = 0 at Δt = 0 events).
    cn          : weight of the log-event-count term in φ.
    """

    def __init__(
                    self,
                    hidden:     int   = 64,
                    num_layers: int   = 2,
                    alpha_min:  float = 1e-4,
                    alpha_max:  float = 0.999,
                    lambda_min: float = 1e-5,
                    delta0:     float = 0.05,
                    cn:         float = 0.25,
                    name:       str   = "adaptive_commit_gate",
                    **kwargs,
                ):
        super().__init__(name=name, **kwargs)
        self.gate_hidden    = int(hidden)
        self.gate_num_layers = int(num_layers)
        self.alpha_min      = float(alpha_min)
        self.alpha_max      = float(alpha_max)
        self.lambda_min     = float(lambda_min)
        self.delta0         = float(delta0)
        self.cn             = float(cn)

        # MLP: maps z [N, 7] → a [N, 1]
        mlp_layers = []
        for i in range(self.gate_num_layers):
            mlp_layers.append(
                layers.Dense(
                    self.gate_hidden,
                    activation="gelu",
                    use_bias=True,
                    name=f"{name}_mlp_{i}",
                )
            )
        # Final projection: [hidden] → [1], NO activation, bias initialised in
        # initialize_bias() after tau_data is known.
        mlp_layers.append(
            layers.Dense(
                1,
                use_bias=True,
                kernel_initializer=keras.initializers.Zeros(),
                bias_initializer=keras.initializers.Zeros(),
                name=f"{name}_mlp_out",
            )
        )
        self.mlp = keras.Sequential(mlp_layers, name=f"{name}_mlp")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def initialize_bias(
                            self,
                            alpha0:   float,
                            tau_data: float,
                            mean_n:   float = 2.0,
                        ) -> None:
        """Set the output layer's bias so the gate starts near the old α_0.

        Solves for b_0 such that, at median exposure (Δt ≈ τ_data, n ≈ mean_n),
        the gate outputs approximately alpha0.

        Call this AFTER the model's dummy build so the Dense weights exist,
        and BEFORE the first training epoch.

        Parameters
        ----------
        alpha0    : the previous uniform α (e.g. 0.999).
        tau_data  : dataset median inter-event time.
        mean_n    : expected events per node per bucket (rough estimate).
        """
        # Exposure at median event: Δt = τ_data → log(1+1) = log(2)
        phi_bar = (
            self.delta0
            + float(np.log(2.0))
            + self.cn * float(np.log(1.0 + max(mean_n, 0.0)))
        )

        # Rescale alpha0 to [0, 1] within [alpha_min, alpha_max]
        span = self.alpha_max - self.alpha_min
        alpha0_scaled = max(
            1e-6, min(1.0 - 1e-6, (float(alpha0) - self.alpha_min) / span)
        )

        # Hazard needed to produce alpha0_scaled at phi_bar:
        #   alpha_scaled = 1 − exp(−λ · φ)  ⟹  λ = −log(1−α_scaled) / φ
        lambda0 = -float(np.log(1.0 - alpha0_scaled)) / max(phi_bar, 1e-8)

        # b0 = softplus_inverse(lambda0 − lambda_min)
        lam_excess = max(lambda0 - self.lambda_min, 1e-8)
        b0 = float(np.log(np.exp(lam_excess) - 1.0 + 1e-8))

        # Assign to the output Dense layer's bias
        out_layer = self.mlp.layers[-1]
        if out_layer.built:
            out_layer.bias.assign([b0])
        else:
            # Store for deferred assignment after first build
            self._pending_b0 = b0

    def _maybe_apply_pending_b0(self) -> None:
        """Apply deferred bias if the layer was built after initialize_bias was called."""
        if hasattr(self, "_pending_b0"):
            out_layer = self.mlp.layers[-1]
            if out_layer.built:
                out_layer.bias.assign([self._pending_b0])
                del self._pending_b0

    # ------------------------------------------------------------------
    # Exposure function
    # ------------------------------------------------------------------

    def exposure(
                    self,
                    delta_t:  tf.Tensor,
                    n:        tf.Tensor,
                    tau_data: float,
                ) -> tf.Tensor:
        """Compute φ_{i,k} = δ_0 + log(1 + Δt/τ) + c_n · log(1+n).

        Parameters
        ----------
        delta_t  : [N] float32 — time since last update.
        n        : [N] float32 — events per node in this bucket.
        tau_data : dataset time scale.

        Returns [N, 1] float32.
        """
        tau = tf.cast(max(tau_data, 1e-8), tf.float32)
        phi = (
            self.delta0
            + tf.math.log1p(delta_t / tau)
            + self.cn * tf.math.log1p(n)
        )
        return tf.reshape(phi, [-1, 1])

    # ------------------------------------------------------------------
    # Prior alpha (for regularisation)
    # ------------------------------------------------------------------

    def prior_alpha(
                        self,
                        phi:    tf.Tensor,
                        lambda0: float,
                    ) -> tf.Tensor:
        """Compute the time-based prior α_prior = scale(1 − exp(−λ_0 · φ)).

        This is what the gate WOULD output with the fixed initialisation
        hazard λ_0.  Used in the prior regularisation loss.

        Parameters
        ----------
        phi     : [N, 1] float32 — exposure tensor.
        lambda0 : target hazard rate (same λ_0 computed in initialize_bias).
        """
        span = self.alpha_max - self.alpha_min
        raw  = 1.0 - tf.exp(-tf.cast(lambda0, tf.float32) * phi)
        return self.alpha_min + span * raw

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
                self,
                s_old:          tf.Tensor,
                s_new:          tf.Tensor,
                delta_t:        tf.Tensor,
                event_count:    tf.Tensor,
                update_count:   tf.Tensor,
                ema_interarrival: tf.Tensor,
                tau_data:       float,
                warmup_eta:     float = 1.0,
                alpha0_uniform: float = 0.2,
                training:       Optional[bool] = None,
            ) -> tf.Tensor:
        """Compute per-node α_{i,k} ∈ [α_min, α_max].

        Parameters
        ----------
        s_old           : [N, D] — old persistent state (constant in tape).
        s_new           : [N, D] — proposed new state (in tape when training).
        delta_t         : [N]    — seconds since last committed update.
        event_count     : [N]    — events per node in current bucket.
        update_count    : [N]    — total committed updates per node.
        ema_interarrival: [N]    — EMA of inter-update intervals.
        tau_data        : float  — dataset time scale.
        warmup_eta      : float ∈ [0,1] — 0 = pure α0, 1 = pure learned.
        alpha0_uniform  : float  — the uniform α_0 used during warmup.
        training        : bool   — passed to sub-layers.

        Returns
        -------
        alpha : [N, 1] float32 ∈ [α_min, α_max]
        """
        self._maybe_apply_pending_b0()

        eps  = 1e-8
        s_old = tf.cast(s_old, tf.float32)
        s_new = tf.cast(s_new, tf.float32)
        dt    = tf.cast(tf.reshape(delta_t,         [-1]), tf.float32)
        n     = tf.cast(tf.reshape(event_count,     [-1]), tf.float32)
        c     = tf.cast(tf.reshape(update_count,    [-1]), tf.float32)
        ema   = tf.cast(tf.reshape(ema_interarrival,[-1]), tf.float32)

        # --- Feature 1-4: time and activity ---
        f_log_dt   = tf.math.log1p(dt)               # log(1+Δt)
        f_log_n    = tf.math.log1p(n)                 # log(1+n)
        f_log_c    = tf.math.log1p(c)                 # log(1+c)
        f_log_ema  = tf.math.log1p(ema)               # log(1+ema_ia)

        # --- Feature 5: normalised time gap r = log(1 + Δt/ema_ia) ---
        f_norm_gap = tf.math.log1p(dt / (ema + eps))

        # --- Feature 6: relative novelty ‖s'−S‖ / (‖S‖ + ε) ---
        diff_norm  = tf.norm(s_new - s_old, axis=-1)           # [N]
        old_norm   = tf.norm(s_old,         axis=-1) + eps      # [N]
        f_rel_nov  = diff_norm / old_norm

        # --- Feature 7: cosine similarity cos(S, s') ---
        dot        = tf.reduce_sum(s_old * s_new, axis=-1)     # [N]
        new_norm   = tf.norm(s_new, axis=-1) + eps              # [N]
        f_cos      = dot / (old_norm * new_norm)

        # Stack into z: [N, 7]
        z = tf.stack(
                [f_log_dt, f_log_n, f_log_c, f_log_ema, f_norm_gap, f_rel_nov, f_cos],
                axis=-1,
            )

        # MLP → a_{i,k}: [N, 1]
        a = self.mlp(z, training=training)          # [N, 1]

        # Hazard λ = softplus(a) + λ_min
        lam = tf.nn.softplus(a) + self.lambda_min   # [N, 1]

        # Exposure φ: [N, 1]
        phi = self.exposure(dt, n, tau_data)

        # Hazard gate (rescaled to [α_min, α_max])
        span  = self.alpha_max - self.alpha_min
        alpha = self.alpha_min + span * (1.0 - tf.exp(-lam * phi))

        # Warmup: blend toward uniform α_0 when warmup_eta < 1.0
        if warmup_eta < 1.0:
            a0 = tf.constant(
                    float(alpha0_uniform), dtype=tf.float32, shape=[1, 1]
                 )
            alpha = (1.0 - warmup_eta) * a0 + warmup_eta * alpha

        return alpha   # [N, 1]

    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
                    dict(
                            hidden=self.gate_hidden,
                            num_layers=self.gate_num_layers,
                            alpha_min=self.alpha_min,
                            alpha_max=self.alpha_max,
                            lambda_min=self.lambda_min,
                            delta0=self.delta0,
                            cn=self.cn,
                        )
                  )
        return cfg


# ---------------------------------------------------------------------------
# Regularisation losses (module-level for easy import)
# ---------------------------------------------------------------------------

def alpha_prior_loss(
                        alpha:       tf.Tensor,
                        alpha_prior: tf.Tensor,
                        eps:         float = 1e-5,
                    ) -> tf.Tensor:
    """Logit-space L2 loss between learned and prior α.

    Pushes the gate toward the time-based hazard prior so that, absent any
    task signal, it defaults to commits proportional to elapsed exposure.

    Both tensors are clipped before logit to avoid ±∞.
    """
    def _logit(x):
        x_c = tf.clip_by_value(tf.cast(x, tf.float32), eps, 1.0 - eps)
        return tf.math.log(x_c / (1.0 - x_c))

    diff = _logit(alpha) - _logit(alpha_prior)
    return tf.reduce_mean(diff ** 2)


def alpha_saturation_loss(
                             alpha: tf.Tensor,
                             eps:   float = 1e-5,
                          ) -> tf.Tensor:
    """Entropy-style penalty to discourage α from collapsing to 0 or 1.

    Equal to −E[log α + log(1−α)], which is large near the boundaries and
    small in the interior.  Acts as a soft lower bound on the variance of α.
    """
    a = tf.clip_by_value(tf.cast(alpha, tf.float32), eps, 1.0 - eps)
    return -tf.reduce_mean(tf.math.log(a) + tf.math.log(1.0 - a))
