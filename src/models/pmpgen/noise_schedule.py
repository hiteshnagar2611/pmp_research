"""
MD-informed anisotropic noise schedule for PMPGen.

Key novelty: noise level σ_i(t) is per-residue and scales with RMSF.

- High RMSF (flexible loops) → more noise → creative sampling
- Low RMSF (rigid helices) → less noise → conservative sampling
- Anchor residues (binding patch) → zero noise (fixed)

This encodes the MD conformational distribution as a prior on generation.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class MDInformedNoiseSchedule(nn.Module):
    """
    Compute per-residue, time-dependent noise schedule.

    σ_i(t) = σ_base(t) · f(RMSF_i)

    where:
      - σ_base(t): base noise schedule (cosine annealing or exponential decay)
      - f(RMSF_i): per-residue scaling factor based on flexibility
    """

    def __init__(
        self,
        schedule_type: str = "cosine",   # "cosine" or "exponential"
        sigma_max: float = 1.0,          # max noise at t=0
        sigma_min: float = 0.01,         # min noise at t=1
        rmsf_scale: float = 2.0,         # multiplier on RMSF for σ scaling
    ):
        """
        Args:
            schedule_type: "cosine" (recommended) or "exponential"
            sigma_max: noise level at t=0
            sigma_min: noise level at t=1
            rmsf_scale: how much RMSF modulates noise (higher = more modulation)
        """
        super().__init__()

        self.schedule_type = schedule_type
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rmsf_scale = rmsf_scale

    def base_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute base noise schedule σ_base(t).

        Args:
            t: (B,) or scalar, time in [0, 1]

        Returns:
            sigma: noise level(s)
        """
        if self.schedule_type == "cosine":
            # Cosine annealing: starts high, goes to low smoothly
            sigma = self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (
                1 + torch.cos(np.pi * t)
            )
        elif self.schedule_type == "exponential":
            # Exponential decay: σ = σ_min + (σ_max - σ_min) · e^(-λt)
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.exp(
                -3.0 * t
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")

        return sigma

    def per_residue_schedule(
        self,
        t: torch.Tensor,     # (B,) or scalar, time in [0, 1]
        rmsf: torch.Tensor,  # (N, 1) or (N,) per-residue RMSF
        anchor_mask: torch.Tensor = None,  # (N, 1) binary mask: 0=generate, 1=anchor
    ) -> torch.Tensor:
        """
        Compute per-residue noise σ_i(t).

        Args:
            t: time in [0, 1]
            rmsf: per-residue flexibility from MD
            anchor_mask: optional binary mask for binding patch (anchored residues get 0 noise)

        Returns:
            sigma: (N,) or (B, N) per-residue noise levels
        """
        # Ensure RMSF is (N, 1)
        if rmsf.dim() == 1:
            rmsf = rmsf.unsqueeze(-1)  # (N, 1)

        # Base schedule: scalar or (B,)
        sigma_base = self.base_schedule(t)

        # Normalize RMSF to [0, 1] approximately
        # RMSF typically ranges 0-5 Å for proteins
        rmsf_norm = torch.tanh(rmsf / 5.0)  # squash to (-1, 1), then map to (0, 1)
        rmsf_norm = (rmsf_norm + 1.0) / 2.0  # map to (0, 1)

        # Per-residue scaling: higher RMSF → more noise
        # Use exponential so flexible regions get much more noise
        f_rmsf = 1.0 + self.rmsf_scale * rmsf_norm  # (N, 1), ranges ~[1, 1+rmsf_scale]

        # Combine base + per-residue scaling
        if sigma_base.dim() == 0:
            # Scalar t → (N,) sigma
            sigma = sigma_base * f_rmsf.squeeze(-1)  # (N,)
        else:
            # (B,) t → (B, N) sigma
            B = sigma_base.shape[0]
            sigma = sigma_base.unsqueeze(-1) * f_rmsf.unsqueeze(0)  # (B, N)

        # ── Apply anchor mask ──────────────────────────────────────────────────
        if anchor_mask is not None:
            if anchor_mask.dim() == 1:
                anchor_mask = anchor_mask.unsqueeze(-1)
            # Anchor residues: sigma = 0 (completely fixed)
            sigma = sigma * (1.0 - anchor_mask)

        return sigma

    def forward(
        self,
        t: torch.Tensor,
        rmsf: torch.Tensor,
        anchor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute noise schedule."""
        return self.per_residue_schedule(t, rmsf, anchor_mask)


class TimeEmbedding(nn.Module):
    """
    Embed continuous time t ∈ [0, 1] into a learnable representation.

    Uses sinusoidal embeddings (like transformer positional encoding).
    """

    def __init__(self, dim: int = 128):
        """
        Args:
            dim: embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed time.

        Args:
            t: (B,) scalar time values in [0, 1]

        Returns:
            t_emb: (B, dim) time embeddings
        """
        # Sinusoidal positional encoding
        device = t.device
        B = t.shape[0]

        # Create frequency bands
        freqs = torch.arange(self.dim // 2, device=device, dtype=torch.float32)
        freqs = 2.0 ** (freqs / (self.dim // 2 - 1)) * np.pi * 2  # log-spaced frequencies

        # Compute sin and cos
        t_expanded = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, dim // 2)
        t_emb = torch.cat([torch.sin(t_expanded), torch.cos(t_expanded)], dim=-1)  # (B, dim)

        return t_emb


class GammaSchedule(nn.Module):
    """
    Noise level schedule as γ(t) = log(σ²(t)).

    Used in some flow matching formulations. Provides alternative parameterization.
    """

    def __init__(self, schedule_type: str = "cosine", sigma_max: float = 1.0, sigma_min: float = 0.01):
        super().__init__()
        self.schedule_type = schedule_type
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute γ(t) = log(σ²(t)).

        Args:
            t: (B,) time in [0, 1]

        Returns:
            gamma: (B,) log-variance
        """
        if self.schedule_type == "cosine":
            sigma = self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + torch.cos(np.pi * t))
        else:
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.exp(-3.0 * t)

        gamma = torch.log(sigma ** 2)
        return gamma

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute dγ/dt.

        Useful for computing velocity fields in flow matching.
        """
        if self.schedule_type == "cosine":
            dσ_dt = -0.5 * (self.sigma_max - self.sigma_min) * np.pi * torch.sin(np.pi * t)
        else:
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.exp(-3.0 * t)
            dσ_dt = -(self.sigma_max - self.sigma_min) * 3.0 * torch.exp(-3.0 * t)

        sigma = self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + torch.cos(np.pi * t))
        dgamma_dt = 2.0 * (dσ_dt / sigma)

        return dgamma_dt
