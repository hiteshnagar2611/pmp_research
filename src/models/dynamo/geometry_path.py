"""
Membrane geometry path for DynaMo Phase 1.

Encodes OPM membrane geometry (normal, depth, tilt, amphipathic score)
into per-residue representations that complement the dynamics path.

Usage:
    geom_path = MembraneGeometryPath(hidden_dim=256)
    H_geom = geom_path(structure, opm_normal, opm_depth)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class MembraneGeometryPath(nn.Module):
    """Encode membrane geometry features into residue representations."""

    def __init__(self, hidden_dim: int = 256, n_residues_max: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Geometry feature embedding ─────────────────────────────────────────
        # Input: 4 scalars (depth, tilt, mem_sasa, amph_score) + 1 vector (n̂)
        self.depth_emb = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim // 4),
        )
        self.tilt_emb = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim // 4),
        )
        self.mem_sasa_emb = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim // 4),
        )
        self.amph_emb = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim // 4),
        )

        # ── Vector feature (membrane normal) → scalar via norm ────────────────
        self.normal_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, hidden_dim // 4),
        )

        # ── Fusion and output ──────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        depth: torch.Tensor,           # (N,) or (N, 1) residue depth relative to membrane
        tilt: torch.Tensor = None,     # (N,) or (N, 1) backbone tilt angle
        mem_sasa: torch.Tensor = None, # (N,) or (N, 1) membrane-facing SASA
        amph_score: torch.Tensor = None,  # (N,) or (N, 1) amphipathic helix score
        normal: torch.Tensor = None,   # (3,) membrane normal unit vector
    ) -> torch.Tensor:
        """
        Encode membrane geometry into per-residue representations.

        Returns:
            H_geom (N, hidden_dim): geometry-encoded representations
        """
        N = depth.shape[0]
        device = depth.device

        # Ensure all inputs are (N, 1) shaped
        if depth.dim() == 1:
            depth = depth.unsqueeze(-1)
        if tilt is None:
            tilt = torch.zeros(N, 1, device=device)
        elif tilt.dim() == 1:
            tilt = tilt.unsqueeze(-1)
        if mem_sasa is None:
            mem_sasa = torch.zeros(N, 1, device=device)
        elif mem_sasa.dim() == 1:
            mem_sasa = mem_sasa.unsqueeze(-1)
        if amph_score is None:
            amph_score = torch.zeros(N, 1, device=device)
        elif amph_score.dim() == 1:
            amph_score = amph_score.unsqueeze(-1)

        # ── Normalise scalar features to ~[0, 1] ──────────────────────────────
        depth = torch.tanh(depth / 10.0)     # normalise depth
        tilt = torch.tanh(tilt)
        mem_sasa = torch.sigmoid(mem_sasa)   # normalise to [0, 1]
        amph_score = torch.sigmoid(amph_score)

        # ── Embed scalar features ──────────────────────────────────────────────
        depth_repr = self.depth_emb(depth)         # (N, hidden_dim // 4)
        tilt_repr = self.tilt_emb(tilt)            # (N, hidden_dim // 4)
        sasa_repr = self.mem_sasa_emb(mem_sasa)    # (N, hidden_dim // 4)
        amph_repr = self.amph_emb(amph_score)      # (N, hidden_dim // 4)

        # ── Concatenate scalar embeddings ──────────────────────────────────────
        scalar_repr = torch.cat([depth_repr, tilt_repr, sasa_repr, amph_repr], dim=-1)

        # ── Embed membrane normal vector ───────────────────────────────────────
        if normal is None:
            normal = torch.tensor([0.0, 0.0, 1.0], device=device)
        if normal.dim() == 1:
            normal = normal.unsqueeze(0).expand(N, -1)  # broadcast to (N, 3)
        normal_repr = self.normal_mlp(normal)      # (N, hidden_dim // 4)

        # ── Concatenate all embeddings ─────────────────────────────────────────
        all_repr = torch.cat([scalar_repr, normal_repr], dim=-1)  # (N, hidden_dim + hidden_dim//4)

        # ── Final fusion ───────────────────────────────────────────────────────
        H_geom = self.fusion(all_repr)  # (N, hidden_dim)

        return H_geom


class MembraneDepthEstimator(nn.Module):
    """Lightweight module to estimate membrane depth from OPM prior."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        CA_coords: torch.Tensor,   # (N, 3) Cα coordinates
        normal: torch.Tensor,      # (3,) membrane normal
    ) -> torch.Tensor:
        """
        Compute signed depth (residue distance along normal direction).

        Returns:
            depth (N,): signed depth relative to protein center
        """
        r_com = CA_coords.mean(dim=0)
        depth = (CA_coords - r_com) @ normal  # dot product
        return depth
