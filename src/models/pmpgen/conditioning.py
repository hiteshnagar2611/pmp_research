"""
Conditioning encoder for PMPGen Phase 2.

Encodes:
  - Query PMP scaffold (structural information)
  - DynaMo binding patch prediction (which residues bind membrane)
  - OPM membrane geometry (normal, depth, tilt)
  - MD ensemble flexibility (RMSF)

All conditioning streams fused via cross-attention into a unified context vector.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditioningEncoder(nn.Module):
    """
    Encode separate conditioning streams: scaffold, binding patch, geometry.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        max_seq_len: int = 1000,
    ):
        """
        Args:
            hidden_dim: output dimension per residue
            max_seq_len: maximum protein length for embeddings
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # ── Scaffold encoder (structure) ───────────────────────────────────────
        # Input: (N, 256) from GVP encoder on reference structure
        self.scaffold_proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ── Binding patch encoder (from DynaMo predictions) ──────────────────
        # Input: (N, 1) binding probability [0, 1]
        self.binding_emb = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ── Membrane geometry encoder (OPM data) ──────────────────────────────
        # Input: 4 scalars (depth, tilt, mem_sasa, amph) + 1 vector (normal)
        self.geo_scalar_emb = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim // 2),
        )

        self.geo_vector_emb = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim // 2),
        )

        # ── RMSF flexibility encoder ───────────────────────────────────────────
        # Input: (N, 1) RMSF value
        self.rmsf_emb = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim // 2),
        )

    def forward(
        self,
        scaffold_h: torch.Tensor,      # (N, 256) GVP encoding of query protein
        binding_prob: torch.Tensor,    # (N, 1) DynaMo binding prediction
        depth: torch.Tensor = None,    # (N, 1) membrane depth
        tilt: torch.Tensor = None,     # (N, 1) backbone tilt angle
        mem_sasa: torch.Tensor = None, # (N, 1) membrane-facing SASA
        amph: torch.Tensor = None,     # (N, 1) amphipathic score
        normal: torch.Tensor = None,   # (3,) membrane normal
        rmsf: torch.Tensor = None,     # (N, 1) MD flexibility
    ) -> torch.Tensor:
        """
        Encode all conditioning streams.

        Returns:
            c: (N, hidden_dim) per-residue conditioning context
        """
        N = scaffold_h.shape[0]
        device = scaffold_h.device

        # ── Scaffold path ──────────────────────────────────────────────────────
        c_scaffold = self.scaffold_proj(scaffold_h)  # (N, hidden_dim)

        # ── Binding patch path ─────────────────────────────────────────────────
        c_binding = self.binding_emb(binding_prob)  # (N, hidden_dim)

        # ── Membrane geometry path ─────────────────────────────────────────────
        c_geo_parts = []

        if depth is not None and tilt is not None and mem_sasa is not None and amph is not None:
            if depth.dim() == 1:
                depth = depth.unsqueeze(-1)
            if tilt.dim() == 1:
                tilt = tilt.unsqueeze(-1)
            if mem_sasa.dim() == 1:
                mem_sasa = mem_sasa.unsqueeze(-1)
            if amph.dim() == 1:
                amph = amph.unsqueeze(-1)

            geo_scalars = torch.cat([depth, tilt, mem_sasa, amph], dim=-1)  # (N, 4)
            c_geo_scalar = self.geo_scalar_emb(geo_scalars)  # (N, hidden_dim // 2)
            c_geo_parts.append(c_geo_scalar)

        if normal is not None:
            if normal.dim() == 1:
                normal = normal.unsqueeze(0).expand(N, -1)  # broadcast to (N, 3)
            c_geo_vector = self.geo_vector_emb(normal)  # (N, hidden_dim // 2)
            c_geo_parts.append(c_geo_vector)

        if c_geo_parts:
            c_geo = torch.cat(c_geo_parts, dim=-1)  # (N, hidden_dim or hidden_dim // 2)
        else:
            c_geo = torch.zeros(N, self.hidden_dim // 2, device=device)

        # ── RMSF flexibility path ──────────────────────────────────────────────
        if rmsf is not None:
            if rmsf.dim() == 1:
                rmsf = rmsf.unsqueeze(-1)
            c_rmsf = self.rmsf_emb(rmsf)  # (N, hidden_dim // 2)
        else:
            c_rmsf = torch.zeros(N, self.hidden_dim // 2, device=device)

        # ── Combine all streams ────────────────────────────────────────────────
        # Concatenate and project down to hidden_dim
        c_all = torch.cat([c_scaffold, c_binding, c_geo, c_rmsf], dim=-1)

        # Project to hidden_dim (may be overcomplete)
        proj = nn.Linear(c_all.shape[-1], self.hidden_dim, device=device)
        c = proj(c_all)

        return c


class ConditioningFusion(nn.Module):
    """
    Fuse scaffold structure with geometry conditioning via cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: dimension of representations
            n_heads: number of attention heads
            dropout: dropout rate
        """
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads

        # ── Cross-attention: scaffold queries, geometry key/value ────────────────
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ── FFN ────────────────────────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout_attn = nn.Dropout(dropout)

    def forward(
        self,
        c_scaffold: torch.Tensor,  # (N, hidden_dim) scaffold conditioning
        c_geometry: torch.Tensor,  # (N, hidden_dim) geometry conditioning
    ) -> torch.Tensor:
        """
        Fuse scaffold and geometry via cross-attention.

        Returns:
            c: (N, hidden_dim) fused conditioning
        """
        N = c_scaffold.shape[0]

        # ── Cross-attention ────────────────────────────────────────────────────
        c_scaffold_norm = self.norm1(c_scaffold)
        c_geometry_norm = self.norm1(c_geometry)

        Q = self.W_q(c_scaffold_norm)  # (N, hidden_dim)
        K = self.W_k(c_geometry_norm)  # (N, hidden_dim)
        V = self.W_v(c_geometry_norm)  # (N, hidden_dim)

        # Reshape for multi-head attention
        Q = Q.reshape(N, self.n_heads, self.d_head).transpose(0, 1)  # (h, N, d_head)
        K = K.reshape(N, self.n_heads, self.d_head).transpose(0, 1)
        V = V.reshape(N, self.n_heads, self.d_head).transpose(0, 1)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)

        # Attend to values
        attn_out = torch.matmul(attn, V)  # (h, N, d_head)
        attn_out = attn_out.transpose(0, 1).reshape(N, self.hidden_dim)  # (N, hidden_dim)
        attn_out = self.W_o(attn_out)

        # ── Residual + FFN ─────────────────────────────────────────────────────
        c = self.norm2(c_scaffold + attn_out)
        c = c + self.ffn(c)

        return c


class BindingPatchMask(nn.Module):
    """
    Convert DynaMo binding predictions to a learnable binary mask for anchoring.

    Binding patch residues (predicted by DynaMo) are fixed during generation,
    while other residues can be modified.
    """

    def __init__(self, threshold: float = 0.5, hard_mask: bool = False):
        """
        Args:
            threshold: probability threshold for binary classification
            hard_mask: if True, use hard thresholding (0/1), else soft (sigmoid)
        """
        super().__init__()
        self.threshold = threshold
        self.hard_mask = hard_mask

    def forward(self, binding_prob: torch.Tensor) -> torch.Tensor:
        """
        Convert binding probabilities to mask.

        Args:
            binding_prob: (N, 1) or (N,) binding probability from DynaMo

        Returns:
            mask: (N, 1) binary mask (0 = generate, 1 = anchor)
        """
        if binding_prob.dim() == 1:
            binding_prob = binding_prob.unsqueeze(-1)

        if self.hard_mask:
            mask = (binding_prob > self.threshold).float()
        else:
            # Soft mask: higher prob → higher mask value → less change
            mask = torch.sigmoid(10.0 * (binding_prob - self.threshold))

        return mask
