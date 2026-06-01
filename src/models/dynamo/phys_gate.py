"""
Physicochemical gate for DynaMo Phase 1.

Injects biophysical priors (hydrophobicity, charge, exposure) as learned gates
on the fused representation. Ensures the model respects domain knowledge:
high hydrophobicity + high membrane depth → high binding likelihood.

Usage:
    gate = PhysiochemicalGate(d_model=256)
    H_gated = gate(H_fused, KD, charge, SASA)
"""

from __future__ import annotations
import torch
import torch.nn as nn


class PhysiochemicalGate(nn.Module):
    """
    Apply physicochemical biophysical gating.

    For each residue:
      gate = σ(W_phys · [KD, charge, SASA] + b)
      H_out = gate ⊙ H_in
    
    This acts as a learned prior that upweights hydrophobic residues
    at the membrane interface.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # ── Gate computation: 4 physicochemical scalars → d_model gate values ──
        # Input: KD_score, net_charge, SASA, amph_score (4 dims)
        self.gate_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, d_model),
            nn.Sigmoid(),  # gate in (0, 1)
        )

    def forward(
        self,
        H: torch.Tensor,        # (N, d_model) fused representation
        kd: torch.Tensor,       # (N, 1) or (N,) Kyte-Doolittle hydrophobicity
        charge: torch.Tensor,   # (N, 1) or (N,) net charge at pH 7
        sasa: torch.Tensor,     # (N, 1) or (N,) relative SASA
        amph: torch.Tensor = None,  # (N, 1) or (N,) amphipathic score
    ) -> torch.Tensor:
        """
        Apply physicochemical gate.

        Returns:
            H_gated (N, d_model): element-wise gated representation
        """
        N = H.shape[0]

        # Ensure all inputs are (N, 1)
        if kd.dim() == 1:
            kd = kd.unsqueeze(-1)
        if charge.dim() == 1:
            charge = charge.unsqueeze(-1)
        if sasa.dim() == 1:
            sasa = sasa.unsqueeze(-1)
        if amph is None:
            amph = torch.zeros(N, 1, device=H.device)
        elif amph.dim() == 1:
            amph = amph.unsqueeze(-1)

        # Concatenate physicochemical features
        phys_feats = torch.cat([kd, charge, sasa, amph], dim=-1)  # (N, 4)

        # Compute gate: (N, d_model), values in (0, 1)
        gate = self.gate_mlp(phys_feats)  # (N, d_model)

        # Element-wise modulation
        H_gated = gate * H  # (N, d_model)

        return H_gated


class BoundingBoxGate(nn.Module):
    """
    Stricter gate: hard constraint on membrane-binding residues.
    Residues outside the membrane bounding box get zero activation.
    """

    def __init__(self, d_model: int = 256, membrane_thickness: float = 30.0):
        super().__init__()
        self.membrane_thickness = membrane_thickness
        self.d_model = d_model

    def forward(
        self,
        H: torch.Tensor,      # (N, d_model)
        depth: torch.Tensor,  # (N,) or (N, 1) signed depth from membrane center
    ) -> torch.Tensor:
        """
        Apply hard bounding box constraint.
        Only residues within ±membrane_thickness/2 of the plane get full activation.

        Returns:
            H_gated (N, d_model): zero activation outside bounding box
        """
        if depth.dim() == 1:
            depth = depth.unsqueeze(-1)

        # Soft bounding box: sigmoid with sharp transition
        threshold = self.membrane_thickness / 2.0
        box_mask = torch.sigmoid(10.0 * (threshold - torch.abs(depth)))  # sharp sigmoid

        H_gated = box_mask * H

        return H_gated


class CombinedPhysiochemicalGate(nn.Module):
    """Combined physicochemical + bounding box gating."""

    def __init__(self, d_model: int = 256, membrane_thickness: float = 30.0):
        super().__init__()
        self.phys_gate = PhysiochemicalGate(d_model)
        self.box_gate = BoundingBoxGate(d_model, membrane_thickness)

    def forward(
        self,
        H: torch.Tensor,
        kd: torch.Tensor,
        charge: torch.Tensor,
        sasa: torch.Tensor,
        depth: torch.Tensor,
        amph: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply both gates in sequence."""
        H_phys = self.phys_gate(H, kd, charge, sasa, amph)
        H_gated = self.box_gate(H_phys, depth)
        return H_gated
