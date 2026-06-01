"""
Membrane plane gradient guidance for PMPGen Phase 2.

Novel contribution: steer diffusion sampling toward correct membrane geometry
without retraining, using gradient of a geometric energy function.

At each denoising step:
  1. Predict depth of each residue from current backbone
  2. Compute energy: E_mem = ||depth_pred - depth_target||²
  3. Compute gradient: ∇E wrt coordinates
  4. Update: x ← x - λ∇E (classifier-free guidance)

No separate classifier needed, just geometric energy.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthPredictor(nn.Module):
    """
    Predict residue depth from backbone coordinates.

    Depth is computed relative to membrane plane:
    depth_i = (r_i - r_COM) · n̂

    where n̂ is membrane normal (from OPM).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        coords: torch.Tensor,  # (N, 3) Cα coordinates
        normal: torch.Tensor,  # (3,) membrane normal unit vector
    ) -> torch.Tensor:
        """
        Predict depth for each residue.

        Args:
            coords: (N, 3) Cα coordinates
            normal: (3,) membrane normal (should be unit vector)

        Returns:
            depth: (N, 1) signed depth relative to protein center
        """
        # Compute center of mass
        r_com = coords.mean(dim=0, keepdim=True)  # (1, 3)

        # Compute depth: (r_i - COM) · n̂
        displacement = coords - r_com  # (N, 3)
        depth = torch.matmul(displacement, normal.unsqueeze(-1))  # (N, 1)

        return depth


class MembraneGuidance(nn.Module):
    """
    Apply membrane plane gradient guidance during diffusion sampling.

    Steers sampling toward target membrane depth profile.
    """

    def __init__(
        self,
        scale: float = 0.3,           # guidance strength
        use_soft_constraint: bool = True,  # soft vs hard constraint
    ):
        """
        Args:
            scale: gradient guidance strength (λ in update rule)
            use_soft_constraint: if True, use smooth energy; if False, hard threshold
        """
        super().__init__()

        self.scale = scale
        self.use_soft_constraint = use_soft_constraint
        self.depth_predictor = DepthPredictor()

    def energy(
        self,
        depth_pred: torch.Tensor,   # (N, 1) predicted depth
        depth_target: torch.Tensor, # (N, 1) target depth
        anchor_mask: torch.Tensor = None,  # (N, 1) which residues are anchored
    ) -> torch.Tensor:
        """
        Compute membrane geometric energy.

        E_mem = ||depth_pred - depth_target||²

        Anchored residues (binding patch) have no energy (fully constrained).

        Args:
            depth_pred: predicted depth
            depth_target: target depth from OPM
            anchor_mask: optional mask (1 = anchor, 0 = free)

        Returns:
            energy: scalar loss value
        """
        error = depth_pred - depth_target  # (N, 1)

        if self.use_soft_constraint:
            # Soft energy: quadratic
            energy = torch.mean(error ** 2)
        else:
            # Hard energy: threshold-based
            threshold = 5.0  # Angstroms
            penalty = torch.clamp(torch.abs(error) - threshold, min=0.0)
            energy = torch.mean(penalty ** 2)

        # Apply anchor mask: anchored residues contribute zero energy
        if anchor_mask is not None:
            energy = energy * (1.0 - anchor_mask).mean()

        return energy

    def gradient_guidance(
        self,
        coords: torch.Tensor,       # (N, 3) residue coordinates (requires grad)
        depth_target: torch.Tensor, # (N, 1) target depth profile
        normal: torch.Tensor,       # (3,) membrane normal
        anchor_mask: torch.Tensor = None,  # (N, 1) anchor constraint
    ) -> torch.Tensor:
        """
        Compute gradient of energy wrt coordinates.

        ∇E = dE/d(coords)

        Returns:
            grad: (N, 3) gradient to subtract from coordinates
        """
        # Enable gradient computation if needed
        coords_req_grad = coords.requires_grad
        if not coords_req_grad:
            coords = coords.detach().requires_grad_(True)

        # Predict depth (differentiable)
        depth_pred = self.depth_predictor(coords, normal)

        # Compute energy
        E = self.energy(depth_pred, depth_target, anchor_mask)

        # Backward to get gradient
        E.backward()

        grad = coords.grad  # (N, 3)

        if not coords_req_grad:
            coords = coords.detach()

        return grad

    def forward(
        self,
        coords: torch.Tensor,       # (N, 3) current coordinates
        depth_target: torch.Tensor, # (N, 1) target depth
        normal: torch.Tensor,       # (3,) membrane normal
        anchor_mask: torch.Tensor = None,  # (N, 1) anchor mask
        return_energy: bool = False,  # if True, also return energy value
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply guidance step.

        coords_new = coords - λ·∇E

        Args:
            coords: current coordinates
            depth_target: target depth profile
            normal: membrane normal
            anchor_mask: optional anchor mask
            return_energy: if True, return (coords_updated, energy)

        Returns:
            coords_updated: (N, 3) updated coordinates
            energy (optional): scalar energy value
        """
        # Compute gradient
        grad = self.gradient_guidance(coords, depth_target, normal, anchor_mask)

        # Apply guidance: move in negative gradient direction
        coords_updated = coords - self.scale * grad

        # Optionally compute and return energy
        if return_energy:
            with torch.no_grad():
                depth_pred = self.depth_predictor(coords, normal)
                energy = self.energy(depth_pred, depth_target, anchor_mask)
            return coords_updated, energy
        else:
            return coords_updated


class HydrophobicPatchGuidance(nn.Module):
    """
    Classifier-free guidance using DynaMo binding patch predictions.

    Steers diffusion to place hydrophobic residues at predicted binding regions.
    """

    def __init__(self, scale: float = 0.1):
        """
        Args:
            scale: guidance strength
        """
        super().__init__()
        self.scale = scale

    def forward(
        self,
        coords: torch.Tensor,       # (N, 3) coordinates
        binding_prob: torch.Tensor, # (N, 1) DynaMo binding probability
        depth_target: torch.Tensor, # (N, 1) target depth
        normal: torch.Tensor,       # (3,) membrane normal
        kd_score: torch.Tensor = None,  # (N, 1) hydrophobicity (optional)
    ) -> torch.Tensor:
        """
        Apply hydrophobic patch guidance.

        Residues with high binding probability should be placed at correct depths.
        Additionally, if hydrophobicity scores provided, high KD + high binding_prob
        should get extra guidance toward membrane.

        Returns:
            guidance_vec: (N, 3) gradient vector to subtract
        """
        N = coords.shape[0]

        # Compute depth
        r_com = coords.mean(dim=0, keepdim=True)
        depth_pred = torch.matmul(coords - r_com, normal.unsqueeze(-1))  # (N, 1)

        # Energy: penalise binding residues that are NOT at correct depth
        binding_error = binding_prob * (depth_pred - depth_target) ** 2  # (N, 1)

        # If hydrophobicity provided, boost penalty for hydrophobic binding residues
        if kd_score is not None:
            hydrophobic_factor = torch.sigmoid(5.0 * kd_score)  # (N, 1) in (0, 1)
            binding_error = binding_error * (1.0 + hydrophobic_factor)

        energy = binding_error.mean()

        # Gradient
        coords_grad = coords.clone().detach().requires_grad_(True)
        depth_grad = torch.matmul(coords_grad - r_com, normal.unsqueeze(-1))
        binding_error_grad = binding_prob * (depth_grad - depth_target) ** 2
        energy_grad = binding_error_grad.mean()
        energy_grad.backward()

        grad = coords_grad.grad  # (N, 3)

        return self.scale * grad


class CombinedGuidance(nn.Module):
    """
    Combine membrane plane guidance + hydrophobic patch guidance.
    """

    def __init__(self, mem_scale: float = 0.3, patch_scale: float = 0.1):
        super().__init__()
        self.mem_guidance = MembraneGuidance(scale=mem_scale)
        self.patch_guidance = HydrophobicPatchGuidance(scale=patch_scale)

    def forward(
        self,
        coords: torch.Tensor,
        depth_target: torch.Tensor,
        normal: torch.Tensor,
        binding_prob: torch.Tensor,
        anchor_mask: torch.Tensor = None,
        kd_score: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply combined guidance."""
        # Membrane plane guidance
        coords_mem = self.mem_guidance(coords, depth_target, normal, anchor_mask)

        # Hydrophobic patch guidance
        patch_grad = self.patch_guidance(coords_mem, binding_prob, depth_target, normal, kd_score)
        coords_final = coords_mem - patch_grad

        return coords_final
