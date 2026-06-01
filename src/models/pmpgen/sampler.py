"""
Sampler: Iterative denoising for protein generation.

Orchestrates the full generation process:
1. Initialize random coordinates
2. Iteratively denoise with flow matching
3. Apply membrane plane gradient guidance
4. Decode sequences with ProteinMPNN
5. Validate with 3-stage cascade
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class PMPGenSampler(nn.Module):
    """
    Iterative sampling for PMPGen.

    Denoises from noise to realistic protein structures via flow matching.

    Args:
        denoiser: IPA denoiser model
        noise_schedule: noise schedule (sigma(t))
        sequence_decoder: sequence design model
        n_steps (int): number of denoising steps (default: 100)
        use_guidance (bool): whether to use membrane guidance (default: True)
        guidance_scale (float): strength of membrane guidance (default: 1.0)
        temperature (float): sampling temperature (default: 1.0)
    """

    def __init__(
        self,
        denoiser: nn.Module,
        noise_schedule: nn.Module,
        sequence_decoder: nn.Module,
        n_steps: int = 100,
        use_guidance: bool = True,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
    ):
        """Initialize sampler."""
        super().__init__()

        self.denoiser = denoiser
        self.noise_schedule = noise_schedule
        self.sequence_decoder = sequence_decoder

        self.n_steps = n_steps
        self.use_guidance = use_guidance
        self.guidance_scale = guidance_scale
        self.temperature = temperature

    def sample(
        self,
        conditioning: Dict[str, torch.Tensor],
        device: str = 'cuda',
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate proteins via iterative denoising.

        Args:
            conditioning: dict containing:
                - 'scaffold_coords': (B, N, 3) query coordinates
                - 'membrane_normal': (3,) or (B, 3) membrane normal
                - 'anchor_mask': (B, N, 1) anchored residues
                - 'binding_mask': (B, N, 1) binding region
            device: device to run on
            verbose: whether to show progress bar

        Returns:
            results: dict containing:
                - 'coords': (B, N, 3) final coordinates
                - 'sequences': (B, N) amino acid sequences
                - 'logits': (B, N, 21) sequence logits
                - 'trajectory': list of intermediate structures
        """
        B = conditioning['scaffold_coords'].shape[0]
        N = conditioning['scaffold_coords'].shape[1]

        # ─────────────────────────────────────────────────────────────────────
        # Initialize
        # ─────────────────────────────────────────────────────────────────────

        coords_t = torch.randn(B, N, 3, device=device)  # start from noise
        trajectory = [coords_t.clone().detach().cpu()]

        # ─────────────────────────────────────────────────────────────────────
        # Iterative Denoising
        # ─────────────────────────────────────────────────────────────────────

        # Time steps: t goes from 1 to 0 (reverse diffusion)
        time_steps = torch.linspace(1.0, 0.0, self.n_steps, device=device)

        iterator = tqdm(time_steps, desc="Denoising", disable=not verbose)

        for t in iterator:
            t_batch = t.expand(B)

            # ─────────────────────────────────────────────────────────────────
            # Denoise Step
            # ─────────────────────────────────────────────────────────────────

            with torch.no_grad():
                # Predict velocity
                v_pred = self.denoiser(
                    coords_t,
                    t_batch,
                    conditioning=conditioning,
                )  # (B, N, 3)

                # Euler step: x_{t-dt} ≈ x_t + v * dt
                dt = 1.0 / self.n_steps
                coords_t = coords_t + v_pred * dt

            # ─────────────────────────────────────────────────────────────────
            # Membrane Plane Guidance
            # ─────────────────────────────────────────────────────────────────

            if self.use_guidance and 'membrane_normal' in conditioning:
                coords_t = self._apply_guidance(
                    coords_t,
                    conditioning,
                    t,
                )

            # ─────────────────────────────────────────────────────────────────
            # Anchor Constraint
            # ─────────────────────────────────────────────────────────────────

            if 'anchor_mask' in conditioning and 'scaffold_coords' in conditioning:
                anchor_mask = conditioning['anchor_mask']  # (B, N, 1)
                scaffold = conditioning['scaffold_coords']  # (B, N, 3)

                # Keep anchored residues fixed to scaffold
                coords_t = coords_t * (1 - anchor_mask) + scaffold * anchor_mask

            # ─────────────────────────────────────────────────────────────────
            # Save Trajectory
            # ─────────────────────────────────────────────────────────────────

            if int(t.item() * self.n_steps) % (self.n_steps // 10) == 0:
                trajectory.append(coords_t.clone().detach().cpu())

        # ─────────────────────────────────────────────────────────────────────
        # Sequence Design
        # ─────────────────────────────────────────────────────────────────────

        with torch.no_grad():
            sequences, logits = self.sequence_decoder(
                coords_t,
                binding_mask=conditioning.get('binding_mask', None),
                temperature=self.temperature,
            )

        # ─────────────────────────────────────────────────────────────────────
        # Return Results
        # ─────────────────────────────────────────────────────────────────────

        results = {
            'coords': coords_t.detach().cpu(),
            'sequences': sequences.cpu(),
            'logits': logits.detach().cpu(),
            'trajectory': trajectory,
        }

        return results

    def _apply_guidance(
        self,
        coords: torch.Tensor,
        conditioning: Dict[str, torch.Tensor],
        t: float,
    ) -> torch.Tensor:
        """
        Apply membrane plane gradient guidance.

        Steer predictions toward correct membrane depth.

        Args:
            coords: (B, N, 3) current coordinates
            conditioning: conditioning dict
            t: current time step (0-1)

        Returns:
            coords_guided: updated coordinates
        """
        coords = coords.requires_grad_(True)

        # Compute depth (z-coordinate in membrane-aligned frame)
        membrane_normal = conditioning['membrane_normal']
        if membrane_normal.ndim == 1:
            membrane_normal = membrane_normal.unsqueeze(0)  # (1, 3)

        # Target depth: binding residues near membrane, scaffold residues deeper
        if 'binding_mask' in conditioning:
            binding_mask = conditioning['binding_mask']  # (B, N, 1)
            # Target depth: 0 for binding (at membrane), -5 Å for non-binding (deeper)
            target_depth = -5.0 * (1 - binding_mask)
        else:
            # Default: all residues at membrane
            target_depth = torch.zeros_like(coords[..., :1])

        # Compute depth error
        depth = torch.sum(coords * membrane_normal, dim=-1, keepdim=True)
        depth_error = (depth - target_depth).pow(2).mean()

        # Gradient ascent to minimize error
        grad = torch.autograd.grad(depth_error, coords)[0]

        # Update coordinates with guidance
        guidance_step = self.guidance_scale * grad
        coords = coords.detach() - guidance_step * 0.01  # small step size

        return coords

    def sample_batch(
        self,
        batch: Dict[str, torch.Tensor],
        device: str = 'cuda',
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Sample multiple proteins from a batch.

        Args:
            batch: batch dict
            device: device
            verbose: show progress

        Returns:
            results: list of result dicts
        """
        results = self.sample(batch, device=device, verbose=verbose)
        
        # Convert batch results to list of individual results
        B = results['coords'].shape[0]
        results_list = []

        for i in range(B):
            result = {
                'coords': results['coords'][i],
                'sequences': results['sequences'][i],
                'logits': results['logits'][i],
            }
            results_list.append(result)

        return results_list


class GuidanceSteeringMixin:
    """Mixin for guidance steering during sampling."""

    def compute_membrane_guidance(
        self,
        coords: torch.Tensor,
        target_depth: torch.Tensor,
        membrane_normal: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute membrane plane guidance gradient.

        Args:
            coords: (B, N, 3) coordinates
            target_depth: (B, N, 1) target depths
            membrane_normal: (3,) membrane normal
            strength: guidance strength

        Returns:
            guidance: (B, N, 3) guidance vectors
        """
        # Project coordinates onto membrane normal
        depth = torch.sum(coords * membrane_normal, dim=-1, keepdim=True)

        # Compute error
        error = depth - target_depth  # (B, N, 1)

        # Gradient points opposite to error direction
        guidance = -strength * error * membrane_normal.unsqueeze(0).unsqueeze(0)

        return guidance

    def compute_binding_guidance(
        self,
        coords: torch.Tensor,
        binding_mask: torch.Tensor,
        binding_target: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute binding region guidance.

        Keep binding residues clustered and compact.

        Args:
            coords: (B, N, 3) coordinates
            binding_mask: (B, N, 1) binding regions
            binding_target: (B, 3) center of binding region
            strength: guidance strength

        Returns:
            guidance: (B, N, 3) guidance vectors
        """
        # Distances from binding target
        distances = torch.linalg.norm(coords - binding_target.unsqueeze(1), dim=-1, keepdim=True)

        # Target: binding residues close to target, others far away
        target_distance = binding_mask * 5.0 + (1 - binding_mask) * 20.0

        # Error in distance
        distance_error = distances - target_distance

        # Direction toward/away from target
        direction = (coords - binding_target.unsqueeze(1)) / (distances + 1e-8)

        # Guidance
        guidance = -strength * distance_error * direction

        return guidance


class ValidationCascade(nn.Module):
    """
    3-Stage validation cascade for generated proteins.

    Validates quality before accepting generation.
    """

    def __init__(
        self,
        plddt_threshold: float = 70.0,
        binding_recall_threshold: float = 0.8,
    ):
        """Initialize validation cascade."""
        super().__init__()

        self.plddt_threshold = plddt_threshold
        self.binding_recall_threshold = binding_recall_threshold

    def validate(
        self,
        coords: torch.Tensor,
        plddt: torch.Tensor,
        binding_pred: torch.Tensor,
        binding_target: torch.Tensor,
        rosetta_ddg: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Run 3-stage validation.

        Args:
            coords: (N, 3) coordinates
            plddt: (N,) per-residue confidence
            binding_pred: (N,) predicted binding
            binding_target: (N,) target binding
            rosetta_ddg: binding free energy (optional)

        Returns:
            passes: whether all validations pass
            scores: validation scores dict
        """
        scores = {}

        # ─────────────────────────────────────────────────────────────────────
        # Stage 1: Foldability (pLDDT)
        # ─────────────────────────────────────────────────────────────────────

        mean_plddt = plddt.mean().item()
        high_conf_fraction = (plddt > self.plddt_threshold).float().mean().item()

        scores['plddt_mean'] = mean_plddt
        scores['plddt_high_conf'] = high_conf_fraction

        stage1_pass = mean_plddt > self.plddt_threshold

        # ─────────────────────────────────────────────────────────────────────
        # Stage 2: Binding Patch (DynaMo recall)
        # ─────────────────────────────────────────────────────────────────────

        if binding_target.sum() > 0:
            true_positives = (binding_pred > 0.5) & (binding_target > 0.5)
            recall = true_positives.sum().float() / (binding_target > 0.5).sum().float()
            scores['binding_recall'] = recall.item()
            stage2_pass = recall > self.binding_recall_threshold
        else:
            scores['binding_recall'] = 1.0
            stage2_pass = True

        # ─────────────────────────────────────────────────────────────────────
        # Stage 3: Energy & Diversity
        # ─────────────────────────────────────────────────────────────────────

        if rosetta_ddg is not None:
            ddg_favorable = rosetta_ddg < -5.0  # favorable binding
            scores['rosetta_ddg'] = rosetta_ddg.item()
            stage3_pass = ddg_favorable
        else:
            stage3_pass = True

        # ─────────────────────────────────────────────────────────────────────
        # Overall
        # ─────────────────────────────────────────────────────────────────────

        passes = stage1_pass and stage2_pass and stage3_pass
        scores['passes'] = passes
        scores['stage1'] = stage1_pass
        scores['stage2'] = stage2_pass
        scores['stage3'] = stage3_pass

        return passes, scores


if __name__ == "__main__":
    # Test sampler (requires full model setup)
    print("Sampler module loaded successfully")
    print("\nKey classes:")
    print("- PMPGenSampler: iterative denoising")
    print("- GuidanceSteeringMixin: guidance computation")
    print("- ValidationCascade: 3-stage validation")
