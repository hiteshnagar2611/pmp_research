"""
SE(3) flow matching for protein backbone generation.

Implements optimal transport (OT) flow matching between Gaussian noise on SE(3)
and target protein backbone frames.

Key components:
  - OT interpolant: linear interpolation on SE(3) manifold
  - Velocity field computation: closed-form target velocity
  - Frame parameterization: rotation + translation per residue
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple
import sys
sys.path.insert(0, '/home/claude/pmp_research/src')
from models.shared.se3_utils import so3_exp, so3_log, interpolate_frames, frame_velocity


class OTFlowInterpolant(nn.Module):
    """
    Optimal transport flow interpolant on SE(3).

    For each residue, parametrize backbone as (R ∈ SO(3), t ∈ ℝ³):
      x(t) = (1-t) · x_0 + t · x_1

    where x_0 ~ N(0, I) on SE(3) and x_1 is target protein structure.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x0_R: torch.Tensor,      # (B, N, 3, 3) or (N, 3, 3) noisy rotation matrices
        x0_t: torch.Tensor,      # (B, N, 3) or (N, 3) noisy translations
        x1_R: torch.Tensor,      # (B, N, 3, 3) or (N, 3, 3) target rotations
        x1_t: torch.Tensor,      # (B, N, 3) or (N, 3) target translations
        time: torch.Tensor,      # (B,) or scalar, interpolation time ∈ [0, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between x_0 (noise) and x_1 (target) at time t.

        Uses geodesic interpolation on SO(3) and linear interpolation on ℝ³.

        Returns:
            x_R, x_t: interpolated frames at time t
        """
        # Handle batching
        scalar_time = time.dim() == 0
        if scalar_time:
            time = time.unsqueeze(0)

        B = time.shape[0]

        # Ensure rotations are normalized
        x0_R = x0_R / (torch.linalg.norm(x0_R, dim=(-2, -1), keepdim=True) + 1e-8)
        x1_R = x1_R / (torch.linalg.norm(x1_R, dim=(-2, -1), keepdim=True) + 1e-8)

        # Use pre-implemented geodesic interpolation from se3_utils
        x_R, x_t = interpolate_frames(x0_R, x0_t, x1_R, x1_t, time)

        return x_R, x_t

    def compute_velocity(
        self,
        x0_R: torch.Tensor,
        x0_t: torch.Tensor,
        x1_R: torch.Tensor,
        x1_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute target velocity field for flow matching.

        For OT flow: v* = x_1 - x_0 (constant velocity field).

        Returns:
            v_R: (N, 3) rotation velocity (as rotation vector)
            v_t: (N, 3) translation velocity
        """
        v_R, v_t = frame_velocity(x0_R, x0_t, x1_R, x1_t)
        return v_R, v_t


class SE3FlowMatcher(nn.Module):
    """
    SE(3) flow matching trainer.

    Matches predicted velocity field against target velocity field.
    Loss: ||v_θ(x_t, t) - v*(x_0, x_1)||²
    """

    def __init__(self):
        super().__init__()
        self.interpolant = OTFlowInterpolant()

    def forward(
        self,
        x0_R: torch.Tensor,        # (B, N, 3, 3) noisy rotations
        x0_t: torch.Tensor,        # (B, N, 3) noisy translations
        x1_R: torch.Tensor,        # (B, N, 3, 3) target rotations
        x1_t: torch.Tensor,        # (B, N, 3) target translations
        time: torch.Tensor,        # (B,) time ∈ [0, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare flow matching training data.

        Args:
            x0_R, x0_t: noise frames
            x1_R, x1_t: target frames
            time: interpolation times

        Returns:
            x_R, x_t: interpolated frames at time t
            v_R, v_t: target velocities
        """
        # Interpolate
        x_R, x_t = self.interpolant(x0_R, x0_t, x1_R, x1_t, time)

        # Compute target velocity
        v_R, v_t = self.interpolant.compute_velocity(x0_R, x0_t, x1_R, x1_t)

        return x_R, x_t, v_R, v_t

    def velocity_loss(
        self,
        v_pred_R: torch.Tensor,   # (B, N, 3) predicted rotation velocity
        v_pred_t: torch.Tensor,   # (B, N, 3) predicted translation velocity
        v_target_R: torch.Tensor, # (B, N, 3) target rotation velocity
        v_target_t: torch.Tensor, # (B, N, 3) target translation velocity
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        L = ||v_pred - v_target||²
        """
        loss_R = torch.mean((v_pred_R - v_target_R) ** 2)
        loss_t = torch.mean((v_pred_t - v_target_t) ** 2)
        return loss_R + loss_t


class FrameParameterization(nn.Module):
    """
    Convert between different frame representations.

    - (R, t): rotation matrix + translation vector (SE(3))
    - (ω, t): rotation vector (so(3)) + translation vector (ℝ³)
    - (θ, n, t): angle-axis representation
    """

    @staticmethod
    def matrix_to_vector(R: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert (R ∈ SO(3), t ∈ ℝ³) → (ω ∈ so(3), t ∈ ℝ³).

        Uses logarithm map on SO(3).
        """
        omega = so3_log(R)  # (N, 3)
        return omega, t

    @staticmethod
    def vector_to_matrix(omega: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert (ω ∈ so(3), t ∈ ℝ³) → (R ∈ SO(3), t ∈ ℝ³).

        Uses exponential map on SO(3).
        """
        R = so3_exp(omega)  # (N, 3, 3)
        return R, t

    @staticmethod
    def angle_axis_to_matrix(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        """
        Convert angle-axis representation to SO(3).

        ω = angle · axis  (axis is unit vector)
        """
        omega = angle.unsqueeze(-1) * axis  # (N, 3)
        R = so3_exp(omega)
        return R
