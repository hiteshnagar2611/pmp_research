"""
SE(3) / SO(3) utilities shared by DynaMo and PMPGen.

Covers:
  - unit vector helpers
  - local backbone frame construction
  - virtual Cβ computation
  - RBF distance encoding
  - SO(3) exponential / logarithm maps (for flow matching)
  - IGSO3 sampling (isotropic Gaussian on SO(3))
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Basic vector helpers
# ─────────────────────────────────────────────────────────────────────────────

def unit(v: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalise last dimension to unit length."""
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def safe_cross(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Cross product with fallback for near-parallel vectors."""
    c = torch.cross(a, b, dim=-1)
    norm = c.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(c)
    fallback[..., 0] = 1.0          # arbitrary non-zero fallback axis
    return torch.where(norm > eps, c / (norm + eps), fallback)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone geometry
# ─────────────────────────────────────────────────────────────────────────────

def build_backbone_frames(
    N_pos: Tensor,   # (B, 3) or (3,)
    CA_pos: Tensor,
    C_pos: Tensor,
) -> Tensor:
    """
    Construct local orthonormal frame per residue from backbone atoms.
    Returns rotation matrix R ∈ ℝ^(...×3×3) where columns are (u1, u2, u3).

    u1 = unit(CA - N)       backbone N→Cα direction
    u2 = unit(C  - CA)      backbone Cα→C direction
    u3 = u1 × u2            normal to local backbone plane
    """
    u1 = unit(CA_pos - N_pos)
    u2 = unit(C_pos  - CA_pos)
    u3 = safe_cross(u1, u2)
    # re-orthogonalise u2 to ensure true ONB
    u2 = safe_cross(u3, u1)
    R  = torch.stack([u1, u2, u3], dim=-1)   # (..., 3, 3)
    return R


def virtual_cbeta(
    N_pos: Tensor,
    CA_pos: Tensor,
    C_pos: Tensor,
) -> Tensor:
    """
    Compute virtual Cβ position from backbone atoms (works for Gly).
    Formula from Jing et al. GVP paper (ICLR 2021).
    """
    b = CA_pos - N_pos
    c = C_pos  - CA_pos
    a = torch.cross(b, c, dim=-1)
    CB = (-0.58273431 * a
          + 0.56802827 * b
          - 0.54067466 * c
          + CA_pos)
    return CB


def dihedral_angle(
    p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor
) -> Tensor:
    """
    Compute dihedral angle (radians) defined by four points.
    Works on batched input (..., 3).
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = safe_cross(b1, b2)
    n2 = safe_cross(b2, b3)

    cos_angle = (n1 * n2).sum(dim=-1)
    sin_angle = (safe_cross(n1, n2) * unit(b2)).sum(dim=-1)
    return torch.atan2(sin_angle, cos_angle)


# ─────────────────────────────────────────────────────────────────────────────
# RBF distance encoding
# ─────────────────────────────────────────────────────────────────────────────

def rbf_encode(
    dist: Tensor,
    d_min: float = 2.0,
    d_max: float = 22.0,
    n_bins: int = 16,
) -> Tensor:
    """
    Radial basis function distance encoding.
    dist : (...,)  Euclidean distance in Ångströms
    Returns (..., n_bins) Gaussian activations.
    """
    centers = torch.linspace(d_min, d_max, n_bins, device=dist.device, dtype=dist.dtype)
    sigma   = (d_max - d_min) / n_bins
    return torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (2 * sigma ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# SO(3) exponential and logarithm maps
# ─────────────────────────────────────────────────────────────────────────────

def so3_exp(omega: Tensor, eps: float = 1e-7) -> Tensor:
    """
    SO(3) exponential map.
    omega : (..., 3)  rotation vector (axis × angle)
    Returns rotation matrices (..., 3, 3).
    Uses Rodrigues' formula.
    """
    theta  = omega.norm(dim=-1, keepdim=True).clamp(min=eps)   # (..., 1)
    k      = omega / theta                                       # (..., 3) unit axis
    theta  = theta.squeeze(-1)                                   # (...,)

    # Build skew-symmetric matrix K
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([ zero, -kz,  ky], dim=-1),
        torch.stack([  kz, zero, -kx], dim=-1),
        torch.stack([ -ky,  kx, zero], dim=-1),
    ], dim=-2)   # (..., 3, 3)

    I   = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    c   = torch.cos(theta)[..., None, None]
    s   = torch.sin(theta)[..., None, None]
    R   = c * I + s * K + (1 - c) * torch.einsum('...i,...j->...ij', k, k)
    return R


def so3_log(R: Tensor, eps: float = 1e-7) -> Tensor:
    """
    SO(3) logarithm map.
    R : (..., 3, 3)  rotation matrices
    Returns rotation vectors (..., 3).
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = torch.acos(((trace - 1) / 2).clamp(-1 + eps, 1 - eps))

    denom = (2 * torch.sin(theta) + eps)[..., None, None]
    log_R = (R - R.transpose(-1, -2)) / denom

    omega = torch.stack([
        log_R[..., 2, 1],
        log_R[..., 0, 2],
        log_R[..., 1, 0],
    ], dim=-1) * theta[..., None]
    return omega


def so3_geodesic_distance(R1: Tensor, R2: Tensor) -> Tensor:
    """Geodesic distance between two rotation matrices."""
    dR    = R1.transpose(-1, -2) @ R2
    omega = so3_log(dR)
    return omega.norm(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# IGSO3 — isotropic Gaussian on SO(3) for forward noising
# ─────────────────────────────────────────────────────────────────────────────

def igso3_sample(sigma: Tensor, n_samples: int = 1) -> Tensor:
    """
    Sample from isotropic Gaussian on SO(3) with concentration σ.
    sigma  : (...,) scalar noise level
    Returns rotation matrices (..., n_samples, 3, 3).

    Implementation: sample rotation vector from N(0, σ²I₃), apply exp map.
    """
    shape  = sigma.shape + (n_samples, 3)
    omega  = torch.randn(shape, device=sigma.device, dtype=sigma.dtype)
    omega  = omega * sigma[..., None, None]          # scale by σ
    return so3_exp(omega.reshape(-1, 3)).reshape(shape[:-1] + (3, 3))


# ─────────────────────────────────────────────────────────────────────────────
# OT-Flow interpolant on SE(3) frames
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_frames(
    x0_R: Tensor, x0_t: Tensor,   # noisy (source): rotation, translation
    x1_R: Tensor, x1_t: Tensor,   # clean (target): rotation, translation
    time: Tensor,                  # (B,) in [0, 1]
) -> tuple[Tensor, Tensor]:
    """
    Linear interpolation of backbone frames for OT-flow matching.
    Translations: linear interpolation in ℝ³.
    Rotations: geodesic SLERP on SO(3).

    Returns interpolated (R_t, t_t) at time t.
    """
    B = time.shape[0]
    t = time.view(B, *([1] * (x0_t.dim() - 1)))

    # Translation: straight line
    trans_t = (1 - t) * x0_t + t * x1_t

    # Rotation: geodesic interpolation via log/exp
    dR      = x0_R.transpose(-1, -2) @ x1_R      # relative rotation
    omega   = so3_log(dR)                          # rotation vector
    omega_t = t[..., None] * omega.view(B, -1, 3) # scale by time
    dR_t    = so3_exp(omega_t.reshape(-1, 3)).reshape(x0_R.shape)
    rot_t   = x0_R @ dR_t

    return rot_t, trans_t


def frame_velocity(
    x0_R: Tensor, x0_t: Tensor,
    x1_R: Tensor, x1_t: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Target velocity for OT-flow matching: v* = x1 - x0.
    Returns (omega_velocity, trans_velocity) — the constant field.
    """
    trans_vel = x1_t - x0_t

    dR        = x0_R.transpose(-1, -2) @ x1_R
    rot_vel   = so3_log(dR)

    return rot_vel, trans_vel
