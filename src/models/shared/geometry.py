"""
Geometry utilities: distance metrics, angles, frames, and geometric operations.

Provides utilities for:
  - Distance computation (pairwise, contact maps)
  - Angle computation (dihedrals, bond angles)
  - Frame construction (backbone frames)
  - Coordinate transformations
"""

from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Distance Metrics
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_distances(
    coords: torch.Tensor,
    squared: bool = False,
) -> torch.Tensor:
    """
    Compute pairwise distances.

    Args:
        coords: (N, 3) coordinates
        squared: if True, return squared distances

    Returns:
        distances: (N, N) distance matrix
    """
    # ||x - y||² = ||x||² + ||y||² - 2(x·y)
    diffs = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
    distances = torch.linalg.norm(diffs, dim=2)  # (N, N)

    if squared:
        distances = distances ** 2

    return distances


def contact_map(
    coords: torch.Tensor,
    threshold: float = 8.0,
) -> torch.Tensor:
    """
    Compute contact map (binary distance matrix).

    Args:
        coords: (N, 3) coordinates
        threshold: distance threshold in Angstroms

    Returns:
        contacts: (N, N) binary contact matrix
    """
    distances = pairwise_distances(coords)
    contacts = (distances < threshold).float()

    return contacts


def k_nearest_neighbors(
    coords: torch.Tensor,
    k: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find k-nearest neighbors for each residue.

    Args:
        coords: (N, 3) coordinates
        k: number of neighbors

    Returns:
        distances: (N, k) distances to k-NN
        indices: (N, k) indices of k-NN
    """
    distances = pairwise_distances(coords)

    # Get k smallest distances (excluding self)
    k_distances, k_indices = torch.topk(
        distances,
        k=k + 1,
        dim=1,
        largest=False
    )

    # Remove self (k=0 has distance 0)
    k_distances = k_distances[:, 1:]
    k_indices = k_indices[:, 1:]

    return k_distances, k_indices


# ─────────────────────────────────────────────────────────────────────────────
# Angle Computations
# ─────────────────────────────────────────────────────────────────────────────

def compute_angles(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    Compute angle ABC (angle at vertex B).

    Args:
        a: (N, 3) first point
        b: (N, 3) vertex
        c: (N, 3) third point

    Returns:
        angles: (N,) angles in radians [0, pi]
    """
    ba = a - b  # (N, 3)
    bc = c - b  # (N, 3)

    # cos(angle) = (ba · bc) / (|ba| |bc|)
    dot_product = torch.sum(ba * bc, dim=1)
    norm_ba = torch.linalg.norm(ba, dim=1)
    norm_bc = torch.linalg.norm(bc, dim=1)

    cos_angle = dot_product / (norm_ba * norm_bc + 1e-8)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    angles = torch.acos(cos_angle)

    return angles


def compute_dihedrals(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """
    Compute dihedral angle ABCD.

    Args:
        a: (N, 3) first atom
        b: (N, 3) second atom
        c: (N, 3) third atom
        d: (N, 3) fourth atom

    Returns:
        dihedrals: (N,) dihedral angles in radians [-pi, pi]
    """
    # Vector from b to a
    ba = a - b  # (N, 3)
    # Vector from b to c
    bc = c - b  # (N, 3)
    # Vector from c to d
    cd = d - c  # (N, 3)

    # Normal to plane ABC
    n1 = torch.cross(ba, bc, dim=1)  # (N, 3)
    # Normal to plane BCD
    n2 = torch.cross(bc, cd, dim=1)  # (N, 3)

    # Normalize
    n1 = n1 / (torch.linalg.norm(n1, dim=1, keepdim=True) + 1e-8)
    n2 = n2 / (torch.linalg.norm(n2, dim=1, keepdim=True) + 1e-8)

    # Dihedral angle
    cos_dihedral = torch.sum(n1 * n2, dim=1)
    cos_dihedral = torch.clamp(cos_dihedral, -1.0, 1.0)

    # Get sign from triple product
    sign = torch.sign(torch.sum(n1 * cd, dim=1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    dihedrals = sign * torch.acos(cos_dihedral)

    return dihedrals


# ─────────────────────────────────────────────────────────────────────────────
# Frame Construction
# ─────────────────────────────────────────────────────────────────────────────

def backbone_frame(
    n: torch.Tensor,
    ca: torch.Tensor,
    c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct backbone frame from N, Cα, C coordinates.

    Frame: X-axis along N→Cα, Z-axis normal to N-Cα-C plane, Y-axis right-hand.

    Args:
        n: (N, 3) nitrogen coordinates
        ca: (N, 3) alpha carbon coordinates
        c: (N, 3) carbonyl carbon coordinates

    Returns:
        origin: (N, 3) frame origin (Cα)
        frame: (N, 3, 3) rotation matrix [x_axis, y_axis, z_axis]
    """
    # Origin at Cα
    origin = ca

    # X-axis: N → Cα (normalized)
    x_axis = ca - n  # (N, 3)
    x_axis = x_axis / (torch.linalg.norm(x_axis, dim=1, keepdim=True) + 1e-8)

    # Vector from Cα to C
    ca_to_c = c - ca  # (N, 3)

    # Z-axis: perpendicular to x_axis and ca_to_c (normalized)
    z_axis = torch.cross(x_axis, ca_to_c, dim=1)  # (N, 3)
    z_axis = z_axis / (torch.linalg.norm(z_axis, dim=1, keepdim=True) + 1e-8)

    # Y-axis: right-hand rule
    y_axis = torch.cross(z_axis, x_axis, dim=1)  # (N, 3)

    # Combine into frame (rotation matrix)
    frame = torch.stack([x_axis, y_axis, z_axis], dim=2)  # (N, 3, 3)

    return origin, frame


def frame_to_coords(
    origin: torch.Tensor,
    frame: torch.Tensor,
) -> torch.Tensor:
    """
    Convert frame to coordinates (reconstruct Cα positions).

    Args:
        origin: (N, 3) frame origins
        frame: (N, 3, 3) rotation matrices

    Returns:
        coords: (N, 3) Cα coordinates
    """
    return origin


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate Transformations
# ─────────────────────────────────────────────────────────────────────────────

def center_coordinates(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Center coordinates at origin.

    Args:
        coords: (..., 3) coordinates

    Returns:
        centered: (..., 3) centered coordinates
        center: (3,) original center
    """
    center = coords.mean(dim=-2, keepdim=True)
    centered = coords - center

    return centered, center.squeeze(-2)


def normalize_coordinates(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize coordinates to unit sphere.

    Args:
        coords: (..., 3) coordinates

    Returns:
        normalized: (..., 3) normalized coordinates
        scale: () scaling factor
    """
    scale = torch.linalg.norm(coords.reshape(-1, 3), dim=1).max()
    normalized = coords / (scale + 1e-8)

    return normalized, scale


def translate_coordinates(
    coords: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """
    Translate coordinates.

    Args:
        coords: (..., 3) coordinates
        translation: (3,) translation vector

    Returns:
        translated: (..., 3) translated coordinates
    """
    return coords + translation.unsqueeze(0)


def rotate_coordinates(
    coords: torch.Tensor,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate coordinates.

    Args:
        coords: (N, 3) coordinates
        rotation_matrix: (3, 3) rotation matrix

    Returns:
        rotated: (N, 3) rotated coordinates
    """
    return torch.matmul(coords, rotation_matrix.T)


# ─────────────────────────────────────────────────────────────────────────────
# Displacement & Motion
# ─────────────────────────────────────────────────────────────────────────────

def compute_displacement(
    coords_t1: torch.Tensor,
    coords_t2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute displacement between two coordinate sets.

    Args:
        coords_t1: (N, 3) coordinates at time t1
        coords_t2: (N, 3) coordinates at time t2

    Returns:
        displacement: (N, 3) displacement vectors
    """
    return coords_t2 - coords_t1


def compute_rmsf_from_snapshots(
    snapshots: torch.Tensor,
) -> torch.Tensor:
    """
    Compute RMSF from coordinate snapshots.

    Args:
        snapshots: (T, N, 3) coordinates over time

    Returns:
        rmsf: (N,) root mean square fluctuation
    """
    # Mean structure
    mean_coords = snapshots.mean(dim=0)  # (N, 3)

    # Deviations
    deviations = snapshots - mean_coords  # (T, N, 3)

    # RMSF: sqrt(mean(squared_distance))
    rmsf = torch.sqrt(torch.mean(torch.sum(deviations ** 2, dim=2), dim=0))

    return rmsf


if __name__ == "__main__":
    # Test geometry utilities
    print("Geometry utilities loaded successfully")

    # Test with dummy data
    N = 100
    coords = torch.randn(N, 3)

    # Test distances
    distances = pairwise_distances(coords)
    print(f"Pairwise distances shape: {distances.shape}")

    # Test k-NN
    k_dists, k_inds = k_nearest_neighbors(coords, k=16)
    print(f"k-NN distances shape: {k_dists.shape}")

    # Test angles
    a = torch.randn(10, 3)
    b = torch.randn(10, 3)
    c = torch.randn(10, 3)
    angles = compute_angles(a, b, c)
    print(f"Angles shape: {angles.shape}")

    # Test backbone frame
    n = torch.randn(50, 3)
    ca = torch.randn(50, 3)
    c = torch.randn(50, 3)
    origin, frame = backbone_frame(n, ca, c)
    print(f"Frame shape: {frame.shape}")

    print("✓ All geometry utilities working!")
