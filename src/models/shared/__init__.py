"""
Shared utilities for all models.

Includes:
  - SE(3) math utilities (rotations, translations, etc.)
  - Rotation matrices and quaternions
  - Graph utilities
"""

from .se3_utils import (
    rotate_vector,
    rotate_coords,
    compute_rotation_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)

__all__ = [
    'rotate_vector',
    'rotate_coords',
    'compute_rotation_matrix',
    'quaternion_to_matrix',
    'matrix_to_quaternion',
]
