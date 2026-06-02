"""
Deep learning models: DynaMo (Phase 1) and PMPGen (Phase 2).

Modules:
  - dynamo: binding prediction model
  - pmpgen: de novo generation model
  - shared: utilities (SE(3) math, etc)

Key features:
  - SE(3)-equivariant architectures
  - Novel attention mechanisms
  - Membrane-aware encoders
  - Flow matching for generation
"""

from .dynamo import DynaMo
from .pmpgen import PMPGen

__all__ = [
    'DynaMo',
    'PMPGen',
]
