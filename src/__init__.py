"""
PMP Research: Peripheral Membrane Protein Design via Deep Learning

Two-phase system:
  Phase 1 (DynaMo): Predict binding residues
  Phase 2 (PMPGen): Generate de novo proteins

Novel contributions:
  - Conformational attention pool (RMSF-adaptive)
  - Membrane geometry path (OPM-aware)
  - Physicochemical gating (biophysical priors)
  - MD-informed noise schedule (per-residue anisotropy)
  - Membrane plane guidance (geometric steering)
  - 3-stage validation cascade (quality filtering)
"""

__version__ = "1.0.0"
__author__ = "PMP Research Team"

from . import models
from . import data
from . import training
from . import evaluation

__all__ = [
    'models',
    'data',
    'training',
    'evaluation',
]
