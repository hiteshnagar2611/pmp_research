"""
Training infrastructure for PMP models.

Modules:
  - losses: loss functions for both phases
  - metrics: evaluation metrics
  - trainer_phase1: DynaMo training loop
  - trainer_phase2: PMPGen training loop
  - callbacks: PyTorch Lightning callbacks
"""

from .losses import (
    FocalLoss, PatchContiguityLoss, ContrastiveLoss,
    FlowMatchingLoss, CombinedPMPGenLoss
)
from .metrics import (
    compute_mcc, compute_auroc, compute_f1,
    GenerationQualityMetrics
)
from .trainer_phase1 import DynaMoTrainer
from .trainer_phase2 import PMPGenTrainer
from .callbacks import CheckpointCallback, EarlyStoppingCallback

__all__ = [
    'FocalLoss',
    'PatchContiguityLoss',
    'ContrastiveLoss',
    'DynaMoTrainer',
    'PMPGenTrainer',
    'CheckpointCallback',
]
