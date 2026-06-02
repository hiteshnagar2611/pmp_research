"""Unit tests for PMP research project.

Tests cover:
  - SE(3) equivariance (critical for geometric deep learning)
  - Model layer shapes and dimensions
  - Loss function correctness
  - Metrics computation
  - Data pipeline functionality
"""

import pytest
import torch
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set random seeds for reproducibility
torch.manual_seed(42)
