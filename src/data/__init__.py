"""
Data pipeline: load, preprocess, and augment protein data.

Modules:
  - graph_builder: construct kNN graphs with edge features
  - feature_extractor: extract residue-level features
  - md_processor: process MD trajectories for dynamics
  - plm_embedder: compute ESM-2 embeddings
  - pmp_dataset: PyTorch Geometric dataset loader
  - transforms: data augmentation and preprocessing
"""

from .graph_builder import GraphBuilder, build_graph
from .feature_extractor import FeatureExtractor
from .md_processor import MDProcessor, RMSFComputer, VelocityExtractor
from .plm_embedder import ESM2Embedder, DummyEmbedder, get_embedder
from .pmp_dataset import PMPDataset, PMPDataModule
from .transforms import (
    RandomRotation, RandomNoise, NormalizeFeatures,
    get_train_transforms, get_test_transforms
)

__all__ = [
    'GraphBuilder',
    'FeatureExtractor',
    'MDProcessor',
    'ESM2Embedder',
    'PMPDataset',
    'PMPDataModule',
    'RandomRotation',
    'get_train_transforms',
]
