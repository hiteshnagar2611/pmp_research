"""
Normalization utilities: feature normalization, standardization, and scaling.

Provides:
  - Layer normalization with learnable parameters
  - Graph normalization
  - Feature standardization
  - Min-max scaling
"""

from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Feature Normalization
# ─────────────────────────────────────────────────────────────────────────────

class FeatureNormalization(nn.Module):
    """
    Normalize features to zero mean, unit variance.

    Args:
        input_dim: input feature dimension
        epsilon: small value for numerical stability
        momentum: momentum for running statistics
    """

    def __init__(
        self,
        input_dim: int,
        epsilon: float = 1e-5,
        momentum: float = 0.1,
    ):
        """Initialize normalization."""
        super().__init__()

        self.input_dim = input_dim
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_var', torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize features.

        Args:
            x: (..., input_dim) input features

        Returns:
            normalized: (..., input_dim) normalized features
        """
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics during evaluation
            mean = self.running_mean
            var = self.running_var

        # Normalize
        normalized = (x - mean) / torch.sqrt(var + self.epsilon)

        # Scale and shift
        output = normalized * self.weight + self.bias

        return output


class GraphNormalization(nn.Module):
    """
    Normalize features per graph.

    Useful for message passing networks where each graph may have different statistics.

    Args:
        input_dim: input feature dimension
    """

    def __init__(self, input_dim: int):
        """Initialize graph normalization."""
        super().__init__()

        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Normalize features per graph.

        Args:
            x: (N, input_dim) node features
            batch: (N,) batch indices (optional)

        Returns:
            normalized: (N, input_dim) normalized features
        """
        if batch is None:
            # Single graph
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
        else:
            # Multiple graphs
            unique_batches = torch.unique(batch)
            mean = torch.zeros_like(x)
            var = torch.zeros_like(x)

            for b in unique_batches:
                mask = batch == b
                mean[mask] = x[mask].mean(dim=0)
                var[mask] = x[mask].var(dim=0, unbiased=False)

        # Normalize
        normalized = (x - mean) / torch.sqrt(var + 1e-5)

        # Scale and shift
        output = normalized * self.weight + self.bias

        return output


# ─────────────────────────────────────────────────────────────────────────────
# Functional Normalization
# ─────────────────────────────────────────────────────────────────────────────

def standardize(
    x: torch.Tensor,
    dim: int = 0,
    epsilon: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize tensor to zero mean, unit variance.

    Args:
        x: input tensor
        dim: dimension to compute statistics
        epsilon: small value for stability

    Returns:
        standardized: standardized tensor
        mean: mean values
        std: standard deviation
    """
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, unbiased=False, keepdim=True)

    standardized = (x - mean) / (std + epsilon)

    return standardized, mean.squeeze(dim), std.squeeze(dim)


def normalize_by_norm(
    x: torch.Tensor,
    p: float = 2.0,
    dim: int = -1,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    Normalize by Lp norm.

    Args:
        x: (..., D) input tensor
        p: norm power (1, 2, etc.)
        dim: dimension to compute norm
        epsilon: small value for stability

    Returns:
        normalized: normalized tensor
    """
    norm = torch.norm(x, p=p, dim=dim, keepdim=True)
    normalized = x / (norm + epsilon)

    return normalized


def min_max_scale(
    x: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0,
    dim: int = 0,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    Min-max scaling.

    Args:
        x: input tensor
        min_val: minimum value of output range
        max_val: maximum value of output range
        dim: dimension to compute min/max
        epsilon: small value for stability

    Returns:
        scaled: scaled tensor in [min_val, max_val]
    """
    x_min = x.min(dim=dim, keepdim=True)[0]
    x_max = x.max(dim=dim, keepdim=True)[0]

    x_scaled = (x - x_min) / (x_max - x_min + epsilon)
    x_scaled = x_scaled * (max_val - min_val) + min_val

    return x_scaled


def layer_normalize(
    x: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    Layer normalization (simplified).

    Args:
        x: (..., *normalized_shape) input tensor
        normalized_shape: shape of features to normalize over
        epsilon: small value for stability

    Returns:
        normalized: normalized tensor
    """
    mean = x.mean(dim=list(range(-len(normalized_shape), 0)), keepdim=True)
    var = x.var(dim=list(range(-len(normalized_shape), 0)), unbiased=False, keepdim=True)

    normalized = (x - mean) / torch.sqrt(var + epsilon)

    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# Vector/Matrix Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_vectors(
    vectors: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize vectors to unit length.

    Args:
        vectors: (..., 3) vectors
        epsilon: small value for stability

    Returns:
        normalized: (..., 3) unit vectors
    """
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    normalized = vectors / (norms + epsilon)

    return normalized


def normalize_frames(
    frames: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize rotation matrices (frame).

    Uses Gram-Schmidt orthogonalization.

    Args:
        frames: (..., 3, 3) rotation matrices
        epsilon: small value for stability

    Returns:
        orthogonal: (..., 3, 3) orthonormal frames
    """
    # Extract columns
    col1 = frames[..., :, 0]  # (..., 3)
    col2 = frames[..., :, 1]
    col3 = frames[..., :, 2]

    # Orthogonalize
    col1 = normalize_vectors(col1, epsilon)

    # col2 orthogonal to col1
    col2 = col2 - (col2 * col1).sum(dim=-1, keepdim=True) * col1
    col2 = normalize_vectors(col2, epsilon)

    # col3 orthogonal to col1, col2
    col3 = col3 - (col3 * col1).sum(dim=-1, keepdim=True) * col1
    col3 = col3 - (col3 * col2).sum(dim=-1, keepdim=True) * col2
    col3 = normalize_vectors(col3, epsilon)

    # Stack back
    orthogonal = torch.stack([col1, col2, col3], dim=-1)

    return orthogonal


# ─────────────────────────────────────────────────────────────────────────────
# Batch Normalization Variants
# ─────────────────────────────────────────────────────────────────────────────

class MaskedLayerNorm(nn.Module):
    """
    Layer normalization with masking.

    Useful for handling variable-length sequences.

    Args:
        normalized_shape: shape of normalized features
        epsilon: small value for stability
    """

    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        epsilon: float = 1e-5,
    ):
        """Initialize masked layer norm."""
        super().__init__()

        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply masked layer normalization.

        Args:
            x: (..., *normalized_shape) input tensor
            mask: (..., 1) mask tensor (optional)

        Returns:
            normalized: normalized tensor
        """
        if mask is not None:
            # Zero out masked positions
            x = x * mask

        # Compute statistics (ignoring masked positions)
        if mask is not None:
            n_valid = mask.sum()
            mean = (x * mask).sum() / n_valid
            var = ((x - mean) ** 2 * mask).sum() / n_valid
        else:
            mean = x.mean(dim=list(range(-len(self.normalized_shape), 0)), keepdim=True)
            var = x.var(dim=list(range(-len(self.normalized_shape), 0)), unbiased=False, keepdim=True)

        # Normalize
        normalized = (x - mean) / torch.sqrt(var + self.epsilon)

        # Apply scale and shift
        normalized = normalized * self.weight + self.bias

        # Mask again
        if mask is not None:
            normalized = normalized * mask

        return normalized


if __name__ == "__main__":
    # Test normalization utilities
    print("Normalization utilities loaded successfully")

    # Test feature normalization
    feat_norm = FeatureNormalization(input_dim=64)
    x = torch.randn(10, 64)
    y = feat_norm(x)
    print(f"Feature normalization output shape: {y.shape}")
    print(f"Output mean: {y.mean().item():.6f}, std: {y.std().item():.6f}")

    # Test standardization
    standardized, mean, std = standardize(x)
    print(f"Standardized shape: {standardized.shape}")

    # Test vector normalization
    vectors = torch.randn(5, 3)
    normalized = normalize_vectors(vectors)
    norms = torch.linalg.norm(normalized, dim=-1)
    print(f"Vector norms (should be ~1): {norms}")

    # Test min-max scaling
    scaled = min_max_scale(x, min_val=0, max_val=1)
    print(f"Scaled min: {scaled.min().item():.4f}, max: {scaled.max().item():.4f}")

    print("✓ All normalization utilities working!")
