"""
Per-Residue Binding Classifier

Final layer that converts learned representations to binding predictions.

Architecture:
    H_in (N, 256)
    ↓
    Linear(256 → 64) + GELU
    ↓
    LayerNorm
    ↓
    Dropout
    ↓
    Linear(64 → 1)
    ↓
    Sigmoid → p_binding (N, 1)

Output is per-residue binding probability (0-1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidueClassifier(nn.Module):
    """
    Classify whether each residue binds the membrane.

    Per-residue binary classification with MLP head.

    Args:
        hidden_dim (int): Input hidden dimension (default: 256)
        intermediate_dim (int): Intermediate hidden dimension (default: 64)
        n_classes (int): Number of output classes (default: 1 for binary)
        dropout (float): Dropout probability (default: 0.1)
        use_layer_norm (bool): Whether to use LayerNorm (default: True)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        intermediate_dim: int = 64,
        n_classes: int = 1,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """Initialize classifier."""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.n_classes = n_classes
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # ─────────────────────────────────────────────────────────────────────
        # MLP Layers
        # ─────────────────────────────────────────────────────────────────────

        # First dense layer: reduce dimension
        self.dense1 = nn.Linear(hidden_dim, intermediate_dim)

        # Activation
        self.gelu = nn.GELU()

        # Layer normalization (optional)
        if use_layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Output layer: per-residue prediction
        self.dense2 = nn.Linear(intermediate_dim, n_classes)

        # ─────────────────────────────────────────────────────────────────────
        # Initialization
        # ─────────────────────────────────────────────────────────────────────

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.dense1.weight, a=0.01, mode='fan_in')
        nn.init.kaiming_uniform_(self.dense2.weight, a=0.01, mode='fan_in')

        # Initialize biases to zero
        if self.dense1.bias is not None:
            nn.init.zeros_(self.dense1.bias)
        if self.dense2.bias is not None:
            nn.init.zeros_(self.dense2.bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Classify binding residues.

        Args:
            H: (N, hidden_dim) learned representations from fusion layer

        Returns:
            logits: (N, n_classes) per-residue predictions
                For binary classification: (N, 1) with values in (-inf, +inf)
                Apply sigmoid during loss computation (BCEWithLogitsLoss)
        """
        # (N, hidden_dim) → (N, intermediate_dim)
        x = self.dense1(H)

        # Activation
        x = self.gelu(x)

        # Layer normalization
        if self.ln is not None:
            x = self.ln(x)

        # Dropout
        x = self.dropout_layer(x)

        # Output layer
        # (N, intermediate_dim) → (N, n_classes)
        logits = self.dense2(x)

        return logits

    def get_predictions(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get binding predictions and probabilities.

        Args:
            H: (N, hidden_dim) representations

        Returns:
            logits: (N, 1) raw predictions
            probs: (N, 1) sigmoid probabilities
        """
        logits = self.forward(H)
        probs = torch.sigmoid(logits)
        return logits, probs


class DeepResidueClassifier(nn.Module):
    """
    Deeper binding classifier with multiple hidden layers.

    Allows for more complex decision boundaries at the cost of more parameters.

    Args:
        hidden_dim (int): Input hidden dimension (default: 256)
        n_layers (int): Number of hidden layers (default: 2)
        hidden_scale (float): Scale for hidden dimensions (default: 0.5)
        n_classes (int): Output classes (default: 1)
        dropout (float): Dropout probability (default: 0.2)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 2,
        hidden_scale: float = 0.5,
        n_classes: int = 1,
        dropout: float = 0.2,
    ):
        """Initialize deep classifier."""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes

        # ─────────────────────────────────────────────────────────────────────
        # Build layers
        # ─────────────────────────────────────────────────────────────────────

        layers = []
        prev_dim = hidden_dim

        for i in range(n_layers):
            # Compute hidden dimension (linear decay)
            hidden_i = max(int(hidden_dim * (hidden_scale ** (i + 1))), n_classes * 4)

            # Add dense layer
            layers.append(nn.Linear(prev_dim, hidden_i))

            # Add activation
            layers.append(nn.GELU())

            # Add layer norm
            layers.append(nn.LayerNorm(hidden_i))

            # Add dropout
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_i

        # Output layer
        layers.append(nn.Linear(prev_dim, n_classes))

        self.mlp = nn.Sequential(*layers)

        # ─────────────────────────────────────────────────────────────────────
        # Initialization
        # ─────────────────────────────────────────────────────────────────────

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01, mode='fan_in')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mlp(H)


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple classifiers for robustness.

    Trains multiple classifier heads and averages predictions.

    Args:
        hidden_dim (int): Input dimension (default: 256)
        n_classifiers (int): Number of classifiers to ensemble (default: 3)
        **kwargs: Arguments passed to ResidueClassifier
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_classifiers: int = 3,
        **kwargs
    ):
        """Initialize ensemble classifier."""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_classifiers = n_classifiers

        # Create multiple classifiers
        self.classifiers = nn.ModuleList([
            ResidueClassifier(hidden_dim=hidden_dim, **kwargs)
            for _ in range(n_classifiers)
        ])

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ensemble averaging.

        Args:
            H: (N, hidden_dim) representations

        Returns:
            logits: (N, 1) averaged predictions
        """
        # Forward through all classifiers
        logits_list = [clf(H) for clf in self.classifiers]

        # Average predictions
        # Stack: (n_classifiers, N, 1) → mean → (N, 1)
        logits = torch.stack(logits_list, dim=0).mean(dim=0)

        return logits

    def get_ensemble_variance(self, H: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction uncertainty from ensemble variance.

        High variance = uncertain predictions

        Args:
            H: (N, hidden_dim) representations

        Returns:
            variance: (N, 1) per-residue uncertainty
        """
        logits_list = [clf(H) for clf in self.classifiers]
        logits_stack = torch.stack(logits_list, dim=0)  # (n_classifiers, N, 1)

        variance = logits_stack.var(dim=0)  # (N, 1)

        return variance


# ─────────────────────────────────────────────────────────────────────────────
# Loss-aware Classifier
# ─────────────────────────────────────────────────────────────────────────────

class FocalLossClassifier(nn.Module):
    """
    Classifier with integrated focal loss weighting.

    Learns to focus on hard examples during training.

    Args:
        hidden_dim (int): Input dimension (default: 256)
        focal_gamma (float): Focal loss focusing parameter (default: 2.0)
        **kwargs: Arguments passed to ResidueClassifier
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        focal_gamma: float = 2.0,
        **kwargs
    ):
        """Initialize focal loss classifier."""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.focal_gamma = focal_gamma

        # Base classifier
        self.classifier = ResidueClassifier(hidden_dim=hidden_dim, **kwargs)

        # Learnable focal gamma (optional: make gamma adaptive)
        # self.log_gamma = nn.Parameter(torch.tensor(math.log(focal_gamma)))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(H)

    def compute_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: (N,) raw predictions
            targets: (N,) binary labels
            reduction: 'mean' or 'sum'

        Returns:
            loss: scalar focal loss
        """
        # Compute probabilities
        p = torch.sigmoid(logits)

        # Focal loss: -α * (1-p_t)^γ * log(p_t)
        # where p_t = p if y=1, else 1-p
        p_t = torch.where(targets == 1, p, 1 - p)

        # Focal term: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.focal_gamma

        # Binary cross entropy (already includes sigmoid)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Apply focal weight
        focal_loss = focal_weight * bce

        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def create_classifier(
    classifier_type: str = 'simple',
    hidden_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create classifier.

    Args:
        classifier_type: 'simple', 'deep', or 'ensemble'
        hidden_dim: Input dimension
        **kwargs: Additional arguments

    Returns:
        Classifier module
    """
    if classifier_type == 'simple':
        return ResidueClassifier(hidden_dim=hidden_dim, **kwargs)
    elif classifier_type == 'deep':
        return DeepResidueClassifier(hidden_dim=hidden_dim, **kwargs)
    elif classifier_type == 'ensemble':
        return EnsembleClassifier(hidden_dim=hidden_dim, **kwargs)
    elif classifier_type == 'focal':
        return FocalLossClassifier(hidden_dim=hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


if __name__ == "__main__":
    # Test classifier
    import torch

    # Create dummy input
    H = torch.randn(10, 256)  # 10 residues, 256-dim hidden

    # Test simple classifier
    clf = ResidueClassifier(hidden_dim=256, intermediate_dim=64)
    logits = clf(H)
    logits, probs = clf.get_predictions(H)

    print(f"Input shape: {H.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"Probs range: [{probs.min():.3f}, {probs.max():.3f}]")

    # Test deep classifier
    deep_clf = DeepResidueClassifier(hidden_dim=256, n_layers=3)
    deep_logits = deep_clf(H)
    print(f"\nDeep classifier logits shape: {deep_logits.shape}")

    # Test ensemble classifier
    ensemble_clf = EnsembleClassifier(hidden_dim=256, n_classifiers=5)
    ensemble_logits = ensemble_clf(H)
    ensemble_var = ensemble_clf.get_ensemble_variance(H)
    print(f"\nEnsemble logits shape: {ensemble_logits.shape}")
    print(f"Ensemble variance shape: {ensemble_var.shape}")
