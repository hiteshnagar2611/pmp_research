"""
Sequence Decoder: Design amino acid sequences for protein backbones.

Wraps ProteinMPNN for sequence design conditioned on backbone structure
and optional binding patch constraints.

ProteinMPNN References:
    - Original: https://github.com/dauparas/ProteinMPNN
    - Paper: "Robust deep learning based protein sequence design..."
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class SequenceDecoder(nn.Module):
    """
    Design amino acid sequences for protein backbones.

    Uses ProteinMPNN-like architecture to predict sequences given:
    - Backbone coordinates (frames or Cα atoms)
    - Optional binding residue mask
    - Optional family/scaffold constraints

    Args:
        n_tokens (int): Vocabulary size (default: 21 for standard 20 AAs + stop)
        hidden_dim (int): Hidden dimension (default: 128)
        n_layers (int): Number of encoder layers (default: 3)
        n_heads (int): Attention heads (default: 8)
        dropout (float): Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        n_tokens: int = 21,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize sequence decoder."""
        super().__init__()

        self.n_tokens = n_tokens
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        # ─────────────────────────────────────────────────────────────────────
        # Backbone Encoder
        # ─────────────────────────────────────────────────────────────────────

        # Project backbone features to embedding dimension
        # Input: (N, 3) Cα coordinates or (N, 3, 3) backbone frames
        self.backbone_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ─────────────────────────────────────────────────────────────────────
        # Graph Attention Encoder
        # ─────────────────────────────────────────────────────────────────────

        # Multi-layer attention over residues
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ─────────────────────────────────────────────────────────────────────
        # Decoder Head
        # ─────────────────────────────────────────────────────────────────────

        # Project to amino acid logits
        self.sequence_head = nn.Linear(hidden_dim, n_tokens)

        # ─────────────────────────────────────────────────────────────────────
        # Binding Constraint Encoder
        # ─────────────────────────────────────────────────────────────────────

        # Optional: modify predictions based on binding mask
        self.binding_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        coords: torch.Tensor,
        binding_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Design sequences for protein backbones.

        Args:
            coords: (B, N, 3) Cα coordinates or (B, N, 3, 3) backbone frames
            binding_mask: (B, N, 1) binding region mask (0-1)
            temperature: sampling temperature (higher = more random)
            return_logits: if True, return raw logits; else return probabilities

        Returns:
            sequences: (B, N) predicted amino acid indices
            logits: (B, N, n_tokens) per-residue logits
        """
        B, N = coords.shape[:2]

        # ─────────────────────────────────────────────────────────────────────
        # Encode Backbone
        # ─────────────────────────────────────────────────────────────────────

        # Handle both Cα coordinates and full backbone frames
        if coords.ndim == 3:
            # (B, N, 3) Cα coordinates
            backbone_feat = coords
        elif coords.ndim == 4:
            # (B, N, 3, 3) backbone frames → flatten to (B, N, 9)
            backbone_feat = coords.reshape(B, N, -1)
            # Project to 3D for encoder
            backbone_feat = backbone_feat[:, :, :3]
        else:
            raise ValueError(f"Expected coords shape (B, N, 3) or (B, N, 3, 3), got {coords.shape}")

        # Encode backbone features
        h = self.backbone_encoder(backbone_feat)  # (B, N, hidden_dim)

        # ─────────────────────────────────────────────────────────────────────
        # Graph Attention
        # ─────────────────────────────────────────────────────────────────────

        # Apply transformer encoder
        h = self.encoder(h)  # (B, N, hidden_dim)

        # ─────────────────────────────────────────────────────────────────────
        # Binding Constraints
        # ─────────────────────────────────────────────────────────────────────

        if binding_mask is not None:
            # Modify hidden state based on binding mask
            # High binding mask → preserve/reinforce current prediction
            binding_feat = self.binding_encoder(binding_mask)  # (B, N, 1)
            h = h + binding_feat.squeeze(-1).unsqueeze(-1) * h  # (B, N, hidden_dim)

        # ─────────────────────────────────────────────────────────────────────
        # Sequence Prediction
        # ─────────────────────────────────────────────────────────────────────

        # Predict logits for each residue
        logits = self.sequence_head(h)  # (B, N, n_tokens)

        # Apply temperature
        logits = logits / temperature

        if return_logits:
            return logits, logits

        # Sample or take argmax
        probs = torch.softmax(logits, dim=-1)
        sequences = torch.argmax(logits, dim=-1)  # (B, N)

        return sequences, logits

    def sample_sequences(
        self,
        coords: torch.Tensor,
        binding_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample multiple sequence designs.

        Args:
            coords: (B, N, 3) backbone coordinates
            binding_mask: (B, N, 1) optional binding mask
            temperature: sampling temperature
            num_samples: number of sequences to sample per input

        Returns:
            sequences: (B, num_samples, N) sampled sequences
            logits: (B, N, n_tokens) logits
        """
        B, N = coords.shape[:2]

        # Forward pass
        _, logits = self.forward(coords, binding_mask, temperature, return_logits=True)

        # Sample from categorical distribution
        logits_flat = logits.reshape(B * N, self.n_tokens)
        dist = torch.distributions.Categorical(logits=logits_flat)
        samples = dist.sample((num_samples,))  # (num_samples, B*N)
        samples = samples.transpose(0, 1).reshape(B, num_samples, N)

        return samples, logits


class ProteinMPNNWrapper(nn.Module):
    """
    Thin wrapper around ProteinMPNN checkpoint.

    Use this to load pre-trained ProteinMPNN weights if available.
    Falls back to SequenceDecoder if ProteinMPNN not available.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_pretrained: bool = False,
    ):
        """
        Initialize ProteinMPNN wrapper.

        Args:
            checkpoint_path: path to ProteinMPNN checkpoint
            use_pretrained: whether to load pre-trained weights
        """
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.use_pretrained = use_pretrained

        # Create base decoder
        self.decoder = SequenceDecoder(
            n_tokens=21,
            hidden_dim=128,
            n_layers=3,
            n_heads=8,
        )

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str):
        """Load ProteinMPNN weights from checkpoint."""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            self.decoder.load_state_dict(state_dict, strict=False)
            print(f"Loaded ProteinMPNN checkpoint from {path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint {path}: {e}")
            print("Using randomly initialized SequenceDecoder instead")

    def forward(
        self,
        coords: torch.Tensor,
        binding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        return self.decoder(coords, binding_mask, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

# Amino acid vocabulary
AA_VOCAB = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
    '*'  # stop token
]

TOKEN_TO_AA = {i: aa for i, aa in enumerate(AA_VOCAB)}
AA_TO_TOKEN = {aa: i for i, aa in enumerate(AA_VOCAB)}


def tokens_to_sequence(tokens: torch.Tensor) -> str:
    """
    Convert token indices to amino acid sequence.

    Args:
        tokens: (N,) token indices

    Returns:
        sequence: amino acid string
    """
    tokens_np = tokens.cpu().numpy()
    sequence = ''.join([TOKEN_TO_AA.get(int(t), 'X') for t in tokens_np])
    return sequence


def sequence_to_tokens(sequence: str) -> torch.Tensor:
    """
    Convert amino acid sequence to token indices.

    Args:
        sequence: amino acid string

    Returns:
        tokens: (N,) token indices
    """
    tokens = [AA_TO_TOKEN.get(aa, AA_TO_TOKEN['X']) for aa in sequence.upper()]
    return torch.tensor(tokens, dtype=torch.long)


if __name__ == "__main__":
    # Test sequence decoder
    import torch

    # Create dummy data
    B, N = 2, 100  # batch size 2, 100 residues
    coords = torch.randn(B, N, 3)
    binding_mask = torch.bernoulli(torch.full((B, N, 1), 0.2))

    # Create decoder
    decoder = SequenceDecoder(n_tokens=21, hidden_dim=64, n_layers=2)

    # Decode sequences
    sequences, logits = decoder(coords, binding_mask, temperature=1.0)

    print(f"Input coords shape: {coords.shape}")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Example sequences:")
    for i in range(B):
        seq = tokens_to_sequence(sequences[i])
        print(f"  {seq}")

    # Sample multiple sequences
    samples, _ = decoder.sample_sequences(coords, binding_mask, num_samples=3)
    print(f"\nSampled sequences shape: {samples.shape}")
