"""
GVP-GNN Encoder for DynaMo Phase 1.

SE(3)-equivariant encoder using Geometric Vector Perceptrons (GVP) for
message passing on protein residue graphs.

Key properties:
  - Scalar node features (invariant): backbone dihedrals, SASA, KD, etc.
  - Vector node features (equivariant): backbone frame, Cβ direction, etc.
  - Shared weights across all MD snapshots
  - 3 message-passing layers with residual connections
  - Output: per-residue representations H ∈ ℝ^(N×256)

Reference: Jing et al., "Learning Protein Structure with Geometric Vector Perceptrons"
(ICLR 2021)

Usage:
    encoder = GVPEncoder(hidden_s=256, hidden_v=16, n_layers=3)
    H = encoder(node_s, node_v, edge_index, edge_s, edge_v)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Geometric Vector Perceptron (GVP) Layer
# ─────────────────────────────────────────────────────────────────────────────

class GVP(nn.Module):
    """
    Geometric Vector Perceptron layer.
    
    Takes scalar and vector features, applies learnable transformations
    while preserving SE(3) equivariance.
    
    Key: all nonlinearities applied to scalars only. Vectors use linear maps
    and gating. This ensures rotation equivariance.
    """

    def __init__(
        self,
        input_dims: Tuple[int, int],     # (n_scalar_in, n_vector_in)
        output_dims: Tuple[int, int],    # (n_scalar_out, n_vector_out)
        activations: Tuple[str, str] = ("relu", "sigmoid"),
        vector_gate: bool = True,
    ):
        """
        Args:
            input_dims: (input scalar dim, input vector dim in)
            output_dims: (output scalar dim, output vector dim)
            activations: (scalar activation, vector gate activation)
            vector_gate: if True, use gating on vector output
        """
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.vector_gate = vector_gate

        n_scalar_in, n_vector_in = input_dims
        n_scalar_out, n_vector_out = output_dims

        # ── Scalar transformation ──────────────────────────────────────────────
        # Input: [s, ||W_v · v||] (concatenate scalar + norms of vector outputs)
        # We need to compute norms of vector outputs first, so dimension is:
        # n_scalar_in + n_vector_out (norms of each output vector)
        self.W_s = nn.Linear(n_scalar_in + n_vector_out, n_scalar_out)

        # ── Vector transformation ──────────────────────────────────────────────
        # Linear map on each vector component (equivariant)
        # v_out = W_v · v_in  where W_v ∈ ℝ^(n_vector_out × n_vector_in)
        self.W_v = nn.Linear(n_vector_in, n_vector_out, bias=False)

        # ── Gating (applied after computing norms) ─────────────────────────────
        if vector_gate:
            # Compute gate from scalars: g = σ(W_g · s)
            self.W_g = nn.Linear(n_scalar_out, n_vector_out, bias=True)

        # ── Activation ─────────────────────────────────────────────────────────
        self.activation = self._get_activation(activations[0]) if activations[0] else None
        self.vector_gate_activation = (
            self._get_activation(activations[1]) if activations[1] else None
        )

    @staticmethod
    def _get_activation(name: str):
        """Get activation function by name."""
        if name == "relu":
            return F.relu
        elif name == "sigmoid":
            return torch.sigmoid
        elif name == "tanh":
            return torch.tanh
        elif name == "silu":
            return F.silu
        else:
            return None

    def forward(
        self,
        s: torch.Tensor,  # (N, n_scalar_in)
        v: torch.Tensor,  # (N, n_vector_in, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply GVP transformation.

        Args:
            s: scalar features (N, n_scalar_in)
            v: vector features (N, n_vector_in, 3)

        Returns:
            s_out: scalar output (N, n_scalar_out)
            v_out: vector output (N, n_vector_out, 3)
        """
        N = s.shape[0]

        # ── Transform vectors (equivariant) ────────────────────────────────────
        # v_out = W_v · v:  (N, n_vector_out, 3)
        v_out = torch.matmul(v, self.W_v.weight.T)  # (N, n_vector_in, 3) @ (n_vector_in, n_vector_out)^T

        # ── Compute norms of vector outputs (invariant) ────────────────────────
        # v_norms: (N, n_vector_out)  per-vector norms
        v_norms = torch.linalg.norm(v_out, dim=-1)  # (N, n_vector_out)

        # ── Scalar transformation: concat original + norms ───────────────────
        s_with_norms = torch.cat([s, v_norms], dim=-1)  # (N, n_scalar_in + n_vector_out)
        s_out = self.W_s(s_with_norms)  # (N, n_scalar_out)

        # ── Apply activation to scalars ────────────────────────────────────────
        if self.activation is not None:
            s_out = self.activation(s_out)

        # ── Gating on vectors ──────────────────────────────────────────────────
        if self.vector_gate:
            # Gate computed from output scalars: g_i = σ(W_g · s_out)_i
            gate = self.vector_gate_activation(self.W_g(s_out))  # (N, n_vector_out)
            # Expand gate and apply: v_out *= gate[..., None]
            v_out = gate.unsqueeze(-1) * v_out  # (N, n_vector_out) × (N, n_vector_out, 3)

        return s_out, v_out


class GVPConvLayer(nn.Module):
    """
    GVP convolution layer for message passing on protein graphs.

    Takes node and edge features, applies GVP transformations, and aggregates
    messages from neighbours.
    """

    def __init__(
        self,
        node_dims: Tuple[int, int],   # (node_scalar_dim, node_vector_dim)
        edge_dims: Tuple[int, int],   # (edge_scalar_dim, edge_vector_dim)
        hidden_dims: Optional[Tuple[int, int]] = None,  # intermediate dims
        activation_s: str = "relu",
        activation_v: str = "sigmoid",
        dropout: float = 0.1,
    ):
        """
        Args:
            node_dims: (input node scalar dim, input node vector dim)
            edge_dims: (input edge scalar dim, input edge vector dim)
            hidden_dims: intermediate hidden dims (default: same as node_dims)
            activation_s: scalar activation
            activation_v: vector gating activation
            dropout: dropout rate
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = node_dims

        n_s_in, n_v_in = node_dims
        n_s_hidden, n_v_hidden = hidden_dims
        n_s_edge, n_v_edge = edge_dims

        # ── Edge message computation (node + edge) ────────────────────────────
        # Concatenate source node scalar + target node scalar + edge scalar
        # for message computation
        self.edge_mlp_s = nn.Sequential(
            nn.Linear(n_s_in + n_s_in + n_s_edge, n_s_hidden),
            nn.GELU(),
            nn.Linear(n_s_hidden, n_s_hidden),
        )

        # ── GVP layers for message passing ────────────────────────────────────
        # Message from edge: GVP takes node + edge vector features
        self.message_gvp = GVP(
            input_dims=(n_s_hidden, n_v_in + n_v_edge),  # s from MLP, v from nodes + edges
            output_dims=(n_s_hidden, n_v_hidden),
            activations=(activation_s, activation_v),
            vector_gate=True,
        )

        # ── Node update: GVP on aggregated messages ────────────────────────────
        self.node_gvp = GVP(
            input_dims=(n_s_hidden + n_s_in, n_v_hidden + n_v_in),  # aggregated + skip connection
            output_dims=(n_s_in, n_v_in),  # output same dim as input
            activations=(activation_s, activation_v),
            vector_gate=True,
        )

        # ── Normalisation ──────────────────────────────────────────────────────
        self.norm_s = nn.LayerNorm(n_s_in)
        self.norm_v_scale = nn.LayerNorm(n_s_in)  # normalise norms of vectors

        # ── Dropout ────────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_s: torch.Tensor,    # (N, n_s_in)
        node_v: torch.Tensor,    # (N, n_v_in, 3)
        edge_index: torch.Tensor,  # (2, E)
        edge_s: torch.Tensor,    # (E, n_s_edge)
        edge_v: torch.Tensor,    # (E, n_v_edge, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Message passing + node update.

        Args:
            node_s: scalar node features (N, n_s_in)
            node_v: vector node features (N, n_v_in, 3)
            edge_index: edge indices (2, E) — [src, dst]
            edge_s: scalar edge features (E, n_s_edge)
            edge_v: vector edge features (E, n_v_edge, 3)

        Returns:
            node_s_out: updated scalar node features (N, n_s_in)
            node_v_out: updated vector node features (N, n_v_in, 3)
        """
        N = node_s.shape[0]
        E = edge_index.shape[1]

        src, dst = edge_index[0], edge_index[1]

        # ── Edge message computation ───────────────────────────────────────────
        # Combine source node + destination node + edge features
        node_s_src = node_s[src]  # (E, n_s_in)
        node_s_dst = node_s[dst]  # (E, n_s_in)

        # Concatenate for edge MLP
        edge_mlp_input = torch.cat([node_s_src, node_s_dst, edge_s], dim=-1)  # (E, 2*n_s_in + n_s_edge)
        edge_s_msg = self.edge_mlp_s(edge_mlp_input)  # (E, n_s_hidden)

        # Combine node and edge vectors for message GVP
        node_v_src = node_v[src]  # (E, n_v_in, 3)
        edge_v_msg = torch.cat([node_v_src, edge_v], dim=1)  # (E, n_v_in + n_v_edge, 3)

        # ── Message GVP ────────────────────────────────────────────────────────
        msg_s, msg_v = self.message_gvp(edge_s_msg, edge_v_msg)  # (E, n_s_hidden), (E, n_v_hidden, 3)
        msg_s = self.dropout(msg_s)
        msg_v = self.dropout(msg_v)

        # ── Aggregate messages by destination node ─────────────────────────────
        # Sum all messages arriving at each node
        agg_s = torch.zeros_like(node_s[:, :msg_s.shape[1]])  # (N, n_s_hidden)
        agg_v = torch.zeros(N, msg_v.shape[1], 3, device=msg_v.device, dtype=msg_v.dtype)

        agg_s.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_s), msg_s)  # scatter by destination
        for i in range(msg_v.shape[1]):
            agg_v[:, i, :].scatter_add_(0, dst.unsqueeze(1), msg_v[:, i, :])

        # ── Node update: GVP on aggregated + residual ──────────────────────────
        # Concatenate aggregated with original node features
        update_s = torch.cat([agg_s, node_s], dim=-1)  # (N, n_s_hidden + n_s_in)
        update_v = torch.cat([agg_v, node_v], dim=1)  # (N, n_v_hidden + n_v_in, 3)

        node_s_out, node_v_out = self.node_gvp(update_s, update_v)

        # ── Residual connection (already in GVP output) ──────────────────────

        # ── Normalisation ──────────────────────────────────────────────────────
        node_s_out = self.norm_s(node_s_out)
        node_v_norms = torch.linalg.norm(node_v_out, dim=-1)  # (N, n_v_in)
        node_v_norms = self.norm_v_scale(node_v_norms)  # normalise
        node_v_out = node_v_out / (torch.linalg.norm(node_v_out, dim=-1, keepdim=True) + 1e-8)
        node_v_out = node_v_norms.unsqueeze(-1) * node_v_out

        return node_s_out, node_v_out


# ─────────────────────────────────────────────────────────────────────────────
# Full GVP Encoder
# ─────────────────────────────────────────────────────────────────────────────

class GVPEncoder(nn.Module):
    """
    Full GVP-GNN encoder for protein residue graphs.

    3-layer message passing with skip connections and normalisation.
    """

    def __init__(
        self,
        node_s_dim: int = 147,    # scalar node features (PLM + struct)
        node_v_dim: int = 6,      # vector node features
        edge_s_dim: int = 40,     # scalar edge features
        edge_v_dim: int = 4,      # vector edge features
        hidden_s: int = 256,      # hidden scalar dimension
        hidden_v: int = 16,       # hidden vector dimension
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            node_s_dim: input scalar node feature dimension
            node_v_dim: input vector node feature dimension (number of 3D vectors)
            edge_s_dim: input scalar edge feature dimension
            edge_v_dim: input vector edge feature dimension
            hidden_s: hidden dimension for scalars
            hidden_v: hidden dimension for vectors
            n_layers: number of GVP layers
            dropout: dropout rate
        """
        super().__init__()

        self.node_s_dim = node_s_dim
        self.node_v_dim = node_v_dim
        self.hidden_s = hidden_s
        self.hidden_v = hidden_v
        self.output_dim = hidden_s

        # ── Input projection to hidden dimensions ──────────────────────────────
        self.node_embedding_s = nn.Sequential(
            nn.Linear(node_s_dim, hidden_s),
            nn.GELU(),
            nn.LayerNorm(hidden_s),
        )

        self.node_embedding_v = nn.Linear(node_v_dim * 3, hidden_v * 3, bias=False)

        # ── GVP layers ─────────────────────────────────────────────────────────
        self.gvp_layers = nn.ModuleList([
            GVPConvLayer(
                node_dims=(hidden_s, hidden_v),
                edge_dims=(edge_s_dim, edge_v_dim),
                hidden_dims=(hidden_s, hidden_v),
                activation_s="relu",
                activation_v="sigmoid",
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # ── Output projection (collapse vectors to invariant) ────────────────
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_s + hidden_v, hidden_s),
            nn.GELU(),
            nn.LayerNorm(hidden_s),
        )

    def forward(
        self,
        node_s: torch.Tensor,      # (N, node_s_dim)
        node_v: torch.Tensor,      # (N, node_v_dim, 3)
        edge_index: torch.Tensor,  # (2, E)
        edge_s: torch.Tensor,      # (E, edge_s_dim)
        edge_v: torch.Tensor,      # (E, edge_v_dim, 3)
    ) -> torch.Tensor:
        """
        Forward pass through GVP encoder.

        Args:
            node_s: scalar node features (N, node_s_dim)
            node_v: vector node features (N, node_v_dim, 3)
            edge_index: edge indices (2, E)
            edge_s: scalar edge features (E, edge_s_dim)
            edge_v: vector edge features (E, edge_v_dim, 3)

        Returns:
            H: per-residue representations (N, hidden_s)
        """
        N = node_s.shape[0]

        # ── Project inputs to hidden dimensions ─────────────────────────────────
        h_s = self.node_embedding_s(node_s)  # (N, hidden_s)

        # Reshape vectors for projection
        node_v_flat = node_v.reshape(N, -1)  # (N, node_v_dim * 3)
        h_v_flat = self.node_embedding_v(node_v_flat)  # (N, hidden_v * 3)
        h_v = h_v_flat.reshape(N, self.hidden_v, 3)  # (N, hidden_v, 3)

        # ── Message passing ────────────────────────────────────────────────────
        for layer in self.gvp_layers:
            h_s_skip = h_s  # skip connection
            h_v_skip = h_v

            h_s, h_v = layer(h_s, h_v, edge_index, edge_s, edge_v)

            # Residual connections
            h_s = h_s + h_s_skip
            h_v = h_v + h_v_skip

        # ── Output: collapse vectors (invariant) ────────────────────────────────
        # Compute norms of vectors (rotation-invariant)
        v_norms = torch.linalg.norm(h_v, dim=-1)  # (N, hidden_v)

        # Concatenate scalar + norms
        h_combined = torch.cat([h_s, v_norms], dim=-1)  # (N, hidden_s + hidden_v)

        # Project to output dimension
        H = self.output_proj(h_combined)  # (N, hidden_s)

        return H


class GVPEncoderWithResiduals(nn.Module):
    """
    Enhanced GVP encoder with explicit residual connections and layer skip paths.

    Provides stronger gradient flow and better feature mixing.
    """

    def __init__(
        self,
        node_s_dim: int = 147,
        node_v_dim: int = 6,
        edge_s_dim: int = 40,
        edge_v_dim: int = 4,
        hidden_s: int = 256,
        hidden_v: int = 16,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_s_dim = node_s_dim
        self.node_v_dim = node_v_dim
        self.hidden_s = hidden_s
        self.hidden_v = hidden_v

        # ── Input projection ───────────────────────────────────────────────────
        self.node_embedding_s = nn.Sequential(
            nn.Linear(node_s_dim, hidden_s),
            nn.GELU(),
            nn.LayerNorm(hidden_s),
        )

        self.node_embedding_v = nn.Linear(node_v_dim * 3, hidden_v * 3, bias=False)

        # ── GVP layers with dense connections ──────────────────────────────────
        self.gvp_layers = nn.ModuleList([
            GVPConvLayer(
                node_dims=(hidden_s, hidden_v),
                edge_dims=(edge_s_dim, edge_v_dim),
                hidden_dims=(hidden_s, hidden_v),
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # ── Dense skip connections between layers ──────────────────────────────
        self.skip_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_s * (i + 2), hidden_s),  # concatenated with all previous
                nn.GELU(),
                nn.LayerNorm(hidden_s),
            )
            for i in range(n_layers - 1)
        ])

        # ── Output ─────────────────────────────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_s + hidden_v, hidden_s),
            nn.GELU(),
            nn.LayerNorm(hidden_s),
        )

    def forward(
        self,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with dense connections."""
        N = node_s.shape[0]

        # Input projection
        h_s = self.node_embedding_s(node_s)
        node_v_flat = node_v.reshape(N, -1)
        h_v_flat = self.node_embedding_v(node_v_flat)
        h_v = h_v_flat.reshape(N, self.hidden_v, 3)

        # Track all hidden states for dense connections
        h_s_history = [h_s]

        # Message passing with dense skip connections
        for i, layer in enumerate(self.gvp_layers):
            h_s_new, h_v_new = layer(h_s, h_v, edge_index, edge_s, edge_v)

            # Residual
            h_s = h_s_new + h_s
            h_v = h_v_new + h_v

            h_s_history.append(h_s)

            # Dense skip: if not last layer, fuse with previous layers
            if i < len(self.gvp_layers) - 1:
                h_s_concat = torch.cat(h_s_history, dim=-1)
                h_s = self.skip_fusions[i](h_s_concat)

        # Output: collapse vectors
        v_norms = torch.linalg.norm(h_v, dim=-1)
        h_combined = torch.cat([h_s, v_norms], dim=-1)
        H = self.output_proj(h_combined)

        return H
