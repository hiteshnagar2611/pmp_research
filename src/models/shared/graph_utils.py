"""
Graph utilities: construction, manipulation, and analysis.

Provides utilities for:
  - Graph construction (k-NN, complete, custom)
  - Edge feature computation
  - Graph analysis and statistics
  - Adjacency matrix operations
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# K-NN Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

def knn_graph(
    coords: torch.Tensor,
    k: int = 16,
) -> torch.Tensor:
    """
    Construct k-nearest neighbor graph.

    Args:
        coords: (N, 3) coordinates
        k: number of neighbors

    Returns:
        edge_index: (2, E) edge indices [source, target]
    """
    N = coords.shape[0]

    # Pairwise distances
    diffs = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
    distances = torch.linalg.norm(diffs, dim=2)  # (N, N)

    # Set diagonal to infinity (exclude self-loops)
    distances = distances + torch.eye(N, device=coords.device) * 1e10

    # Find k-nearest neighbors
    _, indices = torch.topk(distances, k=k, dim=1, largest=False)  # (N, k)

    # Build edge list
    sources = torch.repeat_interleave(torch.arange(N, device=coords.device), k)
    targets = indices.reshape(-1)

    edge_index = torch.stack([sources, targets], dim=0)

    return edge_index


def radius_graph(
    coords: torch.Tensor,
    radius: float = 8.0,
) -> torch.Tensor:
    """
    Construct radius graph (edges within radius threshold).

    Args:
        coords: (N, 3) coordinates
        radius: distance threshold

    Returns:
        edge_index: (2, E) edge indices
    """
    # Pairwise distances
    diffs = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
    distances = torch.linalg.norm(diffs, dim=2)  # (N, N)

    # Find edges within radius (exclude self)
    mask = (distances < radius) & (distances > 0)

    edge_index = torch.nonzero(mask, as_tuple=True)
    edge_index = torch.stack(edge_index, dim=0)

    return edge_index


def fully_connected_graph(n_nodes: int, device: str = 'cpu') -> torch.Tensor:
    """
    Construct fully connected graph.

    Args:
        n_nodes: number of nodes
        device: device to create graph on

    Returns:
        edge_index: (2, N*(N-1)) edge indices
    """
    sources = []
    targets = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                sources.append(i)
                targets.append(j)

    edge_index = torch.tensor(
        [sources, targets],
        dtype=torch.long,
        device=device
    )

    return edge_index


# ─────────────────────────────────────────────────────────────────────────────
# Edge Features
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_features(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Compute edge features from coordinates.

    Features:
      - RBF distance (16-dim)
      - Relative orientation (9-dim)
      - Multi-scale distances (4-dim)
      - Sequence separation (4-dim)
      - Edge type indicators (7-dim)

    Total: 40-dim

    Args:
        coords: (N, 3) coordinates
        edge_index: (2, E) edge indices

    Returns:
        edge_attr: (E, 40) edge features
    """
    sources, targets = edge_index

    # Source and target coordinates
    src_coords = coords[sources]  # (E, 3)
    tgt_coords = coords[targets]  # (E, 3)

    # 1. Distance vector
    diff = tgt_coords - src_coords  # (E, 3)
    distance = torch.linalg.norm(diff, dim=1, keepdim=True)  # (E, 1)

    # 2. RBF distance encoding (16-dim)
    rbf_centers = torch.linspace(0, 8, 16, device=coords.device)
    rbf_distance = torch.exp(-((distance - rbf_centers) ** 2) / 0.1)  # (E, 16)

    # 3. Relative orientation (9-dim)
    # Use relative positions as features
    relative_pos = diff / (distance + 1e-8)  # (E, 3) normalized direction
    # Add higher order terms
    relative_orient = torch.cat([
        relative_pos,  # (E, 3)
        relative_pos ** 2,  # (E, 3)
        relative_pos ** 3,  # (E, 3)
    ], dim=1)  # (E, 9)

    # 4. Multi-scale distances (4-dim)
    multi_scale = torch.cat([
        distance,
        distance ** 2,
        torch.log(distance + 1),
        torch.sqrt(distance),
    ], dim=1)  # (E, 4)

    # 5. Sequence separation (4-dim)
    seq_sep = (targets - sources).float().unsqueeze(1)  # (E, 1)
    seq_features = torch.cat([
        seq_sep,
        torch.log(torch.abs(seq_sep) + 1),
        torch.sign(seq_sep) * torch.ones_like(seq_sep),
        (seq_sep > 0).float(),
    ], dim=1)  # (E, 4)

    # 6. Edge type indicators (7-dim)
    edge_types = torch.zeros(len(sources), 7, device=coords.device)
    # Could classify edge types based on distance
    edge_types[:, 0] = (distance.squeeze() < 5).float()  # close
    edge_types[:, 1] = ((distance.squeeze() >= 5) & (distance.squeeze() < 10)).float()  # medium
    edge_types[:, 2] = (distance.squeeze() >= 10).float()  # far
    edge_types[:, 3] = ((targets - sources) > 0).float()  # forward
    edge_types[:, 4] = ((targets - sources) < 0).float()  # backward
    edge_types[:, 5] = (torch.abs(targets - sources) < 3).float()  # sequential
    edge_types[:, 6] = (torch.abs(targets - sources) >= 3).float()  # non-sequential

    # Concatenate all features
    edge_attr = torch.cat([
        rbf_distance,
        relative_orient,
        multi_scale,
        seq_features,
        edge_types,
    ], dim=1)  # (E, 40)

    return edge_attr


# ─────────────────────────────────────────────────────────────────────────────
# Graph Statistics
# ─────────────────────────────────────────────────────────────────────────────

def graph_statistics(edge_index: torch.Tensor, n_nodes: int) -> Dict[str, float]:
    """
    Compute graph statistics.

    Args:
        edge_index: (2, E) edge indices
        n_nodes: number of nodes

    Returns:
        stats: dictionary of statistics
    """
    n_edges = edge_index.shape[1]

    # Degree distribution
    degrees = torch.bincount(edge_index[0], minlength=n_nodes)
    mean_degree = degrees.float().mean().item()
    max_degree = degrees.max().item()
    min_degree = degrees.min().item()

    # Density
    max_edges = n_nodes * (n_nodes - 1)
    density = n_edges / max_edges

    stats = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'mean_degree': mean_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'density': density,
    }

    return stats


def degree_distribution(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """
    Compute degree distribution.

    Args:
        edge_index: (2, E) edge indices
        n_nodes: number of nodes

    Returns:
        degrees: (N,) node degrees
    """
    degrees = torch.bincount(edge_index[0], minlength=n_nodes)
    return degrees


# ─────────────────────────────────────────────────────────────────────────────
# Adjacency Operations
# ─────────────────────────────────────────────────────────────────────────────

def edge_index_to_adjacency(
    edge_index: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """
    Convert edge index to adjacency matrix.

    Args:
        edge_index: (2, E) edge indices
        n_nodes: number of nodes

    Returns:
        adj_matrix: (N, N) adjacency matrix
    """
    adj_matrix = torch.zeros(n_nodes, n_nodes, device=edge_index.device)
    sources, targets = edge_index
    adj_matrix[sources, targets] = 1

    return adj_matrix


def adjacency_to_edge_index(adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert adjacency matrix to edge index.

    Args:
        adj_matrix: (N, N) adjacency matrix

    Returns:
        edge_index: (2, E) edge indices
    """
    sources, targets = torch.nonzero(adj_matrix, as_tuple=True)
    edge_index = torch.stack([sources, targets], dim=0)

    return edge_index


def add_self_loops(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """
    Add self-loops to edge index.

    Args:
        edge_index: (2, E) edge indices
        n_nodes: number of nodes

    Returns:
        edge_index_with_loops: (2, E + N) edge indices
    """
    self_loops = torch.arange(n_nodes, device=edge_index.device)
    self_loops = torch.stack([self_loops, self_loops], dim=0)

    edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)

    return edge_index_with_loops


def remove_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Remove self-loops from edge index.

    Args:
        edge_index: (2, E) edge indices

    Returns:
        edge_index_no_loops: (2, E') edge indices
    """
    mask = edge_index[0] != edge_index[1]
    edge_index_no_loops = edge_index[:, mask]

    return edge_index_no_loops


# ─────────────────────────────────────────────────────────────────────────────
# Graph Filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_edges_by_distance(
    edge_index: torch.Tensor,
    coords: torch.Tensor,
    max_distance: float,
) -> torch.Tensor:
    """
    Filter edges by maximum distance.

    Args:
        edge_index: (2, E) edge indices
        coords: (N, 3) coordinates
        max_distance: maximum distance threshold

    Returns:
        filtered_edge_index: (2, E') filtered edge indices
    """
    sources, targets = edge_index
    src_coords = coords[sources]
    tgt_coords = coords[targets]

    distances = torch.linalg.norm(tgt_coords - src_coords, dim=1)
    mask = distances < max_distance

    filtered_edge_index = edge_index[:, mask]

    return filtered_edge_index


def filter_edges_by_degree(
    edge_index: torch.Tensor,
    n_nodes: int,
    max_degree: int,
) -> torch.Tensor:
    """
    Filter edges to limit maximum degree.

    Args:
        edge_index: (2, E) edge indices
        n_nodes: number of nodes
        max_degree: maximum degree per node

    Returns:
        filtered_edge_index: (2, E') filtered edge indices
    """
    # Count edges per node
    degrees = torch.bincount(edge_index[0], minlength=n_nodes)

    # Filter edges
    mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)

    for source_idx in range(n_nodes):
        source_edges = (edge_index[0] == source_idx).nonzero(as_tuple=True)[0]

        if len(source_edges) > max_degree:
            # Keep only first max_degree edges
            mask[source_edges[max_degree:]] = False

    filtered_edge_index = edge_index[:, mask]

    return filtered_edge_index


if __name__ == "__main__":
    # Test graph utilities
    print("Graph utilities loaded successfully")

    # Test with dummy data
    N = 100
    coords = torch.randn(N, 3)

    # Test k-NN graph
    edge_index = knn_graph(coords, k=16)
    print(f"k-NN graph edges: {edge_index.shape[1]}")

    # Test edge features
    edge_attr = compute_edge_features(coords, edge_index)
    print(f"Edge features shape: {edge_attr.shape}")

    # Test graph statistics
    stats = graph_statistics(edge_index, N)
    print(f"Graph stats: {stats}")

    # Test adjacency conversion
    adj_matrix = edge_index_to_adjacency(edge_index, N)
    print(f"Adjacency matrix shape: {adj_matrix.shape}")

    print("✓ All graph utilities working!")
