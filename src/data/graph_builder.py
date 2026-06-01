"""
Graph builder for protein residue-level kNN graphs.

Constructs:
  - kNN residue graph based on Cα distances
  - Edge features (distance, orientation, sequence separation, etc.)
  - Node features scaffold (to be filled by feature_extractor)
  - PyG Data objects ready for GVP-GNN

Usage:
    builder = ProteinGraphBuilder(k=16, max_radius=22.0, rbf_bins=16)
    graph_data = builder.build(structure, features)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import distance_matrix
import warnings


@dataclass
class GraphConfig:
    """Configuration for graph construction."""
    k: int = 16              # number of nearest neighbours
    max_radius: float = 22.0 # maximum distance in Angstroms
    rbf_bins: int = 16       # number of RBF distance bins
    include_sequential_edges: bool = True  # add i→i+1, i→i+2 edges
    pad_to_full: bool = False  # ensure all k edges present (pad with self if needed)


class ProteinGraphBuilder:
    """Build kNN residue graphs from protein structures."""

    def __init__(self, config: GraphConfig = None):
        self.config = config or GraphConfig()

    def build(
        self,
        CA_coords: np.ndarray,      # (N, 3) Cα coordinates
        node_scalars: np.ndarray,   # (N, n_scalar) scalar node features
        node_vectors: np.ndarray,   # (N, n_vec, 3) vector node features
        residue_names: list[str],   # (N,) amino acid names
        seq_separation: np.ndarray = None,  # (N,) sequence position
    ) -> Data:
        """
        Build PyG Data object for a single protein.

        Returns:
            torch_geometric.data.Data with:
              - x_s, x_v: node features
              - edge_index, edge_attr_s, edge_attr_v: edge features
              - n_nodes: number of residues
        """
        N = CA_coords.shape[0]

        # Default sequence positions
        if seq_separation is None:
            seq_separation = np.arange(N)

        # Build kNN graph
        src, dst, distances = self._build_knn_graph(CA_coords)

        # Add sequential edges if requested
        if self.config.include_sequential_edges:
            src, dst, distances = self._add_sequential_edges(
                src, dst, distances, CA_coords, N
            )

        # Remove duplicate edges (keep shorter distance)
        src, dst, distances = self._deduplicate_edges(src, dst, distances)

        # Compute edge features
        edge_attr_s, edge_attr_v = self._compute_edge_features(
            src, dst, distances, CA_coords, node_vectors, residue_names, seq_separation
        )

        # Build PyG Data
        edge_index = torch.LongTensor([src, dst])
        x_s = torch.FloatTensor(node_scalars)
        x_v = torch.FloatTensor(node_vectors)

        data = Data(
            x_s=x_s,
            x_v=x_v,
            edge_index=edge_index,
            edge_attr_s=edge_attr_s,
            edge_attr_v=edge_attr_v,
            n_nodes=N,
        )

        return data

    def _build_knn_graph(
        self,
        CA_coords: np.ndarray,  # (N, 3)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build kNN graph from Cα coordinates.

        Returns:
            src, dst, distances: edge sources, destinations, distances
        """
        N = CA_coords.shape[0]
        dist_matrix = distance_matrix(CA_coords, CA_coords)

        # For each residue, find k nearest neighbours (excluding self)
        src_list, dst_list, dist_list = [], [], []

        for i in range(N):
            # Get distances to all other residues
            dists = dist_matrix[i, :]
            dists[i] = np.inf  # exclude self

            # Find k nearest within max_radius
            valid_mask = dists <= self.config.max_radius
            valid_indices = np.where(valid_mask)[0]
            valid_dists = dists[valid_indices]

            # Sort and take top k
            sorted_idx = np.argsort(valid_dists)
            k_to_use = min(self.config.k, len(sorted_idx))

            if k_to_use == 0:
                warnings.warn(f"Residue {i} has no neighbours within {self.config.max_radius}Å")
                continue

            for j_rel in range(k_to_use):
                j_abs = valid_indices[sorted_idx[j_rel]]
                src_list.append(i)
                dst_list.append(j_abs)
                dist_list.append(dists[j_abs])

        src = np.array(src_list, dtype=np.int64)
        dst = np.array(dst_list, dtype=np.int64)
        distances = np.array(dist_list, dtype=np.float32)

        return src, dst, distances

    def _add_sequential_edges(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        distances: np.ndarray,
        CA_coords: np.ndarray,
        N: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add edges between sequential residues (i→i+1, i→i+2) regardless of distance."""
        seq_src, seq_dst, seq_dist = [], [], []

        for i in range(N):
            # i → i+1
            if i + 1 < N:
                seq_src.append(i)
                seq_dst.append(i + 1)
                d = np.linalg.norm(CA_coords[i + 1] - CA_coords[i])
                seq_dist.append(d)

                # i+1 → i (undirected becomes bidirectional)
                seq_src.append(i + 1)
                seq_dst.append(i)
                seq_dist.append(d)

            # i → i+2
            if i + 2 < N:
                seq_src.append(i)
                seq_dst.append(i + 2)
                d = np.linalg.norm(CA_coords[i + 2] - CA_coords[i])
                seq_dist.append(d)

                # i+2 → i
                seq_src.append(i + 2)
                seq_dst.append(i)
                seq_dist.append(d)

        # Concatenate with existing edges
        src = np.concatenate([src, np.array(seq_src, dtype=np.int64)])
        dst = np.concatenate([dst, np.array(seq_dst, dtype=np.int64)])
        distances = np.concatenate([distances, np.array(seq_dist, dtype=np.float32)])

        return src, dst, distances

    def _deduplicate_edges(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        distances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove duplicate edges, keeping the one with smallest distance."""
        # Create edge tuples
        edges = set()
        edge_dict = {}  # (src, dst) → min_distance

        for s, d, dist in zip(src, dst, distances):
            key = (s, d)
            if key not in edge_dict or dist < edge_dict[key]:
                edge_dict[key] = dist

        # Reconstruct arrays
        src_out, dst_out, dist_out = [], [], []
        for (s, d), dist in edge_dict.items():
            src_out.append(s)
            dst_out.append(d)
            dist_out.append(dist)

        return (
            np.array(src_out, dtype=np.int64),
            np.array(dst_out, dtype=np.int64),
            np.array(dist_out, dtype=np.float32),
        )

    def _compute_edge_features(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        distances: np.ndarray,
        CA_coords: np.ndarray,
        node_vectors: np.ndarray,
        residue_names: list[str],
        seq_separation: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scalar and vector edge features.

        Returns:
            edge_attr_s (E, 40): scalar edge features
            edge_attr_v (E, 4, 3): vector edge features
        """
        E = len(src)
        N = CA_coords.shape[0]

        # Preallocate
        edge_attr_s = np.zeros((E, 40), dtype=np.float32)
        edge_attr_v = np.zeros((E, 4, 3), dtype=np.float32)

        # RBF distance encoding
        rbf_feat = self._rbf_encode(distances)  # (E, 16)
        edge_attr_s[:, :16] = rbf_feat

        # Relative orientation: rotation matrix element-wise
        rel_rot = np.zeros((E, 9), dtype=np.float32)
        for edge_idx, (i, j) in enumerate(zip(src, dst)):
            # Local frame at i
            R_i = node_vectors[i, :3, :]  # (3, 3) — first 3 vectors form frame
            # Local frame at j
            R_j = node_vectors[j, :3, :]  # (3, 3)
            # Relative: R_i^T @ R_j (flattened)
            rel_R = R_i.T @ R_j
            rel_rot[edge_idx] = rel_R.flatten()

        edge_attr_s[:, 16:25] = rel_rot

        # Multi-atom distances
        multi_dist = np.zeros((E, 4), dtype=np.float32)
        for edge_idx, (i, j) in enumerate(zip(src, dst)):
            # Cα-Cα (already have this)
            multi_dist[edge_idx, 0] = distances[edge_idx]
            # Cβ-Cβ — use virtual Cβ from node_vectors[*, 2, :]
            CB_i = node_vectors[i, 2, :]  # virtual Cβ unit direction
            CB_j = node_vectors[j, 2, :]
            # Need actual Cβ coords — approximate from Cα + direction
            CB_i_pos = CA_coords[i] + CB_i  # heuristic: unit step in Cβ direction
            CB_j_pos = CA_coords[j] + CB_j
            multi_dist[edge_idx, 1] = np.linalg.norm(CB_i_pos - CB_j_pos)
            # N-O (backbone N of i to C of j) — use backbone frame
            N_i = CA_coords[i] - node_vectors[i, 0, :] * 1.5  # heuristic: N is ~1.5Å from CA along frame
            O_j = CA_coords[j] + node_vectors[j, 1, :] * 1.2   # O is ~1.2Å from CA along frame
            multi_dist[edge_idx, 2] = np.linalg.norm(N_i - O_j)
            # Cα-Cβ distance for target node
            multi_dist[edge_idx, 3] = np.linalg.norm(CA_coords[i] - CB_j_pos)

        edge_attr_s[:, 25:29] = multi_dist

        # Sequence separation (binned)
        sep_bin = np.zeros((E, 4), dtype=np.float32)
        for edge_idx, (i, j) in enumerate(zip(src, dst)):
            sep = abs(i - j)
            if sep == 1:
                sep_bin[edge_idx, 0] = 1
            elif 2 <= sep <= 5:
                sep_bin[edge_idx, 1] = 1
            elif 6 <= sep <= 12:
                sep_bin[edge_idx, 2] = 1
            elif sep > 12:
                sep_bin[edge_idx, 3] = 1

        edge_attr_s[:, 29:33] = sep_bin

        # Physicochemical interaction flags
        H_BOND_DIST = 3.5
        HYDROPHOBIC_DIST = 5.0
        SALT_BRIDGE_DIST = 4.0
        HYDROPHOBIC_AAS = {'ILE', 'LEU', 'VAL', 'PHE', 'TRP', 'MET', 'ALA'}
        POSITIVE_AAS = {'ARG', 'LYS', 'HIS'}
        NEGATIVE_AAS = {'ASP', 'GLU'}

        for edge_idx, (i, j) in enumerate(zip(src, dst)):
            d = distances[edge_idx]
            res_i = residue_names[i]
            res_j = residue_names[j]

            # H-bond indicator (simplified: just distance threshold)
            h_bond = 1.0 if d < H_BOND_DIST else 0.0
            edge_attr_s[edge_idx, 33] = h_bond

            # Hydrophobic contact
            hydrophobic = (
                1.0 if d < HYDROPHOBIC_DIST
                and res_i in HYDROPHOBIC_AAS
                and res_j in HYDROPHOBIC_AAS
                else 0.0
            )
            edge_attr_s[edge_idx, 34] = hydrophobic

            # Salt bridge
            salt_bridge = (
                1.0 if d < SALT_BRIDGE_DIST
                and (
                    (res_i in POSITIVE_AAS and res_j in NEGATIVE_AAS) or
                    (res_i in NEGATIVE_AAS and res_j in POSITIVE_AAS)
                )
                else 0.0
            )
            edge_attr_s[edge_idx, 35] = salt_bridge

            # vdW contact score (heuristic: Lennard-Jones-like)
            vdw = max(0.0, 1.0 - (d / HYDROPHOBIC_DIST) ** 6)
            edge_attr_s[edge_idx, 36] = vdw

        # Remaining 3 columns unused (padding)
        edge_attr_s[:, 37:40] = 0.0

        # Vector edge features
        for edge_idx, (i, j) in enumerate(zip(src, dst)):
            # Unit displacement Cα→Cα
            diff = CA_coords[j] - CA_coords[i]
            edge_attr_v[edge_idx, 0, :] = diff / (np.linalg.norm(diff) + 1e-8)
            # Backbone frame vectors from residue i
            edge_attr_v[edge_idx, 1:4, :] = node_vectors[i, 0:3, :]  # first 3 vectors

        return torch.FloatTensor(edge_attr_s), torch.FloatTensor(edge_attr_v)

    def _rbf_encode(self, distances: np.ndarray) -> np.ndarray:
        """Radial basis function distance encoding."""
        d_min = 2.0
        d_max = self.config.max_radius
        n_bins = self.config.rbf_bins

        centers = np.linspace(d_min, d_max, n_bins)
        sigma = (d_max - d_min) / n_bins

        rbf = np.exp(-((distances[:, None] - centers) ** 2) / (2 * sigma ** 2))
        return rbf.astype(np.float32)
