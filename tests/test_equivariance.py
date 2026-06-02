"""
SE(3) Equivariance Tests

Critical tests to verify that geometric operations preserve rotation invariance.
If these tests fail, the model is not SE(3)-equivariant and will not generalize
across protein orientations.

Tests:
  - GVP layer equivariance under SO(3) rotations
  - Vector norms are invariant
  - Scalar features are invariant
  - Message passing preserves equivariance
"""

import torch
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R


class TestSO3Invariance:
    """Test that scalar operations are SO(3)-invariant (unchanged by rotations)."""

    def test_vector_norm_invariance(self):
        """Vector norms should be invariant to rotations."""
        # Create random vectors
        v = torch.randn(10, 6, 3)  # 10 nodes, 6 vectors, 3D

        # Random rotation matrix
        rotation = R.random().as_matrix()
        R_torch = torch.from_numpy(rotation).float()

        # Rotate vectors
        v_rotated = torch.einsum('ij,klj->kli', R_torch, v)  # (10, 6, 3)

        # Vector norms should be identical
        norm_original = torch.linalg.norm(v, dim=-1)  # (10, 6)
        norm_rotated = torch.linalg.norm(v_rotated, dim=-1)

        assert torch.allclose(norm_original, norm_rotated, atol=1e-5), \
            "Vector norms not invariant to rotation"

    def test_dot_product_invariance(self):
        """Dot products between vectors should be invariant."""
        v1 = torch.randn(5, 3)
        v2 = torch.randn(5, 3)

        # Compute dot products
        dot_original = torch.sum(v1 * v2, dim=-1)

        # Rotate both vectors
        rotation = R.random().as_matrix()
        R_torch = torch.from_numpy(rotation).float()

        v1_rot = torch.matmul(v1, R_torch.T)
        v2_rot = torch.matmul(v2, R_torch.T)

        dot_rotated = torch.sum(v1_rot * v2_rot, dim=-1)

        assert torch.allclose(dot_original, dot_rotated, atol=1e-5), \
            "Dot products not invariant to rotation"

    def test_distance_invariance(self):
        """Pairwise distances should be invariant."""
        positions = torch.randn(10, 3)

        # Compute distances
        dists_original = torch.cdist(positions, positions)  # (10, 10)

        # Rotate all positions
        rotation = R.random().as_matrix()
        R_torch = torch.from_numpy(rotation).float()
        positions_rot = torch.matmul(positions, R_torch.T)

        dists_rotated = torch.cdist(positions_rot, positions_rot)

        assert torch.allclose(dists_original, dists_rotated, atol=1e-5), \
            "Distances not invariant to rotation"


class TestGVPEquivariance:
    """Test GVP layer equivariance properties."""

    @pytest.fixture
    def sample_inputs(self):
        """Create sample scalar and vector features."""
        N = 10  # number of nodes
        s = torch.randn(N, 32)  # scalar features
        v = torch.randn(N, 4, 3)  # vector features (4 vectors per node)
        return s, v

    def test_gvp_scalar_invariance(self):
        """Scalar outputs of GVP should be invariant to input vector rotations."""
        from src.models.shared.gvp_primitives import GVP

        N = 10
        hidden_dim = 64

        gvp = GVP(node_s_in=32, node_v_in=4, hidden_dim=hidden_dim)

        # Original input
        s = torch.randn(N, 32)
        v = torch.randn(N, 4, 3)

        # Forward pass
        s_out_1, _ = gvp(s, v)

        # Rotate input vectors
        rotation = R.random().as_matrix()
        R_torch = torch.from_numpy(rotation).float()

        # v: (N, 4, 3) → rotate each 3D vector
        v_rotated = torch.einsum('ij,klj->kli', R_torch, v)

        # Forward pass with rotated input
        with torch.no_grad():
            s_out_2, _ = gvp(s, v_rotated)

        # Scalar outputs should be identical
        assert torch.allclose(s_out_1, s_out_2, atol=1e-5), \
            "GVP scalar output not invariant to vector rotation"

    def test_gvp_vector_equivariance(self):
        """Vector outputs of GVP should rotate the same way as inputs."""
        from src.models.shared.gvp_primitives import GVP

        N = 10
        hidden_dim = 64

        gvp = GVP(node_s_in=32, node_v_in=4, hidden_dim=hidden_dim)

        s = torch.randn(N, 32)
        v = torch.randn(N, 4, 3)

        # Original output
        with torch.no_grad():
            _, v_out_1 = gvp(s, v)  # (N, v_out, 3)

        # Rotate input
        rotation = R.random().as_matrix()
        R_torch = torch.from_numpy(rotation).float()
        v_rotated = torch.einsum('ij,klj->kli', R_torch, v)

        # Forward with rotated input
        with torch.no_grad():
            _, v_out_2_rot = gvp(s, v_rotated)

        # Rotate the original output the same way
        v_out_1_rotated = torch.einsum('ij,klj->kli', R_torch, v_out_1)

        # They should match (output rotates same as input)
        assert torch.allclose(v_out_1_rotated, v_out_2_rot, atol=1e-4), \
            "GVP vector output not equivariant to rotation"


class TestGraphEquivariance:
    """Test that graph operations preserve equivariance."""

    def test_knn_graph_rotation_invariance(self):
        """KNN graph connectivity should be invariant to rotations."""
        from src.data.graph_builder import build_knn_graph

        coords = torch.randn(20, 3)
        k = 5

        # Build graph on original coords
        edge_index_1 = build_knn_graph(coords, k=k)

        # Rotate coordinates
        rotation = R.random().as_matrix()
        R_torch = torch.from_numpy(rotation).float()
        coords_rotated = torch.matmul(coords, R_torch.T)

        # Build graph on rotated coords
        edge_index_2 = build_knn_graph(coords_rotated, k=k)

        # Graph structure should be identical
        assert torch.equal(edge_index_1, edge_index_2), \
            "KNN graph connectivity changed after rotation"

    def test_edge_features_invariance(self):
        """Edge scalar features (distances) should be invariant."""
        from src.data.graph_builder import compute_edge_features

        coords = torch.randn(10, 3)
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 4]
        ])

        # Compute edge features
        edge_features_1 = compute_edge_features(coords, edge_index)

        # Rotate coordinates
        rotation = R.random().as_matrix()
        R_torch = torch.from_numpy(rotation).float()
        coords_rotated = torch.matmul(coords, R_torch.T)

        # Compute edge features on rotated coords
        edge_features_2 = compute_edge_features(coords_rotated, edge_index)

        # Distance features should match
        assert torch.allclose(edge_features_1[:, :16], edge_features_2[:, :16], atol=1e-4), \
            "RBF distance features not invariant to rotation"


class TestTranslationInvariance:
    """Test that translation does not affect geometric features."""

    def test_distance_translation_invariance(self):
        """Distances should be invariant to translation."""
        coords = torch.randn(10, 3)
        translation = torch.randn(1, 3)

        dists_1 = torch.cdist(coords, coords)
        dists_2 = torch.cdist(coords + translation, coords + translation)

        assert torch.allclose(dists_1, dists_2, atol=1e-5), \
            "Distances not invariant to translation"

    def test_dihedral_translation_invariance(self):
        """Backbone dihedrals should be invariant to translation."""
        # Backbone atoms: N, CA, C (3 atoms define a plane + dihedral)
        positions = torch.randn(20, 3, 3)  # 20 residues, 3 atoms, 3D coords

        def compute_dihedral(p1, p2, p3, p4):
            """Compute dihedral angle between 4 points."""
            b1 = p2 - p1
            b2 = p3 - p2
            b3 = p4 - p3

            n1 = torch.cross(b1, b2)
            n2 = torch.cross(b2, b3)

            return torch.atan2(
                torch.sum(torch.cross(n1, n2) * b2 / torch.norm(b2, dim=-1, keepdim=True), dim=-1),
                torch.sum(n1 * n2, dim=-1)
            )

        # Compute dihedrals on original
        dihedral_1 = compute_dihedral(
            positions[:-3, 2],  # C of i-1
            positions[:-2, 0],  # N of i
            positions[:-2, 1],  # CA of i
            positions[:-2, 2],  # C of i
        )

        # Translate and compute
        translation = torch.randn(1, 1, 3)
        dihedral_2 = compute_dihedral(
            positions[:-3, 2] + translation,
            positions[:-2, 0] + translation,
            positions[:-2, 1] + translation,
            positions[:-2, 2] + translation,
        )

        assert torch.allclose(dihedral_1, dihedral_2, atol=1e-5), \
            "Dihedrals not invariant to translation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
