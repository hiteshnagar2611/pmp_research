"""
GVP Layer Shape Tests

Verify that tensor shapes flow correctly through GVP layers.
Shape mismatches are common bugs in geometric deep learning.

Tests:
  - Input/output shapes match specifications
  - Batch dimension handling
  - Edge case handling (single node, single edge)
  - Message passing aggregation shapes
"""

import torch
import pytest
from typing import Tuple


class TestGVPLayerShapes:
    """Test GVP layer input/output shapes."""

    @pytest.fixture
    def gvp_layer(self):
        """Create a GVP layer for testing."""
        from src.models.dynamo.gvp_encoder import GVP
        return GVP(
            node_s_in=147,
            node_v_in=6,
            hidden_dim=256,
            hidden_v_out=16
        )

    def test_forward_shapes(self, gvp_layer):
        """Test basic forward pass shapes."""
        N = 10  # number of nodes
        s = torch.randn(N, 147)
        v = torch.randn(N, 6, 3)

        s_out, v_out = gvp_layer(s, v)

        # Check output shapes
        assert s_out.shape == (N, 256), f"Unexpected s_out shape: {s_out.shape}"
        assert v_out.shape == (N, 16, 3), f"Unexpected v_out shape: {v_out.shape}"

    def test_batch_shapes(self, gvp_layer):
        """Test with different batch sizes."""
        for N in [1, 5, 10, 100]:
            s = torch.randn(N, 147)
            v = torch.randn(N, 6, 3)

            s_out, v_out = gvp_layer(s, v)

            assert s_out.shape[0] == N, f"Batch size mismatch for N={N}"
            assert v_out.shape[0] == N, f"Batch size mismatch for N={N}"

    def test_variable_hidden_dims(self):
        """Test with different hidden dimensions."""
        from src.models.dynamo.gvp_encoder import GVP

        for hidden_dim in [64, 128, 256, 512]:
            for hidden_v in [4, 8, 16]:
                gvp = GVP(
                    node_s_in=147,
                    node_v_in=6,
                    hidden_dim=hidden_dim,
                    hidden_v_out=hidden_v
                )

                s = torch.randn(10, 147)
                v = torch.randn(10, 6, 3)

                s_out, v_out = gvp(s, v)

                assert s_out.shape == (10, hidden_dim)
                assert v_out.shape == (10, hidden_v, 3)


class TestGVPEncoderShapes:
    """Test full GVP encoder pipeline."""

    @pytest.fixture
    def gvp_encoder(self):
        """Create a GVP encoder."""
        from src.models.dynamo.gvp_encoder import GVPEncoder
        return GVPEncoder(
            node_s_dim=147,
            node_v_dim=6,
            hidden_s_dim=256,
            hidden_v_dim=16,
            n_layers=3
        )

    def test_encoder_shapes(self, gvp_encoder):
        """Test encoder output shapes."""
        N = 20
        s = torch.randn(N, 147)
        v = torch.randn(N, 6, 3)
        edge_index = torch.tensor([[i, (i+1) % N] for i in range(N)])

        output = gvp_encoder(s, v, edge_index.t().contiguous())

        # Encoder output is aggregated representation
        assert output.shape == (N, 256), f"Unexpected encoder output shape: {output.shape}"

    def test_encoder_with_different_graph_sizes(self, gvp_encoder):
        """Test encoder with varying graph sizes."""
        for N in [5, 10, 50, 200]:
            s = torch.randn(N, 147)
            v = torch.randn(N, 6, 3)

            # Create k-NN graph (k=5)
            k = min(5, N - 1)
            edge_index_list = []
            for i in range(N):
                neighbors = torch.randperm(N)[:k]
                for j in neighbors:
                    if i != j:
                        edge_index_list.append([i, j])

            edge_index = torch.tensor(edge_index_list).t().contiguous() if edge_index_list else torch.zeros((2, 0), dtype=torch.long)

            output = gvp_encoder(s, v, edge_index)

            assert output.shape == (N, 256), f"Shape mismatch for N={N}"


class TestMessagePassingShapes:
    """Test message passing layer shapes."""

    @pytest.fixture
    def mp_layer(self):
        """Create a message passing layer."""
        from src.models.dynamo.gvp_encoder import GVPConvLayer
        return GVPConvLayer(
            hidden_s_dim=256,
            hidden_v_dim=16,
            edge_s_dim=40,
            edge_v_dim=4
        )

    def test_message_passing_forward(self, mp_layer):
        """Test message passing forward pass."""
        N = 10
        E = 20  # number of edges

        h_s = torch.randn(N, 256)
        h_v = torch.randn(N, 16, 3)
        edge_index = torch.randint(0, N, (2, E))
        edge_s = torch.randn(E, 40)
        edge_v = torch.randn(E, 4, 3)

        h_s_out, h_v_out = mp_layer(h_s, h_v, edge_index, edge_s, edge_v)

        # Output shapes should match input node shapes
        assert h_s_out.shape == (N, 256), f"h_s_out shape: {h_s_out.shape}"
        assert h_v_out.shape == (N, 16, 3), f"h_v_out shape: {h_v_out.shape}"

    def test_scatter_aggregation(self):
        """Test that scatter aggregation handles shapes correctly."""
        from torch_scatter import scatter_add

        N = 10
        E = 20

        # Random message tensor
        msg = torch.randn(E, 256)

        # Random destination indices
        dst = torch.randint(0, N, (E,))

        # Scatter aggregation
        aggregated = scatter_add(msg, dst, dim=0, dim_size=N)

        assert aggregated.shape == (N, 256), f"Scatter output shape: {aggregated.shape}"


class TestFeatureFusionShapes:
    """Test feature fusion layer shapes."""

    @pytest.fixture
    def fusion_layer(self):
        """Create feature fusion layer."""
        from src.models.dynamo.fusion import FeatureFusion
        return FeatureFusion(
            plm_dim=1280,
            struct_dim=19,
            vec_dim=6,
            hidden_dim=128
        )

    def test_fusion_forward(self, fusion_layer):
        """Test fusion layer forward pass."""
        N = 20

        plm_feat = torch.randn(N, 1280)
        struct_feat = torch.randn(N, 19)
        vec_feat = torch.randn(N, 6, 3)

        fused = fusion_layer(plm_feat, struct_feat, vec_feat)

        # Fused output should be scalar features only
        assert fused.shape == (N, 128), f"Fused shape: {fused.shape}"


class TestAttentionShapes:
    """Test attention layer shapes."""

    @pytest.fixture
    def cross_attention(self):
        """Create cross-attention layer."""
        from src.models.dynamo.cross_attention import StructureDynamicsCrossAttention
        return StructureDynamicsCrossAttention(
            hidden_dim=256,
            n_heads=8,
            dropout=0.1
        )

    def test_cross_attention_forward(self, cross_attention):
        """Test cross-attention forward pass."""
        N = 20

        Q = torch.randn(N, 256)  # geometry path (query)
        K = torch.randn(N, 256)  # dynamics path (key)
        V = torch.randn(N, 256)  # dynamics path (value)

        out = cross_attention(Q, K, V)

        assert out.shape == (N, 256), f"Attention output shape: {out.shape}"

    def test_attention_head_split(self):
        """Test that attention correctly splits into heads."""
        from src.models.dynamo.cross_attention import StructureDynamicsCrossAttention

        N = 10
        n_heads = 8
        hidden_dim = 256
        d_head = hidden_dim // n_heads

        attn = StructureDynamicsCrossAttention(hidden_dim, n_heads)

        Q = torch.randn(N, hidden_dim)
        K = torch.randn(N, hidden_dim)
        V = torch.randn(N, hidden_dim)

        # Project to heads
        Q_heads = attn.W_q(Q).reshape(N, n_heads, d_head)
        K_heads = attn.W_k(K).reshape(N, n_heads, d_head)

        assert Q_heads.shape == (N, n_heads, d_head)
        assert K_heads.shape == (N, n_heads, d_head)


class TestClassifierHeadShapes:
    """Test final classification head shapes."""

    def test_classifier_forward(self):
        """Test classifier head."""
        from src.models.dynamo.classifier import ResidueClassifier

        classifier = ResidueClassifier(hidden_dim=256, n_classes=1)

        N = 20
        H = torch.randn(N, 256)

        logits = classifier(H)

        assert logits.shape == (N, 1), f"Classifier output shape: {logits.shape}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_node(self):
        """Test with single node."""
        from src.models.dynamo.gvp_encoder import GVP

        gvp = GVP(147, 6, 256, 16)

        s = torch.randn(1, 147)
        v = torch.randn(1, 6, 3)

        s_out, v_out = gvp(s, v)

        assert s_out.shape == (1, 256)
        assert v_out.shape == (1, 16, 3)

    def test_single_edge(self):
        """Test with single edge."""
        from src.models.dynamo.gvp_encoder import GVPConvLayer

        mp = GVPConvLayer(256, 16, 40, 4)

        h_s = torch.randn(2, 256)
        h_v = torch.randn(2, 16, 3)
        edge_index = torch.tensor([[0], [1]])
        edge_s = torch.randn(1, 40)
        edge_v = torch.randn(1, 4, 3)

        h_s_out, h_v_out = mp(h_s, h_v, edge_index, edge_s, edge_v)

        assert h_s_out.shape == (2, 256)
        assert h_v_out.shape == (2, 16, 3)

    def test_no_edges(self):
        """Test with no edges (single component)."""
        from src.models.dynamo.gvp_encoder import GVPConvLayer

        mp = GVPConvLayer(256, 16, 40, 4)

        h_s = torch.randn(5, 256)
        h_v = torch.randn(5, 16, 3)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_s = torch.randn(0, 40)
        edge_v = torch.randn(0, 4, 3)

        h_s_out, h_v_out = mp(h_s, h_v, edge_index, edge_s, edge_v)

        # With no edges, output should be zeros or identity
        assert h_s_out.shape == (5, 256)
        assert h_v_out.shape == (5, 16, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
