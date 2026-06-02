"""
Pytest configuration and shared fixtures for all tests.

Provides:
  - Device selection (GPU/CPU)
  - Random seed initialization
  - Temporary directories
  - Mock data generators
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile


# ─────────────────────────────────────────────────────────────────────────────
# Device Setup
# ─────────────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Configure pytest."""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Select device
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"Running tests on device: {config.device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")


@pytest.fixture
def device():
    """Get the device to run tests on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Random Seed Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def set_seed():
    """Reset random seeds before each test."""
    torch.manual_seed(42)
    np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Temporary Directories
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def data_dir(tmp_dir):
    """Create a data directory structure for testing."""
    raw_dir = tmp_dir / "raw"
    processed_dir = tmp_dir / "processed"
    
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    
    (raw_dir / "pdb").mkdir()
    (raw_dir / "opm").mkdir()
    (raw_dir / "labels").mkdir()
    
    (processed_dir / "graphs").mkdir()
    (processed_dir / "embeddings").mkdir()
    
    return tmp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Mock Data Generators
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_protein():
    """Generate sample protein data."""
    N = 250  # residues
    
    return {
        'pdb_id': 'TEST0001',
        'coords': np.random.randn(N, 3).astype(np.float32),
        'sequence': ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), N)),
        'binding_labels': np.random.binomial(1, 0.15, N),  # ~15% binding
        'rmsf': np.random.exponential(1.0, N),
        'plm_embedding': np.random.randn(N, 1280).astype(np.float32),
    }


@pytest.fixture
def sample_batch(device):
    """Generate sample batch for training."""
    B = 4  # batch size
    N = 250  # max sequence length
    
    return {
        'H_static': torch.randn(B, N, 256, device=device),
        'H_snapshots': torch.randn(5, B, N, 256, device=device),  # 5 MD snapshots
        'rmsf': torch.randn(B, N, 1, device=device).abs() + 0.5,
        'depth': torch.randn(B, N, 1, device=device),
        'kd': torch.randn(B, N, 1, device=device),
        'charge': torch.randn(B, N, 1, device=device),
        'sasa': torch.rand(B, N, 1, device=device),
        'targets': torch.bernoulli(torch.full((B, N), 0.15)).long(),
    }


@pytest.fixture
def sample_graph():
    """Generate sample protein graph."""
    N = 50
    
    # k-NN graph
    k = 5
    edges = []
    for i in range(N):
        neighbors = np.random.choice(N, k, replace=False)
        for j in neighbors:
            if i != j:
                edges.append([i, j])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return {
        'edge_index': edge_index,
        'num_nodes': N,
        'num_edges': edge_index.shape[1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_gvp(device):
    """Create a small GVP layer for testing."""
    from src.models.dynamo.gvp_encoder import GVP
    
    gvp = GVP(
        node_s_in=64,
        node_v_in=4,
        hidden_dim=128,
        hidden_v_out=8
    ).to(device)
    
    return gvp


@pytest.fixture
def small_encoder(device):
    """Create a small GVP encoder for testing."""
    from src.models.dynamo.gvp_encoder import GVPEncoder
    
    encoder = GVPEncoder(
        node_s_dim=64,
        node_v_dim=4,
        hidden_s_dim=128,
        hidden_v_dim=8,
        n_layers=2
    ).to(device)
    
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# Markers
# ─────────────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "equivariance: marks equivariance tests"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pytest Options
# ─────────────────────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--device", action="store", default="auto",
        help="Device to run tests on: cpu, cuda, or auto (default: auto)"
    )
    parser.addoption(
        "--slow", action="store_true",
        help="Run slow tests"
    )


@pytest.fixture
def setup_device(request):
    """Setup device based on command-line option."""
    device_opt = request.config.getoption("--device")
    
    if device_opt == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_opt
    
    return torch.device(device)
