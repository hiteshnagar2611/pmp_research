"""
PLM Embedder: Get protein language model (ESM-2) embeddings.

ESM-2 (Evolutionary Scale Modeling) from Meta:
  - 1280-dimensional embeddings per residue
  - Trained on 2.7B protein sequences
  - Captures evolutionary and functional information
  - Fast inference (~1000 proteins/hour)

References:
  - Paper: "Language models of protein sequences at the scale of evolution"
  - Code: https://github.com/facebookresearch/esmfold
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
import pickle

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# Try to import ESM (optional dependency)
try:
    import esm
    HAS_ESM = True
except ImportError:
    HAS_ESM = False
    logger.warning("ESM not installed. Run: pip install fair-esm")


class ESM2Embedder:
    """
    Get ESM-2 embeddings for protein sequences.

    Uses pre-trained ESM-2 model to compute 1280-dim embeddings per residue.

    Args:
        model_name (str): ESM model name (default: 'esm2_t33_650M_UR50D')
        device (str): device to use (default: 'cuda' if available)
        cache_dir (str): directory to cache embeddings (default: None)
    """

    def __init__(
        self,
        model_name: str = 'esm2_t33_650M_UR50D',
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize ESM-2 embedder."""
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load model
        self._model = None
        self._alphabet = None

    @property
    def model(self):
        """Lazy load ESM-2 model."""
        if self._model is None:
            if not HAS_ESM:
                raise ImportError("ESM required for embeddings. Install: pip install fair-esm")

            logger.info(f"Loading {self.model_name}...")
            self._model, self._alphabet = esm.pretrained.load_model_and_alphabet_local(
                self.model_name
            )
            self._model = self._model.to(self.device)
            self._model.eval()

        return self._model

    @property
    def alphabet(self):
        """Get alphabet."""
        if self._alphabet is None:
            _ = self.model  # triggers loading
        return self._alphabet

    def get_cache_path(self, pdb_id: str) -> Path:
        """Get cache path for a PDB ID."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{pdb_id}_esm2.pkl"

    def load_from_cache(self, pdb_id: str) -> Optional[np.ndarray]:
        """Load embeddings from cache."""
        cache_path = self.get_cache_path(pdb_id)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                return data['embeddings']
            except Exception as e:
                logger.warning(f"Could not load cache for {pdb_id}: {e}")
        return None

    def save_to_cache(self, pdb_id: str, embeddings: np.ndarray):
        """Save embeddings to cache."""
        cache_path = self.get_cache_path(pdb_id)
        if cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({'embeddings': embeddings}, f)
            except Exception as e:
                logger.warning(f"Could not save cache for {pdb_id}: {e}")

    @torch.no_grad()
    def embed(
        self,
        sequence: str,
        pdb_id: Optional[str] = None,
        return_token_embeddings: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get ESM-2 embeddings for a sequence.

        Args:
            sequence: amino acid sequence
            pdb_id: optional PDB ID for caching
            return_token_embeddings: whether to return token-level embeddings

        Returns:
            embeddings: (L, 1280) per-residue embeddings
            contact_predictions: (L, L) optional contact map
        """
        # Check cache
        if pdb_id:
            cached = self.load_from_cache(pdb_id)
            if cached is not None:
                return cached, None

        # Prepare sequence
        sequence = sequence.upper()
        if not all(aa in self.alphabet.symbols for aa in sequence):
            logger.warning(f"Invalid characters in sequence, skipping special tokens")

        # Batch format for ESM
        data = [(pdb_id or "protein", sequence)]

        # Get batch converter
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        # Move to device
        batch_tokens = batch_tokens.to(self.device)

        # Forward pass
        model = self.model
        results = model(batch_tokens, repr_layers=[33])

        # Extract embeddings
        token_embeddings = results["representations"][33]  # (1, L+2, 1280)

        # Remove special tokens (CLS, EOS)
        token_embeddings = token_embeddings[0, 1:-1, :].cpu().numpy()  # (L, 1280)

        embeddings = token_embeddings.astype(np.float32)

        # Cache
        if pdb_id:
            self.save_to_cache(pdb_id, embeddings)

        return embeddings, None

    @torch.no_grad()
    def embed_batch(
        self,
        sequences: List[str],
        pdb_ids: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """
        Get embeddings for multiple sequences.

        Args:
            sequences: list of amino acid sequences
            pdb_ids: optional list of PDB IDs for caching

        Returns:
            embeddings_list: list of (L_i, 1280) embeddings
        """
        embeddings_list = []
        pdb_ids = pdb_ids or [None] * len(sequences)

        for sequence, pdb_id in zip(sequences, pdb_ids):
            embeddings, _ = self.embed(sequence, pdb_id)
            embeddings_list.append(embeddings)

        return embeddings_list


class DummyEmbedder:
    """
    Dummy embedder when ESM is not available.

    Returns random embeddings of correct shape.
    """

    def __init__(self, dim: int = 1280, **kwargs):
        """Initialize dummy embedder."""
        self.dim = dim

    def embed(
        self,
        sequence: str,
        pdb_id: Optional[str] = None,
        return_token_embeddings: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return random embeddings."""
        L = len(sequence)
        embeddings = np.random.randn(L, self.dim).astype(np.float32)
        return embeddings, None

    def embed_batch(
        self,
        sequences: List[str],
        pdb_ids: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """Return random embeddings for batch."""
        embeddings_list = [self.embed(seq)[0] for seq in sequences]
        return embeddings_list


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_embedder(
    use_esm: bool = True,
    cache_dir: Optional[str] = None,
) -> ESM2Embedder | DummyEmbedder:
    """
    Get appropriate embedder.

    Args:
        use_esm: whether to use ESM (fallback to dummy if unavailable)
        cache_dir: cache directory for embeddings

    Returns:
        embedder: ESM2Embedder or DummyEmbedder
    """
    if use_esm and HAS_ESM:
        return ESM2Embedder(cache_dir=cache_dir)
    else:
        logger.warning("Using DummyEmbedder - install ESM for real embeddings")
        return DummyEmbedder()


def embed_sequence(
    sequence: str,
    embedder: Optional[ESM2Embedder] = None,
) -> np.ndarray:
    """
    Embed a single sequence.

    Args:
        sequence: amino acid sequence
        embedder: embedder instance (creates new if None)

    Returns:
        embeddings: (L, 1280) embeddings
    """
    if embedder is None:
        embedder = get_embedder()

    embeddings, _ = embedder.embed(sequence)
    return embeddings


def embed_file(
    fasta_file: str,
    cache_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Embed sequences from FASTA file.

    Args:
        fasta_file: path to FASTA file
        cache_dir: cache directory

    Returns:
        embeddings_dict: {sequence_id: embeddings}
    """
    embedder = get_embedder(cache_dir=cache_dir)

    embeddings_dict = {}

    try:
        from Bio import SeqIO

        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_id = record.id
            sequence = str(record.seq)

            embeddings, _ = embedder.embed(sequence, pdb_id=seq_id)
            embeddings_dict[seq_id] = embeddings

    except ImportError:
        logger.error("BioPython required. Install: pip install biopython")

    return embeddings_dict


if __name__ == "__main__":
    # Test embedder
    print("PLM Embedder loaded successfully")
    print("\nKey classes:")
    print("- ESM2Embedder: real ESM-2 embeddings (requires fair-esm)")
    print("- DummyEmbedder: random embeddings for testing")

    if HAS_ESM:
        print("\n✓ ESM available - using real embeddings")
        embedder = ESM2Embedder(cache_dir="/tmp/esm_cache")
        seq = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTG"
        embeddings, _ = embedder.embed(seq)
        print(f"  Sequence length: {len(seq)}")
        print(f"  Embeddings shape: {embeddings.shape}")
    else:
        print("\n✗ ESM not available - install with: pip install fair-esm")
