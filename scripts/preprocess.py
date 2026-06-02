#!/usr/bin/env python3
"""
Preprocess PMP data: build graphs, extract features, compute embeddings.

Pipeline:
    1. Load PDB files and sequences
    2. Compute ESM-2 embeddings
    3. Extract structural features
    4. Process MD trajectories (if available)
    5. Build kNN graphs with edge features
    6. Create train/val/test splits
    7. Cache everything for fast loading

Usage:
    python scripts/preprocess.py --pdb-dir data/raw/pdb --output-dir data/processed
    python scripts/preprocess.py --fasta-dir data/raw/sequences --n-workers 8
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional
import multiprocessing as mp
from tqdm import tqdm
import pickle

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DataPreprocessor:
    """Preprocess PMP data."""

    def __init__(
        self,
        pdb_dir: str,
        output_dir: str,
        n_workers: int = 4,
    ):
        """Initialize preprocessor."""
        self.pdb_dir = Path(pdb_dir)
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers

        # Create output directories
        self.processed_dir = self.output_dir / 'processed'
        self.graph_dir = self.processed_dir / 'graphs'
        self.embedding_dir = self.processed_dir / 'embeddings'
        self.feature_dir = self.processed_dir / 'features'

        for d in [self.processed_dir, self.graph_dir, self.embedding_dir, self.feature_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def preprocess_all(self):
        """Run full preprocessing pipeline."""
        # Get PDB files
        pdb_files = list(self.pdb_dir.glob('*.pdb'))
        logger.info(f"Found {len(pdb_files)} PDB files")

        # Step 1: Load sequences and compute embeddings
        logger.info("Step 1: Computing ESM-2 embeddings...")
        self._compute_embeddings(pdb_files)

        # Step 2: Extract structural features
        logger.info("Step 2: Extracting structural features...")
        self._extract_features(pdb_files)

        # Step 3: Build graphs
        logger.info("Step 3: Building kNN graphs...")
        self._build_graphs(pdb_files)

        # Step 4: Create splits
        logger.info("Step 4: Creating train/val/test splits...")
        self._create_splits(pdb_files)

        logger.info("Preprocessing complete!")

    def _compute_embeddings(self, pdb_files: List[Path]):
        """Compute ESM-2 embeddings for all proteins."""
        from src.data.plm_embedder import get_embedder

        embedder = get_embedder(cache_dir=str(self.embedding_dir))

        for pdb_file in tqdm(pdb_files, desc="Computing embeddings"):
            try:
                pdb_id = pdb_file.stem

                # Load sequence from PDB
                sequence = self._extract_sequence_from_pdb(pdb_file)
                if sequence is None:
                    logger.warning(f"Could not extract sequence from {pdb_file}")
                    continue

                # Embed
                embeddings, _ = embedder.embed(sequence, pdb_id=pdb_id)

                # Save
                embedding_file = self.embedding_dir / f'{pdb_id}_embeddings.pkl'
                with open(embedding_file, 'wb') as f:
                    pickle.dump({'embeddings': embeddings}, f)

            except Exception as e:
                logger.error(f"Error processing {pdb_file}: {e}")

    def _extract_features(self, pdb_files: List[Path]):
        """Extract structural features."""
        from src.data.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor()

        for pdb_file in tqdm(pdb_files, desc="Extracting features"):
            try:
                pdb_id = pdb_file.stem

                # Load structure
                coords, sequence = self._load_pdb_structure(pdb_file)
                if coords is None:
                    logger.warning(f"Could not load structure from {pdb_file}")
                    continue

                # Extract features
                features = extractor.extract_all(coords, sequence)

                # Save
                feature_file = self.feature_dir / f'{pdb_id}_features.pkl'
                with open(feature_file, 'wb') as f:
                    pickle.dump(features, f)

            except Exception as e:
                logger.error(f"Error processing {pdb_file}: {e}")

    def _build_graphs(self, pdb_files: List[Path]):
        """Build kNN graphs for all proteins."""
        from src.data.graph_builder import GraphBuilder

        builder = GraphBuilder(k=16)

        for pdb_file in tqdm(pdb_files, desc="Building graphs"):
            try:
                pdb_id = pdb_file.stem

                # Load structure
                coords, sequence = self._load_pdb_structure(pdb_file)
                if coords is None:
                    continue

                # Load features
                feature_file = self.feature_dir / f'{pdb_id}_features.pkl'
                if not feature_file.exists():
                    logger.warning(f"Features not found for {pdb_id}")
                    continue

                with open(feature_file, 'rb') as f:
                    features = pickle.load(f)

                # Build graph
                graph = builder.build_graph(coords, sequence, features)

                # Save
                graph_file = self.graph_dir / f'{pdb_id}.pkl'
                with open(graph_file, 'wb') as f:
                    pickle.dump(graph, f)

            except Exception as e:
                logger.error(f"Error processing {pdb_file}: {e}")

    def _create_splits(self, pdb_files: List[Path]):
        """Create train/val/test splits."""
        pdb_ids = [f.stem for f in pdb_files]
        n_proteins = len(pdb_ids)

        # Shuffle
        np.random.shuffle(pdb_ids)

        # Split
        train_frac = 0.8
        val_frac = 0.1
        test_frac = 0.1

        n_train = int(n_proteins * train_frac)
        n_val = int(n_proteins * val_frac)

        train_ids = pdb_ids[:n_train]
        val_ids = pdb_ids[n_train:n_train+n_val]
        test_ids = pdb_ids[n_train+n_val:]

        # Save
        for split_name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
            split_file = self.processed_dir / f'{split_name}_split.txt'
            with open(split_file, 'w') as f:
                for pdb_id in ids:
                    f.write(f"{pdb_id}\n")

            logger.info(f"{split_name.upper()}: {len(ids)} proteins")

    @staticmethod
    def _extract_sequence_from_pdb(pdb_file: Path) -> Optional[str]:
        """Extract sequence from PDB file."""
        try:
            from Bio import PDB

            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_file.stem, str(pdb_file))

            # Get CA coordinates for each residue
            ppb = PDB.PPBuilder()
            pp_list = ppb.build_peptides(structure)

            if not pp_list:
                return None

            sequence = pp_list[0].get_sequence()
            return str(sequence)

        except Exception as e:
            logger.warning(f"Could not extract sequence from {pdb_file}: {e}")
            return None

    @staticmethod
    def _load_pdb_structure(pdb_file: Path) -> tuple:
        """Load Cα coordinates and sequence from PDB."""
        try:
            from Bio import PDB

            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_file.stem, str(pdb_file))

            # Get CA coordinates
            ca_coords = []
            sequence = []

            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            ca_coords.append(residue['CA'].coord)
                            seq = PDB.Polypeptide.three_to_one(residue.resname)
                            sequence.append(seq)

            if not ca_coords:
                return None, None

            coords = np.array(ca_coords)
            sequence = ''.join(sequence)

            return coords, sequence

        except Exception as e:
            logger.warning(f"Could not load structure from {pdb_file}: {e}")
            return None, None


def main():
    """Main preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocess PMP data")

    parser.add_argument(
        '--pdb-dir', type=str, default='data/raw/pdb',
        help='Directory with PDB files'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data',
        help='Output directory'
    )
    parser.add_argument(
        '--n-workers', type=int, default=4,
        help='Number of workers for parallel processing'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from checkpoint'
    )

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = DataPreprocessor(
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
    )

    # Run preprocessing
    logger.info("Starting PMP data preprocessing...")
    logger.info(f"PDB directory: {args.pdb_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        preprocessor.preprocess_all()
        logger.info("✓ Preprocessing complete!")
    except KeyboardInterrupt:
        logger.warning("Preprocessing interrupted by user")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
