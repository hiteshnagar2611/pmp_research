#!/usr/bin/env python3
"""
Download and prepare data for PMP research.

Supports:
  - Downloading PDB files from RCSB
  - Downloading OPM annotations
  - Creating dummy data for testing
  - Generating binding labels

Usage:
    python scripts/download_data.py --dummy              # Create test data (10 proteins)
    python scripts/download_data.py --pdb-ids 1BX7 2L6P # Download specific PDB files
    python scripts/download_data.py --all                # Everything (PDB + OPM + dummy)
    python scripts/download_data.py --opm                # Download OPM only
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Try to import optional dependencies
try:
    from Bio.PDB import PDBList
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    logger.warning("BioPython not installed. Install with: pip install biopython")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("Requests not installed. Install with: pip install requests")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("Pandas not installed. Install with: pip install pandas")


class DataDownloader:
    """Download and prepare data for PMP research."""

    def __init__(self, output_dir: str = 'data/raw'):
        """Initialize downloader."""
        self.output_dir = Path(output_dir)
        self.pdb_dir = self.output_dir / 'pdb'
        self.opm_dir = self.output_dir / 'opm'
        self.md_dir = self.output_dir / 'md'
        self.labels_dir = self.output_dir / 'labels'

        # Create directories
        for d in [self.pdb_dir, self.opm_dir, self.labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def download_pdb_files(self, pdb_ids: List[str]) -> None:
        """Download PDB files from RCSB PDB."""
        if not HAS_BIOPYTHON:
            logger.error("BioPython required for PDB download. Install with: pip install biopython")
            return

        logger.info(f"Downloading {len(pdb_ids)} PDB files from RCSB...")

        pdbl = PDBList()

        for pdb_id in pdb_ids:
            try:
                logger.info(f"  Downloading {pdb_id}...")
                pdbl.retrieve_pdb_file(
                    pdb_id,
                    pdir=str(self.pdb_dir),
                    file_format='pdb'
                )
            except Exception as e:
                logger.warning(f"  Failed to download {pdb_id}: {e}")

        logger.info(f"✓ Downloaded PDB files to {self.pdb_dir}")

    def download_opm_data(self, n_proteins: int = 100) -> None:
        """Download OPM (Orientations of Proteins in Membranes) data."""
        if not HAS_REQUESTS:
            logger.error("Requests required for OPM download. Install with: pip install requests")
            return

        logger.info(f"Downloading OPM annotations (first {n_proteins})...")

        try:
            response = requests.get('https://opm.psu.edu/api/proteins', timeout=30)
            proteins = response.json()

            downloaded = 0
            for protein in proteins[:n_proteins]:
                try:
                    pdb_id = protein['pdb_id'].lower()

                    opm_data = {
                        'pdb_id': protein['pdb_id'],
                        'membrane_normal': protein.get('normal', [0, 0, 1]),
                        'membrane_thickness': protein.get('thickness', 30),
                        'membrane_center': protein.get('center'),
                    }

                    with open(self.opm_dir / f'{pdb_id}_opm.json', 'w') as f:
                        json.dump(opm_data, f, indent=2)

                    downloaded += 1

                except Exception as e:
                    logger.warning(f"  Failed to process {protein.get('pdb_id')}: {e}")

            logger.info(f"✓ Downloaded {downloaded} OPM annotations to {self.opm_dir}")

        except Exception as e:
            logger.error(f"Failed to download OPM data: {e}")

    def create_dummy_pdb(self, pdb_id: str, n_residues: int = 250) -> None:
        """Create a dummy PDB file for testing."""
        pdb_content = f"HEADER Generated dummy PMP {pdb_id}\n"
        pdb_content += f"REMARK Test protein with {n_residues} residues\n"

        for res_idx in range(n_residues):
            # Random Cα coordinates
            x = np.random.randn() * 10 + res_idx
            y = np.random.randn() * 10
            z = np.random.randn() * 10

            # PDB format
            pdb_content += (
                f"ATOM  {res_idx+1:5d}  CA  ALA A{res_idx+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )

        pdb_content += "END\n"

        with open(self.pdb_dir / f'{pdb_id}.pdb', 'w') as f:
            f.write(pdb_content)

    def create_dummy_opm(self, pdb_id: str) -> None:
        """Create dummy OPM annotation."""
        opm_data = {
            'pdb_id': pdb_id,
            'membrane_normal': [0.0, 0.0, 1.0],
            'membrane_thickness': 30.0,
            'membrane_center': [0.0, 0.0, 0.0],
        }

        with open(self.opm_dir / f'{pdb_id}_opm.json', 'w') as f:
            json.dump(opm_data, f, indent=2)

    def create_dummy_labels(self, pdb_ids: List[str], binding_fraction: float = 0.15) -> None:
        """Create dummy binding labels."""
        if not HAS_PANDAS:
            logger.error("Pandas required. Install with: pip install pandas")
            return

        logger.info(f"Creating binding labels for {len(pdb_ids)} proteins...")

        labels_data = []

        for pdb_id in pdb_ids:
            # Random number of residues
            n_residues = np.random.randint(150, 400)

            # ~binding_fraction are binding residues
            binding_labels = np.random.binomial(1, binding_fraction, n_residues)

            for res_idx, label in enumerate(binding_labels):
                labels_data.append({
                    'pdb_id': pdb_id,
                    'residue_index': res_idx,
                    'binding_label': int(label)
                })

        labels_df = pd.DataFrame(labels_data)
        labels_file = self.labels_dir / 'binding_labels.csv'
        labels_df.to_csv(labels_file, index=False)

        logger.info(f"✓ Created {len(labels_df)} label entries")
        logger.info(f"  Binding fraction: {labels_df['binding_label'].mean():.1%}")
        logger.info(f"  Unique proteins: {labels_df['pdb_id'].nunique()}")
        logger.info(f"  Saved to {labels_file}")

    def create_dummy_data(self, n_proteins: int = 20) -> None:
        """Create complete dummy dataset for testing."""
        logger.info(f"Creating dummy data for {n_proteins} proteins...")

        pdb_ids = [f'TEST{i:04d}' for i in range(n_proteins)]

        # Create dummy PDB files
        logger.info("  Creating PDB files...")
        for pdb_id in pdb_ids:
            n_residues = np.random.randint(150, 350)
            self.create_dummy_pdb(pdb_id, n_residues)

        # Create dummy OPM data
        logger.info("  Creating OPM annotations...")
        for pdb_id in pdb_ids:
            self.create_dummy_opm(pdb_id)

        # Create dummy labels
        if HAS_PANDAS:
            self.create_dummy_labels(pdb_ids)

        logger.info(f"✓ Created dummy dataset with {n_proteins} proteins")
        logger.info(f"  PDB files: {self.pdb_dir}")
        logger.info(f"  OPM files: {self.opm_dir}")
        logger.info(f"  Labels: {self.labels_dir / 'binding_labels.csv'}")

    def verify_data(self) -> None:
        """Verify downloaded data."""
        logger.info("Verifying data...")

        pdb_files = list(self.pdb_dir.glob('*.pdb'))
        opm_files = list(self.opm_dir.glob('*.json'))
        labels_file = self.labels_dir / 'binding_labels.csv'

        logger.info(f"  PDB files: {len(pdb_files)}")
        logger.info(f"  OPM files: {len(opm_files)}")

        if labels_file.exists() and HAS_PANDAS:
            labels_df = pd.read_csv(labels_file)
            logger.info(f"  Label entries: {len(labels_df)}")
            logger.info(f"  Unique proteins: {labels_df['pdb_id'].nunique()}")
            logger.info(f"  Binding fraction: {labels_df['binding_label'].mean():.1%}")

        logger.info("✓ Data verification complete")


def main():
    """Main download script."""
    parser = argparse.ArgumentParser(
        description="Download and prepare data for PMP research"
    )

    parser.add_argument(
        '--dummy',
        action='store_true',
        help='Create dummy data for testing (10 proteins)'
    )
    parser.add_argument(
        '--pdb-ids',
        nargs='+',
        default=None,
        help='List of PDB IDs to download (e.g., 1BX7 2L6P 4DKL)'
    )
    parser.add_argument(
        '--opm',
        action='store_true',
        help='Download OPM annotations'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download everything (dummy + PDB + OPM)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for data'
    )
    parser.add_argument(
        '--n-proteins',
        type=int,
        default=20,
        help='Number of proteins for dummy data'
    )

    args = parser.parse_args()

    # Create downloader
    downloader = DataDownloader(output_dir=args.output_dir)

    # Default: create dummy data if no args
    if not any([args.dummy, args.pdb_ids, args.opm, args.all]):
        args.dummy = True

    # Download data
    if args.all:
        downloader.create_dummy_data(n_proteins=args.n_proteins)
        downloader.download_opm_data()
        if args.pdb_ids:
            downloader.download_pdb_files(args.pdb_ids)

    elif args.dummy:
        downloader.create_dummy_data(n_proteins=args.n_proteins)

    elif args.pdb_ids:
        downloader.download_pdb_files(args.pdb_ids)

    if args.opm:
        downloader.download_opm_data()

    # Verify
    downloader.verify_data()

    logger.info("\n" + "="*70)
    logger.info("✓ Data download complete!")
    logger.info("="*70)
    logger.info(f"Next steps:")
    logger.info(f"  1. Run preprocessing: python scripts/preprocess.py")
    logger.info(f"  2. Start training: python scripts/train_phase1.py")


if __name__ == "__main__":
    main()
