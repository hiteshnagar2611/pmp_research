#!/usr/bin/env python3
"""
Generate proteins with PMPGen: de novo peripheral membrane protein design.

Generation pipeline:
    1. Load trained PMPGen model
    2. Sample from scaffold templates
    3. Run iterative denoising
    4. Validate with ESMFold + DynaMo + Rosetta (optional)
    5. Save generated structures and sequences

Usage:
    python scripts/generate.py --checkpoint outputs/pmpgen_best.pt --n-generate 10
    python scripts/generate.py --scaffold pdb/query.pdb --n-samples 5 --validate
    python scripts/generate.py --binding-pattern 0.3 --output outputs/generated
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict
import pickle

import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ProteinGenerator:
    """Generate proteins with PMPGen."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        output_dir: str = 'outputs/generated',
    ):
        """Initialize generator."""
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'structures').mkdir(exist_ok=True)
        (self.output_dir / 'sequences').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)

        # Load model
        self.model = self._load_model()

    def _load_model(self):
        """Load PMPGen model."""
        from src.models.pmpgen import PMPGen

        logger.info(f"Loading model from {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Create model
        model = PMPGen(
            n_res_in=256,
            n_res_out=256,
            hidden_dim=256,
            n_layers=6,
        )

        # Load weights
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()

        logger.info("✓ Model loaded successfully")
        return model

    def generate_from_scaffold(
        self,
        scaffold_coords: np.ndarray,
        n_samples: int = 1,
        membrane_normal: Optional[np.ndarray] = None,
        binding_mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Generate proteins from scaffold template.

        Args:
            scaffold_coords: (N, 3) scaffold coordinates
            n_samples: number of samples to generate
            membrane_normal: (3,) membrane orientation
            binding_mask: (N, 1) binding region specification

        Returns:
            results: generation results
        """
        B = n_samples
        N = scaffold_coords.shape[0]

        # Prepare conditioning
        conditioning = {
            'scaffold_coords': torch.from_numpy(
                np.repeat(scaffold_coords[np.newaxis], B, axis=0)
            ).float().to(self.device),
            'membrane_normal': torch.from_numpy(
                membrane_normal or np.array([0., 0., 1.])
            ).float().to(self.device),
            'anchor_mask': torch.zeros(B, N, 1).to(self.device),  # no anchors
        }

        if binding_mask is not None:
            conditioning['binding_mask'] = torch.from_numpy(
                np.repeat(binding_mask[np.newaxis], B, axis=0)
            ).float().to(self.device)

        # Generate
        logger.info(f"Generating {n_samples} proteins...")
        results = self.model.generate(
            conditioning=conditioning,
            num_samples=n_samples,
            verbose=True,
        )

        return results

    def generate_random(
        self,
        n_residues: int = 150,
        n_samples: int = 1,
        binding_fraction: float = 0.15,
    ) -> Dict:
        """
        Generate proteins from random scaffold.

        Args:
            n_residues: number of residues
            n_samples: number of samples
            binding_fraction: fraction of residues to bind

        Returns:
            results: generation results
        """
        logger.info(f"Generating {n_samples} random proteins ({n_residues} residues)...")

        # Random scaffold coordinates
        scaffold_coords = np.random.randn(n_residues, 3).astype(np.float32) * 10

        # Random binding mask
        binding_mask = np.random.binomial(1, binding_fraction, (n_residues, 1)).astype(np.float32)

        results = self.generate_from_scaffold(
            scaffold_coords=scaffold_coords,
            n_samples=n_samples,
            binding_mask=binding_mask,
        )

        return results

    def save_results(
        self,
        results: Dict,
        sample_ids: Optional[list] = None,
    ):
        """
        Save generated proteins.

        Args:
            results: generation results
            sample_ids: optional sample identifiers
        """
        B = results['coords'].shape[0]

        for i in range(B):
            sample_id = sample_ids[i] if sample_ids else f"sample_{i:03d}"

            # Save coordinates (PDB-like format)
            coords = results['coords'][i].numpy()
            sequence = self._tokens_to_sequence(results['sequences'][i])

            self._save_pdb(
                coords,
                sequence,
                self.output_dir / 'structures' / f"{sample_id}.pdb"
            )

            # Save sequence
            self._save_sequence(
                sequence,
                self.output_dir / 'sequences' / f"{sample_id}.fasta"
            )

            # Save metadata
            metadata = {
                'sample_id': sample_id,
                'sequence': sequence,
                'n_residues': len(sequence),
                'binding_prediction': results['validation']['binding_pred'][i].numpy(),
                'plddt': results['validation']['plddt'][i].numpy(),
            }

            with open(self.output_dir / 'metadata' / f"{sample_id}.pkl", 'wb') as f:
                pickle.dump(metadata, f)

        logger.info(f"✓ Saved {B} generated proteins to {self.output_dir}")

    @staticmethod
    def _tokens_to_sequence(tokens: torch.Tensor) -> str:
        """Convert token indices to sequence."""
        from src.data.sequence_decoder import TOKEN_TO_AA

        tokens_np = tokens.cpu().numpy()
        sequence = ''.join([TOKEN_TO_AA.get(int(t), 'X') for t in tokens_np])
        return sequence

    @staticmethod
    def _save_pdb(
        coords: np.ndarray,
        sequence: str,
        output_file: Path,
    ):
        """Save coordinates as PDB file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("REMARK Generated by PMPGen\n")

            for i, (coord, aa) in enumerate(zip(coords, sequence)):
                x, y, z = coord
                f.write(
                    f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )

            f.write("END\n")

    @staticmethod
    def _save_sequence(sequence: str, output_file: Path):
        """Save sequence as FASTA file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(f">{output_file.stem}\n")
            # Write sequence in 80-char lines
            for i in range(0, len(sequence), 80):
                f.write(f"{sequence[i:i+80]}\n")


def main():
    """Main generation script."""
    parser = argparse.ArgumentParser(description="Generate proteins with PMPGen")

    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained PMPGen checkpoint'
    )
    parser.add_argument(
        '--scaffold', type=str, default=None,
        help='Path to scaffold PDB file (if None, use random)'
    )
    parser.add_argument(
        '--n-generate', type=int, default=10,
        help='Number of proteins to generate'
    )
    parser.add_argument(
        '--n-residues', type=int, default=150,
        help='Number of residues (for random scaffold)'
    )
    parser.add_argument(
        '--binding-fraction', type=float, default=0.15,
        help='Fraction of residues to bind membrane'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs/generated',
        help='Output directory'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Run validation pipeline (ESMFold + DynaMo + Rosetta)'
    )

    args = parser.parse_args()

    # Create generator
    generator = ProteinGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
    )

    try:
        # Generate proteins
        if args.scaffold:
            logger.info(f"Loading scaffold from {args.scaffold}")
            # TODO: load scaffold from PDB
            # scaffold_coords = ...
            # results = generator.generate_from_scaffold(scaffold_coords, args.n_generate)
            logger.warning("Scaffold loading not yet implemented")
            return
        else:
            results = generator.generate_random(
                n_residues=args.n_residues,
                n_samples=args.n_generate,
                binding_fraction=args.binding_fraction,
            )

        # Save results
        generator.save_results(results)

        # Optional: validate
        if args.validate:
            logger.info("Running validation pipeline...")
            logger.warning("Validation not yet implemented")
            # TODO: run ESMFold, DynaMo, Rosetta

        logger.info("✓ Generation complete!")

    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
