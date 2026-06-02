"""
Benchmark Phase 2: Evaluate PMPGen protein generation quality.

Metrics:
  - pLDDT: foldability confidence from ESMFold
  - TM-score: structural similarity to template
  - Patch recall: does generated protein have binding patch?
  - Sequence novelty: how different from training set?
  - Structural diversity: how different from other generated?
  - Rosetta ΔG: binding energy predictions
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class GeneratedProtein:
    """Container for generated protein and metrics."""

    pdb_id: str
    sequence: str
    coords: np.ndarray  # (N, 3) Cα coordinates
    plddt: np.ndarray  # (N,) per-residue confidence
    binding_patch_pred: np.ndarray  # (N,) binding predictions
    tm_score: float = None
    rosetta_ddg: float = None
    novelty: float = None
    diversity: float = None


class Phase2Benchmark:
    """
    Comprehensive benchmarking for Phase 2 generation.
    """

    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: "cuda" or "cpu"
        """
        self.device = device
        self.generated_proteins = []
        self.metrics_df = None

    def add_generated_protein(self, protein: GeneratedProtein):
        """Add a generated protein to benchmark."""
        self.generated_proteins.append(protein)

    def compute_all_metrics(
        self,
        training_sequences: List[str] = None,
    ) -> pd.DataFrame:
        """
        Compute all quality metrics for generated proteins.

        Args:
            training_sequences: list of training sequences for novelty comparison

        Returns:
            metrics_df: DataFrame with metrics for all generated proteins
        """
        results = []

        for protein in self.generated_proteins:
            metrics = {
                "pdb_id": protein.pdb_id,
                "sequence_length": len(protein.sequence),
                "plddt_mean": float(protein.plddt.mean()),
                "plddt_high_conf": float((protein.plddt > 70).mean()),
                "binding_coverage": float(protein.binding_patch_pred.mean()),
                "binding_high_conf": float((protein.binding_patch_pred > 0.5).mean()),
            }

            # TM-score if available
            if protein.tm_score is not None:
                metrics["tm_score"] = protein.tm_score

            # Rosetta if available
            if protein.rosetta_ddg is not None:
                metrics["rosetta_ddg"] = protein.rosetta_ddg

            # Sequence novelty
            if training_sequences is not None:
                novelty = self._compute_novelty(protein.sequence, training_sequences)
                metrics["sequence_novelty"] = novelty

            results.append(metrics)

        self.metrics_df = pd.DataFrame(results)
        return self.metrics_df

    @staticmethod
    def _compute_novelty(seq_gen: str, seq_train: List[str]) -> float:
        """
        Compute sequence novelty: 1 - max_identity to training set.

        Higher = more novel.
        """
        max_identity = 0.0

        for seq_t in seq_train:
            if len(seq_gen) != len(seq_t):
                continue
            identity = sum(s1 == s2 for s1, s2 in zip(seq_gen, seq_t)) / len(seq_gen)
            max_identity = max(max_identity, identity)

        novelty = 1.0 - max_identity
        return novelty

    def compute_diversity(self) -> Dict[str, float]:
        """
        Compute structural diversity among generated proteins.

        Uses pairwise TM-scores: lower average = more diverse.

        Returns:
            diversity_metrics: dict with mean_tm, std_tm, etc.
        """
        from src.training.metrics import TMScore

        tm_scorer = TMScore()
        pairwise_tms = []

        for i, p1 in enumerate(self.generated_proteins):
            for j, p2 in enumerate(self.generated_proteins):
                if i >= j:
                    continue

                # Align to same length (truncate if needed)
                min_len = min(len(p1.coords), len(p2.coords))
                coords1 = torch.from_numpy(p1.coords[:min_len]).float()
                coords2 = torch.from_numpy(p2.coords[:min_len]).float()

                tm = tm_scorer(coords1, coords2)
                pairwise_tms.append(tm)

        if not pairwise_tms:
            return {}

        return {
            "mean_pairwise_tm": np.mean(pairwise_tms),
            "std_pairwise_tm": np.std(pairwise_tms),
            "min_pairwise_tm": np.min(pairwise_tms),
            "max_pairwise_tm": np.max(pairwise_tms),
        }

    def foldability_analysis(self) -> Dict:
        """
        Analyze foldability of generated proteins based on pLDDT.

        Criteria:
          - Very high (pLDDT > 90): high confidence
          - High (70-90): confident
          - Medium (50-70): moderate confidence
          - Low (<50): low confidence

        Returns:
            analysis: dict with confidence statistics
        """
        high_conf = []
        med_conf = []
        low_conf = []

        for protein in self.generated_proteins:
            high = (protein.plddt > 70).mean()
            med = ((protein.plddt > 50) & (protein.plddt <= 70)).mean()
            low = (protein.plddt <= 50).mean()

            high_conf.append(high)
            med_conf.append(med)
            low_conf.append(low)

        return {
            "avg_high_confidence": np.mean(high_conf),
            "avg_medium_confidence": np.mean(med_conf),
            "avg_low_confidence": np.mean(low_conf),
            "percent_very_confident": np.mean([p > 0.7 for p in high_conf]) * 100,
        }

    def binding_patch_analysis(self) -> Dict:
        """
        Analyze binding patch predictions on generated proteins.

        Returns:
            analysis: dict with binding statistics
        """
        patch_coverages = []
        patch_confidences = []

        for protein in self.generated_proteins:
            coverage = protein.binding_patch_pred.mean()
            confidence = (protein.binding_patch_pred > 0.5).mean()

            patch_coverages.append(coverage)
            patch_confidences.append(confidence)

        return {
            "avg_binding_coverage": np.mean(patch_coverages),
            "std_binding_coverage": np.std(patch_coverages),
            "avg_binding_confidence": np.mean(patch_confidences),
            "percent_with_binding": np.mean([p > 0.3 for p in patch_coverages]) * 100,
        }

    def print_summary(self):
        """Print comprehensive summary."""
        if self.metrics_df is None:
            print("No metrics computed. Call compute_all_metrics() first.")
            return

        print("\n" + "=" * 100)
        print("PHASE 2 BENCHMARK: PROTEIN GENERATION QUALITY")
        print("=" * 100)

        print(f"\nGenerated {len(self.generated_proteins)} proteins")
        print(self.metrics_df.to_string(index=False))

        # Foldability
        fold_analysis = self.foldability_analysis()
        print("\nFOLDABILITY ANALYSIS (pLDDT)")
        print(f"  Average high confidence: {fold_analysis['avg_high_confidence']:.2%}")
        print(f"  Average medium confidence: {fold_analysis['avg_medium_confidence']:.2%}")
        print(f"  Average low confidence: {fold_analysis['avg_low_confidence']:.2%}")
        print(f"  Proteins with >70% high-confidence residues: {fold_analysis['percent_very_confident']:.1f}%")

        # Binding patch
        binding_analysis = self.binding_patch_analysis()
        print("\nBINDING PATCH ANALYSIS")
        print(f"  Average binding coverage: {binding_analysis['avg_binding_coverage']:.2%}")
        print(f"  Proteins with binding patch: {binding_analysis['percent_with_binding']:.1f}%")

        # Diversity
        diversity = self.compute_diversity()
        if diversity:
            print("\nSTRUCTURAL DIVERSITY (Pairwise TM-scores)")
            print(f"  Mean pairwise TM-score: {diversity['mean_pairwise_tm']:.4f}")
            print(f"  Std pairwise TM-score: {diversity['std_pairwise_tm']:.4f}")
            print(f"  Min TM-score: {diversity['min_pairwise_tm']:.4f}")
            print(f"  Max TM-score: {diversity['max_pairwise_tm']:.4f}")

        print("=" * 100 + "\n")

    def plot_metrics(self, save_path: str = None):
        """
        Plot generation metrics.

        Args:
            save_path: where to save figure
        """
        if self.metrics_df is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # pLDDT distribution
        if "plddt_mean" in self.metrics_df.columns:
            axes[0, 0].hist(self.metrics_df["plddt_mean"], bins=15, edgecolor="black")
            axes[0, 0].set_xlabel("Mean pLDDT")
            axes[0, 0].set_ylabel("Count")
            axes[0, 0].set_title("pLDDT Distribution")
            axes[0, 0].axvline(70, color="r", linestyle="--", label="Confident threshold")
            axes[0, 0].legend()

        # Binding coverage
        if "binding_coverage" in self.metrics_df.columns:
            axes[0, 1].hist(self.metrics_df["binding_coverage"], bins=15, edgecolor="black")
            axes[0, 1].set_xlabel("Binding Coverage")
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].set_title("Binding Patch Coverage Distribution")

        # High confidence fraction
        if "plddt_high_conf" in self.metrics_df.columns:
            axes[1, 0].hist(self.metrics_df["plddt_high_conf"], bins=15, edgecolor="black")
            axes[1, 0].set_xlabel("Fraction pLDDT > 70")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("High Confidence Residue Fraction")

        # TM-score if available
        if "tm_score" in self.metrics_df.columns:
            axes[1, 1].hist(self.metrics_df["tm_score"], bins=15, edgecolor="black")
            axes[1, 1].set_xlabel("TM-score")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("TM-score Distribution")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()


def generation_quality_report(
    benchmark: Phase2Benchmark,
    output_path: str = "outputs/phase2_generation_report.txt",
) -> str:
    """
    Generate comprehensive generation quality report.

    Args:
        benchmark: Phase2Benchmark instance
        output_path: where to save report

    Returns:
        report: formatted report string
    """
    fold_analysis = benchmark.foldability_analysis()
    binding_analysis = benchmark.binding_patch_analysis()
    diversity = benchmark.compute_diversity()

    report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║           PMPGEN PHASE 2 GENERATION QUALITY REPORT                        ║
╚════════════════════════════════════════════════════════════════════════════╝

SUMMARY
──────────────────────────────────────────────────────────────────────────────
Generated proteins: {len(benchmark.generated_proteins)}

FOLDABILITY (pLDDT from ESMFold)
──────────────────────────────────────────────────────────────────────────────
✓ Average high confidence (>70%): {fold_analysis['avg_high_confidence']:.1%}
✓ Average medium confidence (50-70%): {fold_analysis['avg_medium_confidence']:.1%}
✓ Average low confidence (<50%): {fold_analysis['avg_low_confidence']:.1%}
✓ Proteins with >70% confident residues: {fold_analysis['percent_very_confident']:.0f}%

BINDING PATCH PREDICTIONS
──────────────────────────────────────────────────────────────────────────────
✓ Average binding coverage: {binding_analysis['avg_binding_coverage']:.1%}
✓ Proteins with predicted binding patch: {binding_analysis['percent_with_binding']:.0f}%
✓ Average binding confidence: {binding_analysis['avg_binding_confidence']:.1%}

STRUCTURAL DIVERSITY
──────────────────────────────────────────────────────────────────────────────
✓ Mean pairwise TM-score: {diversity.get('mean_pairwise_tm', 'N/A')}
✓ Std pairwise TM-score: {diversity.get('std_pairwise_tm', 'N/A')}
✓ (Lower = more diverse structures)

QUALITY ASSESSMENT
──────────────────────────────────────────────────────────────────────────────
"""

    # Assess quality tiers
    high_quality = sum(
        1 for _, row in benchmark.metrics_df.iterrows()
        if row['plddt_mean'] > 75 and row['binding_coverage'] > 0.2
    )
    medium_quality = sum(
        1 for _, row in benchmark.metrics_df.iterrows()
        if 70 <= row['plddt_mean'] <= 75 or 0.1 < row['binding_coverage'] <= 0.2
    )
    low_quality = len(benchmark.generated_proteins) - high_quality - medium_quality

    report += f"""
High quality:    {high_quality} proteins ({high_quality/len(benchmark.generated_proteins)*100:.0f}%)
Medium quality:  {medium_quality} proteins ({medium_quality/len(benchmark.generated_proteins)*100:.0f}%)
Low quality:     {low_quality} proteins ({low_quality/len(benchmark.generated_proteins)*100:.0f}%)

RECOMMENDATIONS
──────────────────────────────────────────────────────────────────────────────
✓ Iterate on proteins with pLDDT < 70 (low foldability)
✓ Improve binding patch predictions if coverage too high/low
✓ Ensure diversity with pairwise TM-scores
✓ Validate with MD simulations before submission
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Saved report: {output_path}")

    return report
