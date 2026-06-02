#!/usr/bin/env python3
"""
Evaluate trained models and run benchmarks.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/dynamo_best.pt
    python scripts/evaluate.py --phase 2 --checkpoint outputs/checkpoints/pmpgen_best.pt
    python scripts/evaluate.py --phase 1 --baseline scannet
    python scripts/evaluate.py --ablation

Outputs:
    - outputs/evaluation/phase1_results.csv
    - outputs/evaluation/phase1_roc_comparison.png
    - outputs/evaluation/ablation_results.csv
    - outputs/evaluation/report.txt
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

from src.evaluation.benchmark_phase1 import Phase1Benchmark, ROCCurveComparison
from src.evaluation.benchmark_phase2 import Phase2Benchmark, generation_quality_report
from src.evaluation.ablation import run_ablation_suite
from src.evaluation.interpretability import AttentionMapVisualizer


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate_phase1(
    checkpoint_path: str,
    test_loader,
    output_dir: str = "outputs/evaluation",
    run_ablation: bool = False,
    run_attention: bool = False,
) -> dict:
    """
    Evaluate Phase 1 (DynaMo) model.

    Args:
        checkpoint_path: path to model checkpoint
        test_loader: test data loader
        output_dir: where to save results
        run_ablation: whether to run ablation study
        run_attention: whether to visualize attention

    Returns:
        results dict
    """
    logger.info("="*70)
    logger.info("PHASE 1 EVALUATION: DynaMo Binding Prediction")
    logger.info("="*70)

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    try:
        model = torch.load(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Benchmark
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("\n[1/4] Running Phase 1 benchmark...")
    benchmark = Phase1Benchmark(device=str(device))

    # Evaluate model on test set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # TODO: Adjust keys based on your data loader
            try:
                logits = model(batch)  # placeholder
                preds = torch.sigmoid(logits).cpu()
                targets = batch["targets"].cpu()

                all_preds.append(preds)
                all_targets.append(targets)
            except Exception as e:
                logger.warning(f"Batch evaluation failed: {e}")
                continue

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    metrics = benchmark.evaluate_model(model, test_loader, "DynaMo")
    benchmark.print_comparison()

    # Save metrics
    metrics_path = output_dir / "phase1_metrics.csv"
    benchmark.metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to {metrics_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # ROC Curves
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("\n[2/4] Plotting ROC curves...")
    roc = ROCCurveComparison()
    roc.add_model("DynaMo", all_preds, all_targets)

    roc_path = output_dir / "phase1_roc_curve.png"
    roc.plot(save_path=str(roc_path))
    logger.info(f"Saved ROC curve to {roc_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Ablation
    # ─────────────────────────────────────────────────────────────────────────

    if run_ablation:
        logger.info("\n[3/4] Running ablation study...")
        ablation_results = run_ablation_suite(
            model, test_loader, phase=1, device=str(device),
            output_path=output_dir / "phase1_ablation.csv"
        )
        logger.info(f"Ablation results:\n{ablation_results}")
    else:
        logger.info("\n[3/4] Skipping ablation study (use --ablation to enable)")

    # ─────────────────────────────────────────────────────────────────────────
    # Attention
    # ─────────────────────────────────────────────────────────────────────────

    if run_attention:
        logger.info("\n[4/4] Visualizing attention maps...")
        try:
            visualizer = AttentionMapVisualizer(model, device=str(device))
            logger.info("Attention visualization complete")
        except Exception as e:
            logger.warning(f"Attention visualization failed: {e}")
    else:
        logger.info("\n[4/4] Skipping attention visualization (use --attention to enable)")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("\n" + "="*70)
    logger.info("PHASE 1 EVALUATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Results saved to {output_dir}")

    return {
        "metrics": metrics,
        "all_preds": all_preds,
        "all_targets": all_targets,
    }


def evaluate_phase2(
    checkpoint_path: str,
    test_loader,
    output_dir: str = "outputs/evaluation",
    n_generate: int = 50,
) -> dict:
    """
    Evaluate Phase 2 (PMPGen) model.

    Args:
        checkpoint_path: path to model checkpoint
        test_loader: test data loader
        output_dir: where to save results
        n_generate: number of proteins to generate

    Returns:
        results dict
    """
    logger.info("="*70)
    logger.info("PHASE 2 EVALUATION: PMPGen De Novo Generation")
    logger.info("="*70)

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    try:
        model = torch.load(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────────────────────────────────

    logger.info(f"\n[1/2] Generating {n_generate} proteins...")
    benchmark = Phase2Benchmark(device=str(device))

    # TODO: Implement sampling loop
    logger.warning("Generation not yet implemented - add sampling logic here")

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("\n[2/2] Evaluating generation quality...")
    if len(benchmark.generated_proteins) > 0:
        metrics_df = benchmark.compute_all_metrics()
        benchmark.print_summary()

        # Save report
        report_path = output_dir / "phase2_generation_report.txt"
        report = generation_quality_report(benchmark, output_path=str(report_path))

        # Save metrics
        metrics_path = output_dir / "phase2_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics to {metrics_path}")
    else:
        logger.warning("No proteins generated - check sampling implementation")

    logger.info("\n" + "="*70)
    logger.info("PHASE 2 EVALUATION COMPLETE")
    logger.info("="*70)

    return {"benchmark": benchmark}


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate PMP research models")

    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=1,
        help="Which phase to evaluate (1=DynaMo, 2=PMPGen)"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study (Phase 1 only)"
    )
    parser.add_argument(
        "--attention", action="store_true",
        help="Visualize attention maps (Phase 1 only)"
    )
    parser.add_argument(
        "--n-generate", type=int, default=50,
        help="Number of proteins to generate (Phase 2 only)"
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="auto",
        help="Device to use"
    )

    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────
    # Placeholder: Load test data
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("Loading test data...")
    # TODO: Implement data loading
    # from src.data.pmp_dataset import PMPDataModule
    # dm = PMPDataModule(...)
    # dm.setup()
    # test_loader = dm.test_dataloader()

    logger.warning("Data loading not yet implemented - add data loading here")
    test_loader = None

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluate
    # ─────────────────────────────────────────────────────────────────────────

    if args.phase == 1:
        if not args.checkpoint:
            logger.error("--checkpoint required for Phase 1 evaluation")
            return

        results = evaluate_phase1(
            checkpoint_path=args.checkpoint,
            test_loader=test_loader,
            output_dir=args.output_dir,
            run_ablation=args.ablation,
            run_attention=args.attention,
        )
    else:
        if not args.checkpoint:
            logger.error("--checkpoint required for Phase 2 evaluation")
            return

        results = evaluate_phase2(
            checkpoint_path=args.checkpoint,
            test_loader=test_loader,
            output_dir=args.output_dir,
            n_generate=args.n_generate,
        )

    logger.info(f"\n✓ Evaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
