"""
Ablation study: systematically remove components to measure their contribution.

Tests for Phase 1 (DynaMo):
  1. Remove conformational attention → use only static structure
  2. Remove membrane geometry path → only use dynamics
  3. Remove physicochemical gate → linear output
  4. Remove cross-attention → concatenate instead of fusing

Tests for Phase 2 (PMPGen):
  1. Remove MD-informed noise schedule → constant noise
  2. Remove membrane guidance → no geometric steering
  3. Remove anchor preservation loss → all residues can change
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class AblationResult:
    """Container for ablation study results."""

    component: str
    metric_name: str
    metric_value: float
    delta: float  # difference from full model
    percent_change: float  # (delta / full_model) * 100


class AblationStudy:
    """
    Run ablation study on DynaMo or PMPGen.

    Systematically remove components and measure performance drop.
    """

    def __init__(self, metric_name: str = "auroc"):
        """
        Args:
            metric_name: metric to optimize for (e.g., "mcc", "auroc", "f1")
        """
        self.metric_name = metric_name
        self.results = []
        self.full_model_score = None

    def ablate_phase1(
        self,
        model: nn.Module,
        test_loader,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """
        Run ablation study on Phase 1 (DynaMo).

        Systematically removes components:
          1. Conformational attention pool
          2. Membrane geometry path
          3. Physicochemical gate
          4. Cross-attention fusion

        Args:
            model: DynaMo model
            test_loader: test data loader
            device: "cuda" or "cpu"

        Returns:
            results_df: DataFrame with ablation results
        """
        # Get baseline (full model) score
        self.full_model_score = self._evaluate_model(model, test_loader, device)
        print(f"Full Model {self.metric_name}: {self.full_model_score:.4f}")

        # Ablation 1: Remove conformational attention
        # Set all snapshot weights to equal (uniform pooling)
        print("\nAblation 1: Remove conformational attention pool...")
        model_ablated = self._ablate_conf_attention(model)
        score = self._evaluate_model(model_ablated, test_loader, device)
        self._record_result("Conformational Attention Pool", score)

        # Ablation 2: Remove membrane geometry path
        # Zero out geometry path contributions
        print("Ablation 2: Remove membrane geometry path...")
        model_ablated = self._ablate_geometry_path(model)
        score = self._evaluate_model(model_ablated, test_loader, device)
        self._record_result("Membrane Geometry Path", score)

        # Ablation 3: Remove physicochemical gate
        # Set gate to 1.0 (no modulation)
        print("Ablation 3: Remove physicochemical gate...")
        model_ablated = self._ablate_phys_gate(model)
        score = self._evaluate_model(model_ablated, test_loader, device)
        self._record_result("Physicochemical Gate", score)

        # Ablation 4: Remove cross-attention
        # Simple concatenation instead of attention
        print("Ablation 4: Remove cross-attention fusion...")
        model_ablated = self._ablate_cross_attention(model)
        score = self._evaluate_model(model_ablated, test_loader, device)
        self._record_result("Cross-Attention Fusion", score)

        return self._compile_results()

    def ablate_phase2(
        self,
        model: nn.Module,
        test_loader,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """
        Run ablation study on Phase 2 (PMPGen).

        Removes:
          1. MD-informed noise schedule
          2. Membrane plane gradient guidance
          3. Anchor preservation loss

        Args:
            model: PMPGen model
            test_loader: test data loader
            device: "cuda" or "cpu"

        Returns:
            results_df: DataFrame with ablation results
        """
        # Get baseline score (overall loss)
        self.full_model_score = self._evaluate_model(model, test_loader, device)
        print(f"Full Model {self.metric_name}: {self.full_model_score:.4f}")

        # Ablation 1: Remove MD-informed noise
        print("\nAblation 1: Remove MD-informed noise schedule...")
        model_ablated = self._ablate_md_noise(model)
        score = self._evaluate_model(model_ablated, test_loader, device)
        self._record_result("MD-Informed Noise Schedule", score)

        # Ablation 2: Remove membrane guidance
        print("Ablation 2: Remove membrane gradient guidance...")
        model_ablated = self._ablate_mem_guidance(model)
        score = self._evaluate_model(model_ablated, test_loader, device)
        self._record_result("Membrane Plane Guidance", score)

        # Ablation 3: Remove anchor preservation
        print("Ablation 3: Remove anchor preservation loss...")
        # Set lambda_anchor = 0
        model.loss_fn.lambda_anchor = 0.0
        score = self._evaluate_model(model, test_loader, device)
        self._record_result("Anchor Preservation Loss", score)

        return self._compile_results()

    def _evaluate_model(
        self,
        model: nn.Module,
        test_loader,
        device: str,
    ) -> float:
        """
        Evaluate model and return single metric score.

        Implementation depends on model type. Here's a template.
        """
        # This is a placeholder - actual implementation depends on your data format
        from src.training.metrics import BindingMetrics

        model.eval()
        metrics_fn = BindingMetrics()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                # Forward pass (adjust keys based on your data)
                logits = model(...)  # placeholder
                preds = torch.sigmoid(logits.reshape(-1)).cpu()
                targets = batch["targets"].reshape(-1).cpu()

                all_preds.append(preds)
                all_targets.append(targets)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        metrics = metrics_fn(all_preds, all_targets)
        return metrics[self.metric_name]

    def _record_result(self, component: str, metric_value: float):
        """Record ablation result."""
        delta = self.full_model_score - metric_value
        percent_change = (delta / self.full_model_score * 100) if self.full_model_score != 0 else 0

        result = AblationResult(
            component=component,
            metric_name=self.metric_name,
            metric_value=metric_value,
            delta=delta,
            percent_change=percent_change,
        )

        self.results.append(result)
        print(f"  {component}: {metric_value:.4f} (Δ = {delta:+.4f}, {percent_change:+.1f}%)")

    def _compile_results(self) -> pd.DataFrame:
        """Compile results into DataFrame."""
        data = [
            {
                "Component": r.component,
                "Score": r.metric_value,
                "Delta (Full - Ablated)": r.delta,
                "% Change": r.percent_change,
                "Importance": "HIGH" if abs(r.delta) > 0.05 else "MEDIUM" if abs(r.delta) > 0.01 else "LOW",
            }
            for r in self.results
        ]

        df = pd.DataFrame(data)
        df = df.sort_values("Delta (Full - Ablated)", ascending=False)

        return df

    def print_summary(self):
        """Print ablation study summary."""
        print("\n" + "=" * 100)
        print("ABLATION STUDY SUMMARY")
        print("=" * 100)
        print(f"Full model {self.metric_name}: {self.full_model_score:.4f}\n")

        df = self._compile_results()
        print(df.to_string(index=False))

        print("\n" + "=" * 100)
        print("INTERPRETATION")
        print("=" * 100)
        print("Larger Δ = more important component")
        print("Components sorted by importance (descending)")
        print("=" * 100 + "\n")

    # ── Ablation helpers for Phase 1 ────────────────────────────────────────

    @staticmethod
    def _ablate_conf_attention(model: nn.Module) -> nn.Module:
        """Disable conformational attention pool."""
        # Clone model
        model_ablated = type(model)(**model.config) if hasattr(model, 'config') else model
        model_ablated.load_state_dict(model.state_dict())

        # Modify: set all snapshot weights to 1/T (uniform)
        # This disables the learned temperature scaling
        def uniform_pool_hook(module, input, output):
            # Replace conf_pool output with simple average
            H_static, H_snapshots, rmsf = input[0], input[1], input[2]
            return H_snapshots.mean(dim=0)  # uniform pooling

        # Register hook on conf_pool
        model_ablated.conf_pool.register_forward_hook(uniform_pool_hook)

        return model_ablated

    @staticmethod
    def _ablate_geometry_path(model: nn.Module) -> nn.Module:
        """Disable membrane geometry path."""
        model_ablated = type(model)(**model.config) if hasattr(model, 'config') else model
        model_ablated.load_state_dict(model.state_dict())

        # Zero out geometry path
        def zero_geometry_hook(module, input, output):
            return torch.zeros_like(output)

        model_ablated.geom_path.register_forward_hook(zero_geometry_hook)

        return model_ablated

    @staticmethod
    def _ablate_phys_gate(model: nn.Module) -> nn.Module:
        """Disable physicochemical gate."""
        model_ablated = type(model)(**model.config) if hasattr(model, 'config') else model
        model_ablated.load_state_dict(model.state_dict())

        # Set gate to identity (multiply by 1.0)
        def identity_gate_hook(module, input, output):
            H = input[0]
            # Skip the gate, return input unchanged
            return H

        model_ablated.phys_gate.register_forward_hook(identity_gate_hook)

        return model_ablated

    @staticmethod
    def _ablate_cross_attention(model: nn.Module) -> nn.Module:
        """Disable cross-attention fusion."""
        model_ablated = type(model)(**model.config) if hasattr(model, 'config') else model
        model_ablated.load_state_dict(model.state_dict())

        # Replace cross-attention with concatenation
        def concat_instead_of_attn(module, input, output):
            H_geom, H_star = input[0], input[1]
            # Concatenate instead of attending
            return torch.cat([H_geom, H_star], dim=-1)[:, :H_geom.shape[-1]]

        model_ablated.cross_attn.register_forward_hook(concat_instead_of_attn)

        return model_ablated

    # ── Ablation helpers for Phase 2 ────────────────────────────────────────

    @staticmethod
    def _ablate_md_noise(model: nn.Module) -> nn.Module:
        """Use constant noise schedule instead of MD-informed."""
        model_ablated = type(model)(**model.config) if hasattr(model, 'config') else model
        model_ablated.load_state_dict(model.state_dict())

        # Replace MD schedule with constant
        def constant_noise_hook(module, input, output):
            # Replace RMSF-dependent noise with constant
            t, rmsf = input[0], input[1]
            # Return base schedule only (ignore RMSF)
            return module.base_schedule(t)

        model_ablated.noise_schedule.register_forward_hook(constant_noise_hook)

        return model_ablated

    @staticmethod
    def _ablate_mem_guidance(model: nn.Module) -> nn.Module:
        """Disable membrane plane gradient guidance."""
        model_ablated = type(model)(**model.config) if hasattr(model, 'config') else model
        model_ablated.load_state_dict(model.state_dict())

        # Zero out guidance
        def zero_guidance_hook(module, input, output):
            return torch.zeros_like(output)

        model_ablated.mem_guidance.register_forward_hook(zero_guidance_hook)

        return model_ablated


def run_ablation_suite(
    model: nn.Module,
    test_loader,
    phase: int = 1,
    device: str = "cuda",
    output_path: str = None,
) -> pd.DataFrame:
    """
    Run complete ablation study and save results.

    Args:
        model: model to ablate
        test_loader: test data
        phase: 1 for DynaMo, 2 for PMPGen
        device: "cuda" or "cpu"
        output_path: where to save results

    Returns:
        results_df: ablation results
    """
    ablation = AblationStudy(metric_name="auroc" if phase == 1 else "loss_total")

    if phase == 1:
        results_df = ablation.ablate_phase1(model, test_loader, device)
    else:
        results_df = ablation.ablate_phase2(model, test_loader, device)

    ablation.print_summary()

    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"Saved ablation results: {output_path}")

    return results_df


def plot_ablation_results(
    results_df: pd.DataFrame,
    save_path: str = None,
    figsize: Tuple = (10, 6),
):
    """
    Plot ablation study results as bar chart.

    Args:
        results_df: from run_ablation_suite
        save_path: where to save figure
        figsize: figure size
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    components = results_df["Component"]
    deltas = results_df["Delta (Full - Ablated)"]
    colors = ["red" if d > 0.05 else "orange" if d > 0.01 else "yellow" for d in deltas]

    ax.barh(components, deltas, color=colors, edgecolor="black")
    ax.set_xlabel("Performance Drop (Full Model - Ablated)")
    ax.set_title("Component Importance: Ablation Study")
    ax.invert_yaxis()

    # Add value labels
    for i, (component, delta) in enumerate(zip(components, deltas)):
        ax.text(delta, i, f"  {delta:.4f}", va="center")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
