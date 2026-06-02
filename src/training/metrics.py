"""
Evaluation metrics for protein binding prediction and generation.

DynaMo metrics:
  - MCC (Matthews Correlation Coefficient)
  - AUROC (Area Under ROC Curve)
  - Patch F1 Score (spatial contiguity)

PMPGen metrics:
  - TM-score (structural similarity)
  - Patch recall/precision (generation success)
  - Foldability (pLDDT)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    auc,
)
from scipy.spatial.distance import cdist


class MatthewsCorrectionCoefficient(nn.Module):
    """
    Matthews Correlation Coefficient (MCC) for binary classification.

    Balanced metric for imbalanced datasets:
    MCC = (TP·TN - FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Range: [-1, 1], where 1 = perfect, 0 = random, -1 = opposite
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        preds: torch.Tensor,   # (N,) predictions in [0, 1]
        targets: torch.Tensor, # (N,) binary targets {0, 1}
        threshold: float = 0.5,  # binary classification threshold
    ) -> float:
        """
        Compute MCC.

        Args:
            preds: predicted probabilities
            targets: binary labels
            threshold: classification threshold

        Returns:
            mcc: scalar MCC value
        """
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # Binarize predictions
        preds_binary = (preds_np >= threshold).astype(int)

        # Compute MCC
        mcc = matthews_corrcoef(targets_np, preds_binary)

        return mcc


class ROCCurveAUC(nn.Module):
    """
    Area Under the ROC Curve (AUROC).

    Measures classification performance across all thresholds.
    Invariant to class imbalance (unlike accuracy).

    Range: [0, 1], where 1 = perfect, 0.5 = random
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        preds: torch.Tensor,   # (N,) predictions in [0, 1]
        targets: torch.Tensor, # (N,) binary targets {0, 1}
    ) -> float:
        """
        Compute AUROC.

        Args:
            preds: predicted probabilities
            targets: binary labels

        Returns:
            auroc: scalar AUC score
        """
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # Require variance in both dimensions
        if len(np.unique(targets_np)) < 2 or len(np.unique(preds_np)) < 2:
            return np.nan

        auroc = roc_auc_score(targets_np, preds_np)

        return auroc


class PatchF1Score(nn.Module):
    """
    Patch-level F1 score for spatial contiguity.

    Groups contiguous binding residues into "patches" and evaluates
    precision + recall at patch level (not residue level).

    This rewards spatially cohesive predictions.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        preds: torch.Tensor,   # (N,) predictions in [0, 1]
        targets: torch.Tensor, # (N,) binary targets {0, 1}
    ) -> dict:
        """
        Compute patch-level metrics.

        Args:
            preds: predicted probabilities
            targets: binary labels

        Returns:
            dict with keys: patch_f1, patch_precision, patch_recall, n_pred_patches, n_true_patches
        """
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # Binarize
        preds_binary = (preds_np >= self.threshold).astype(int)

        # Extract patches (contiguous regions)
        pred_patches = self._extract_patches(preds_binary)
        true_patches = self._extract_patches(targets_np)

        # Evaluate patches: TP = overlap > 50%
        tp = 0
        for pred_patch in pred_patches:
            for true_patch in true_patches:
                overlap = len(set(pred_patch) & set(true_patch))
                if overlap > max(len(pred_patch), len(true_patch)) * 0.5:
                    tp += 1
                    break

        fp = len(pred_patches) - tp
        fn = len(true_patches) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "patch_f1": f1,
            "patch_precision": precision,
            "patch_recall": recall,
            "n_pred_patches": len(pred_patches),
            "n_true_patches": len(true_patches),
        }

    @staticmethod
    def _extract_patches(binary_array: np.ndarray) -> list:
        """
        Extract contiguous patches from binary array.

        Returns list of patches, each patch is a list of residue indices.
        """
        patches = []
        current_patch = []

        for i, val in enumerate(binary_array):
            if val == 1:
                current_patch.append(i)
            else:
                if current_patch:
                    patches.append(current_patch)
                    current_patch = []

        if current_patch:
            patches.append(current_patch)

        return patches


class TMScore(nn.Module):
    """
    TM-score for structural similarity.

    Measures quality of protein structure alignment.
    Range: [0, 1], where 1 = perfect alignment, 0.5 = random

    Formula: TM = (1/L) * Σ_i 1 / (1 + (d_i/d_0)²)

    where:
      - L: length of native protein
      - d_i: distance between aligned residues
      - d_0: normalisation constant depending on L
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        coords_pred: torch.Tensor,   # (N, 3) predicted Cα coordinates
        coords_target: torch.Tensor, # (N, 3) target Cα coordinates
    ) -> float:
        """
        Compute TM-score.

        Assumes 1-to-1 residue correspondence (no gap alignment).

        Args:
            coords_pred: predicted backbone coordinates
            coords_target: target backbone coordinates

        Returns:
            tmscore: scalar TM-score in [0, 1]
        """
        coords_pred_np = coords_pred.detach().cpu().numpy()
        coords_target_np = coords_target.detach().cpu().numpy()

        L = len(coords_pred_np)
        d0 = 1.24 * (L - 15) ** (1/3) - 1.8  # standard d0 formula

        # Compute distances between corresponding residues
        distances = np.linalg.norm(coords_pred_np - coords_target_np, axis=1)

        # Compute TM-score
        tm_scores = 1.0 / (1.0 + (distances / d0) ** 2)
        tm_score = np.mean(tm_scores)

        return float(tm_score)


class BindingMetrics(nn.Module):
    """
    Comprehensive binding metrics for Phase 1 evaluation.

    Computes: MCC, AUROC, F1, Patch F1, Sensitivity, Specificity
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.mcc = MatthewsCorrectionCoefficient()
        self.auroc = ROCCurveAUC()
        self.patch_f1 = PatchF1Score(threshold)

    def forward(
        self,
        preds: torch.Tensor,   # (N,) predictions
        targets: torch.Tensor, # (N,) binary labels
    ) -> dict:
        """
        Compute comprehensive binding metrics.

        Returns:
            dict with all metrics
        """
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        preds_binary = (preds_np >= self.threshold).astype(int)

        # Standard metrics
        mcc = self.mcc(preds, targets, self.threshold)
        auroc = self.auroc(preds, targets)

        # Patch metrics
        patch_metrics = self.patch_f1(preds, targets)

        # Sensitivity and Specificity
        tp = np.sum((preds_binary == 1) & (targets_np == 1))
        fn = np.sum((preds_binary == 0) & (targets_np == 1))
        fp = np.sum((preds_binary == 1) & (targets_np == 0))
        tn = np.sum((preds_binary == 0) & (targets_np == 0))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # F1 score
        f1 = f1_score(targets_np, preds_binary, zero_division=0.0)

        return {
            "mcc": mcc,
            "auroc": auroc,
            "f1": f1,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "patch_f1": patch_metrics["patch_f1"],
            "patch_precision": patch_metrics["patch_precision"],
            "patch_recall": patch_metrics["patch_recall"],
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }


class GenerationQualityMetrics(nn.Module):
    """
    Metrics for evaluating generated PMPs.

    - pLDDT: foldability (from ESMFold)
    - TM-score: structural similarity
    - Patch recall: does generated protein have binding patch?
    - Sequence novelty: how different from training set?
    - Structural diversity: how different from other generated?
    """

    def __init__(self):
        super().__init__()
        self.tmscore = TMScore()

    def forward(
        self,
        coords_gen: torch.Tensor,      # (N, 3) generated coordinates
        coords_template: torch.Tensor, # (N, 3) template coordinates
        plddt_gen: torch.Tensor = None,  # (N,) pLDDT per residue from ESMFold
        binding_pred: torch.Tensor = None,  # (N,) binding prediction
        sequence_gen: str = None,      # generated sequence
        sequence_train: list = None,   # training sequences for novelty
    ) -> dict:
        """
        Compute generation quality metrics.

        Returns:
            dict with generation metrics
        """
        metrics = {}

        # TM-score
        if coords_template is not None:
            tm = self.tmscore(coords_gen, coords_template)
            metrics["tm_score"] = tm

        # pLDDT confidence
        if plddt_gen is not None:
            metrics["plddt_mean"] = plddt_gen.mean().item()
            metrics["plddt_high_conf"] = (plddt_gen > 70).float().mean().item()

        # Binding patch prediction on generated
        if binding_pred is not None:
            binding_coverage = binding_pred.mean().item()
            metrics["binding_coverage"] = binding_coverage

        # Sequence novelty
        if sequence_gen is not None and sequence_train is not None:
            novelty = self._compute_sequence_novelty(sequence_gen, sequence_train)
            metrics["sequence_novelty"] = novelty

        return metrics

    @staticmethod
    def _compute_sequence_novelty(seq_gen: str, seq_train: list) -> float:
        """
        Compute sequence novelty: % identity to closest training sequence.

        Return 1 - max_identity: 1 = completely novel, 0 = identical to training
        """
        max_identity = 0.0

        for seq_t in seq_train:
            # Simple identity metric
            identity = sum(s1 == s2 for s1, s2 in zip(seq_gen, seq_t)) / len(seq_gen)
            max_identity = max(max_identity, identity)

        novelty = 1.0 - max_identity

        return novelty
