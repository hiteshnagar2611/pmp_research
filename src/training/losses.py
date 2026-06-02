"""
Loss functions for DynaMo Phase 1 and PMPGen Phase 2.

DynaMo losses:
  - Focal loss: handles severe class imbalance
  - Patch contiguity: spatial smoothness of binding regions
  - Contrastive: family-level alignment

PMPGen losses:
  - Flow matching: velocity field regression
  - Anchor preservation: keep binding patch fixed
  - Membrane geometry: correct depth profile
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# DynaMo Phase 1 Losses
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for severe class imbalance.

    From Tsung-Yi Lin et al., "Focal Loss for Dense Object Detection"
    (ICCV 2017)

    L = -α_t · (1 - p_t)^γ · log(p_t)

    where:
      - p_t: model probability for true class
      - α_t: per-class weight (higher for rare class)
      - γ: focusing parameter (γ=2 typical)

    Helpful for PMP binding prediction: ~10-15% binding residues.
    """

    def __init__(
        self,
        alpha: float = None,  # weight for positive class (default: auto-computed)
        gamma: float = 2.0,   # focusing parameter
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: weight for positive class (if None, auto-computed from pos_weight)
            gamma: focusing exponent
            reduction: "mean" or "sum"
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,      # (N,) or (B, N) raw predictions
        targets: torch.Tensor,     # (N,) or (B, N) binary labels {0, 1}
        pos_weight: float = None,  # weight for positive class
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: model predictions (before sigmoid)
            targets: binary labels
            pos_weight: optional weight for positive class

        Returns:
            loss: scalar focal loss
        """
        # Convert logits to probabilities
        p = torch.sigmoid(logits)  # (N,) or (B, N)

        # Clamp to avoid log(0)
        p = torch.clamp(p, min=1e-7, max=1 - 1e-7)

        # Compute focal term
        # For positive samples: (1 - p)^γ
        # For negative samples: p^γ
        ce_loss = F.binary_cross_entropy(p, targets.float(), reduction="none")

        # Focal weight
        alpha = self.alpha or (1.0 / (1.0 + pos_weight) if pos_weight else 0.5)
        focal_weight = torch.where(
            targets.bool(),
            alpha * (1 - p) ** self.gamma,
            (1 - alpha) * p ** self.gamma,
        )

        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class PatchContiguityLoss(nn.Module):
    """
    Loss to encourage spatial contiguity of binding residues.

    Binding regions should form clusters, not scattered individual residues.
    Uses cosine similarity of representations for neighbouring residues.

    L_patch = -Σ_{(i,j) both binding} cos_sim(H_i, H_j)
    """

    def __init__(self, margin: float = 0.5):
        """
        Args:
            margin: cosine similarity threshold above which we encourage alignment
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        H: torch.Tensor,           # (N, hidden_dim) residue representations
        targets: torch.Tensor,     # (N,) binary binding labels
        edge_index: torch.Tensor,  # (2, E) edges in sequence/spatial graph
    ) -> torch.Tensor:
        """
        Compute patch contiguity loss.

        Args:
            H: per-residue hidden representations
            targets: binary binding labels
            edge_index: edges in the protein graph (typically sequential neighbors)

        Returns:
            loss: scalar patch contiguity loss
        """
        src, dst = edge_index[0], edge_index[1]

        # Get representations for connected pairs
        H_src = H[src]  # (E, hidden_dim)
        H_dst = H[dst]  # (E, hidden_dim)

        # Cosine similarity
        cos_sim = F.cosine_similarity(H_src, H_dst, dim=-1)  # (E,)

        # Get labels for connected pairs
        targets_src = targets[src].float()  # (E,)
        targets_dst = targets[dst].float()  # (E,)

        # Both endpoints binding → encourage similarity
        both_binding = (targets_src * targets_dst).bool()  # (E,)

        if not both_binding.any():
            return torch.tensor(0.0, device=H.device)

        # Loss: -cos_sim for binding pairs (want them similar)
        loss = -cos_sim[both_binding].mean()

        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for family-level binding representation alignment.

    Pulls together representations from same PMP family with similar binding patterns.
    Pushes apart representations from different families or different binding profiles.

    Uses InfoNCE (Contrastive Predictive Coding) style loss.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: temperature for softmax (smaller = sharper)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        H: torch.Tensor,         # (N, hidden_dim) residue representations
        targets: torch.Tensor,   # (N,) binary binding labels
        family_ids: torch.Tensor,  # (N,) family assignment (int)
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.

        For each residue, treat its representation as anchor and other binding
        residues from same family as positives, all others as negatives.

        Args:
            H: representations
            targets: binding labels
            family_ids: family assignment for each residue

        Returns:
            loss: scalar contrastive loss
        """
        N = H.shape[0]
        device = H.device

        # Normalize representations
        H_norm = F.normalize(H, p=2, dim=-1)  # (N, hidden_dim)

        # Pairwise similarity matrix
        sim_matrix = torch.matmul(H_norm, H_norm.T) / self.temperature  # (N, N)

        # Create positive mask: same family AND both binding
        targets_expanded = targets.unsqueeze(0) * targets.unsqueeze(1)  # (N, N)
        family_match = (family_ids.unsqueeze(0) == family_ids.unsqueeze(1)).float()  # (N, N)
        pos_mask = targets_expanded.bool() & family_match.bool()  # (N, N)
        pos_mask.fill_diagonal_(False)  # exclude self-similarity

        # Create negative mask: all pairs except positives
        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)

        # For each anchor, compute loss
        loss = 0.0
        count = 0

        for i in range(N):
            if targets[i] == 0 or pos_mask[i].sum() == 0:
                continue  # skip if not binding or no positives

            # Positives and negatives
            pos_sim = sim_matrix[i][pos_mask[i]]
            neg_sim = sim_matrix[i][neg_mask[i]]

            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue

            # InfoNCE loss
            pos_sum = torch.logsumexp(pos_sim, dim=0)
            neg_sum = torch.logsumexp(neg_sim, dim=0)
            loss += -(pos_sum - neg_sum)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)

        return loss / count


# ─────────────────────────────────────────────────────────────────────────────
# PMPGen Phase 2 Losses
# ─────────────────────────────────────────────────────────────────────────────

class FlowMatchingLoss(nn.Module):
    """
    Flow matching loss for SE(3) denoising.

    Matches predicted velocity field against target OT velocity.

    L_flow = ||v_θ(x_t, t, c) - v*(x_0, x_1)||²
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: "mean" or "sum"
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        v_pred_R: torch.Tensor,   # (B, N, 3) predicted rotation velocity
        v_pred_t: torch.Tensor,   # (B, N, 3) predicted translation velocity
        v_target_R: torch.Tensor, # (B, N, 3) target rotation velocity
        v_target_t: torch.Tensor, # (B, N, 3) target translation velocity
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            v_pred_*: predicted velocities
            v_target_*: target velocities

        Returns:
            loss: scalar MSE loss
        """
        loss_R = F.mse_loss(v_pred_R, v_target_R, reduction=self.reduction)
        loss_t = F.mse_loss(v_pred_t, v_target_t, reduction=self.reduction)

        return loss_R + loss_t


class AnchorPreservationLoss(nn.Module):
    """
    Loss to keep binding patch residues fixed during generation.

    Penalises deviation of anchor residues from query structure.

    L_anchor = Σ_i [anchor_mask_i] · ||x_gen_i - x_query_i||²
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        coords_gen: torch.Tensor,     # (B, N, 3) generated coordinates
        coords_query: torch.Tensor,   # (B, N, 3) query coordinates
        anchor_mask: torch.Tensor,    # (B, N, 1) binary mask (1 = anchor)
    ) -> torch.Tensor:
        """
        Compute anchor preservation loss.

        Args:
            coords_gen: generated backbone coordinates
            coords_query: query structure coordinates
            anchor_mask: binary mask for anchored residues

        Returns:
            loss: scalar MSE on anchored residues only
        """
        # Compute per-residue distance
        distance = torch.norm(coords_gen - coords_query, dim=-1, keepdim=True)  # (B, N, 1)

        # Apply anchor mask: only penalise anchored residues
        masked_distance = distance * anchor_mask
        loss = (masked_distance ** 2).mean()

        return loss


class MembraneGeometryLoss(nn.Module):
    """
    Loss for correct membrane depth profile.

    Penalises deviation of predicted residue depths from OPM target.

    L_mem = ||depth_pred - depth_target||²
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        depth_pred: torch.Tensor,     # (B, N, 1) predicted depth
        depth_target: torch.Tensor,   # (B, N, 1) target depth from OPM
    ) -> torch.Tensor:
        """
        Compute membrane geometry loss.

        Args:
            depth_pred: predicted depth
            depth_target: target depth

        Returns:
            loss: scalar MSE
        """
        loss = F.mse_loss(depth_pred, depth_target, reduction=self.reduction)
        return loss


class StructuralValidityLoss(nn.Module):
    """
    Penalty for violated structural constraints.

    Encourages:
      - Correct bond lengths (Cα-Cα ~3.8Å)
      - Correct angles (Cα-Cα-Cα ~110°)
      - No steric clashes
    """

    def __init__(self, bond_length_target: float = 3.8):
        super().__init__()
        self.bond_length_target = bond_length_target

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute structural validity loss.

        Args:
            coords: (B, N, 3) Cα coordinates

        Returns:
            loss: scalar penalty for structural violations
        """
        # Consecutive Cα distances
        if coords.dim() == 3:
            coords_i = coords[:, :-1, :]  # (B, N-1, 3)
            coords_j = coords[:, 1:, :]   # (B, N-1, 3)
        else:
            coords_i = coords[:-1, :]
            coords_j = coords[1:, :]

        distances = torch.norm(coords_j - coords_i, dim=-1)  # (B, N-1) or (N-1,)

        # Penalty for deviating from target bond length
        bond_penalty = (distances - self.bond_length_target) ** 2

        # No extreme outliers
        outlier_penalty = torch.clamp(torch.abs(distances - self.bond_length_target) - 1.0, min=0.0)

        loss = bond_penalty.mean() + outlier_penalty.mean()

        return loss


class CombinedPMPGenLoss(nn.Module):
    """
    Combined loss for PMPGen Phase 2 training.

    L_total = λ_flow · L_flow + λ_anchor · L_anchor + λ_mem · L_mem + λ_struct · L_struct
    """

    def __init__(
        self,
        lambda_flow: float = 1.0,
        lambda_anchor: float = 0.5,
        lambda_mem: float = 0.3,
        lambda_struct: float = 0.1,
    ):
        super().__init__()

        self.lambda_flow = lambda_flow
        self.lambda_anchor = lambda_anchor
        self.lambda_mem = lambda_mem
        self.lambda_struct = lambda_struct

        self.loss_flow = FlowMatchingLoss()
        self.loss_anchor = AnchorPreservationLoss()
        self.loss_mem = MembraneGeometryLoss()
        self.loss_struct = StructuralValidityLoss()

    def forward(
        self,
        v_pred_R: torch.Tensor,
        v_pred_t: torch.Tensor,
        v_target_R: torch.Tensor,
        v_target_t: torch.Tensor,
        coords_gen: torch.Tensor,
        coords_query: torch.Tensor,
        anchor_mask: torch.Tensor,
        depth_pred: torch.Tensor,
        depth_target: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss and return component losses.

        Returns:
            dict with keys: loss_total, loss_flow, loss_anchor, loss_mem, loss_struct
        """
        loss_flow = self.loss_flow(v_pred_R, v_pred_t, v_target_R, v_target_t)
        loss_anchor = self.loss_anchor(coords_gen, coords_query, anchor_mask)
        loss_mem = self.loss_mem(depth_pred, depth_target)
        loss_struct = self.loss_struct(coords_gen)

        loss_total = (
            self.lambda_flow * loss_flow
            + self.lambda_anchor * loss_anchor
            + self.lambda_mem * loss_mem
            + self.lambda_struct * loss_struct
        )

        return {
            "loss_total": loss_total,
            "loss_flow": loss_flow.detach(),
            "loss_anchor": loss_anchor.detach(),
            "loss_mem": loss_mem.detach(),
            "loss_struct": loss_struct.detach(),
        }
