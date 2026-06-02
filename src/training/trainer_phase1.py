"""
DynaMo Phase 1 training loop.

Trains binding residue predictor with multi-term loss:
  - Focal loss (class imbalance)
  - Patch contiguity loss (spatial smoothness)
  - Contrastive loss (family alignment)

Uses Weights & Biases for logging.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class DynaMoTrainingConfig:
    """Configuration for DynaMo Phase 1 training."""

    # Optimization
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    max_epochs: int = 100
    batch_size: int = 4
    grad_clip: float = 1.0
    warmup_steps: int = 500

    # Loss weights
    lambda_focal: float = 1.0
    lambda_patch: float = 0.2
    lambda_contrast: float = 0.1
    focal_gamma: float = 2.0

    # Validation
    val_every_n_epochs: int = 5
    save_top_k: int = 3
    monitor: str = "val/mcc"  # metric to monitor for checkpointing
    monitor_mode: str = "max"  # "max" for MCC/AUROC, "min" for loss

    # Logging
    log_every_n_steps: int = 50
    project_name: str = "pmp-research"
    run_name: str = "dynamo-phase1"
    use_wandb: bool = True

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4


class DynaMoTrainer:
    """
    Trainer for DynaMo Phase 1 binding prediction model.

    Handles:
      - Forward pass, loss computation, backward pass
      - Gradient clipping, learning rate scheduling
      - Validation and checkpoint saving
      - Weights & Biases logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config: DynaMoTrainingConfig,
        device: str = "cuda",
    ):
        """
        Args:
            model: DynaMo model to train
            optimizer: torch optimizer
            config: training configuration
            device: "cuda" or "cpu"
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device

        # Loss functions
        from .losses import FocalLoss, PatchContiguityLoss, ContrastiveLoss

        self.loss_focal = FocalLoss(gamma=config.focal_gamma)
        self.loss_patch = PatchContiguityLoss()
        self.loss_contrast = ContrastiveLoss()

        # Metrics
        from .metrics import BindingMetrics

        self.metrics = BindingMetrics()

        # Logging
        self.global_step = 0
        self.best_val_score = None
        self.checkpoint_dir = "outputs/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # W&B
        if config.use_wandb and wandb is not None:
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.__dict__,
            )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
    ):
        """
        Full training loop.

        Args:
            train_loader: training data loader
            val_loader: validation data loader (optional)
        """
        # Learning rate scheduler
        scheduler = self._setup_scheduler()

        # Training loop
        for epoch in range(self.config.max_epochs):
            # Train epoch
            train_loss = self.train_epoch(train_loader, scheduler)

            # Validation
            if val_loader is not None and (epoch + 1) % self.config.val_every_n_epochs == 0:
                val_metrics = self.validate(val_loader)
                self._log_validation(val_metrics, epoch)

                # Checkpoint best model
                self._save_checkpoint(epoch, val_metrics)

            self._log_epoch(epoch, train_loss)

    def train_epoch(self, loader: DataLoader, scheduler=None) -> float:
        """
        Train for one epoch.

        Args:
            loader: training data loader
            scheduler: learning rate scheduler

        Returns:
            avg_loss: average loss over epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Train", leave=False)

        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            loss = self._train_step(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()

            # Learning rate warmup
            if scheduler is not None:
                scheduler.step()

            # Logging
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_every_n_steps == 0:
                avg_loss = total_loss / n_batches
                pbar.set_postfix({"loss": avg_loss})

                if wandb is not None:
                    wandb.log({"train/loss": avg_loss}, step=self.global_step)

        return total_loss / n_batches

    def _train_step(self, batch) -> torch.Tensor:
        """
        Single training step.

        Returns:
            loss: scalar loss tensor
        """
        # Unpack batch (structure depends on your DataLoader)
        # Assume: batch has keys: H_static, H_snapshots, rmsf, depth, kd, charge, sasa, targets, edge_index, ...

        H_static = batch["H_static"].to(self.device)  # (B, N, 256)
        H_snapshots = batch["H_snapshots"].to(self.device)  # (T, B, N, 256)
        rmsf = batch["rmsf"].to(self.device)  # (B, N, 1)
        depth = batch["depth"].to(self.device)  # (B, N, 1)
        kd = batch["kd"].to(self.device)  # (B, N, 1)
        charge = batch["charge"].to(self.device)  # (B, N, 1)
        sasa = batch["sasa"].to(self.device)  # (B, N, 1)
        targets = batch["targets"].to(self.device)  # (B, N) binary
        edge_index = batch.get("edge_index")  # optional

        # Forward pass
        logits, attn_info = self.model(
            H_static=H_static,
            H_snapshots=H_snapshots,
            rmsf=rmsf,
            depth=depth,
            kd=kd,
            charge=charge,
            sasa=sasa,
        )

        # Reshape for loss computation (flatten batch dimension)
        B, N = logits.shape
        logits_flat = logits.reshape(-1)
        targets_flat = targets.reshape(-1)

        # Focal loss (main task)
        loss_focal = self.loss_focal(logits_flat, targets_flat)

        # Patch contiguity loss (if edge_index provided)
        loss_patch = torch.tensor(0.0, device=self.device)
        if edge_index is not None:
            H_fused = attn_info["H_fused"]  # (B, N, 256)
            H_fused_flat = H_fused.reshape(B * N, -1)
            loss_patch = self.loss_patch(H_fused_flat, targets_flat, edge_index)

        # Contrastive loss (if family_ids provided)
        loss_contrast = torch.tensor(0.0, device=self.device)
        if "family_ids" in batch:
            family_ids = batch["family_ids"].to(self.device).reshape(-1)
            H_fused_flat = attn_info["H_fused"].reshape(B * N, -1)
            loss_contrast = self.loss_contrast(H_fused_flat, targets_flat, family_ids)

        # Combined loss
        loss = (
            self.config.lambda_focal * loss_focal
            + self.config.lambda_patch * loss_patch
            + self.config.lambda_contrast * loss_contrast
        )

        return loss

    def validate(self, loader: DataLoader) -> dict:
        """
        Validate on hold-out set.

        Returns:
            metrics: dict of validation metrics
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Val", leave=False):
                H_static = batch["H_static"].to(self.device)
                H_snapshots = batch["H_snapshots"].to(self.device)
                rmsf = batch["rmsf"].to(self.device)
                depth = batch["depth"].to(self.device)
                kd = batch["kd"].to(self.device)
                charge = batch["charge"].to(self.device)
                sasa = batch["sasa"].to(self.device)
                targets = batch["targets"].to(self.device)

                # Forward pass
                logits, _ = self.model(
                    H_static=H_static,
                    H_snapshots=H_snapshots,
                    rmsf=rmsf,
                    depth=depth,
                    kd=kd,
                    charge=charge,
                    sasa=sasa,
                )

                # Collect predictions
                preds = torch.sigmoid(logits.reshape(-1)).cpu()
                all_preds.append(preds)
                all_targets.append(targets.reshape(-1).cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Compute metrics
        metrics = self.metrics(all_preds, all_targets)

        return metrics

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        # Warmup: linear increase from 0 to lr over warmup_steps
        base_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs * 1000,  # approximate
        )

        # Wrap with warmup
        class WarmupScheduler:
            def __init__(self, optimizer, warmup_steps, base_scheduler):
                self.optimizer = optimizer
                self.warmup_steps = warmup_steps
                self.base_scheduler = base_scheduler
                self.step_count = 0

            def step(self):
                self.step_count += 1

                if self.step_count <= self.warmup_steps:
                    # Linear warmup
                    lr = self.optimizer.defaults["lr"] * (self.step_count / self.warmup_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                else:
                    self.base_scheduler.step()

        return WarmupScheduler(self.optimizer, self.config.warmup_steps, base_scheduler)

    def _log_validation(self, metrics: dict, epoch: int):
        """Log validation metrics."""
        if wandb is not None:
            log_dict = {f"val/{k}": v for k, v in metrics.items()}
            wandb.log(log_dict, step=epoch)

    def _log_epoch(self, epoch: int, train_loss: float):
        """Log epoch summary."""
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")

    def _save_checkpoint(self, epoch: int, metrics: dict):
        """Save best checkpoint."""
        monitor_key = self.config.monitor.split("/")[-1]

        if monitor_key not in metrics:
            return

        current_score = metrics[monitor_key]

        # Check if this is best so far
        if self.best_val_score is None:
            is_best = True
        elif self.config.monitor_mode == "max":
            is_best = current_score > self.best_val_score
        else:
            is_best = current_score < self.best_val_score

        if is_best:
            self.best_val_score = current_score
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"dynamo_best_{monitor_key}_{current_score:.4f}.pt"
            )
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
