"""
PMPGen Phase 2 training loop.

Trains SE(3) flow matching denoiser with multi-term loss:
  - Flow matching loss (velocity field regression)
  - Anchor preservation loss (binding patch fixed)
  - Membrane geometry loss (correct depth profile)
  - Structural validity loss (bond lengths, angles)

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
class PMPGenTrainingConfig:
    """Configuration for PMPGen Phase 2 training."""

    # Optimization
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    max_epochs: int = 200
    batch_size: int = 2  # smaller batch for generation
    grad_clip: float = 1.0
    warmup_steps: int = 1000

    # Loss weights
    lambda_flow: float = 1.0
    lambda_anchor: float = 0.5
    lambda_mem: float = 0.3
    lambda_struct: float = 0.1

    # Flow matching
    num_flow_steps: int = 500
    min_time: float = 0.01  # minimum diffusion time

    # Validation
    val_every_n_epochs: int = 10
    save_top_k: int = 3
    monitor: str = "val/loss_total"
    monitor_mode: str = "min"

    # Logging
    log_every_n_steps: int = 50
    project_name: str = "pmp-research"
    run_name: str = "pmpgen-phase2"
    use_wandb: bool = True

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4


class PMPGenTrainer:
    """
    Trainer for PMPGen Phase 2 protein generation model.

    Handles:
      - Forward pass with flow matching
      - Multi-term loss computation
      - Validation on generation quality
      - Checkpoint saving
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config: PMPGenTrainingConfig,
        device: str = "cuda",
    ):
        """
        Args:
            model: PMPGen model to train
            optimizer: torch optimizer
            config: training configuration
            device: "cuda" or "cpu"
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device

        # Loss functions
        from .losses import CombinedPMPGenLoss

        self.loss_fn = CombinedPMPGenLoss(
            lambda_flow=config.lambda_flow,
            lambda_anchor=config.lambda_anchor,
            lambda_mem=config.lambda_mem,
            lambda_struct=config.lambda_struct,
        )

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
        scheduler = self._setup_scheduler()

        for epoch in range(self.config.max_epochs):
            # Train epoch
            train_losses = self.train_epoch(train_loader, scheduler)

            # Validation
            if val_loader is not None and (epoch + 1) % self.config.val_every_n_epochs == 0:
                val_losses = self.validate(val_loader)
                self._log_validation(val_losses, epoch)

                # Checkpoint
                self._save_checkpoint(epoch, val_losses)

            self._log_epoch(epoch, train_losses)

    def train_epoch(self, loader: DataLoader, scheduler=None) -> dict:
        """
        Train for one epoch.

        Returns:
            loss_dict: dict of average losses
        """
        self.model.train()
        loss_sums = {}
        n_batches = 0

        pbar = tqdm(loader, desc="Train", leave=False)

        for batch in pbar:
            loss_dict = self._train_step(batch)

            # Backward
            self.optimizer.zero_grad()
            loss_dict["loss_total"].backward()

            if self.config.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Accumulate losses
            for key, val in loss_dict.items():
                if key not in loss_sums:
                    loss_sums[key] = 0.0
                loss_sums[key] += val.item()

            n_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_every_n_steps == 0:
                avg_losses = {k: v / n_batches for k, v in loss_sums.items()}
                pbar.set_postfix({"loss": avg_losses["loss_total"]})

                if wandb is not None:
                    log_dict = {f"train/{k}": v for k, v in avg_losses.items()}
                    wandb.log(log_dict, step=self.global_step)

        avg_losses = {k: v / n_batches for k, v in loss_sums.items()}
        return avg_losses

    def _train_step(self, batch) -> dict:
        """
        Single training step for flow matching.

        Returns:
            loss_dict: dict with loss_total, loss_flow, loss_anchor, loss_mem, loss_struct
        """
        # Unpack batch
        # Assume: batch has x0_R, x0_t, x1_R, x1_t (noisy and target frames)
        #         time, coords_query, anchor_mask, depth_target, etc.

        x0_R = batch["x0_R"].to(self.device)  # (B, N, 3, 3)
        x0_t = batch["x0_t"].to(self.device)  # (B, N, 3)
        x1_R = batch["x1_R"].to(self.device)  # (B, N, 3, 3) target
        x1_t = batch["x1_t"].to(self.device)  # (B, N, 3)
        time = batch["time"].to(self.device)  # (B,)
        coords_query = batch["coords_query"].to(self.device)  # (B, N, 3)
        anchor_mask = batch["anchor_mask"].to(self.device)  # (B, N, 1)
        depth_target = batch["depth_target"].to(self.device)  # (B, N, 1)
        c = batch.get("conditioning")  # (B, N, 256) optional conditioning
        if c is not None:
            c = c.to(self.device)

        # Interpolate between x0 and x1 at time t
        x_R, x_t = self.model.interpolant(x0_R, x0_t, x1_R, x1_t, time)

        # Compute target velocities
        v_target_R, v_target_t = self.model.interpolant.compute_velocity(
            x0_R, x0_t, x1_R, x1_t
        )

        # Forward pass: predict velocities
        coords_current = x_t  # use translation as coordinates
        v_pred_R, v_pred_t = self.model.denoiser(coords_current, coords_current, time, c)

        # Compute depth from current coordinates
        depth_pred = self.model.depth_predictor(coords_current, batch["normal"].to(self.device))

        # Combined loss
        loss_dict = self.loss_fn(
            v_pred_R=v_pred_R,
            v_pred_t=v_pred_t,
            v_target_R=v_target_R,
            v_target_t=v_target_t,
            coords_gen=coords_current,
            coords_query=coords_query,
            anchor_mask=anchor_mask,
            depth_pred=depth_pred,
            depth_target=depth_target,
        )

        return loss_dict

    def validate(self, loader: DataLoader) -> dict:
        """
        Validate on hold-out set.

        Returns:
            loss_dict: average losses
        """
        self.model.eval()
        loss_sums = {}
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Val", leave=False):
                loss_dict = self._train_step(batch)

                for key, val in loss_dict.items():
                    if key not in loss_sums:
                        loss_sums[key] = 0.0
                    loss_sums[key] += val.item()

                n_batches += 1

        avg_losses = {k: v / n_batches for k, v in loss_sums.items()}
        return avg_losses

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        base_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs * 1000,
        )

        class WarmupScheduler:
            def __init__(self, optimizer, warmup_steps, base_scheduler):
                self.optimizer = optimizer
                self.warmup_steps = warmup_steps
                self.base_scheduler = base_scheduler
                self.step_count = 0

            def step(self):
                self.step_count += 1

                if self.step_count <= self.warmup_steps:
                    lr = self.optimizer.defaults["lr"] * (self.step_count / self.warmup_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                else:
                    self.base_scheduler.step()

        return WarmupScheduler(self.optimizer, self.config.warmup_steps, base_scheduler)

    def _log_validation(self, losses: dict, epoch: int):
        """Log validation metrics."""
        if wandb is not None:
            log_dict = {f"val/{k}": v for k, v in losses.items()}
            wandb.log(log_dict, step=epoch)

    def _log_epoch(self, epoch: int, losses: dict):
        """Log epoch summary."""
        loss_str = ", ".join([f"{k}={v:.4f}" for k, v in losses.items()])
        print(f"Epoch {epoch+1}: {loss_str}")

    def _save_checkpoint(self, epoch: int, losses: dict):
        """Save best checkpoint."""
        monitor_key = self.config.monitor.split("/")[-1]

        if monitor_key not in losses:
            return

        current_score = losses[monitor_key]

        if self.best_val_score is None:
            is_best = True
        elif self.config.monitor_mode == "max":
            is_best = current_score > self.best_val_score
        else:
            is_best = current_score < self.best_val_score

        if is_best:
            self.best_val_score = current_score
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"pmpgen_best_{monitor_key}_{current_score:.4f}.pt"
            )
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
