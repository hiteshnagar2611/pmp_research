"""
Training callbacks for DynaMo and PMPGen.

Provides:
  - Learning rate scheduling (warmup, cosine annealing)
  - Checkpoint management (save best, top-k)
  - Early stopping
  - Weights & Biases logging
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import os
from typing import Optional


class Callback(ABC):
    """Base callback class."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: dict):
        pass


class LearningRateScheduler(Callback):
    """
    Learning rate scheduler with linear warmup + cosine annealing.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        base_lr: float = 1.0e-4,
        min_lr: float = 1.0e-6,
    ):
        """
        Args:
            optimizer: torch optimizer
            warmup_epochs: number of warmup epochs
            max_epochs: total training epochs
            base_lr: initial learning rate
            min_lr: minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

        # Warmup scheduler
        def warmup_fn(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 1.0

        self.warmup_scheduler = LambdaLR(optimizer, warmup_fn)

        # Cosine annealing scheduler (after warmup)
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=min_lr / base_lr,
        )

    def on_epoch_end(self, epoch: int, metrics: dict = None):
        """Update learning rate."""
        if epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class CheckpointCallback(Callback):
    """
    Save model checkpoints based on monitored metric.

    Keeps top-k best checkpoints based on monitored metric.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: str = "outputs/checkpoints",
        monitor: str = "val/loss",
        mode: str = "min",  # "min" or "max"
        save_top_k: int = 3,
    ):
        """
        Args:
            model: model to save
            checkpoint_dir: directory to save checkpoints
            monitor: metric key to monitor (e.g., "val/mcc", "val/loss")
            mode: "min" for loss, "max" for metrics like AUROC
            save_top_k: keep only top k checkpoints
        """
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_score = None
        self.checkpoint_list = []  # list of (score, path) tuples

    def on_epoch_end(self, epoch: int, metrics: dict):
        """Save checkpoint if metric improved."""
        if self.monitor not in metrics:
            return

        current_score = metrics[self.monitor]

        # Check if this is best so far
        if self.best_score is None:
            is_improvement = True
        elif self.mode == "max":
            is_improvement = current_score > self.best_score
        else:
            is_improvement = current_score < self.best_score

        if is_improvement:
            self.best_score = current_score

        # Save checkpoint
        checkpoint_name = f"checkpoint_epoch{epoch:04d}_{self.monitor.replace('/', '_')}_{current_score:.4f}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        torch.save(self.model.state_dict(), checkpoint_path)

        # Track checkpoint
        self.checkpoint_list.append((current_score, checkpoint_path))

        # Keep only top-k
        if self.mode == "max":
            self.checkpoint_list.sort(reverse=True)
        else:
            self.checkpoint_list.sort()

        # Remove old checkpoints
        while len(self.checkpoint_list) > self.save_top_k:
            _, old_path = self.checkpoint_list.pop()
            if os.path.exists(old_path):
                os.remove(old_path)

        if is_improvement:
            print(f"Epoch {epoch}: {self.monitor}={current_score:.4f} (saved)")
        else:
            print(f"Epoch {epoch}: {self.monitor}={current_score:.4f}")


class EarlyStoppingCallback(Callback):
    """
    Early stopping based on monitored metric.

    Stops training if metric doesn't improve for N consecutive epochs.
    """

    def __init__(
        self,
        monitor: str = "val/loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 1e-4,
    ):
        """
        Args:
            monitor: metric key to monitor
            patience: number of epochs without improvement before stopping
            mode: "min" or "max"
            min_delta: minimum change to qualify as improvement
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score = None
        self.wait_count = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, metrics: dict):
        """Check if should stop."""
        if self.monitor not in metrics:
            return

        current_score = metrics[self.monitor]

        if self.best_score is None:
            self.best_score = current_score
        else:
            if self.mode == "max":
                is_improvement = current_score - self.best_score > self.min_delta
            else:
                is_improvement = self.best_score - current_score > self.min_delta

            if is_improvement:
                self.best_score = current_score
                self.wait_count = 0
            else:
                self.wait_count += 1

        if self.wait_count >= self.patience:
            self.should_stop = True
            print(f"Early stopping: {self.monitor} didn't improve for {self.patience} epochs")


class WandBLogger(Callback):
    """
    Log metrics to Weights & Biases.
    """

    def __init__(self, use_wandb: bool = True):
        """
        Args:
            use_wandb: whether to use wandb (gracefully skips if not installed)
        """
        self.use_wandb = use_wandb
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            self.use_wandb = False

    def on_epoch_end(self, epoch: int, metrics: dict):
        """Log metrics to wandb."""
        if not self.use_wandb:
            return

        self.wandb.log(metrics, step=epoch)


class LoggingCallback(Callback):
    """Simple logging callback."""

    def on_epoch_end(self, epoch: int, metrics: dict):
        """Print metrics."""
        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])
        print(f"Epoch {epoch}: {metric_str}")


class CallbackRunner:
    """
    Manages multiple callbacks.

    Calls on_epoch_end for all registered callbacks.
    """

    def __init__(self):
        self.callbacks = []

    def register(self, callback: Callback):
        """Register a callback."""
        self.callbacks.append(callback)

    def on_epoch_end(self, epoch: int, metrics: dict):
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics)

            # Check early stopping
            if isinstance(callback, EarlyStoppingCallback) and callback.should_stop:
                return True  # signal to stop training

        return False  # continue training
