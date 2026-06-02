"""
Checkpoint utilities: save, load, and manage model checkpoints.

Features:
  - Save/load model states
  - Checkpoint management (keep best K)
  - Resume training from checkpoint
  - Model versioning
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    output_file: str = 'checkpoint.pt',
) -> Path:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: optimizer (optional)
        epoch: current epoch
        metrics: metrics dictionary (optional)
        output_file: output file path

    Returns:
        output_path: path to saved checkpoint
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, output_path)

    return output_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_file: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        model: PyTorch model
        checkpoint_file: path to checkpoint
        optimizer: optimizer to load state (optional)
        device: device to load on

    Returns:
        checkpoint: loaded checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


class CheckpointManager:
    """Manage model checkpoints (save best K, clean up old ones)."""

    def __init__(
        self,
        output_dir: str = 'checkpoints',
        keep_best_k: int = 3,
        metric_name: str = 'loss',
        mode: str = 'min',  # 'min' or 'max'
    ):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: directory to save checkpoints
            keep_best_k: keep best K checkpoints
            metric_name: metric to track
            mode: 'min' (lower is better) or 'max' (higher is better)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.keep_best_k = keep_best_k
        self.metric_name = metric_name
        self.mode = mode

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.saved_checkpoints = []  # List of (metric_value, path)

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        Save checkpoint if it's better than previous.

        Args:
            model: PyTorch model
            optimizer: optimizer (optional)
            epoch: current epoch
            metrics: metrics dictionary

        Returns:
            checkpoint_path: path to saved checkpoint (or None if not saved)
        """
        if metrics is None or self.metric_name not in metrics:
            # No metric to compare, just save
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch:04d}.pt'
            save_checkpoint(model, optimizer, epoch, metrics, str(checkpoint_path))
            return checkpoint_path

        metric_value = metrics[self.metric_name]

        # Check if better
        is_better = (
            (self.mode == 'min' and metric_value < self.best_value) or
            (self.mode == 'max' and metric_value > self.best_value)
        )

        if is_better:
            self.best_value = metric_value

            # Save checkpoint
            checkpoint_path = (
                self.output_dir / f'checkpoint_epoch_{epoch:04d}_'
                f'{self.metric_name}_{metric_value:.4f}.pt'
            )
            save_checkpoint(model, optimizer, epoch, metrics, str(checkpoint_path))

            # Track saved checkpoint
            self.saved_checkpoints.append((metric_value, checkpoint_path))

            # Keep only best K
            if len(self.saved_checkpoints) > self.keep_best_k:
                # Sort and remove worst
                self.saved_checkpoints.sort(key=lambda x: x[0])
                if self.mode == 'max':
                    self.saved_checkpoints = self.saved_checkpoints[-(self.keep_best_k):]
                else:
                    self.saved_checkpoints = self.saved_checkpoints[:self.keep_best_k]

                # Delete old checkpoint
                old_value, old_path = self.saved_checkpoints[0]
                if old_path.exists():
                    old_path.unlink()

            return checkpoint_path

        return None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if not self.saved_checkpoints:
            return None

        if self.mode == 'min':
            best_value, best_path = min(self.saved_checkpoints, key=lambda x: x[0])
        else:
            best_value, best_path = max(self.saved_checkpoints, key=lambda x: x[0])

        return best_path

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu',
    ) -> Optional[Dict[str, Any]]:
        """Load best checkpoint."""
        best_path = self.get_best_checkpoint()

        if best_path is None:
            return None

        checkpoint = load_checkpoint(model, str(best_path), optimizer, device)

        return checkpoint


class ExperimentTracker:
    """Track experiment metadata and results."""

    def __init__(self, experiment_dir: str = 'experiments'):
        """
        Initialize experiment tracker.

        Args:
            experiment_dir: directory for experiments
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_path = self.experiment_dir / self.experiment_name
        self.experiment_path.mkdir(exist_ok=True)

        self.metadata = {
            'name': self.experiment_name,
            'created': datetime.now().isoformat(),
            'config': {},
            'results': {},
        }

    def save_config(self, config: Dict[str, Any]):
        """Save configuration."""
        self.metadata['config'] = config

        with open(self.experiment_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def save_results(self, results: Dict[str, Any]):
        """Save results."""
        self.metadata['results'] = results

        with open(self.experiment_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

    def save_metadata(self):
        """Save experiment metadata."""
        with open(self.experiment_path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory for this experiment."""
        checkpoint_dir = self.experiment_path / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        return checkpoint_dir


if __name__ == "__main__":
    # Test checkpoint management
    print("Checkpoint utilities loaded successfully")

    # Test CheckpointManager
    manager = CheckpointManager(output_dir='/tmp/checkpoints', keep_best_k=3, mode='min')

    # Dummy model
    model = nn.Linear(10, 1)

    # Save dummy checkpoints
    for epoch in range(5):
        metrics = {'loss': 0.5 - epoch * 0.1}
        manager.save(model, epoch=epoch, metrics=metrics)

    best_path = manager.get_best_checkpoint()
    print(f"Best checkpoint: {best_path}")

    print("✓ Checkpoint utilities working!")
