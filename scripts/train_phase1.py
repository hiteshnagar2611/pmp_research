#!/usr/bin/env python3
"""
Train DynaMo Phase 1: Binding Residue Prediction

Usage:
    python scripts/train_phase1.py                          # Use default config
    python scripts/train_phase1.py train.max_epochs=100     # Override epoch count
    python scripts/train_phase1.py data.batch_size=8        # Override batch size
    python scripts/train_phase1.py --config configs/train/phase1.yaml

Hydra will create an output directory with:
    - checkpoints/           (best model weights)
    - logs/                  (TensorBoard logs)
    - config.yaml            (saved configuration)
"""

import os
from pathlib import Path
import logging
from typing import Dict

import hydra
from hydra.utils import instantiate
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

from src.models.dynamo import DynaMo
from src.training.trainer_phase1 import DynaMoTrainer
from src.training.callbacks import CheckpointCallback, EarlyStoppingCallback
from src.data.pmp_dataset import PMPDataModule


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train/phase1")
def train(cfg) -> Dict:
    """
    Train DynaMo Phase 1.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary with training results
    """
    logger.info(f"Configuration:\n{cfg}")

    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seeds
    pl.seed_everything(cfg.seed, workers=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("Loading data...")
    try:
        dm = PMPDataModule(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
        dm.setup()
        logger.info(f"Train: {len(dm.train_dataset)} proteins")
        logger.info(f"Val: {len(dm.val_dataset)} proteins")
        logger.info(f"Test: {len(dm.test_dataset)} proteins")
    except Exception as e:
        logger.warning(f"Data loading failed: {e}")
        logger.warning("Note: Implement PMPDataModule in src/data/pmp_dataset.py")
        raise

    # ─────────────────────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("Creating model...")
    model_cfg = cfg.model.dynamo
    model = DynaMo(
        node_s_in=model_cfg.node_s_in,
        node_v_in=model_cfg.node_v_in,
        hidden_s_dim=model_cfg.hidden_s_dim,
        hidden_v_dim=model_cfg.hidden_v_dim,
        n_layers=model_cfg.n_layers,
        dropout=model_cfg.dropout,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ─────────────────────────────────────────────────────────────────────────
    # Trainer
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("Creating trainer...")
    trainer_cfg = cfg.train

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(trainer_cfg.checkpoint_dir),
        filename="dynamo-{epoch:02d}-{val_mcc:.4f}",
        monitor="val_mcc",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_mcc",
        patience=trainer_cfg.early_stopping_patience,
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Loggers
    loggers = [TensorBoardLogger("logs", name="dynamo_phase1")]

    if trainer_cfg.use_wandb:
        loggers.append(
            WandbLogger(
                project=trainer_cfg.wandb_project,
                name="DynaMo Phase 1",
                config=cfg,
            )
        )

    # PyTorch Lightning Trainer
    pl_trainer = pl.Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=trainer_cfg.devices if torch.cuda.is_available() else 1,
        logger=loggers,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
        deterministic=True,
        benchmark=False,
    )

    # Trainer wrapper
    trainer = DynaMoTrainer(
        model=model,
        optimizer_cfg=trainer_cfg.optimizer,
        lr_scheduler_cfg=trainer_cfg.lr_scheduler,
        loss_cfg=trainer_cfg.losses,
        device=device,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("Starting training...")
    try:
        pl_trainer.fit(trainer, dm)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        raise

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("Evaluating on test set...")
    test_results = pl_trainer.test(trainer, dm)

    logger.info(f"Test results:\n{test_results}")

    # ─────────────────────────────────────────────────────────────────────────
    # Results
    # ─────────────────────────────────────────────────────────────────────────

    results = {
        "best_model_path": checkpoint_callback.best_model_path,
        "best_score": checkpoint_callback.best_model_score.item() if hasattr(checkpoint_callback.best_model_score, 'item') else checkpoint_callback.best_model_score,
        "test_results": test_results[0] if test_results else {},
    }

    logger.info(f"Best model saved to: {results['best_model_path']}")
    logger.info(f"Best validation MCC: {results['best_score']:.4f}")

    return results


if __name__ == "__main__":
    train()
