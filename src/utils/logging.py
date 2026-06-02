"""
Logging utilities: experiment tracking, progress monitoring, result logging.

Features:
  - Structured logging
  - Experiment tracking (local and W&B)
  - Training progress monitoring
  - Result logging and reporting
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class Logger:
    """Experiment logger with console and file output."""

    def __init__(
        self,
        name: str,
        log_dir: str = 'logs',
        level: int = logging.INFO,
    ):
        """
        Initialize logger.

        Args:
            name: logger name
            log_dir: directory for log files
            level: logging level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        log_file = self.log_dir / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        self.log_file = log_file

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        self.info("Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        prefix = f"Step {step} - " if step else ""
        for key, value in metrics.items():
            self.info(f"{prefix}{key}: {value:.6f}")

    def log_table(self, data: list, headers: list):
        """Log table of results."""
        self.info("Results:")
        self.info("  " + " | ".join(headers))
        self.info("  " + "-" * (len(headers) * 15))
        for row in data:
            self.info("  " + " | ".join(f"{v:>14}" for v in row))


def setup_logger(
    name: str,
    log_dir: str = 'logs',
    level: str = 'INFO',
) -> Logger:
    """
    Setup logger.

    Args:
        name: logger name
        log_dir: directory for log files
        level: logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        logger: Logger instance
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }

    log_level = level_map.get(level.upper(), logging.INFO)

    logger = Logger(name, log_dir=log_dir, level=log_level)

    return logger


class MetricLogger:
    """Log metrics to JSON file."""

    def __init__(self, output_file: str):
        """
        Initialize metric logger.

        Args:
            output_file: path to output JSON file
        """
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = {}

    def log(self, key: str, value: float, step: int):
        """
        Log metric.

        Args:
            key: metric name
            value: metric value
            step: step/epoch number
        """
        if key not in self.metrics:
            self.metrics[key] = []

        self.metrics[key].append({
            'step': step,
            'value': value,
        })

    def save(self):
        """Save metrics to JSON file."""
        with open(self.output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def load(self):
        """Load metrics from JSON file."""
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                self.metrics = json.load(f)


class ProgressMonitor:
    """Monitor training progress."""

    def __init__(self, total_epochs: int, samples_per_epoch: int):
        """
        Initialize progress monitor.

        Args:
            total_epochs: total number of epochs
            samples_per_epoch: samples per epoch
        """
        self.total_epochs = total_epochs
        self.samples_per_epoch = samples_per_epoch
        self.current_epoch = 0
        self.current_sample = 0

    def update(self, n_samples: int = 1):
        """Update progress."""
        self.current_sample += n_samples

        if self.current_sample >= self.samples_per_epoch:
            self.current_epoch += 1
            self.current_sample = 0

    def get_progress(self) -> float:
        """Get progress as fraction [0, 1]."""
        epoch_progress = self.current_sample / self.samples_per_epoch
        total_progress = (self.current_epoch + epoch_progress) / self.total_epochs
        return min(total_progress, 1.0)

    def get_status(self) -> str:
        """Get status string."""
        progress = self.get_progress()
        percentage = progress * 100
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        return f"[{bar}] {percentage:.1f}% (Epoch {self.current_epoch}/{self.total_epochs})"


if __name__ == "__main__":
    # Test logging
    logger = setup_logger('test_logger')
    logger.info("Logger initialized")
    logger.log_config({'lr': 1e-4, 'batch_size': 32})
    logger.log_metrics({'loss': 0.5, 'accuracy': 0.95}, step=1)

    print("✓ Logging utilities working!")
