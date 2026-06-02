"""
Metric utilities: tracking, aggregation, and computation.

Features:
  - Average meter for online statistics
  - Metric tracker for aggregating values
  - Metric history logging
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


class AverageMeter:
    """Track average value and other statistics."""

    def __init__(self, name: str = ''):
        """
        Initialize average meter.

        Args:
            name: meter name
        """
        self.name = name
        self.reset()

    def reset(self):
        """Reset statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.values = []

    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.

        Args:
            val: value to add
            n: weight/count of this value
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # Track all values for std computation
        self.values.extend([val] * n)

        if len(self.values) > 1:
            self.std = np.std(self.values)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}: {self.avg:.4f} (±{self.std:.4f})"


class MetricTracker:
    """Track multiple metrics."""

    def __init__(self):
        """Initialize metric tracker."""
        self.meters = {}
        self.history = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        """
        Update metrics.

        Args:
            metrics: dictionary of metric_name -> value
            n: weight/count
        """
        for key, val in metrics.items():
            if key not in self.meters:
                self.meters[key] = AverageMeter(name=key)
                self.history[key] = []

            self.meters[key].update(val, n)
            self.history[key].append(val)

    def reset(self):
        """Reset all meters."""
        for meter in self.meters.values():
            meter.reset()

    def get_averages(self) -> Dict[str, float]:
        """Get average values."""
        return {name: meter.avg for name, meter in self.meters.items()}

    def get_values(self) -> Dict[str, float]:
        """Get current values."""
        return {name: meter.val for name, meter in self.meters.items()}

    def __str__(self) -> str:
        """String representation."""
        return " | ".join(str(meter) for meter in self.meters.values())


class EpochMetrics:
    """Track metrics for one epoch."""

    def __init__(self, epoch: int):
        """
        Initialize epoch metrics.

        Args:
            epoch: epoch number
        """
        self.epoch = epoch
        self.metrics = {}

    def add(self, name: str, values: List[float]):
        """
        Add metric values.

        Args:
            name: metric name
            values: list of values
        """
        self.metrics[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values,
        }

    def get_metric(self, name: str) -> Optional[float]:
        """Get metric mean value."""
        if name in self.metrics:
            return self.metrics[name]['mean']
        return None

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        summary = {}
        for name, stats in self.metrics.items():
            summary[f'{name}_mean'] = stats['mean']
            summary[f'{name}_std'] = stats['std']

        return summary

    def __str__(self) -> str:
        """String representation."""
        lines = [f"Epoch {self.epoch}:"]
        for name, stats in self.metrics.items():
            lines.append(f"  {name}: {stats['mean']:.4f} (±{stats['std']:.4f})")

        return "\n".join(lines)


class MetricHistory:
    """Track metric history across epochs."""

    def __init__(self):
        """Initialize metric history."""
        self.history = {}

    def add_epoch(self, epoch_metrics: EpochMetrics):
        """
        Add epoch metrics.

        Args:
            epoch_metrics: EpochMetrics object
        """
        epoch = epoch_metrics.epoch

        for name, stats in epoch_metrics.metrics.items():
            if name not in self.history:
                self.history[name] = {
                    'epochs': [],
                    'values': [],
                    'means': [],
                    'stds': [],
                }

            self.history[name]['epochs'].append(epoch)
            self.history[name]['values'].append(stats['values'])
            self.history[name]['means'].append(stats['mean'])
            self.history[name]['stds'].append(stats['std'])

    def get_best(self, metric_name: str, mode: str = 'min') -> tuple:
        """
        Get best epoch and value for metric.

        Args:
            metric_name: name of metric
            mode: 'min' or 'max'

        Returns:
            (best_epoch, best_value)
        """
        if metric_name not in self.history:
            return None, None

        means = self.history[metric_name]['means']
        epochs = self.history[metric_name]['epochs']

        if mode == 'min':
            best_idx = np.argmin(means)
        else:
            best_idx = np.argmax(means)

        return epochs[best_idx], means[best_idx]

    def get_improvement(self, metric_name: str, mode: str = 'min') -> float:
        """
        Get total improvement over all epochs.

        Args:
            metric_name: name of metric
            mode: 'min' or 'max'

        Returns:
            improvement: total improvement
        """
        if metric_name not in self.history:
            return 0

        means = self.history[metric_name]['means']

        if len(means) < 2:
            return 0

        if mode == 'min':
            improvement = means[0] - means[-1]
        else:
            improvement = means[-1] - means[0]

        return improvement

    def __str__(self) -> str:
        """String representation."""
        lines = ["Metric History:"]

        for name, stats in self.history.items():
            best_epoch, best_value = self.get_best(name, mode='min')
            improvement = self.get_improvement(name, mode='min')

            lines.append(f"  {name}:")
            lines.append(f"    Best: epoch {best_epoch}, value {best_value:.4f}")
            lines.append(f"    Improvement: {improvement:.4f}")

        return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    print("Metrics utilities loaded successfully")

    # Test AverageMeter
    meter = AverageMeter(name='loss')
    for i in range(10):
        meter.update(1.0 / (i + 1))
    print(f"Meter: {meter}")

    # Test MetricTracker
    tracker = MetricTracker()
    for i in range(5):
        tracker.update({'loss': 0.5 - i * 0.1, 'accuracy': 0.8 + i * 0.05})
    print(f"\nTracker:\n{tracker}")
    print(f"Averages: {tracker.get_averages()}")

    print("✓ Metrics utilities working!")
