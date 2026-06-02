"""
Evaluation and benchmarking framework.

Modules:
  - benchmark_phase1: DynaMo binding prediction evaluation
  - benchmark_phase2: PMPGen generation quality assessment
  - ablation: component importance studies
  - interpretability: attention visualization and analysis
"""

from .benchmark_phase1 import Phase1Benchmark, ROCCurveComparison
from .benchmark_phase2 import Phase2Benchmark
from .ablation import run_ablation_suite
from .interpretability import AttentionMapVisualizer

__all__ = [
    'Phase1Benchmark',
    'Phase2Benchmark',
    'ROCCurveComparison',
    'run_ablation_suite',
    'AttentionMapVisualizer',
]
