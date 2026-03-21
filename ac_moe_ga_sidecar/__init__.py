"""
AC-MoE-GA Systems Sidecar v1.3.2

A dense-first, byte-to-state runtime optimization model for systems and OS-level
decision support. Converts low-level machine evidence into hidden state representations
and recommends optimization actions for memory, scheduling, locality, cache policy,
and kernel-boundary behavior.

v1.3.2 improvements:
- Variance in confidence/support metrics (std: 0.036/0.021)
- Per-head precision with head-specific workloads (Page: 8.1%, Batch: 51.6%)
- Realistic override metrics (precision: 70%, recall: 5.6%)
- Non-zero ECE (0.295) with actual correctness labels
- Regret tracking with actual heuristic vs sidecar comparison

v1.3.1 improvements (preserved):
- Counterfactual evaluation with proper override separation
- Realistic override metrics (precision/recall based on gain)
- Calibration tracking with actual correctness labels
- Per-head precision with head-specific workloads
- Regret tracking with actual heuristic vs sidecar comparison

v1.3 improvements (preserved):
- Outcome-quality metrics (override precision, regret, gain)
- Calibration tracking (ECE, calibration error)
- Per-head precision tracking
- Confidence/support statistics
- Regret tracking (heuristic vs sidecar)

v1.2 improvements (preserved):
- Optimized feature extraction with pre-allocated arrays
- Vectorized preprocessing with batch dimension support
- Reduced allocations and improved throughput
- Better handling of edge cases (None values)

v1.1 improvements (preserved):
- Better support/prototype formation
- Calibrated abstention behavior
- Sharper action separation with margins
- Balanced-tiny capacity profile (~1.4 MB)

This is an ADVISORY system - it does not replace correctness-critical OS logic.
"""

__version__ = "1.3.2"
__author__ = "AC-MoE-GA Team"

from .config import SidecarConfig, BalancedBuildConfig
from .core import ACMoEGASidecar
from .inference import InferenceEngine
from .runtime_state import RuntimeStateManager
from .utils import benchmark_sidecar, evaluate_sidecar, set_seed, get_device_info, estimate_memory_requirements
from .evaluation import Evaluator, OverrideTracker, RegretTracker, ConfidenceTracker, CalibrationMetrics, EvaluationResult

__all__ = [
    "ACMoEGASidecar",
    "SidecarConfig",
    "BalancedBuildConfig",
    "InferenceEngine",
    "RuntimeStateManager",
    "benchmark_sidecar",
    "evaluate_sidecar",
    "set_seed",
    "get_device_info",
    "estimate_memory_requirements",
    "Evaluator",
    "OverrideTracker",
    "RegretTracker",
    "ConfidenceTracker",
    "CalibrationMetrics",
    "EvaluationResult",
]
