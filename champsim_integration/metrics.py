"""
Metrics collection for ChampSim + sidecar simulation.

Tracks cache performance, sidecar decision quality, and
deployment effectiveness metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import json
from pathlib import Path


@dataclass
class SimulationMetrics:
    """Aggregate simulation metrics."""
    # Trace info
    trace_name: str = ""
    total_instructions: int = 0
    total_memory_accesses: int = 0
    unique_pages: int = 0

    # Cache performance (baseline LRU)
    baseline_l1d_hit_rate: float = 0.0
    baseline_l2_hit_rate: float = 0.0
    baseline_llc_hit_rate: float = 0.0
    baseline_ipc_estimate: float = 0.0

    # Cache performance (sidecar-guided)
    sidecar_l1d_hit_rate: float = 0.0
    sidecar_l2_hit_rate: float = 0.0
    sidecar_llc_hit_rate: float = 0.0
    sidecar_ipc_estimate: float = 0.0

    # Sidecar decision stats
    total_inferences: int = 0
    total_overrides: int = 0
    total_abstentions: int = 0
    override_rate: float = 0.0
    abstention_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_support_density: float = 0.0
    avg_action_margin: float = 0.0

    # Page policy effectiveness
    preserve_decisions: int = 0
    reclaim_decisions: int = 0
    preserve_hit_rate: float = 0.0  # Hit rate on pages sidecar said to preserve
    reclaim_miss_rate: float = 0.0  # Miss rate on pages sidecar said to reclaim

    # Prefetch effectiveness
    prefetch_issued: int = 0
    prefetch_useful: int = 0
    prefetch_accuracy: float = 0.0

    # Latency
    baseline_avg_latency: float = 0.0
    sidecar_avg_latency: float = 0.0
    latency_improvement_pct: float = 0.0

    # Sidecar overhead
    sidecar_inference_latency_us: float = 0.0
    sidecar_memory_mb: float = 0.0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(s)
        return s


class MetricsCollector:
    """Collects per-access metrics during simulation."""

    def __init__(self):
        self._latencies_baseline: List[int] = []
        self._latencies_sidecar: List[int] = []
        self._confidences: List[float] = []
        self._support_densities: List[float] = []
        self._action_margins: List[float] = []
        self._inference_latencies: List[float] = []
        self._override_count: int = 0
        self._abstention_count: int = 0
        self._inference_count: int = 0
        self._preserve_hits: int = 0
        self._preserve_total: int = 0
        self._reclaim_misses: int = 0
        self._reclaim_total: int = 0
        self._prefetch_issued: int = 0
        self._prefetch_useful: int = 0

    def record_baseline_access(self, latency: int):
        self._latencies_baseline.append(latency)

    def record_sidecar_access(self, latency: int):
        self._latencies_sidecar.append(latency)

    def record_inference(self, confidence: float, support: float,
                         margin: float, latency_us: float,
                         override: bool, abstain: bool):
        self._inference_count += 1
        self._confidences.append(confidence)
        self._support_densities.append(support)
        self._action_margins.append(margin)
        self._inference_latencies.append(latency_us)
        if override:
            self._override_count += 1
        if abstain:
            self._abstention_count += 1

    def record_preserve_outcome(self, was_hit: bool):
        self._preserve_total += 1
        if was_hit:
            self._preserve_hits += 1

    def record_reclaim_outcome(self, was_miss: bool):
        self._reclaim_total += 1
        if was_miss:
            self._reclaim_misses += 1

    def record_prefetch(self, useful: bool):
        self._prefetch_issued += 1
        if useful:
            self._prefetch_useful += 1

    def finalize(self, trace_name: str, total_instructions: int,
                 unique_pages: int, baseline_stats: Dict,
                 sidecar_stats: Dict, sidecar_memory_mb: float) -> SimulationMetrics:
        """Compute final metrics."""
        m = SimulationMetrics()
        m.trace_name = trace_name
        m.total_instructions = total_instructions
        m.total_memory_accesses = len(self._latencies_baseline)
        m.unique_pages = unique_pages

        # Baseline cache
        m.baseline_l1d_hit_rate = baseline_stats.get("L1D", 0.0)
        m.baseline_l2_hit_rate = baseline_stats.get("L2", 0.0)
        m.baseline_llc_hit_rate = baseline_stats.get("LLC", 0.0)

        # Sidecar cache
        m.sidecar_l1d_hit_rate = sidecar_stats.get("L1D", 0.0)
        m.sidecar_l2_hit_rate = sidecar_stats.get("L2", 0.0)
        m.sidecar_llc_hit_rate = sidecar_stats.get("LLC", 0.0)

        # Sidecar decisions
        m.total_inferences = self._inference_count
        m.total_overrides = self._override_count
        m.total_abstentions = self._abstention_count
        m.override_rate = self._override_count / max(1, self._inference_count)
        m.abstention_rate = self._abstention_count / max(1, self._inference_count)
        m.avg_confidence = float(np.mean(self._confidences)) if self._confidences else 0.0
        m.avg_support_density = float(np.mean(self._support_densities)) if self._support_densities else 0.0
        m.avg_action_margin = float(np.mean(self._action_margins)) if self._action_margins else 0.0

        # Page policy
        m.preserve_decisions = self._preserve_total
        m.reclaim_decisions = self._reclaim_total
        m.preserve_hit_rate = self._preserve_hits / max(1, self._preserve_total)
        m.reclaim_miss_rate = self._reclaim_misses / max(1, self._reclaim_total)

        # Prefetch
        m.prefetch_issued = self._prefetch_issued
        m.prefetch_useful = self._prefetch_useful
        m.prefetch_accuracy = self._prefetch_useful / max(1, self._prefetch_issued)

        # Latency
        m.baseline_avg_latency = float(np.mean(self._latencies_baseline)) if self._latencies_baseline else 0.0
        m.sidecar_avg_latency = float(np.mean(self._latencies_sidecar)) if self._latencies_sidecar else 0.0
        m.latency_improvement_pct = (
            (m.baseline_avg_latency - m.sidecar_avg_latency) / max(1e-9, m.baseline_avg_latency) * 100
        )

        # IPC estimates (simplified: IPC ∝ 1/avg_latency for memory-bound workloads)
        m.baseline_ipc_estimate = 1.0 / max(1.0, m.baseline_avg_latency / 100)
        m.sidecar_ipc_estimate = 1.0 / max(1.0, m.sidecar_avg_latency / 100)

        # Overhead
        m.sidecar_inference_latency_us = float(np.mean(self._inference_latencies)) if self._inference_latencies else 0.0
        m.sidecar_memory_mb = sidecar_memory_mb

        return m
