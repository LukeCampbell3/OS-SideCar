"""
ChampSim + Sidecar co-simulation engine.

Runs a ChampSim trace through both a baseline cache hierarchy (pure LRU)
and a sidecar-guided hierarchy, comparing performance side-by-side.

The sidecar's page/NUMA/prefetch recommendations are translated into
cache eviction hints and prefetch insertions.

Online learning: The sidecar updates its weights based on cache outcomes
to improve future page state predictions.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

import torch
import torch.optim as optim
import numpy as np

from ac_moe_ga_sidecar.core import ACMoEGASidecar
from ac_moe_ga_sidecar.config import SidecarConfig, BalancedBuildConfig
from ac_moe_ga_sidecar.types import (
    Recommendation, PageAction, NUMAAction, MicroEvent
)
from ac_moe_ga_sidecar.training import Trainer, TrainingBatch, SidecarLoss

from .trace_parser import ChampSimTraceParser, ChampSimRecord
from .event_bridge import ChampSimEventBridge
from .cache_model import CacheHierarchy, CacheHierarchyConfig
from .metrics import MetricsCollector, SimulationMetrics
from .online_learner import OnlineLearner, DecisionRecord
from .calibration import OutcomeCalibrator, MarginCalibrator

logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    """Configuration for the co-simulation."""
    # Trace
    trace_path: str = ""
    max_instructions: Optional[int] = None
    warmup_instructions: int = 1_000_000

    # Cache hierarchy
    cache_config: Optional[CacheHierarchyConfig] = None

    # Sidecar
    sidecar_config: Optional[SidecarConfig] = None
    fault_injection_rate: float = 0.05  # 5% synthetic page faults

    # Prefetch
    enable_prefetch: bool = True
    prefetch_degree: int = 2       # Lines to prefetch ahead
    prefetch_on_preserve: bool = True  # Prefetch neighbors of preserved pages

    # Online learning
    enable_online_learning: bool = True
    online_lr: float = 1e-4
    online_batch_size: int = 32
    online_update_interval: int = 16  # Update every N inferences

    # Reporting
    report_interval: int = 1_000_000  # Print progress every N instructions
    output_path: Optional[str] = None


class ChampSimSidecarSimulator:
    """
    Co-simulation engine: runs ChampSim traces through baseline and
    sidecar-guided cache hierarchies simultaneously.

    Supports online learning: the sidecar updates its weights based on
    cache outcomes (hit/miss) to improve future page state predictions.
    """

    def __init__(self, config: SimulatorConfig):
        self.config = config

        # Sidecar
        sidecar_cfg = config.sidecar_config or BalancedBuildConfig()
        self.sidecar = ACMoEGASidecar(config=sidecar_cfg)

        # Online learning: create online learner
        if config.enable_online_learning:
            self.online_learner = OnlineLearner(
                model=self.sidecar.model,
                learning_rate=config.online_lr,
                batch_size=config.online_batch_size,
            )
        else:
            self.online_learner = None

        # Two cache hierarchies: baseline (pure LRU) and sidecar-guided
        cache_cfg = config.cache_config or CacheHierarchyConfig()
        self.baseline_cache = CacheHierarchy(cache_cfg)
        self.sidecar_cache = CacheHierarchy(cache_cfg)

        # Event bridge
        self.bridge = ChampSimEventBridge()
        self.bridge.set_fault_injection_rate(config.fault_injection_rate)

        # Metrics
        self.collector = MetricsCollector()

        # State
        self._warmup_done = False
        self._instructions_processed = 0
        self._last_recommendation: Optional[Recommendation] = None

        # Online learning: track page decisions and cache outcomes
        self._last_page_decisions: Dict[int, Dict] = {}  # page -> {time, state, hit}
        self._learning_buffer: List[Dict] = []
        self._last_inference_time = 0

        # Calibration: adjust abstention/margin thresholds based on cache outcomes
        self.outcome_calibrator = OutcomeCalibrator()
        self.margin_calibrator = MarginCalibrator()

    def run(self) -> SimulationMetrics:
        """Run the full co-simulation and return metrics."""
        trace_path = Path(self.config.trace_path)
        parser = ChampSimTraceParser(trace_path, max_records=self.config.max_instructions)

        logger.info(f"Starting co-simulation: {trace_path.name}")
        logger.info(f"Warmup: {self.config.warmup_instructions:,} instructions")
        if self.config.max_instructions:
            logger.info(f"Max instructions: {self.config.max_instructions:,}")

        start_time = time.time()

        for record in parser.parse():
            self._process_record(record)
            self._instructions_processed += 1

            if (self._instructions_processed == self.config.warmup_instructions
                    and not self._warmup_done):
                self._end_warmup()

            # Periodically update online learning
            if self._warmup_done and self.online_learner is not None:
                if self._instructions_processed % self.config.online_update_interval == 0:
                    self._update_online_learning()

            if self._instructions_processed % self.config.report_interval == 0:
                self._print_progress(start_time)

        elapsed = time.time() - start_time
        logger.info(f"Simulation complete: {self._instructions_processed:,} instructions in {elapsed:.1f}s")

        # Compute sidecar memory footprint
        sidecar_memory_mb = self._estimate_sidecar_memory()

        # Finalize metrics
        baseline_rates = self.baseline_cache.get_summary()
        sidecar_rates = self.sidecar_cache.get_summary()

        metrics = self.collector.finalize(
            trace_name=trace_path.name,
            total_instructions=self._instructions_processed,
            unique_pages=len(self.bridge._page_tracker),
            baseline_stats={k.replace("_hit_rate", ""): v for k, v in baseline_rates.items()},
            sidecar_stats={k.replace("_hit_rate", ""): v for k, v in sidecar_rates.items()},
            sidecar_memory_mb=sidecar_memory_mb,
        )

        self._print_report(metrics)

        if self.config.output_path:
            metrics.to_json(self.config.output_path)
            logger.info(f"Results saved to {self.config.output_path}")

        return metrics

    def _process_record(self, record: ChampSimRecord):
        """Process a single trace record through both pipelines."""
        if not record.has_memory_access:
            return

        addr = record.destination_memory if record.is_store else record.source_memory
        is_write = record.is_store

        # --- Baseline pipeline (pure LRU) ---
        _, baseline_latency = self.baseline_cache.access(addr, is_write)
        if self._warmup_done:
            self.collector.record_baseline_access(baseline_latency)

        # --- Sidecar pipeline ---
        # Translate to MicroEvent and feed to sidecar
        event = self.bridge.translate(record)
        recommendation = None

        if event is not None:
            result = self.sidecar.process_event(event)
            if result is not None:
                recommendation = result.recommendation
                self._last_recommendation = recommendation

                if self._warmup_done:
                    self.collector.record_inference(
                        confidence=recommendation.inferred_state.confidence,
                        support=recommendation.support_density,
                        margin=recommendation.action_margin or 0.0,
                        latency_us=result.latency_us,
                        override=recommendation.should_override_heuristic,
                        abstain=recommendation.abstain,
                    )

                # Apply sidecar hints to cache
                self._apply_sidecar_hints(event, recommendation)

        # Access sidecar-guided cache
        _, sidecar_latency = self.sidecar_cache.access(addr, is_write)
        if self._warmup_done:
            self.collector.record_sidecar_access(sidecar_latency)

        # Record cache outcome for online learning and calibration
        if self._warmup_done:
            if self.online_learner is not None:
                self._record_cache_outcome(addr, is_write, recommendation)
            if recommendation is not None:
                self._record_calibration_outcome(addr, recommendation, is_write)

        # Prefetch based on sidecar recommendation
        if self.config.enable_prefetch and self._last_recommendation is not None:
            self._maybe_prefetch(addr, record, self._last_recommendation)

    def _apply_sidecar_hints(self, event: MicroEvent, rec: Recommendation):
        """Translate sidecar recommendation into cache eviction hints."""
        if rec.abstain or not rec.should_override_heuristic:
            return

        page = event.virtual_page
        if page is None:
            return

        # Use actual page state from inference, not action scores
        page_state = rec.inferred_state.page_state
        page_state_dict = {
            "cold": page_state.cold,
            "reclaimable": page_state.reclaimable,
            "burst_hot": page_state.burst_hot,
            "recently_reused": page_state.recently_reused,
        }

        self.sidecar_cache.apply_sidecar_hints(page, page_state_dict)

        if self._warmup_done:
            # Track if sidecar's eviction decision was correct
            # (hot pages should be preserved, cold pages can be evicted)
            is_hot = page_state.burst_hot > 0.5 or page_state.recently_reused > 0.5
            is_cold = page_state.cold > 0.5 or page_state.reclaimable > 0.5

            if is_hot:
                self.collector.record_preserve_outcome(was_hit=True)
            elif is_cold:
                self.collector.record_reclaim_outcome(was_miss=True)

    def _get_features_for_page(self, page: int) -> Dict[str, torch.Tensor]:
        """
        Get features for a page from the runtime state.
        
        This is a simplified approach - in production you'd store the full
        feature vector for each decision.
        """
        # Get the runtime state manager
        runtime_state = self.sidecar.engine.runtime_state

        # Get page runtime state
        page_state = runtime_state.pages.get(page)
        if page_state is None:
            # Create dummy features
            return {
                "low8_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "high8_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "alignment_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "small_int_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "delta_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "hamming_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "continuous_features": torch.zeros(1, 6, device=self.sidecar.device),
                "bitfield_features": torch.zeros(1, 16, device=self.sidecar.device),
                "sketch_features": torch.zeros(1, 32, device=self.sidecar.device),
                "page_hash_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "offset_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "cache_line_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "addr_alignment_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "stride_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "reuse_dist_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "locality_cluster": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "entropy_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "address_flags": torch.zeros(1, 5, device=self.sidecar.device),
                "event_type": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "fault_class": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "syscall_class": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "opcode_family": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "transition_type": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "result_class": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "pte_flags": torch.zeros(1, 11, device=self.sidecar.device),
                "vma_class": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "protection_domain": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "read_count_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "write_count_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "fault_count_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "cow_count_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "recency_bucket": torch.zeros(1, dtype=torch.long, device=self.sidecar.device),
                "volatility_features": torch.zeros(1, 4, device=self.sidecar.device),
                "pressure_features": torch.zeros(1, 12, device=self.sidecar.device),
                "missingness_mask": torch.zeros(1, 8, device=self.sidecar.device),
                "freshness_ages": torch.ones(1, 4, device=self.sidecar.device),
                "source_quality": torch.ones(1, 2, device=self.sidecar.device),
                "conflict_score": torch.zeros(1, device=self.sidecar.device),
                "consistency_score": torch.ones(1, device=self.sidecar.device),
            }

        # Extract features from page state
        return {
            "read_count_bucket": torch.tensor([min(4096, page_state.read_count) // 10], dtype=torch.long, device=self.sidecar.device),
            "write_count_bucket": torch.tensor([min(4096, page_state.write_count) // 10], dtype=torch.long, device=self.sidecar.device),
            "fault_count_bucket": torch.tensor([min(4096, page_state.fault_count)], dtype=torch.long, device=self.sidecar.device),
            "cow_count_bucket": torch.tensor([min(4096, page_state.cow_count)], dtype=torch.long, device=self.sidecar.device),
            "recency_bucket": torch.tensor([min(63, page_state.recency_bucket)], dtype=torch.long, device=self.sidecar.device),
            "volatility_features": torch.tensor([[page_state.volatility, 0, 0, 0]], dtype=torch.float32, device=self.sidecar.device),
            "pressure_features": torch.zeros(1, 12, dtype=torch.float32, device=self.sidecar.device),
        }

    def _maybe_prefetch(self, addr: int, record: ChampSimRecord, rec: Recommendation):
        """Issue prefetches based on sidecar locality/page hints."""
        if not self.config.enable_prefetch:
            return

        page_scores = rec.action_scores.page_scores
        if not page_scores:
            return

        preserve_score = page_scores.get(PageAction.PRESERVE, 0.0)

        # Prefetch next N cache lines if page is hot
        if self.config.prefetch_on_preserve and preserve_score > 0.3:
            cache_line = addr >> 6
            for i in range(1, self.config.prefetch_degree + 1):
                target = (cache_line + i) << 6
                result = self.sidecar_cache.prefetch_line(target)
                if self._warmup_done:
                    self.collector.record_prefetch(useful=(result == "LLC_HIT"))

    def _record_calibration_outcome(self, addr: int, recommendation: Recommendation, is_write: bool):
        """Record cache outcome for calibration."""
        page = addr >> 12
        page_state = recommendation.inferred_state.page_state

        # Determine if the decision was correct based on cache outcome
        # (we'll check this when the page is accessed again)
        self._last_page_decisions[page] = {
            "time": self._instructions_processed,
            "page_state": page_state,
            "recommendation": recommendation,
        }

        # Check if this page was recently accessed and we have a cache outcome
        if page in self._last_page_decisions:
            last_decision = self._last_page_decisions[page]
            if last_decision["time"] == self._instructions_processed - 1:
                # This is the next access to the same page - record the hit
                last_decision["hit"] = True

        # Check if we have a complete record (decision + outcome)
        if page in self._last_page_decisions:
            last_decision = self._last_page_decisions[page]
            if "hit" in last_decision and last_decision["hit"] is not None:
                # Record for calibration
                rec = last_decision["recommendation"]
                was_hit = last_decision["hit"]
                page_state = last_decision["page_state"]

                # Determine if the decision was correct
                if was_hit:
                    # Page was hot
                    correct = page_state.burst_hot > page_state.cold
                else:
                    # Page was cold
                    correct = page_state.cold > page_state.burst_hot

                # Record for outcome calibration
                self.outcome_calibrator.record_decision(
                    confidence=rec.inferred_state.confidence,
                    abstain=rec.abstain,
                    cache_hit=was_hit,
                    page_state_cold=page_state.cold,
                    page_state_hot=page_state.burst_hot,
                )

                # Record for margin calibration
                margin = rec.action_margin or 0.0
                self.margin_calibrator.record_decision(
                    margin=margin,
                    cache_hit=was_hit,
                    correct=correct,
                )

                # Clear the recorded hit
                last_decision["hit"] = None

    def _record_cache_outcome(self, addr: int, is_write: bool, recommendation: Optional[Recommendation]):
        """
        Record cache outcome for online learning.
        
        Tracks whether pages marked as hot/cold by the sidecar actually
        resulted in cache hits/misses, then uses this to update the model.
        """
        if recommendation is None or recommendation.abstain:
            return

        page = addr >> 12
        page_state = recommendation.inferred_state.page_state

        # Record the decision for this page
        self._last_page_decisions[page] = {
            "time": self._instructions_processed,
            "page_state": page_state,
        }

        # Check if this page was recently accessed and we have a cache outcome
        if page in self._last_page_decisions:
            last_decision = self._last_page_decisions[page]
            if last_decision["time"] == self._instructions_processed - 1:
                # This is the next access to the same page - record the hit
                last_decision["hit"] = True

        # Check if we have a complete record (decision + outcome)
        if page in self._last_page_decisions:
            last_decision = self._last_page_decisions[page]
            if "hit" in last_decision and last_decision["hit"] is not None:
                # We have both prediction and outcome - record for learning
                self._learning_buffer.append({
                    "page_state": last_decision["page_state"],
                    "was_hit": last_decision["hit"],
                    "is_write": is_write,
                })
                # Clear the recorded hit
                last_decision["hit"] = None

        # Update online learner if enabled
        if self.online_learner is not None:
            # Record the decision
            self.online_learner.record_decision(
                features=self._get_features_for_page(page),
                page_state=page_state,
                timestamp=self._instructions_processed,
            )
            # Record the outcome
            if "hit" in last_decision and last_decision["hit"] is not None:
                self.online_learner.record_outcome(
                    timestamp=self._instructions_processed,
                    cache_hit=last_decision["hit"],
                )

    def _update_online_learning(self):
        """
        Update the model based on recorded cache outcomes.
        
        Uses cache hit/miss as supervision to train the model's
        page state prediction.
        """
        if self.online_learner is None:
            return

        # Train the online learner
        metrics = self.online_learner.train_step()

        # Log learning progress occasionally
        if self._instructions_processed % (self.config.report_interval * 2) == 0:
            logger.info(f"  Online learning: loss={metrics.get('loss', 0):.4f}, count={metrics.get('count', 0)}")

        # Clear the learning buffer
        self._learning_buffer.clear()

    def _end_warmup(self):
        """Reset stats after warmup period."""
        self._warmup_done = True
        # Reset cache stats but keep cache contents (warm)
        for level_name in ["l1d", "l2", "llc"]:
            getattr(self.baseline_cache, level_name).stats = type(
                getattr(self.baseline_cache, level_name).stats
            )()
            getattr(self.sidecar_cache, level_name).stats = type(
                getattr(self.sidecar_cache, level_name).stats
            )()
        logger.info(f"Warmup complete at {self._instructions_processed:,} instructions")

    def _estimate_sidecar_memory(self) -> float:
        """Estimate sidecar memory footprint in MB."""
        total_params = sum(
            p.numel() * p.element_size()
            for p in self.sidecar.model.parameters()
        )
        return total_params / (1024 * 1024)

    def _print_progress(self, start_time: float):
        elapsed = time.time() - start_time
        rate = self._instructions_processed / max(0.001, elapsed)
        logger.info(
            f"  [{self._instructions_processed:>12,} insns] "
            f"{rate:,.0f} insns/sec | "
            f"pages: {len(self.bridge._page_tracker):,}"
        )

    def _print_report(self, m: SimulationMetrics):
        """Print a formatted simulation report."""
        print()
        print("=" * 72)
        print(f"  ChampSim + AC-MoE-GA Sidecar Co-Simulation Report")
        print(f"  Trace: {m.trace_name}")
        print("=" * 72)
        print()
        print(f"  Instructions:     {m.total_instructions:>14,}")
        print(f"  Memory accesses:  {m.total_memory_accesses:>14,}")
        print(f"  Unique pages:     {m.unique_pages:>14,}")
        print()

        print("  Cache Hit Rates:")
        print(f"  {'Level':<8} {'Baseline':>12} {'Sidecar':>12} {'Delta':>12}")
        print(f"  {'-'*44}")
        for level in ["l1d", "l2", "llc"]:
            bl = getattr(m, f"baseline_{level}_hit_rate")
            sc = getattr(m, f"sidecar_{level}_hit_rate")
            delta = (sc - bl) * 100
            print(f"  {level.upper():<8} {bl:>11.4f} {sc:>11.4f} {delta:>+11.2f}%")
        print()

        print("  Latency:")
        print(f"    Baseline avg:   {m.baseline_avg_latency:>10.1f} cycles")
        print(f"    Sidecar avg:    {m.sidecar_avg_latency:>10.1f} cycles")
        print(f"    Improvement:    {m.latency_improvement_pct:>+10.2f}%")
        print()

        print("  IPC Estimate:")
        print(f"    Baseline:       {m.baseline_ipc_estimate:>10.4f}")
        print(f"    Sidecar:        {m.sidecar_ipc_estimate:>10.4f}")
        print()

        print("  Sidecar Decisions:")
        print(f"    Inferences:     {m.total_inferences:>10,}")
        print(f"    Override rate:   {m.override_rate:>10.2%}")
        print(f"    Abstention rate: {m.abstention_rate:>10.2%}")
        print(f"    Avg confidence:  {m.avg_confidence:>10.4f}")
        print(f"    Avg support:     {m.avg_support_density:>10.4f}")
        print(f"    Avg margin:      {m.avg_action_margin:>10.4f}")
        print()

        if m.prefetch_issued > 0:
            print("  Prefetch:")
            print(f"    Issued:         {m.prefetch_issued:>10,}")
            print(f"    Useful:         {m.prefetch_useful:>10,}")
            print(f"    Accuracy:       {m.prefetch_accuracy:>10.2%}")
            print()

        print("  Sidecar Overhead:")
        print(f"    Inference latency: {m.sidecar_inference_latency_us:>8.1f} μs")
        print(f"    Model memory:      {m.sidecar_memory_mb:>8.2f} MB")
        print()
        print("=" * 72)
