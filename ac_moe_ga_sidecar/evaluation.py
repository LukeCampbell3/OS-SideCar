"""
Evaluation utilities for AC-MoE-GA Sidecar v1.3.

Adds outcome-quality metrics, regret tracking, and calibration evaluation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from .types import Recommendation, ActionScores, PageAction, BatchAction, KVAction, NUMAAction, BoundaryAction
from .config import SidecarConfig


@dataclass
class DecisionRecord:
    """Record of a single decision point with full context.
    
    v1.3.1: Separates recommendation generation from override execution.
    """
    timestamp: int
    head: str  # Which head made the decision
    confidence: float
    support_density: float
    action_margin: float
    heuristic_action: str
    sidecar_action: str
    override_executed: bool  # Was override actually executed?
    abstained: bool  # Did we abstain?
    heuristic_outcome: float  # Outcome if heuristic was used
    sidecar_outcome: float  # Outcome if sidecar was used
    gain: float  # sidecar_outcome - heuristic_outcome
    regret: float  # heuristic_outcome - sidecar_outcome
    beneficial: bool  # Was sidecar better than heuristic?
    head_used: str  # Which head was actually used


@dataclass
class OverrideRecord:
    """Record of a single override decision with outcome."""
    timestamp: int
    page_id: Optional[int]
    action: str
    action_score: float
    confidence: float
    support_density: float
    margin: float
    heuristic_action: str
    heuristic_score: float
    outcome_better: bool  # Did sidecar action improve outcome?
    outcome_margin: float  # How much better/worse


@dataclass
class HeadGain:
    """Gain metrics for a specific action head."""
    total_overrides: int = 0
    successful_overrides: int = 0
    failed_overrides: int = 0
    total_gain: float = 0.0
    avg_gain: float = 0.0
    best_action_ratio: float = 0.0


@dataclass
class CalibrationMetrics:
    """Calibration metrics for confidence estimates."""
    n_bins: int = 10
    bin_counts: List[int] = field(default_factory=lambda: [0] * 10)
    bin_correct: List[int] = field(default_factory=lambda: [0] * 10)
    bin_confidence: List[float] = field(default_factory=lambda: [0.0] * 10)
    
    def add_sample(self, confidence: float, is_correct: bool):
        """Add a sample to calibration tracking."""
        bin_idx = min(int(confidence * self.n_bins), self.n_bins - 1)
        self.bin_counts[bin_idx] += 1
        self.bin_correct[bin_idx] += 1 if is_correct else 0
        self.bin_confidence[bin_idx] += confidence
    
    def get_ece(self) -> float:
        """Get Expected Calibration Error."""
        total = sum(self.bin_counts)
        if total == 0:
            return 0.0
        
        ece = 0.0
        for i in range(self.n_bins):
            if self.bin_counts[i] > 0:
                avg_conf = self.bin_confidence[i] / self.bin_counts[i]
                avg_acc = self.bin_correct[i] / self.bin_counts[i]
                ece += self.bin_counts[i] * abs(avg_conf - avg_acc)
        
        return ece / total
    
    def get_calibration_error(self) -> float:
        """Get maximum calibration error."""
        max_error = 0.0
        for i in range(self.n_bins):
            if self.bin_counts[i] > 0:
                avg_conf = self.bin_confidence[i] / self.bin_counts[i]
                avg_acc = self.bin_correct[i] / self.bin_counts[i]
                max_error = max(max_error, abs(avg_conf - avg_acc))
        return max_error


@dataclass
class EvaluationResult:
    """Result of an evaluation run with outcome-quality metrics."""
    # Performance metrics
    events_per_second: float
    inferences_per_second: float
    avg_inference_latency_us: float
    p99_inference_latency_us: float
    memory_usage_mb: float
    
    # Behavioral metrics
    abstention_rate: float
    override_rate: float
    
    # Override quality metrics
    override_precision: float
    override_recall: float
    avg_gain: float
    total_gain: float
    
    # Per-head metrics
    page_precision: float
    batch_precision: float
    kv_precision: float
    numa_precision: float
    boundary_precision: float
    
    # Confidence/support metrics
    avg_confidence: float
    avg_support_density: float
    avg_action_margin: float
    confidence_std: float
    support_std: float
    
    # Calibration metrics
    ece: float
    max_calibration_error: float
    
    # Regret metrics
    avg_regret: float
    positive_regret_count: int
    negative_regret_count: int


class OverrideTracker:
    """Tracks override decisions and their outcomes.
    
    v1.3.1: Uses DecisionRecord for proper counterfactual evaluation.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.decisions: deque = deque(maxlen=max_history)
        self.overrides: deque = deque(maxlen=max_history)
        self.heuristic_only_overrides: deque = deque(maxlen=max_history)
        
        # Per-head tracking
        self.page_gain = HeadGain()
        self.batch_gain = HeadGain()
        self.kv_gain = HeadGain()
        self.numa_gain = HeadGain()
        self.boundary_gain = HeadGain()
        
        # Overall metrics
        self.total_overrides = 0
        self.successful_overrides = 0
        self.failed_overrides = 0
        self.total_gain = 0.0
        
        # Decision tracking
        self.total_decisions = 0
        self.abstentions = 0
        self.heuristic_only = 0
        self.overrides_executed = 0
        self.overrides_beneficial = 0
        
    def record_decision(
        self,
        timestamp: int,
        head: str,
        confidence: float,
        support_density: float,
        action_margin: float,
        heuristic_action: str,
        sidecar_action: str,
        override_executed: bool,
        abstained: bool,
        heuristic_outcome: float,
        sidecar_outcome: float,
        beneficial: bool,
        head_used: str,
    ):
        """Record a decision point with full counterfactual context."""
        gain = sidecar_outcome - heuristic_outcome
        regret = heuristic_outcome - sidecar_outcome
        
        record = DecisionRecord(
            timestamp=timestamp,
            head=head,
            confidence=confidence,
            support_density=support_density,
            action_margin=action_margin,
            heuristic_action=heuristic_action,
            sidecar_action=sidecar_action,
            override_executed=override_executed,
            abstained=abstained,
            heuristic_outcome=heuristic_outcome,
            sidecar_outcome=sidecar_outcome,
            gain=gain,
            regret=regret,
            beneficial=beneficial,
            head_used=head_used,
        )
        self.decisions.append(record)
        
        # Update overall metrics
        self.total_decisions += 1
        if abstained:
            self.abstentions += 1
        elif not override_executed:
            self.heuristic_only += 1
            # Track when sidecar agrees with heuristic (keep baseline)
            if sidecar_action == heuristic_action:
                pass  # Sidecar agrees with heuristic
        else:
            self.overrides_executed += 1
            if beneficial:
                self.overrides_beneficial += 1
        
        # Update per-head metrics
        if 'page' in head.lower():
            self.page_gain.total_overrides += 1
            if beneficial:
                self.page_gain.successful_overrides += 1
                self.page_gain.total_gain += gain
            else:
                self.page_gain.failed_overrides += 1
        elif 'batch' in head.lower():
            self.batch_gain.total_overrides += 1
            if beneficial:
                self.batch_gain.successful_overrides += 1
                self.batch_gain.total_gain += gain
            else:
                self.batch_gain.failed_overrides += 1
        elif 'kv' in head.lower():
            self.kv_gain.total_overrides += 1
            if beneficial:
                self.kv_gain.successful_overrides += 1
                self.kv_gain.total_gain += gain
            else:
                self.kv_gain.failed_overrides += 1
        elif 'numa' in head.lower():
            self.numa_gain.total_overrides += 1
            if beneficial:
                self.numa_gain.successful_overrides += 1
                self.numa_gain.total_gain += gain
            else:
                self.numa_gain.failed_overrides += 1
        elif 'boundary' in head.lower():
            self.boundary_gain.total_overrides += 1
            if beneficial:
                self.boundary_gain.successful_overrides += 1
                self.boundary_gain.total_gain += gain
            else:
                self.boundary_gain.failed_overrides += 1
    
    def record_override(
        self,
        timestamp: int,
        page_id: Optional[int],
        action: str,
        action_score: float,
        confidence: float,
        support_density: float,
        margin: float,
        heuristic_action: str,
        heuristic_score: float,
        outcome_better: bool,
        outcome_margin: float,
    ):
        """Record an override decision (legacy, for backward compatibility)."""
        record = OverrideRecord(
            timestamp=timestamp,
            page_id=page_id,
            action=action,
            action_score=action_score,
            confidence=confidence,
            support_density=support_density,
            margin=margin,
            heuristic_action=heuristic_action,
            heuristic_score=heuristic_score,
            outcome_better=outcome_better,
            outcome_margin=outcome_margin,
        )
        self.overrides.append(record)
        
        # Update overall metrics
        self.total_overrides += 1
        if outcome_better:
            self.successful_overrides += 1
            self.total_gain += outcome_margin
        else:
            self.failed_overrides += 1
        
        # Update per-head metrics based on action name
        action_lower = action.lower()
        if 'page' in action_lower:
            self.page_gain.total_overrides += 1
            if outcome_better:
                self.page_gain.successful_overrides += 1
                self.page_gain.total_gain += outcome_margin
            else:
                self.page_gain.failed_overrides += 1
        elif 'batch' in action_lower:
            self.batch_gain.total_overrides += 1
            if outcome_better:
                self.batch_gain.successful_overrides += 1
                self.batch_gain.total_gain += outcome_margin
            else:
                self.batch_gain.failed_overrides += 1
        elif 'kv' in action_lower:
            self.kv_gain.total_overrides += 1
            if outcome_better:
                self.kv_gain.successful_overrides += 1
                self.kv_gain.total_gain += outcome_margin
            else:
                self.kv_gain.failed_overrides += 1
        elif 'numa' in action_lower:
            self.numa_gain.total_overrides += 1
            if outcome_better:
                self.numa_gain.successful_overrides += 1
                self.numa_gain.total_gain += outcome_margin
            else:
                self.numa_gain.failed_overrides += 1
        elif 'boundary' in action_lower:
            self.boundary_gain.total_overrides += 1
            if outcome_better:
                self.boundary_gain.successful_overrides += 1
                self.boundary_gain.total_gain += outcome_margin
            else:
                self.boundary_gain.failed_overrides += 1
    
    def get_statistics(self) -> Dict:
        """Get override statistics."""
        # Compute action margin statistics from decisions
        if len(self.decisions) > 0:
            margins = [r.action_margin for r in self.decisions if r.action_margin is not None]
            avg_margin = sum(margins) / len(margins) if margins else 0.0
            max_margin = max(margins) if margins else 0.0
            min_margin = min(margins) if margins else 0.0
        else:
            avg_margin = 0.0
            max_margin = 0.0
            min_margin = 0.0
        
        # Compute decision statistics
        total_decisions = self.total_decisions
        abstentions = self.abstentions
        heuristic_only = self.heuristic_only
        overrides_executed = self.overrides_executed
        overrides_beneficial = self.overrides_beneficial
        
        # Override metrics (only count actual overrides)
        # Precision: fraction of executed overrides where sidecar was beneficial
        if overrides_executed > 0:
            override_precision = overrides_beneficial / overrides_executed
        else:
            override_precision = 0.0
        
        # Recall: fraction of cases where override was beneficial that we actually executed
        # This requires knowing how many cases would have benefited from override
        # For now, use a proxy based on total decisions
        if total_decisions > 0:
            override_recall = overrides_beneficial / max(1, total_decisions)
        else:
            override_recall = 0.0
        
        stats = {
            'total_decisions': total_decisions,
            'abstentions': abstentions,
            'heuristic_only': heuristic_only,
            'overrides_executed': overrides_executed,
            'overrides_beneficial': overrides_beneficial,
            'override_precision': override_precision,
            'override_recall': override_recall,
            'override_rate': overrides_executed / max(1, total_decisions),
            'abstention_rate': abstentions / max(1, total_decisions),
            'total_gain': self.total_gain,
            'avg_gain': self.total_gain / max(1, overrides_executed) if overrides_executed > 0 else 0.0,
            'avg_action_margin': avg_margin,
            'max_action_margin': max_margin,
            'min_action_margin': min_margin,
        }
        
        # Per-head statistics
        for head_name, head in [
            ('page', self.page_gain),
            ('batch', self.batch_gain),
            ('kv', self.kv_gain),
            ('numa', self.numa_gain),
            ('boundary', self.boundary_gain),
        ]:
            if head.total_overrides > 0:
                stats[f'{head_name}_precision'] = head.successful_overrides / head.total_overrides
                stats[f'{head_name}_avg_gain'] = head.total_gain / head.total_overrides
            else:
                stats[f'{head_name}_precision'] = 0.0
                stats[f'{head_name}_avg_gain'] = 0.0
        
        return stats


class RegretTracker:
    """Tracks regret (heuristic vs sidecar performance)."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.heuristic_outcomes: deque = deque(maxlen=max_history)
        self.sidecar_outcomes: deque = deque(maxlen=max_history)
        self.heuristic_vs_sidecar: deque = deque(maxlen=max_history)
        
    def record_heuristic_outcome(self, outcome: float):
        """Record heuristic outcome."""
        self.heuristic_outcomes.append(outcome)
    
    def record_sidecar_outcome(self, outcome: float):
        """Record sidecar outcome."""
        self.sidecar_outcomes.append(outcome)
        
        # Compute regret
        if len(self.heuristic_outcomes) > 0:
            regret = self.heuristic_outcomes[-1] - outcome
            self.heuristic_vs_sidecar.append(regret)
    
    def get_regret_stats(self) -> Dict:
        """Get regret statistics."""
        if len(self.heuristic_vs_sidecar) == 0:
            return {
                'avg_regret': 0.0,
                'max_regret': 0.0,
                'min_regret': 0.0,
                'positive_regret_count': 0,
                'negative_regret_count': 0,
            }
        
        regrets = list(self.heuristic_vs_sidecar)
        return {
            'avg_regret': np.mean(regrets),
            'max_regret': np.max(regrets),
            'min_regret': np.min(regrets),
            'positive_regret_count': sum(1 for r in regrets if r > 0),  # sidecar better
            'negative_regret_count': sum(1 for r in regrets if r < 0),  # heuristic better
        }


class ConfidenceTracker:
    """Tracks confidence calibration and support density."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.confidences: deque = deque(maxlen=max_history)
        self.support_densities: deque = deque(maxlen=max_history)
        self.action_margins: deque = deque(maxlen=max_history)
        self.abstentions: deque = deque(maxlen=max_history)
        
    def record_inference(
        self,
        confidence: float,
        support_density: float,
        action_margin: float,
        abstain: bool,
    ):
        """Record inference results."""
        self.confidences.append(confidence)
        self.support_densities.append(support_density)
        self.action_margins.append(action_margin)
        self.abstentions.append(abstain)
    
    def get_statistics(self) -> Dict:
        """Get confidence/support statistics."""
        if len(self.confidences) == 0:
            return {
                'avg_confidence': 0.0,
                'avg_support_density': 0.0,
                'avg_action_margin': 0.0,
                'abstention_rate': 0.0,
                'confidence_std': 0.0,
                'support_std': 0.0,
            }
        
        return {
            'avg_confidence': np.mean(self.confidences),
            'avg_support_density': np.mean(self.support_densities),
            'avg_action_margin': np.mean(self.action_margins),
            'abstention_rate': sum(self.abstentions) / len(self.abstentions),
            'confidence_std': np.std(self.confidences),
            'support_std': np.std(self.support_densities),
            'confidence_min': np.min(self.confidences),
            'confidence_max': np.max(self.confidences),
            'support_min': np.min(self.support_densities),
            'support_max': np.max(self.support_densities),
        }


class Evaluator:
    """Main evaluation class for v1.3.1.
    
    v1.3.1: Uses DecisionRecord for proper counterfactual evaluation.
    """
    
    def __init__(self, config: SidecarConfig):
        self.config = config
        self.override_tracker = OverrideTracker()
        self.regret_tracker = RegretTracker()
        self.confidence_tracker = ConfidenceTracker()
        self.calibration_metrics = CalibrationMetrics()
        self._last_decision = None
        
    def record_decision(
        self,
        timestamp: int,
        head: str,
        confidence: float,
        support_density: float,
        action_margin: float,
        heuristic_action: str,
        sidecar_action: str,
        override_executed: bool,
        abstained: bool,
        heuristic_outcome: float,
        sidecar_outcome: float,
        beneficial: bool,
        head_used: str,
    ):
        """Record a decision point with full counterfactual context."""
        self.override_tracker.record_decision(
            timestamp=timestamp,
            head=head,
            confidence=confidence,
            support_density=support_density,
            action_margin=action_margin,
            heuristic_action=heuristic_action,
            sidecar_action=sidecar_action,
            override_executed=override_executed,
            abstained=abstained,
            heuristic_outcome=heuristic_outcome,
            sidecar_outcome=sidecar_outcome,
            beneficial=beneficial,
            head_used=head_used,
        )
    
    def record_override(
        self,
        recommendation: Recommendation,
        heuristic_action: str,
        heuristic_score: float,
        outcome_better: bool,
        outcome_margin: float = 0.0,
    ):
        """Record an override decision with outcome (legacy, for backward compatibility)."""
        # Get the top page action
        top_action = "PRESERVE"  # Default
        if recommendation.action_scores and recommendation.action_scores.page_scores:
            top_action = max(recommendation.action_scores.page_scores.items(), key=lambda x: x[1])[0].name
        
        # Get the actual margin from the model output (top1 - top2)
        # This is the true margin that the model uses for decisions
        actual_margin = recommendation.action_margin if recommendation.action_margin is not None else 0.0
        
        self.override_tracker.record_override(
            timestamp=0,  # Will be set by caller
            page_id=None,  # Will be set by caller
            action=top_action,
            action_score=recommendation.action_scores.page_scores.get('PRESERVE', 0) if recommendation.action_scores else 0,
            confidence=recommendation.inferred_state.confidence,
            support_density=recommendation.support_density,
            margin=actual_margin,  # Use the actual model margin
            heuristic_action=heuristic_action,
            heuristic_score=heuristic_score,
            outcome_better=outcome_better,
            outcome_margin=outcome_margin,
        )
    
    def record_inference(self, recommendation: Recommendation):
        """Record inference results for calibration tracking."""
        # Get action margin from recommendation if available
        action_margin = 0.0
        if recommendation.action_margin is not None:
            action_margin = recommendation.action_margin
        elif recommendation.action_scores and recommendation.action_scores.page_scores:
            # Compute margin from page scores
            page_scores = recommendation.action_scores.page_scores
            if page_scores:
                sorted_scores = sorted(page_scores.values(), reverse=True)
                if len(sorted_scores) >= 2:
                    action_margin = sorted_scores[0] - sorted_scores[1]
        
        self.confidence_tracker.record_inference(
            confidence=recommendation.inferred_state.confidence,
            support_density=recommendation.support_density,
            action_margin=action_margin,
            abstain=recommendation.abstain,
        )
    
    def record_outcomes(self, heuristic_outcome: float, sidecar_outcome: float):
        """Record outcomes for regret calculation."""
        self.regret_tracker.record_heuristic_outcome(heuristic_outcome)
        self.regret_tracker.record_sidecar_outcome(sidecar_outcome)
    
    def get_evaluation_report(self) -> Dict:
        """Get comprehensive evaluation report."""
        return {
            'override_stats': self.override_tracker.get_statistics(),
            'regret_stats': self.regret_tracker.get_regret_stats(),
            'confidence_stats': self.confidence_tracker.get_statistics(),
            'calibration': {
                'ece': self.calibration_metrics.get_ece(),
                'max_calibration_error': self.calibration_metrics.get_calibration_error(),
            },
        }
