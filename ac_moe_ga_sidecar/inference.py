"""
Inference Engine for AC-MoE-GA Sidecar.

Handles the complete inference pipeline including:
- Event batching and cadence control
- Feature extraction
- Model inference
- Recommendation generation
- Override decision logic
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from .config import SidecarConfig, InferenceCadence, UncertaintyThresholds
from .types import (
    MicroEvent, Recommendation, InferredState, ActionScores,
    PageState, RegionState, ProcessPhase, PressureState, HazardState,
    UncertaintyVector, BatchAction, KVAction, NUMAAction, BoundaryAction, PageAction
)
from .model import ACMoEGAModel, ModelState, ModelOutput
from .runtime_state import RuntimeStateManager
from .feature_extraction import FeatureExtractor, ExtractedFeatures


@dataclass
class InferenceResult:
    """Result of a single inference."""
    recommendation: Recommendation
    model_output: ModelOutput
    latency_us: float
    events_processed: int


class InferenceEngine:
    """
    Main inference engine for the sidecar.
    
    Manages the complete pipeline from raw events to recommendations,
    including cadence control and override decisions.
    """
    
    def __init__(
        self,
        config: SidecarConfig,
        model: Optional[ACMoEGAModel] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        if model is None:
            self.model = ACMoEGAModel(config)
        else:
            self.model = model
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize runtime state manager
        self.runtime_state = RuntimeStateManager(config)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(config)
        
        # Model state for recurrence
        self.model_state: Optional[ModelState] = None
        
        # Event buffer for batching
        self.event_buffer: deque = deque(maxlen=config.cadence.max_events_between_inference)
        self.events_since_inference = 0
        
        # Trigger tracking
        self.last_inference_time = 0
        self.fault_burst_count = 0
        self.syscall_burst_count = 0
        
        # Statistics
        self.total_inferences = 0
        self.total_events = 0
        self.abstention_count = 0
        self.override_count = 0
        
        # Evaluation tracking (v1.3)
        self._evaluator = None
        self._last_recommendation = None
        self._last_heuristic_action = None
        self._last_heuristic_score = None
        
    def process_event(self, event: MicroEvent) -> Optional[InferenceResult]:
        """
        Process a single micro-event.
        
        Returns InferenceResult if inference was triggered, None otherwise.
        """
        self.total_events += 1
        self.events_since_inference += 1
        
        # Update runtime state
        page_summary, region_summary, process_summary = self.runtime_state.process_event(event)
        
        # Extract features
        features = self.feature_extractor.extract(
            event, page_summary, region_summary, process_summary
        )
        
        # Buffer the event
        self.event_buffer.append((event, features, page_summary, region_summary, process_summary))
        
        # Check if inference should be triggered
        if self._should_trigger_inference(event):
            return self._run_inference()
        
        return None
    
    def process_batch(self, events: List[MicroEvent]) -> List[InferenceResult]:
        """Process a batch of events."""
        results = []
        for event in events:
            result = self.process_event(event)
            if result is not None:
                results.append(result)
        return results
    
    def force_inference(self) -> InferenceResult:
        """Force an inference regardless of cadence."""
        return self._run_inference()
    
    def _should_trigger_inference(self, event: MicroEvent) -> bool:
        """Determine if inference should be triggered."""
        cadence = self.config.cadence
        
        # Minimum events threshold
        if self.events_since_inference < cadence.min_events_between_inference:
            return False
        
        # Maximum events threshold
        if self.events_since_inference >= cadence.max_events_between_inference:
            return True
        
        # Fault burst trigger
        if cadence.fault_burst_trigger and event.event_type in [2, 3]:
            self.fault_burst_count += 1
            if self.fault_burst_count >= 3:
                self.fault_burst_count = 0
                return True
        else:
            self.fault_burst_count = max(0, self.fault_burst_count - 1)
        
        # COW fault trigger
        if cadence.cow_fault_trigger and event.event_type == 3:
            return True
        
        # Syscall burst trigger
        if cadence.syscall_burst_trigger and event.event_type in [5, 6]:
            self.syscall_burst_count += 1
            if self.syscall_burst_count >= 5:
                self.syscall_burst_count = 0
                return True
        else:
            self.syscall_burst_count = max(0, self.syscall_burst_count - 1)
        
        # Pressure change trigger
        if cadence.pressure_change_trigger:
            pressure = self.runtime_state.get_system_pressure_vector()
            if np.max(np.abs(np.diff(pressure))) > 0.2:
                return True
        
        return False
    
    def _run_inference(self) -> InferenceResult:
        """Run model inference on buffered events."""
        import time
        start_time = time.perf_counter()
        
        self.total_inferences += 1
        events_processed = self.events_since_inference
        self.events_since_inference = 0
        
        # Get most recent features
        if not self.event_buffer:
            # Create dummy features if buffer is empty
            features = self._create_dummy_features()
        else:
            _, features, page_summary, region_summary, process_summary = self.event_buffer[-1]
        
        # Convert to tensors
        input_tensors = self.feature_extractor.to_tensors(features, self.device)
        
        # Run model
        with torch.no_grad():
            output = self.model(input_tensors, self.model_state)
        
        # Update model state
        self.model_state = output.new_state
        
        # Build recommendation
        recommendation = self._build_recommendation(output)
        
        # Get calibrated confidence for evaluation
        calibrated_conf = recommendation.inferred_state.confidence
        
        # Track statistics
        if recommendation.abstain:
            self.abstention_count += 1
        if recommendation.should_override_heuristic:
            self.override_count += 1
        
        # Store recommendation for outcome tracking (v1.3)
        self._last_recommendation = recommendation
        self._last_heuristic_action = None
        self._last_heuristic_score = None
        
        # Note: Decision recording is handled by evaluate_sidecar, not inference engine
        # The inference engine should not record decisions because it doesn't have
        # access to heuristic actions and outcomes in production
        
        latency_us = (time.perf_counter() - start_time) * 1e6
        
        return InferenceResult(
            recommendation=recommendation,
            model_output=output,
            latency_us=latency_us,
            events_processed=events_processed,
        )
    
    def _build_recommendation(self, output: ModelOutput) -> Recommendation:
        """Build recommendation from model output v1.1."""
        thresholds = self.config.uncertainty
        
        # Extract uncertainty
        uncertainty = output.uncertainty[0].cpu().numpy()
        uncertainty_vec = UncertaintyVector(
            calibration=float(uncertainty[0]),
            selective_prediction=float(uncertainty[1]),
            ranking=float(uncertainty[2]),
            ood=float(uncertainty[3]),
            observability=float(uncertainty[4]),
        )
        
        # Determine abstention (v1.1: use model's decision)
        abstain = bool(output.should_abstain[0].item())
        
        # Get calibrated confidence and action margin
        calibrated_conf = float(output.calibrated_confidence[0].item())
        action_margin = float(output.action_margin[0].item())
        
        # Determine override with stricter criteria (v1.1)
        should_override = self._should_override_heuristic_v11(
            output, thresholds, calibrated_conf, action_margin
        )
        
        # Build inferred state
        inferred_state = self._build_inferred_state(output, calibrated_conf)
        
        # Build action scores
        action_scores = self._build_action_scores(output)
        
        # Ensure boolean types for recommendation fields
        should_override_bool = bool(should_override)
        abstain_bool = bool(abstain)
        
        return Recommendation(
            inferred_state=inferred_state,
            action_scores=action_scores,
            should_override_heuristic=should_override_bool and not abstain_bool,
            abstain=abstain_bool,
            expert_used=output.used_experts[0] if output.used_experts else None,
            prototype_match=int(output.proto_match[0].item()) if output.proto_match is not None else None,
            support_density=float(output.support_density[0].item()),
            drift_score=float(output.drift_score[0].item()),
            action_margin=float(output.action_margin[0].item()) if output.action_margin is not None else None,
        )
    
    def _should_override_heuristic_v11(
        self,
        output: ModelOutput,
        thresholds: UncertaintyThresholds,
        calibrated_conf: float,
        action_margin: float,
    ) -> bool:
        """
        Determine if heuristic should be overridden (v1.1).
        
        Key insight: Action margin is the most reliable indicator of model conviction.
        An untrained model will have margins near 0, while a trained model with
        clear preferences will have larger margins.
        
        v1.4: Support-dependent thresholds to improve recall without sacrificing precision.
        """
        uncertainty = output.uncertainty[0].cpu().numpy()
        
        # Normalize action margin to [0, 1] for comparison
        import math
        normalized_margin = 1.0 / (1.0 + math.exp(-action_margin))  # sigmoid
        
        # Hard blocks - any one of these prevents override
        if uncertainty[1] > 0.75:  # selective uncertainty - relaxed
            return False
        if uncertainty[3] > thresholds.ood_threshold:  # OOD
            return False
        
        # Check support density (minimum bar)
        support = float(output.support_density[0].item())
        if support < 0.10:  # Lowered bar for support
            return False
        
        # Check familiarity
        familiarity = float(output.familiarity[0].item())
        if familiarity < 0.12:  # Lowered bar for familiarity
            return False
        
        # PRIMARY GATE: Action margin must show some preference
        # Raw margin of 0.01 -> normalized ~0.5025
        # Raw margin of 0.05 -> normalized ~0.512
        # Raw margin of 0.1 -> normalized ~0.525
        # Require margin > 0.01 (normalized > 0.5025) for any preference
        if normalized_margin < 0.5025:  # Lowered threshold for any preference
            return False
        
        # SECONDARY GATE: Confidence must be reasonable
        if calibrated_conf < 0.48:  # Lowered threshold for confidence
            return False
        
        # TERTIARY GATE: Ranking uncertainty must not be too high
        if uncertainty[2] > 0.62:  # Relaxed from 0.60
            return False
        
        # If all gates pass, check combined score for final decision
        # This allows some variability in override rate
        combined_score = (
            0.35 * min(1.0, calibrated_conf) +
            0.25 * min(1.0, support) +
            0.2 * min(1.0, familiarity) +
            0.1 * (1.0 - uncertainty[2]) +  # Lower ranking uncertainty is better
            0.1 * normalized_margin
        )
        
        # Support-dependent threshold for recall improvement
        # Higher support = lower threshold (more willing to override)
        # Lower support = higher threshold (more conservative)
        # Linear interpolation between base_threshold (high support) and base_threshold + slope (low support)
        support_factor = min(1.0, max(0.0, (support - 0.10) / 0.30))  # 0 at support=0.10, 1 at support=0.40
        dynamic_threshold = thresholds.override_base_threshold - thresholds.override_threshold_slope * support_factor
        
        # Override if combined score is high enough
        # With typical untrained values (conf=0.62, support=0.43, fam=0.27, rank_unc=0.48, margin=0.503):
        # score = 0.35*0.62 + 0.25*0.43 + 0.2*0.27 + 0.1*0.52 + 0.1*0.503 = 0.217 + 0.108 + 0.054 + 0.052 + 0.050 = 0.481
        # Target: ~10-35% override rate when model is confident and well-supported
        # Threshold tuned to achieve realistic override rate
        return combined_score >= dynamic_threshold
    
    def _build_inferred_state(self, output: ModelOutput, calibrated_conf: float) -> InferredState:
        """Build inferred state from model output."""
        heads = output.head_outputs
        
        # Page state
        page_state = self.model.output_heads.page_state.to_page_state(heads['page_state'])
        
        # Region state
        region_state = self.model.output_heads.region_state.to_region_state(heads['region_state'])
        
        # Process phase
        process_phase = self.model.output_heads.process_phase.to_process_phase(heads['process_phase'])
        
        # Pressure state (from runtime)
        pressure = self.runtime_state.get_system_pressure_vector()
        pressure_state = PressureState(
            memory_pressure=float(pressure[0]),
            bandwidth_pressure=float(pressure[1]),
            reclaim_pressure=float(pressure[2]),
            queue_pressure=float(pressure[3]),
            remote_numa_pressure=float(pressure[4]),
            kv_residency_pressure=float(pressure[5]),
            kernel_crossing_pressure=float(pressure[6]),
        )
        
        # Hazard state
        hazard_state = self.model.output_heads.hazard.to_hazard_state(heads['hazard'])
        
        # Uncertainty
        uncertainty = self.model.output_heads.uncertainty.to_uncertainty_vector(output.uncertainty)
        
        # Confidence - use calibrated confidence from v1.1
        confidence = calibrated_conf
        
        return InferredState(
            page_state=page_state,
            region_state=region_state,
            process_phase=process_phase,
            pressure_state=pressure_state,
            hazard_state=hazard_state,
            uncertainty=uncertainty,
            confidence=confidence,
        )
    
    def _build_action_scores(self, output: ModelOutput) -> ActionScores:
        """Build action scores from model output."""
        heads = output.head_outputs
        
        return ActionScores(
            batch_scores=self.model.output_heads.batch_scheduler.to_scores(heads['batch_actions']),
            kv_scores=self.model.output_heads.kv_policy.to_scores(heads['kv_actions']),
            numa_scores=self.model.output_heads.numa_placement.to_scores(heads['numa_actions']),
            boundary_scores=self.model.output_heads.boundary_control.to_scores(heads['boundary_actions']),
            page_scores=self.model.output_heads.page_policy.to_scores(heads['page_actions']),
        )
    
    def _create_dummy_features(self) -> ExtractedFeatures:
        """Create dummy features when buffer is empty."""
        return ExtractedFeatures(
            low8_bucket=0, high8_bucket=0, alignment_bucket=0,
            small_int_bucket=0, delta_bucket=32, hamming_bucket=0,
            continuous_features=np.zeros(6, dtype=np.float32),
            bitfield_features=np.zeros(16, dtype=np.float32),
            sketch_features=np.zeros(32, dtype=np.float32),
            page_hash_bucket=0, offset_bucket=0, cache_line_bucket=0,
            addr_alignment_bucket=0, stride_bucket=64, reuse_dist_bucket=63,
            locality_cluster=0, entropy_bucket=0,
            address_flags=np.zeros(5, dtype=np.float32),
            event_type=0, fault_class=0, syscall_class=0,
            opcode_family=0, transition_type=0, result_class=0,
            pte_flags=np.zeros(11, dtype=np.float32),
            vma_class=0, protection_domain=0,
            read_count_bucket=0, write_count_bucket=0,
            fault_count_bucket=0, cow_count_bucket=0, recency_bucket=31,
            volatility_features=np.zeros(4, dtype=np.float32),
            pressure_features=np.zeros(12, dtype=np.float32),
            missingness_mask=np.ones(8, dtype=np.float32),
            freshness_ages=np.zeros(4, dtype=np.float32),
            source_quality=np.zeros(2, dtype=np.float32),
            conflict_score=0.0, consistency_score=0.0,
        )
    
    def get_statistics(self) -> Dict:
        """Get inference statistics."""
        return {
            'total_events': self.total_events,
            'total_inferences': self.total_inferences,
            'abstention_count': self.abstention_count,
            'override_count': self.override_count,
            'abstention_rate': self.abstention_count / max(1, self.total_inferences),
            'override_rate': self.override_count / max(1, self.total_inferences),
            'events_per_inference': self.total_events / max(1, self.total_inferences),
            'runtime_memory_mb': self.runtime_state.get_memory_usage_mb(),
            'model_size_mb': self.model.get_model_size_mb(),
        }
    
    def reset_state(self):
        """Reset all state for a fresh start."""
        self.model_state = None
        self.event_buffer.clear()
        self.events_since_inference = 0
        self.fault_burst_count = 0
        self.syscall_burst_count = 0
        self._last_recommendation = None
        self._last_heuristic_action = None
        self._last_heuristic_score = None
    
    def set_evaluator(self, evaluator):
        """Set the evaluator for tracking outcomes."""
        self._evaluator = evaluator
    
    def record_outcome(self, heuristic_action: str, heuristic_score: float, outcome_better: bool, outcome_margin: float = 0.0):
        """Record an outcome for override evaluation."""
        if self._last_recommendation is not None and self._evaluator is not None:
            self._evaluator.record_override(
                recommendation=self._last_recommendation,
                heuristic_action=heuristic_action,
                heuristic_score=heuristic_score,
                outcome_better=outcome_better,
                outcome_margin=outcome_margin,
            )
            self._last_recommendation = None
            self._last_heuristic_action = None
            self._last_heuristic_score = None
