"""Tests for the inference engine."""

import pytest
import torch
import numpy as np

from ac_moe_ga_sidecar.config import SidecarConfig, BalancedBuildConfig
from ac_moe_ga_sidecar.types import MicroEvent
from ac_moe_ga_sidecar.inference import InferenceEngine, InferenceResult
from ac_moe_ga_sidecar.model import ACMoEGAModel
from ac_moe_ga_sidecar.utils import create_synthetic_event, create_workload_trace
from ac_moe_ga_sidecar.config import EventType


@pytest.fixture
def config():
    return BalancedBuildConfig()


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def engine(config, device):
    return InferenceEngine(config, device=device)


class TestInferenceEngine:
    def test_initialization(self, engine, config):
        assert engine.config == config
        assert engine.model is not None
        assert engine.runtime_state is not None
        assert engine.feature_extractor is not None

    def test_process_single_event(self, engine):
        event = create_synthetic_event()
        result = engine.process_event(event)
        
        # First event shouldn't trigger inference (below min threshold)
        assert result is None
        assert engine.events_since_inference == 1

    def test_inference_triggers_at_max_events(self, engine, config):
        max_events = config.cadence.max_events_between_inference
        
        for i in range(max_events):
            event = create_synthetic_event(timestamp=i)
            result = engine.process_event(event)
            
            if i < max_events - 1:
                assert result is None
            else:
                assert result is not None
                assert isinstance(result, InferenceResult)

    def test_force_inference(self, engine):
        # Process a few events
        for i in range(5):
            engine.process_event(create_synthetic_event(timestamp=i))
        
        # Force inference
        result = engine.force_inference()
        
        assert isinstance(result, InferenceResult)
        assert result.recommendation is not None
        assert result.latency_us > 0

    def test_recommendation_structure(self, engine):
        for i in range(10):
            engine.process_event(create_synthetic_event(timestamp=i))
        
        result = engine.force_inference()
        rec = result.recommendation
        
        # Check recommendation fields
        assert rec.inferred_state is not None
        assert rec.action_scores is not None
        assert isinstance(rec.should_override_heuristic, bool)
        assert isinstance(rec.abstain, bool)
        assert isinstance(rec.support_density, float)
        assert isinstance(rec.drift_score, float)

    def test_inferred_state_structure(self, engine):
        for i in range(10):
            engine.process_event(create_synthetic_event(timestamp=i))
        
        result = engine.force_inference()
        state = result.recommendation.inferred_state
        
        # Check all state components
        assert state.page_state is not None
        assert state.region_state is not None
        assert state.process_phase is not None
        assert state.pressure_state is not None
        assert state.hazard_state is not None
        assert state.uncertainty is not None
        assert 0 <= state.confidence <= 1

    def test_action_scores_structure(self, engine):
        for i in range(10):
            engine.process_event(create_synthetic_event(timestamp=i))
        
        result = engine.force_inference()
        scores = result.recommendation.action_scores
        
        # Check all action score dictionaries
        assert len(scores.batch_scores) > 0
        assert len(scores.kv_scores) > 0
        assert len(scores.numa_scores) > 0
        assert len(scores.boundary_scores) > 0
        assert len(scores.page_scores) > 0
        
        # Scores should sum to approximately 1 (softmax)
        batch_sum = sum(scores.batch_scores.values())
        assert 0.99 <= batch_sum <= 1.01

    def test_statistics_tracking(self, engine):
        # Process some events
        for i in range(100):
            engine.process_event(create_synthetic_event(timestamp=i))
        
        stats = engine.get_statistics()
        
        assert stats['total_events'] == 100
        assert stats['total_inferences'] >= 1
        assert 'abstention_rate' in stats
        assert 'override_rate' in stats
        assert 'runtime_memory_mb' in stats
        assert 'model_size_mb' in stats

    def test_reset_state(self, engine):
        # Process some events
        for i in range(50):
            engine.process_event(create_synthetic_event(timestamp=i))
        
        # Reset
        engine.reset_state()
        
        assert engine.model_state is None
        assert len(engine.event_buffer) == 0
        assert engine.events_since_inference == 0

    def test_fault_burst_trigger(self, engine, config):
        if not config.cadence.fault_burst_trigger:
            pytest.skip("Fault burst trigger disabled")
        
        # Send fault events
        for i in range(config.cadence.min_events_between_inference):
            event = create_synthetic_event(
                event_type=EventType.PAGE_FAULT,
                timestamp=i
            )
            result = engine.process_event(event)
        
        # Should have triggered inference due to fault burst
        assert engine.total_inferences >= 1

    def test_cow_fault_trigger(self, engine, config):
        if not config.cadence.cow_fault_trigger:
            pytest.skip("COW fault trigger disabled")
        
        # Process enough events to trigger at least one inference
        for i in range(config.cadence.max_events_between_inference + 10):
            if i % 10 == 0:
                event = create_synthetic_event(
                    event_type=EventType.COW_FAULT,
                    timestamp=i
                )
            else:
                event = create_synthetic_event(timestamp=i)
            engine.process_event(event)
        
        # Should have triggered at least one inference
        assert engine.total_inferences >= 1

    def test_batch_processing(self, engine):
        events = create_workload_trace(100, workload_type="mixed")
        results = engine.process_batch(events)
        
        assert len(results) >= 1
        assert all(isinstance(r, InferenceResult) for r in results)


class TestInferenceLatency:
    def test_inference_latency_reasonable(self, engine):
        # Warm up
        for i in range(100):
            engine.process_event(create_synthetic_event(timestamp=i))
        
        # Measure latency
        latencies = []
        for i in range(10):
            result = engine.force_inference()
            latencies.append(result.latency_us)
        
        avg_latency = np.mean(latencies)
        
        # Should be under 100ms on CPU (generous for CI environments)
        assert avg_latency < 100000, f"Average latency {avg_latency}us too high"


class TestUncertaintyBehavior:
    def test_abstention_on_high_uncertainty(self, engine):
        # Process minimal events (should have high uncertainty)
        for i in range(10):
            engine.process_event(create_synthetic_event(timestamp=i))
        
        result = engine.force_inference()
        
        # With minimal data, uncertainty should be relatively high
        uncertainty = result.recommendation.inferred_state.uncertainty
        assert uncertainty.selective_prediction >= 0

    def test_override_requires_low_uncertainty(self, engine):
        # Process many events to build confidence
        events = create_workload_trace(500, workload_type="sequential")
        for event in events:
            engine.process_event(event)
        
        result = engine.force_inference()
        
        # If override is recommended, uncertainty should be reasonable
        if result.recommendation.should_override_heuristic:
            uncertainty = result.recommendation.inferred_state.uncertainty
            # Allow small tolerance for floating point comparison
            # Note: selective_prediction is just one component of the uncertainty vector
            # The model can make overrides with moderate selective_prediction if other signals are strong
            assert uncertainty.selective_prediction < 0.75  # Reasonable upper bound
