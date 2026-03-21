"""Integration tests for AC-MoE-GA Sidecar."""

import pytest
import torch
import numpy as np
import tempfile
import os

from ac_moe_ga_sidecar import ACMoEGASidecar, SidecarConfig, BalancedBuildConfig
from ac_moe_ga_sidecar.utils import (
    create_synthetic_event, create_workload_trace, benchmark_sidecar,
    set_seed, get_device_info, estimate_memory_requirements
)
from ac_moe_ga_sidecar.config import EventType


@pytest.fixture
def sidecar():
    return ACMoEGASidecar()


class TestACMoEGASidecar:
    def test_initialization_default(self):
        sidecar = ACMoEGASidecar()
        assert sidecar.config is not None
        assert sidecar.model is not None
        assert sidecar.engine is not None

    def test_initialization_with_config(self):
        config = BalancedBuildConfig()
        sidecar = ACMoEGASidecar(config=config)
        assert sidecar.config == config

    def test_initialization_cpu(self):
        sidecar = ACMoEGASidecar(device='cpu')
        assert sidecar.device == torch.device('cpu')

    def test_process_event(self, sidecar):
        event = create_synthetic_event()
        result = sidecar.process_event(event)
        # First event shouldn't trigger inference
        assert result is None

    def test_process_batch(self, sidecar):
        events = create_workload_trace(100)
        results = sidecar.process_batch(events)
        assert len(results) >= 1

    def test_force_inference(self, sidecar):
        # Process some events first
        for i in range(10):
            sidecar.process_event(create_synthetic_event(timestamp=i))
        
        result = sidecar.force_inference()
        assert result is not None
        assert result.recommendation is not None

    def test_get_recommendation(self, sidecar):
        for i in range(10):
            sidecar.process_event(create_synthetic_event(timestamp=i))
        
        rec = sidecar.get_recommendation()
        assert rec is not None
        assert rec.inferred_state is not None
        assert rec.action_scores is not None

    def test_should_override_heuristic(self, sidecar):
        for i in range(10):
            sidecar.process_event(create_synthetic_event(timestamp=i))
        
        result = sidecar.should_override_heuristic()
        assert isinstance(result, bool)

    def test_get_statistics(self, sidecar):
        for i in range(100):
            sidecar.process_event(create_synthetic_event(timestamp=i))
        
        stats = sidecar.get_statistics()
        
        assert 'total_events' in stats
        assert 'total_inferences' in stats
        assert 'abstention_rate' in stats
        assert 'override_rate' in stats
        assert 'runtime_memory_mb' in stats
        assert 'model_size_mb' in stats
        
        assert stats['total_events'] == 100

    def test_reset(self, sidecar):
        for i in range(100):
            sidecar.process_event(create_synthetic_event(timestamp=i))
        
        sidecar.reset()
        
        # After reset, engine state should be cleared
        assert sidecar.engine.model_state is None
        assert len(sidecar.engine.event_buffer) == 0

    def test_save_and_load_model(self, sidecar):
        # Process some events to change state
        for i in range(50):
            sidecar.process_event(create_synthetic_event(timestamp=i))
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        
        try:
            sidecar.save_model(path)
            assert os.path.exists(path)
            
            # Create new sidecar and load
            new_sidecar = ACMoEGASidecar()
            new_sidecar.load_model(path)
            
            # Models should have same parameters
            for (n1, p1), (n2, p2) in zip(
                sidecar.model.named_parameters(),
                new_sidecar.model.named_parameters()
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)

    def test_repr(self, sidecar):
        repr_str = repr(sidecar)
        assert 'ACMoEGASidecar' in repr_str
        assert 'device=' in repr_str
        assert 'model_size_mb=' in repr_str


class TestWorkloadPatterns:
    def test_sequential_workload(self, sidecar):
        events = create_workload_trace(500, workload_type="sequential")
        results = sidecar.process_batch(events)
        
        assert len(results) >= 1
        # Sequential workload should have good locality
        for result in results:
            state = result.recommendation.inferred_state
            # Region should show streaming or clustered behavior
            assert state.region_state.streaming >= 0 or state.region_state.clustered_reuse >= 0

    def test_random_workload(self, sidecar):
        events = create_workload_trace(500, workload_type="random")
        results = sidecar.process_batch(events)
        
        assert len(results) >= 1

    def test_syscall_heavy_workload(self, sidecar):
        events = create_workload_trace(500, workload_type="syscall_heavy")
        results = sidecar.process_batch(events)
        
        assert len(results) >= 1
        # Should detect syscall-heavy phase
        for result in results:
            state = result.recommendation.inferred_state
            # Process phase should reflect syscall activity
            assert state.process_phase is not None

    def test_mixed_workload(self, sidecar):
        events = create_workload_trace(500, workload_type="mixed")
        results = sidecar.process_batch(events)
        
        assert len(results) >= 1


class TestBenchmark:
    def test_benchmark_runs(self, sidecar):
        result = benchmark_sidecar(sidecar, num_events=1000)
        
        assert result.events_per_second > 0
        assert result.inferences_per_second > 0
        assert result.avg_inference_latency_us > 0
        assert result.memory_usage_mb > 0
        assert 0 <= result.abstention_rate <= 1
        assert 0 <= result.override_rate <= 1

    def test_benchmark_different_workloads(self, sidecar):
        for workload in ["sequential", "random", "mixed"]:
            result = benchmark_sidecar(sidecar, num_events=500, workload_type=workload)
            assert result.events_per_second > 0


class TestUtilities:
    def test_set_seed_reproducibility(self):
        set_seed(42)
        events1 = create_workload_trace(10)
        
        set_seed(42)
        events2 = create_workload_trace(10)
        
        # Events should be identical
        for e1, e2 in zip(events1, events2):
            assert e1.virtual_page == e2.virtual_page
            assert e1.event_type == e2.event_type

    def test_get_device_info(self):
        info = get_device_info()
        
        assert 'cuda_available' in info
        assert 'cpu_count' in info
        assert isinstance(info['cuda_available'], bool)

    def test_estimate_memory_requirements(self):
        config = BalancedBuildConfig()
        estimates = estimate_memory_requirements(config)
        
        assert 'model_mb' in estimates
        assert 'runtime_state_mb' in estimates
        assert 'total_mb' in estimates
        
        assert estimates['model_mb'] > 0
        assert estimates['total_mb'] > estimates['model_mb']


class TestEdgeCases:
    def test_empty_event_buffer(self, sidecar):
        # Force inference with no events
        result = sidecar.force_inference()
        assert result is not None
        # Should still produce a recommendation (with high uncertainty)

    def test_missing_virtual_page(self, sidecar):
        event = create_synthetic_event(event_type=EventType.SYSCALL_ENTRY)
        event = MicroEvent(
            timestamp_bucket=0,
            cpu_id=0,
            numa_node=0,
            pid=1000,
            tid=1000,
            pc_bucket=0,
            event_type=EventType.SYSCALL_ENTRY.value,
            opcode_class=0,
            virtual_page=None,  # Missing
            region_id=None,
        )
        
        # Should handle gracefully
        result = sidecar.process_event(event)
        # No crash

    def test_rapid_event_processing(self, sidecar):
        # Process many events rapidly
        for i in range(10000):
            event = create_synthetic_event(timestamp=i)
            sidecar.process_event(event)
        
        stats = sidecar.get_statistics()
        assert stats['total_events'] == 10000

    def test_alternating_pids(self, sidecar):
        # Alternate between different processes
        for i in range(100):
            pid = 1000 + (i % 5)
            event = create_synthetic_event(pid=pid, timestamp=i)
            sidecar.process_event(event)
        
        # Should track multiple processes
        assert len(sidecar.engine.runtime_state.processes) >= 5


# Import MicroEvent for edge case test
from ac_moe_ga_sidecar.types import MicroEvent
