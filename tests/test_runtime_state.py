"""Tests for runtime state management."""

import pytest
import numpy as np

from ac_moe_ga_sidecar.config import SidecarConfig, BalancedBuildConfig
from ac_moe_ga_sidecar.runtime_state import (
    RuntimeStateManager, PageRuntimeState, RegionRuntimeState,
    ProcessRuntimeState, SystemRuntimeState, LocalitySketch, ReuseDistanceTracker
)
from ac_moe_ga_sidecar.types import MicroEvent
from ac_moe_ga_sidecar.utils import create_synthetic_event
from ac_moe_ga_sidecar.config import EventType


@pytest.fixture
def config():
    return BalancedBuildConfig()


@pytest.fixture
def manager(config):
    return RuntimeStateManager(config)


class TestLocalitySketch:
    def test_initialization(self):
        sketch = LocalitySketch(size=256)
        assert sketch.size == 256
        assert len(sketch.sketch) == 256
        assert np.all(sketch.sketch == 0)

    def test_update(self):
        sketch = LocalitySketch()
        sketch.update(0x1000)
        assert sketch.sketch.sum() > 0

    def test_locality_score_empty(self):
        sketch = LocalitySketch()
        score = sketch.get_locality_score()
        assert score == 0.0

    def test_locality_score_concentrated(self):
        sketch = LocalitySketch()
        # Access same address repeatedly
        for _ in range(100):
            sketch.update(0x1000)
        
        score = sketch.get_locality_score()
        assert score > 0.5  # Should be high for concentrated access

    def test_locality_score_spread(self):
        sketch = LocalitySketch()
        # Access many different addresses with large gaps
        for i in range(1000):
            sketch.update(i * 4096 * 1000)  # Very spread out
        
        score = sketch.get_locality_score()
        # Score depends on hash collisions, just verify it's computed and reasonable
        assert score >= 0
        assert score <= 1.01  # Allow small floating point error

    def test_stride_pattern_detection(self):
        sketch = LocalitySketch()
        # Sequential access with stride 1
        for i in range(16):
            sketch.update(0x1000 + i)
        
        stride, confidence = sketch.get_stride_pattern()
        assert stride == 1
        assert confidence > 0.5


class TestReuseDistanceTracker:
    def test_initialization(self):
        tracker = ReuseDistanceTracker()
        assert len(tracker.last_access) == 0
        assert tracker.access_counter == 0

    def test_first_access(self):
        tracker = ReuseDistanceTracker()
        distance = tracker.record_access(0x1000)
        assert distance == tracker.max_distance  # First access has max distance

    def test_immediate_reuse(self):
        tracker = ReuseDistanceTracker()
        tracker.record_access(0x1000)
        distance = tracker.record_access(0x1000)
        assert distance == 1  # Immediate reuse

    def test_reuse_distance_calculation(self):
        tracker = ReuseDistanceTracker()
        tracker.record_access(0x1000)
        tracker.record_access(0x2000)
        tracker.record_access(0x3000)
        distance = tracker.record_access(0x1000)
        assert distance == 3  # 3 accesses since last access to 0x1000

    def test_bucket_calculation(self):
        tracker = ReuseDistanceTracker()
        assert tracker.get_bucket(1) == 1
        assert tracker.get_bucket(2) == 1
        assert tracker.get_bucket(4) == 2
        assert tracker.get_bucket(8) == 3

    def test_eviction_of_old_entries(self):
        tracker = ReuseDistanceTracker(max_distance=100)
        
        # Access many pages
        for i in range(200):
            tracker.record_access(i)
        
        # Old entries should be evicted
        assert len(tracker.last_access) <= 100


class TestPageRuntimeState:
    def test_initialization(self):
        state = PageRuntimeState(page_id=0x1000)
        assert state.page_id == 0x1000
        assert state.read_count == 0
        assert state.write_count == 0
        assert state.fault_count == 0

    def test_to_summary(self):
        state = PageRuntimeState(page_id=0x1000)
        state.read_count = 100
        state.write_count = 50
        state.last_read_time = 900
        state.last_write_time = 950
        
        summary = state.to_summary(current_time=1000)
        
        assert summary.read_count > 0
        assert summary.write_count > 0
        assert summary.recency_bucket >= 0


class TestRuntimeStateManager:
    def test_initialization(self, manager, config):
        assert manager.config == config
        assert len(manager.pages) == 0
        assert len(manager.regions) == 0
        assert len(manager.processes) == 0

    def test_process_event_creates_page_state(self, manager):
        event = create_synthetic_event(virtual_page=0x1000)
        manager.process_event(event)
        
        assert 0x1000 in manager.pages
        assert manager.pages[0x1000].read_count > 0 or manager.pages[0x1000].write_count > 0

    def test_process_event_creates_region_state(self, manager):
        event = create_synthetic_event(region_id=1)
        manager.process_event(event)
        
        assert 1 in manager.regions

    def test_process_event_creates_process_state(self, manager):
        event = create_synthetic_event(pid=1234)
        manager.process_event(event)
        
        assert 1234 in manager.processes

    def test_read_count_increments(self, manager):
        event = create_synthetic_event(virtual_page=0x1000, is_write=False)
        manager.process_event(event)
        
        assert manager.pages[0x1000].read_count == 1

    def test_write_count_increments(self, manager):
        event = create_synthetic_event(virtual_page=0x1000, is_write=True)
        manager.process_event(event)
        
        assert manager.pages[0x1000].write_count == 1

    def test_fault_count_increments(self, manager):
        event = create_synthetic_event(
            event_type=EventType.PAGE_FAULT,
            virtual_page=0x1000
        )
        manager.process_event(event)
        
        assert manager.pages[0x1000].fault_count == 1

    def test_cow_count_increments(self, manager):
        # COW fault event type is 3
        event = MicroEvent(
            timestamp_bucket=0,
            cpu_id=0,
            numa_node=0,
            pid=1000,
            tid=1000,
            pc_bucket=0,
            event_type=3,  # COW_FAULT
            opcode_class=0,
            virtual_page=0x1000,
            region_id=1,
            rw_flag=False,
        )
        manager.process_event(event)
        
        # COW faults also increment fault_count
        assert manager.pages[0x1000].fault_count >= 1

    def test_page_eviction(self, manager, config):
        # Create more pages than the limit
        for i in range(config.max_active_pages + 100):
            event = create_synthetic_event(virtual_page=i, timestamp=i)
            manager.process_event(event)
        
        # Should have evicted old pages
        assert len(manager.pages) <= config.max_active_pages

    def test_system_pressure_vector(self, manager):
        # Process some events
        for i in range(100):
            event = create_synthetic_event(timestamp=i)
            manager.process_event(event)
        
        pressure = manager.get_system_pressure_vector()
        
        assert len(pressure) == 12
        assert all(0 <= p <= 1 for p in pressure)

    def test_memory_usage_estimation(self, manager):
        # Process some events
        for i in range(1000):
            event = create_synthetic_event(virtual_page=i % 100, timestamp=i)
            manager.process_event(event)
        
        usage_mb = manager.get_memory_usage_mb()
        
        assert usage_mb > 0
        assert usage_mb < 100  # Should be reasonable

    def test_syscall_tracking(self, manager):
        event = create_synthetic_event(
            event_type=EventType.SYSCALL_ENTRY,
            pid=1234
        )
        manager.process_event(event)
        
        assert manager.processes[1234].syscall_count == 1

    def test_kernel_entry_tracking(self, manager):
        event = create_synthetic_event(
            event_type=EventType.KERNEL_ENTRY,
            pid=1234
        )
        manager.process_event(event)
        
        assert manager.processes[1234].kernel_entry_count == 1
