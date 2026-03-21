"""
Runtime State Manager for AC-MoE-GA Sidecar.

Manages the deterministic runtime state (Level 0) including counters,
EMAs, recency clocks, and sketches. This is the always-on, cheap,
non-neural state that holds much of the real live signal.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import hashlib
import struct

from .config import SidecarConfig, UpdateConfig, RuntimeMemoryBudget
from .types import MicroEvent, EntitySummary


@dataclass
class PageRuntimeState:
    """Runtime state for a single page."""
    page_id: int
    read_count: int = 0
    write_count: int = 0
    fault_count: int = 0
    cow_count: int = 0
    last_read_time: int = 0
    last_write_time: int = 0
    last_fault_time: int = 0
    read_ema: float = 0.0
    write_ema: float = 0.0
    fault_ema: float = 0.0
    burst_streak: int = 0
    reuse_distance_sum: int = 0
    reuse_count: int = 0
    neighbor_touches: int = 0
    volatility: float = 0.0
    
    def to_summary(self, current_time: int) -> EntitySummary:
        recency = min(63, (current_time - max(self.last_read_time, self.last_write_time)) // 100)
        return EntitySummary(
            read_count=min(63, self.read_count // 10),
            write_count=min(63, self.write_count // 10),
            fault_count=min(63, self.fault_count),
            cow_count=min(63, self.cow_count),
            recency_bucket=recency,
            volatility=self.volatility,
            neighbor_density=self.neighbor_touches / max(1, self.read_count + self.write_count),
            bandwidth_pressure=0.0,
            reclaim_pressure=0.0,
            queue_depth=0.0,
            numa_pressure=0.0,
            kernel_entry_pressure=0.0,
            kv_pressure=0.0,
        )


@dataclass
class RegionRuntimeState:
    """Runtime state for a memory region."""
    region_id: int
    page_count: int = 0
    total_reads: int = 0
    total_writes: int = 0
    total_faults: int = 0
    growth_rate_ema: float = 0.0
    volatility_ema: float = 0.0
    fragmentation_score: float = 0.0
    streaming_score: float = 0.0
    clustering_score: float = 0.0
    last_activity_time: int = 0


@dataclass
class ProcessRuntimeState:
    """Runtime state for a process."""
    pid: int
    syscall_count: int = 0
    fault_count: int = 0
    kernel_entry_count: int = 0
    working_set_size: int = 0
    working_set_ema: float = 0.0
    syscall_density_ema: float = 0.0
    phase_indicator: int = 0  # Encoded phase
    last_activity_time: int = 0


@dataclass
class SystemRuntimeState:
    """Global system runtime state."""
    memory_pressure: float = 0.0
    bandwidth_pressure: float = 0.0
    reclaim_pressure: float = 0.0
    queue_pressure: float = 0.0
    numa_pressure: float = 0.0
    kv_pressure: float = 0.0
    kernel_crossing_pressure: float = 0.0
    total_events: int = 0
    fault_rate_ema: float = 0.0
    syscall_rate_ema: float = 0.0


class LocalitySketch:
    """Compact sketch for tracking locality patterns."""
    
    def __init__(self, size: int = 256):
        self.size = size
        self.sketch = np.zeros(size, dtype=np.uint16)
        self.last_addresses = np.zeros(16, dtype=np.uint64)
        self.last_idx = 0
        
    def update(self, address: int):
        # Update sketch
        bucket = hash(address) % self.size
        self.sketch[bucket] = min(65535, self.sketch[bucket] + 1)
        
        # Track recent addresses for stride detection
        self.last_addresses[self.last_idx % 16] = address
        self.last_idx += 1
        
    def get_locality_score(self) -> float:
        """Compute locality score based on sketch concentration."""
        if self.sketch.sum() == 0:
            return 0.0
        normalized = self.sketch / self.sketch.sum()
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        max_entropy = np.log2(self.size)
        return 1.0 - (entropy / max_entropy)
    
    def get_stride_pattern(self) -> Tuple[int, float]:
        """Detect dominant stride pattern."""
        if self.last_idx < 2:
            return 0, 0.0
        
        strides = np.diff(self.last_addresses[:min(self.last_idx, 16)])
        if len(strides) == 0:
            return 0, 0.0
            
        # Find most common stride
        unique, counts = np.unique(strides, return_counts=True)
        if len(counts) == 0:
            return 0, 0.0
        dominant_idx = counts.argmax()
        confidence = counts[dominant_idx] / len(strides)
        return int(unique[dominant_idx]), confidence


class ReuseDistanceTracker:
    """Tracks reuse distance for pages."""
    
    def __init__(self, max_distance: int = 10000):
        self.max_distance = max_distance
        self.last_access: Dict[int, int] = {}
        self.access_counter = 0
        self.distance_histogram = np.zeros(64, dtype=np.uint32)  # Log-scale buckets
        
    def record_access(self, page_id: int) -> int:
        """Record access and return reuse distance."""
        self.access_counter += 1
        
        if page_id in self.last_access:
            distance = self.access_counter - self.last_access[page_id]
            bucket = min(63, int(np.log2(distance + 1)))
            self.distance_histogram[bucket] += 1
        else:
            distance = self.max_distance
            
        self.last_access[page_id] = self.access_counter
        
        # Evict old entries to bound memory
        if len(self.last_access) > self.max_distance:
            threshold = self.access_counter - self.max_distance
            self.last_access = {k: v for k, v in self.last_access.items() if v > threshold}
            
        return min(distance, self.max_distance)
    
    def get_bucket(self, distance: int) -> int:
        """Convert distance to log-scale bucket."""
        return min(63, int(np.log2(distance + 1)))


class RuntimeStateManager:
    """
    Manages all deterministic runtime state.
    
    This is Level 0 of the memory hierarchy - always-on, cheap, non-neural.
    """
    
    def __init__(self, config: SidecarConfig):
        self.config = config
        self.update_config = config.updates
        
        # Entity states
        self.pages: Dict[int, PageRuntimeState] = {}
        self.regions: Dict[int, RegionRuntimeState] = {}
        self.processes: Dict[int, ProcessRuntimeState] = {}
        self.system = SystemRuntimeState()
        
        # Sketches and trackers
        self.locality_sketch = LocalitySketch()
        self.reuse_tracker = ReuseDistanceTracker()
        
        # Event counters
        self.event_count = 0
        self.current_time = 0
        
        # Rolling windows for pressure computation
        self.fault_window: List[int] = []
        self.syscall_window: List[int] = []
        self.window_size = 1000
        
    def process_event(self, event: MicroEvent) -> Tuple[EntitySummary, EntitySummary, EntitySummary]:
        """
        Process a micro-event and update all relevant state.
        
        Returns summaries for page, region, and process.
        """
        self.event_count += 1
        self.current_time = event.timestamp_bucket
        
        # Update page state
        page_summary = self._update_page_state(event)
        
        # Update region state
        region_summary = self._update_region_state(event)
        
        # Update process state
        process_summary = self._update_process_state(event)
        
        # Update system state
        self._update_system_state(event)
        
        return page_summary, region_summary, process_summary
    
    def _update_page_state(self, event: MicroEvent) -> EntitySummary:
        """Update page-level state."""
        if event.virtual_page is None:
            return self._empty_summary()
            
        page_id = event.virtual_page
        
        if page_id not in self.pages:
            self.pages[page_id] = PageRuntimeState(page_id=page_id)
            
        page = self.pages[page_id]
        
        # Update counters
        is_write = event.rw_flag if event.rw_flag is not None else False
        if is_write:
            page.write_count = min(self.update_config.counter_saturation, page.write_count + 1)
            page.last_write_time = self.current_time
            page.write_ema = self.update_config.ema_decay_fast * page.write_ema + (1 - self.update_config.ema_decay_fast)
        else:
            page.read_count = min(self.update_config.counter_saturation, page.read_count + 1)
            page.last_read_time = self.current_time
            page.read_ema = self.update_config.ema_decay_fast * page.read_ema + (1 - self.update_config.ema_decay_fast)
            
        # Track faults
        if event.event_type in [2, 3]:  # PAGE_FAULT, COW_FAULT
            page.fault_count = min(self.update_config.counter_saturation, page.fault_count + 1)
            page.last_fault_time = self.current_time
            page.fault_ema = self.update_config.ema_decay_fast * page.fault_ema + (1 - self.update_config.ema_decay_fast)
            
            if event.event_type == 3:  # COW_FAULT
                page.cow_count = min(self.update_config.counter_saturation, page.cow_count + 1)
                
        # Update reuse distance
        reuse_dist = self.reuse_tracker.record_access(page_id)
        page.reuse_distance_sum += reuse_dist
        page.reuse_count += 1
        
        # Update locality sketch
        self.locality_sketch.update(page_id)
        
        # Check for neighbor touches
        for neighbor_offset in [-1, 1]:
            neighbor_id = page_id + neighbor_offset
            if neighbor_id in self.pages:
                page.neighbor_touches += 1
                
        # Update volatility
        time_since_last = self.current_time - max(page.last_read_time, page.last_write_time)
        if time_since_last < 100:
            page.volatility = self.update_config.ema_decay_fast * page.volatility + (1 - self.update_config.ema_decay_fast)
        else:
            page.volatility *= self.update_config.ema_decay_slow
            
        # Evict old pages if over limit
        self._evict_old_pages()
        
        return page.to_summary(self.current_time)
    
    def _update_region_state(self, event: MicroEvent) -> EntitySummary:
        """Update region-level state."""
        if event.region_id is None:
            return self._empty_summary()
            
        region_id = event.region_id
        
        if region_id not in self.regions:
            self.regions[region_id] = RegionRuntimeState(region_id=region_id)
            
        region = self.regions[region_id]
        
        # Update counters
        is_write = event.rw_flag if event.rw_flag is not None else False
        if is_write:
            region.total_writes += 1
        else:
            region.total_reads += 1
            
        if event.event_type in [2, 3]:
            region.total_faults += 1
            
        region.last_activity_time = self.current_time
        
        # Update streaming/clustering scores based on locality
        locality_score = self.locality_sketch.get_locality_score()
        stride, stride_conf = self.locality_sketch.get_stride_pattern()
        
        if stride_conf > 0.7 and abs(stride) <= 4096:  # Sequential-ish
            region.streaming_score = self.update_config.ema_decay_fast * region.streaming_score + (1 - self.update_config.ema_decay_fast)
        else:
            region.streaming_score *= self.update_config.ema_decay_slow
            
        if locality_score > 0.7:
            region.clustering_score = self.update_config.ema_decay_fast * region.clustering_score + (1 - self.update_config.ema_decay_fast)
        else:
            region.clustering_score *= self.update_config.ema_decay_slow
            
        # Evict old regions
        self._evict_old_regions()
        
        return EntitySummary(
            read_count=min(63, region.total_reads // 100),
            write_count=min(63, region.total_writes // 100),
            fault_count=min(63, region.total_faults),
            cow_count=0,
            recency_bucket=min(31, (self.current_time - region.last_activity_time) // 100),
            volatility=region.volatility_ema,
            neighbor_density=region.clustering_score,
            bandwidth_pressure=self.system.bandwidth_pressure,
            reclaim_pressure=self.system.reclaim_pressure,
            queue_depth=self.system.queue_pressure,
            numa_pressure=self.system.numa_pressure,
            kernel_entry_pressure=self.system.kernel_crossing_pressure,
            kv_pressure=self.system.kv_pressure,
        )
    
    def _update_process_state(self, event: MicroEvent) -> EntitySummary:
        """Update process-level state."""
        pid = event.pid
        
        if pid not in self.processes:
            self.processes[pid] = ProcessRuntimeState(pid=pid)
            
        proc = self.processes[pid]
        
        # Update counters
        if event.event_type in [5, 6]:  # SYSCALL_ENTRY, SYSCALL_EXIT
            proc.syscall_count += 1
            proc.syscall_density_ema = self.update_config.ema_decay_fast * proc.syscall_density_ema + (1 - self.update_config.ema_decay_fast)
            
        if event.event_type in [7, 8]:  # KERNEL_ENTRY, KERNEL_EXIT
            proc.kernel_entry_count += 1
            
        if event.event_type in [2, 3]:
            proc.fault_count += 1
            
        proc.last_activity_time = self.current_time
        
        # Evict old processes
        self._evict_old_processes()
        
        return EntitySummary(
            read_count=0,
            write_count=0,
            fault_count=min(63, proc.fault_count),
            cow_count=0,
            recency_bucket=min(31, (self.current_time - proc.last_activity_time) // 100),
            volatility=proc.syscall_density_ema,
            neighbor_density=0.0,
            bandwidth_pressure=self.system.bandwidth_pressure,
            reclaim_pressure=self.system.reclaim_pressure,
            queue_depth=self.system.queue_pressure,
            numa_pressure=self.system.numa_pressure,
            kernel_entry_pressure=self.system.kernel_crossing_pressure,
            kv_pressure=self.system.kv_pressure,
        )
    
    def _update_system_state(self, event: MicroEvent):
        """Update global system state."""
        self.system.total_events += 1
        
        # Track fault rate
        if event.event_type in [2, 3]:
            self.fault_window.append(self.current_time)
            self.system.fault_rate_ema = self.update_config.ema_decay_fast * self.system.fault_rate_ema + (1 - self.update_config.ema_decay_fast)
        else:
            self.system.fault_rate_ema *= self.update_config.ema_decay_slow
            
        # Track syscall rate
        if event.event_type in [5, 6]:
            self.syscall_window.append(self.current_time)
            self.system.syscall_rate_ema = self.update_config.ema_decay_fast * self.system.syscall_rate_ema + (1 - self.update_config.ema_decay_fast)
        else:
            self.system.syscall_rate_ema *= self.update_config.ema_decay_slow
            
        # Trim windows
        cutoff = self.current_time - self.window_size
        self.fault_window = [t for t in self.fault_window if t > cutoff]
        self.syscall_window = [t for t in self.syscall_window if t > cutoff]
        
        # Update pressures based on rates
        self.system.kernel_crossing_pressure = min(1.0, len(self.syscall_window) / 100)
        
    def _evict_old_pages(self):
        """Evict old pages to stay within memory budget."""
        if len(self.pages) > self.config.max_active_pages:
            # Sort by last activity and remove oldest
            sorted_pages = sorted(
                self.pages.items(),
                key=lambda x: max(x[1].last_read_time, x[1].last_write_time)
            )
            to_remove = len(self.pages) - self.config.max_active_pages
            for page_id, _ in sorted_pages[:to_remove]:
                del self.pages[page_id]
                
    def _evict_old_regions(self):
        """Evict old regions."""
        if len(self.regions) > self.config.max_active_regions:
            sorted_regions = sorted(
                self.regions.items(),
                key=lambda x: x[1].last_activity_time
            )
            to_remove = len(self.regions) - self.config.max_active_regions
            for region_id, _ in sorted_regions[:to_remove]:
                del self.regions[region_id]
                
    def _evict_old_processes(self):
        """Evict old processes."""
        if len(self.processes) > self.config.max_active_processes:
            sorted_procs = sorted(
                self.processes.items(),
                key=lambda x: x[1].last_activity_time
            )
            to_remove = len(self.processes) - self.config.max_active_processes
            for pid, _ in sorted_procs[:to_remove]:
                del self.processes[pid]
                
    def _empty_summary(self) -> EntitySummary:
        """Return empty summary."""
        return EntitySummary(
            read_count=0, write_count=0, fault_count=0, cow_count=0,
            recency_bucket=31, volatility=0.0, neighbor_density=0.0,
            bandwidth_pressure=0.0, reclaim_pressure=0.0, queue_depth=0.0,
            numa_pressure=0.0, kernel_entry_pressure=0.0, kv_pressure=0.0,
        )
    
    def get_system_pressure_vector(self) -> NDArray[np.float32]:
        """Get current system pressure as a vector."""
        return np.array([
            self.system.memory_pressure,
            self.system.bandwidth_pressure,
            self.system.reclaim_pressure,
            self.system.queue_pressure,
            self.system.numa_pressure,
            self.system.kv_pressure,
            self.system.kernel_crossing_pressure,
            self.system.fault_rate_ema,
            self.system.syscall_rate_ema,
            0.0, 0.0, 0.0,  # Padding to 12
        ], dtype=np.float32)
    
    def get_memory_usage_mb(self) -> float:
        """Estimate current memory usage."""
        # Rough estimates
        page_size = 100  # bytes per page state
        region_size = 80
        process_size = 60
        
        total = (
            len(self.pages) * page_size +
            len(self.regions) * region_size +
            len(self.processes) * process_size +
            self.locality_sketch.size * 2 +
            len(self.reuse_tracker.last_access) * 16
        )
        return total / (1024 * 1024)
