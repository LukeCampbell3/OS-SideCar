"""
Bridge between ChampSim trace records and AC-MoE-GA MicroEvents.

Translates architectural simulation data (IPs, memory addresses, branches)
into the sidecar's MicroEvent schema, maintaining runtime context like
stride detection, page reuse tracking, and phase estimation.
"""

import numpy as np
from collections import defaultdict
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field

from ac_moe_ga_sidecar.types import MicroEvent
from ac_moe_ga_sidecar.config import EventType
from .trace_parser import ChampSimRecord


@dataclass
class AccessTracker:
    """Tracks per-page access patterns for stride/reuse detection."""
    last_access_time: int = 0
    access_count: int = 0
    last_cache_line: int = 0
    stride_history: list = field(default_factory=list)
    is_write: bool = False


class ChampSimEventBridge:
    """
    Converts ChampSim trace records into sidecar MicroEvents.
    
    Maintains enough state to produce meaningful features:
    - Page access history for reuse distance
    - Stride detection across cache lines
    - Phase estimation from branch/syscall patterns
    - Region grouping from address clustering
    """

    def __init__(
        self,
        pid: int = 1000,
        cpu_id: int = 0,
        numa_node: int = 0,
        region_shift: int = 21,  # 2MB region granularity
        max_stride_history: int = 4,
    ):
        self.pid = pid
        self.cpu_id = cpu_id
        self.numa_node = numa_node
        self.region_shift = region_shift
        self.max_stride_history = max_stride_history

        # Tracking state
        self._page_tracker: Dict[int, AccessTracker] = defaultdict(AccessTracker)
        self._last_page: int = 0
        self._last_cache_line: int = 0
        self._timestamp: int = 0
        self._branch_count: int = 0
        self._mem_count: int = 0
        self._fault_injection_rate: float = 0.0

    def set_fault_injection_rate(self, rate: float):
        """Set synthetic fault injection rate (0.0 - 1.0) for page fault simulation."""
        self._fault_injection_rate = max(0.0, min(1.0, rate))

    def translate(self, record: ChampSimRecord) -> Optional[MicroEvent]:
        """
        Translate a ChampSim record into a MicroEvent.
        
        Returns None for non-memory instructions (unless they're branches
        that contribute to phase detection).
        """
        self._timestamp += 1

        if record.is_branch:
            self._branch_count += 1

        if not record.has_memory_access:
            return None

        self._mem_count += 1
        is_write = record.is_store
        addr = record.destination_memory if is_write else record.source_memory
        page = addr >> 12
        cache_line = addr >> 6
        region = addr >> self.region_shift

        # Determine event type
        event_type = self._classify_event(page, is_write)

        # Build opcode class from access pattern
        opcode_class = self._estimate_opcode_class(record, is_write)

        # PTE flags estimation (simulated)
        pte_flags = self._estimate_pte_flags(page, is_write)

        # Update tracking
        tracker = self._page_tracker[page]
        tracker.access_count += 1
        tracker.is_write = is_write

        # Stride detection
        if tracker.last_cache_line != 0:
            stride = cache_line - tracker.last_cache_line
            tracker.stride_history.append(stride)
            if len(tracker.stride_history) > self.max_stride_history:
                tracker.stride_history.pop(0)

        tracker.last_cache_line = cache_line
        tracker.last_access_time = self._timestamp

        event = MicroEvent(
            timestamp_bucket=self._timestamp,
            cpu_id=self.cpu_id,
            numa_node=self.numa_node,
            pid=self.pid,
            tid=self.pid,  # Single-threaded trace
            pc_bucket=(record.ip >> 6) & 0xFFFF,  # Bucket the PC
            event_type=event_type,
            opcode_class=opcode_class,
            virtual_page=page & 0xFFFFF,  # 20-bit page bucket
            region_id=region & 0x3FF,      # 10-bit region bucket
            rw_flag=is_write,
            mode=0,  # User mode
            pte_flags=pte_flags,
        )

        self._last_page = page
        self._last_cache_line = cache_line

        return event

    def _classify_event(self, page: int, is_write: bool) -> int:
        """Classify the memory event, potentially injecting faults."""
        tracker = self._page_tracker[page]

        # First access to a page → simulate page fault
        if tracker.access_count == 0:
            if self._fault_injection_rate > 0 and np.random.random() < self._fault_injection_rate:
                return EventType.PAGE_FAULT.value
            # Even without injection, first-touch is a compulsory miss
            return EventType.PAGE_FAULT.value

        if is_write:
            # COW detection: first write after many reads
            if tracker.access_count > 5 and not tracker.is_write:
                if np.random.random() < 0.1:  # 10% COW probability on write-after-read
                    return EventType.COW_FAULT.value
            return EventType.MEMORY_WRITE.value

        return EventType.MEMORY_READ.value

    def _estimate_opcode_class(self, record: ChampSimRecord, is_write: bool) -> int:
        """Estimate opcode family from trace record."""
        if record.is_branch:
            return 4  # BRANCH
        if is_write:
            return 2  # STORE
        return 1  # LOAD

    def _estimate_pte_flags(self, page: int, is_write: bool) -> int:
        """Estimate PTE flags based on access pattern."""
        flags = 0x01  # PRESENT
        if is_write:
            flags |= 0x02  # WRITABLE
            flags |= 0x20  # DIRTY
        flags |= 0x08  # USER
        flags |= 0x10  # ACCESSED

        tracker = self._page_tracker[page]
        if tracker.access_count > 100:
            flags |= 0x40  # HUGEPAGE candidate (hot page)

        return flags

    @property
    def stats(self) -> Dict:
        return {
            "total_instructions": self._timestamp,
            "memory_accesses": self._mem_count,
            "branches": self._branch_count,
            "unique_pages": len(self._page_tracker),
            "memory_ratio": self._mem_count / max(1, self._timestamp),
        }
