"""
Head-specific workload generators for v1.5.

Each generator creates workloads that stress-test a specific head's decision space,
with distinct reward surfaces that make head specialization learnable.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from .types import MicroEvent
from .config import EventType


@dataclass
class HeadWorkloadConfig:
    """Configuration for a head-specific workload."""
    num_events: int
    pid: int = 1000
    seed: int = 42


def generate_page_workload(config: HeadWorkloadConfig) -> List[MicroEvent]:
    """
    Generate page-policy workload that stresses page-specific decisions.
    
    This workload creates clear page-specific patterns:
    - Hot pages that should be preserved
    - Cold pages that should be reclaimed
    - COW-sensitive transitions
    - Fault-on-reaccess patterns
    
    Reward surface for page head:
    - PRESERVE hot page: +0.8
    - PRESERVE cold page: -0.2
    - RECLAIM_CANDIDATE cold page: +0.7
    - RECLAIM_CANDIDATE hot page: -0.3
    """
    np.random.seed(config.seed)
    events = []
    
    # Track page states
    page_state = {}  # page_id -> state (0=cold, 1=hot)
    current_page = np.random.randint(0, 2**16)
    
    for i in range(config.num_events):
        # Determine page state based on event pattern
        # Hot pages: 40% of the time
        is_hot = np.random.random() < 0.4
        virtual_page = current_page + (i % 16)
        
        # Track page state
        page_state[virtual_page] = 1 if is_hot else 0
        
        # Generate event type based on page state
        r = np.random.random()
        
        if r < 0.35:
            # Hot page read - should preserve
            event_type = EventType.MEMORY_READ
            is_write = False
        elif r < 0.55:
            # Cold page read - could reclaim
            event_type = EventType.MEMORY_READ
            is_write = False
        elif r < 0.70:
            # Hot page write - should preserve
            event_type = EventType.MEMORY_WRITE
            is_write = True
        elif r < 0.85:
            # Cold page write - could reclaim
            event_type = EventType.MEMORY_WRITE
            is_write = True
        elif r < 0.92:
            # Page fault on cold page - good time to reclaim
            event_type = EventType.PAGE_FAULT
            is_write = False
            virtual_page = np.random.randint(0, 2**20)
        else:
            # COW fault - COW-sensitive transition
            event_type = EventType.COW_FAULT
            is_write = False
            virtual_page = current_page + np.random.randint(0, 100)
        
        event = create_synthetic_event(
            event_type=event_type,
            pid=config.pid,
            virtual_page=virtual_page,
            is_write=is_write,
            timestamp=i,
        )
        events.append(event)
        
        # Shift working set periodically
        if i % 500 == 0:
            current_page = np.random.randint(0, 2**16)
    
    return events


def generate_batch_workload(config: HeadWorkloadConfig) -> List[MicroEvent]:
    """
    Generate batch-policy workload that stresses batch-specific decisions.
    
    This workload creates batch-specific patterns:
    - Burst arrivals that benefit from batching
    - Queue growth that requires batch size adjustment
    - Latency/throughput tradeoffs
    
    Reward surface for batch head:
    - Appropriate batch size: +0.6
    - Too small batch (throughput loss): -0.2
    - Too large batch (latency penalty): -0.3
    - Queue stabilization: +0.5
    """
    np.random.seed(config.seed)
    events = []
    
    # Track queue state
    queue_depth = 0
    current_page = np.random.randint(0, 2**16)
    
    for i in range(config.num_events):
        # Simulate queue dynamics
        if i % 10 == 0:
            # Burst phase - queue grows
            queue_depth += np.random.randint(5, 15)
        elif i % 20 == 0:
            # Processing phase - queue shrinks
            queue_depth = max(0, queue_depth - np.random.randint(3, 8))
        else:
            # Normal phase - small changes
            queue_depth += np.random.randint(-2, 3)
            queue_depth = max(0, min(queue_depth, 100))
        
        # Generate event type based on queue state
        r = np.random.random()
        
        if r < 0.25:
            # High queue - batch beneficial
            event_type = EventType.MEMORY_READ
            is_write = False
        elif r < 0.45:
            # Medium queue - normal operation
            event_type = EventType.MEMORY_WRITE
            is_write = True
        elif r < 0.65:
            # Low queue - small batches better
            event_type = EventType.MEMORY_READ
            is_write = False
        elif r < 0.80:
            # Queue spike - batch beneficial
            event_type = EventType.MEMORY_WRITE
            is_write = True
        elif r < 0.90:
            # Queue overflow risk - batch beneficial
            event_type = EventType.MEMORY_READ
            is_write = False
        else:
            # Queue stable - normal operation
            event_type = EventType.MEMORY_WRITE
            is_write = True
        
        event = create_synthetic_event(
            event_type=event_type,
            pid=config.pid,
            virtual_page=current_page + np.random.randint(0, 32),
            is_write=is_write,
            timestamp=i,
        )
        events.append(event)
        
        # Shift working set periodically
        if i % 500 == 0:
            current_page = np.random.randint(0, 2**16)
    
    return events


def generate_kv_workload(config: HeadWorkloadConfig) -> List[MicroEvent]:
    """
    Generate KV-policy workload that stresses KV-specific decisions.
    
    This workload creates KV-specific patterns:
    - Hot vs cold KV segments
    - Cache pressure scenarios
    - Prefix reuse patterns
    
    Reward surface for KV head:
    - Compress cold KV: +0.5
    - Preserve hot KV: +0.6
    - Aggressive compression of hot KV: -0.4
    - Preserve cold KV unnecessarily: -0.2
    """
    np.random.seed(config.seed)
    events = []
    
    # Track KV state
    kv_state = {}  # kv_id -> state (0=cold, 1=hot)
    current_page = np.random.randint(0, 2**16)
    
    for i in range(config.num_events):
        # Determine KV state
        kv_id = i % 64
        is_hot = np.random.random() < 0.3  # 30% hot
        
        # Track KV state
        kv_state[kv_id] = 1 if is_hot else 0
        
        # Generate event type based on KV state
        r = np.random.random()
        
        if r < 0.30:
            # Hot KV read - preserve
            event_type = EventType.MEMORY_READ
            is_write = False
        elif r < 0.50:
            # Cold KV read - could compress
            event_type = EventType.MEMORY_READ
            is_write = False
        elif r < 0.70:
            # Hot KV write - preserve
            event_type = EventType.MEMORY_WRITE
            is_write = True
        elif r < 0.85:
            # Cold KV write - could compress
            event_type = EventType.MEMORY_WRITE
            is_write = True
        elif r < 0.95:
            # Cache pressure - compress cold
            event_type = EventType.MEMORY_READ
            is_write = False
        else:
            # KV policy event
            event_type = EventType.KV_POLICY
            is_write = False
        
        event = create_synthetic_event(
            event_type=event_type,
            pid=config.pid,
            virtual_page=current_page + np.random.randint(0, 16),
            is_write=is_write,
            timestamp=i,
        )
        events.append(event)
        
        # Shift working set periodically
        if i % 500 == 0:
            current_page = np.random.randint(0, 2**16)
    
    return events


def generate_numa_workload(config: HeadWorkloadConfig) -> List[MicroEvent]:
    """
    Generate NUMA-policy workload that stresses NUMA-specific decisions.
    
    This workload creates NUMA-specific patterns:
    - Local vs remote memory access patterns
    - Migration cost scenarios
    - Cross-node contention
    
    Reward surface for NUMA head:
    - Keep local when local is good: +0.5
    - Migrate when remote is better: +0.4
    - Unnecessary migration: -0.3
    - Stay remote when local available: -0.2
    """
    np.random.seed(config.seed)
    events = []
    
    # Track NUMA state
    numa_local = {}  # page_id -> is_local
    current_page = np.random.randint(0, 2**16)
    
    for i in range(config.num_events):
        # Determine NUMA state
        virtual_page = current_page + (i % 32)
        is_local = np.random.random() < 0.6  # 60% local
        
        # Track NUMA state
        numa_local[virtual_page] = is_local
        
        # Generate event type based on NUMA state
        r = np.random.random()
        
        if r < 0.35:
            # Local access - keep local
            event_type = EventType.MEMORY_READ
            is_write = False
        elif r < 0.55:
            # Remote access - consider migration
            event_type = EventType.MEMORY_READ
            is_write = False
        elif r < 0.70:
            # Local write - keep local
            event_type = EventType.MEMORY_WRITE
            is_write = True
        elif r < 0.85:
            # Remote write - consider migration
            event_type = EventType.MEMORY_WRITE
            is_write = True
        elif r < 0.95:
            # Migration event
            event_type = EventType.MIGRATION
            is_write = False
        else:
            # NUMA policy event
            event_type = EventType.QUEUE
            is_write = False
        
        event = create_synthetic_event(
            event_type=event_type,
            pid=config.pid,
            virtual_page=virtual_page,
            is_write=is_write,
            timestamp=i,
        )
        events.append(event)
        
        # Shift working set periodically
        if i % 500 == 0:
            current_page = np.random.randint(0, 2**16)
    
    return events


def generate_boundary_workload(config: HeadWorkloadConfig) -> List[MicroEvent]:
    """
    Generate boundary-policy workload that stresses boundary-specific decisions.
    
    This workload creates boundary-specific patterns:
    - Syscall coalescing opportunities
    - Repeated small operations
    - IO_uring-like batching
    
    Reward surface for boundary head:
    - Coalesce small syscalls: +0.5
    - Batch repeated operations: +0.4
    - Fail to coalesce: -0.2
    - Over-batch and lose responsiveness: -0.3
    """
    np.random.seed(config.seed)
    events = []
    
    # Track syscall state
    syscall_count = 0
    current_page = np.random.randint(0, 2**16)
    
    for i in range(config.num_events):
        # Simulate syscall patterns
        if i % 5 == 0:
            # Syscall burst
            syscall_count += np.random.randint(3, 8)
        else:
            syscall_count = max(0, syscall_count - 1)
        
        # Generate event type based on syscall state
        r = np.random.random()
        
        if r < 0.30:
            # Syscall entry - coalesce if burst
            event_type = EventType.SYSCALL_ENTRY
            is_write = False
        elif r < 0.50:
            # Syscall exit
            event_type = EventType.SYSCALL_EXIT
            is_write = False
        elif r < 0.65:
            # Kernel entry
            event_type = EventType.KERNEL_ENTRY
            is_write = False
        elif r < 0.80:
            # Kernel exit
            event_type = EventType.KERNEL_EXIT
            is_write = False
        elif r < 0.90:
            # Repeated small operation - batch beneficial
            event_type = EventType.MEMORY_READ
            is_write = False
        else:
            # Normal operation
            event_type = EventType.MEMORY_WRITE
            is_write = True
        
        event = create_synthetic_event(
            event_type=event_type,
            pid=config.pid,
            virtual_page=current_page + np.random.randint(0, 8),
            is_write=is_write,
            timestamp=i,
        )
        events.append(event)
        
        # Shift working set periodically
        if i % 500 == 0:
            current_page = np.random.randint(0, 2**16)
    
    return events


def create_synthetic_event(
    event_type: EventType = EventType.MEMORY_READ,
    pid: int = 1000,
    virtual_page: Optional[int] = None,
    region_id: Optional[int] = None,
    is_write: bool = False,
    timestamp: int = 0,
) -> MicroEvent:
    """
    Create a synthetic micro-event for testing.
    
    Args:
        event_type: Type of event
        pid: Process ID
        virtual_page: Virtual page number (random if None)
        region_id: Region ID (derived from page if None)
        is_write: Whether this is a write operation
        timestamp: Timestamp bucket
        
    Returns:
        MicroEvent instance
    """
    if virtual_page is None:
        virtual_page = np.random.randint(0, 2**20)
    if region_id is None:
        region_id = virtual_page // 256
    
    # Create register sketch
    register_sketch = np.random.bytes(16)
    
    return MicroEvent(
        timestamp_bucket=timestamp,
        cpu_id=np.random.randint(0, 8),
        numa_node=np.random.randint(0, 2),
        pid=pid,
        tid=pid,
        pc_bucket=np.random.randint(0, 1024),
        event_type=event_type.value,
        opcode_class=0 if not is_write else 1,
        trap_fault_syscall_code=None if event_type not in [EventType.PAGE_FAULT, EventType.COW_FAULT, EventType.SYSCALL_ENTRY] else np.random.randint(0, 64),
        virtual_page=virtual_page,
        region_id=region_id,
        rw_flag=is_write,
        mode=0,
        pte_flags=0b11111 if event_type != EventType.PAGE_FAULT else 0,
        register_sketch=np.frombuffer(register_sketch, dtype=np.uint8),
        context_flags=0,
        missing_mask=0,
    )


def get_head_for_event_type(event_type: int) -> str:
    """Get the head that should handle this event type."""
    # Map event types to heads based on EventType enum values:
    # MEMORY_READ=1, MEMORY_WRITE=2, PAGE_FAULT=3, COW_FAULT=4, TRAP=5,
    # SYSCALL_ENTRY=6, SYSCALL_EXIT=7, KERNEL_ENTRY=8, KERNEL_EXIT=9,
    # RECLAIM=10, PROMOTION=11, DEMOTION=12, MIGRATION=13, QUEUE=14,
    # CACHE=15, KV_POLICY=16
    
    if event_type in [6, 7, 8, 9]:  # SYSCALL_ENTRY, SYSCALL_EXIT, KERNEL_ENTRY, KERNEL_EXIT
        return 'boundary'
    elif event_type == 10:  # RECLAIM
        return 'batch'
    elif event_type in [13, 14]:  # MIGRATION, QUEUE
        return 'numa'
    elif event_type == 16:  # KV_POLICY
        return 'kv'
    elif event_type in [3, 4]:  # PAGE_FAULT, COW_FAULT
        return 'page'
    else:  # MEMORY_READ, MEMORY_WRITE, TRAP, PROMOTION, DEMOTION, CACHE
        return 'page'
