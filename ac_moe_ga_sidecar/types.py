"""
Type definitions and data structures for AC-MoE-GA Systems Sidecar.

Defines the canonical micro-event schema, entity states, and output structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray


class PTEFlags(Enum):
    """Page Table Entry flags."""
    PRESENT = 1 << 0
    WRITABLE = 1 << 1
    EXECUTABLE = 1 << 2
    USER = 1 << 3
    ACCESSED = 1 << 4
    DIRTY = 1 << 5
    HUGEPAGE = 1 << 6
    SHARED = 1 << 7
    ANONYMOUS = 1 << 8
    FILE_BACKED = 1 << 9
    COW_CANDIDATE = 1 << 10


class VMAClass(Enum):
    """Virtual Memory Area classification."""
    HEAP = auto()
    STACK = auto()
    MMAP_ANON = auto()
    MMAP_FILE = auto()
    SHARED_LIB = auto()
    CODE = auto()
    DATA = auto()
    VDSO = auto()
    UNKNOWN = auto()


class OpcodeFamily(Enum):
    """Opcode classification families."""
    LOAD = auto()
    STORE = auto()
    ATOMIC = auto()
    BRANCH = auto()
    CALL = auto()
    RETURN = auto()
    SYSCALL = auto()
    FENCE = auto()
    ARITHMETIC = auto()
    VECTOR = auto()
    OTHER = auto()


class ModeType(Enum):
    """Execution mode."""
    USER = auto()
    KERNEL = auto()
    HYPERVISOR = auto()


@dataclass
class MicroEvent:
    """
    Canonical micro-event schema.
    
    Not every field must exist for every event.
    Missingness is explicit and preserved.
    """
    timestamp_bucket: int
    cpu_id: int
    numa_node: int
    pid: int
    tid: int
    pc_bucket: int
    event_type: int  # Maps to EventType enum
    opcode_class: int  # Maps to OpcodeFamily enum
    trap_fault_syscall_code: Optional[int] = None
    virtual_page: Optional[int] = None
    region_id: Optional[int] = None
    rw_flag: Optional[bool] = None  # True = write, False = read
    mode: int = 0  # Maps to ModeType enum
    pte_flags: int = 0  # Bitmask of PTEFlags
    register_sketch: Optional[NDArray[np.uint8]] = None
    context_flags: int = 0
    
    # Missingness tracking
    missing_mask: int = 0  # Bitmask indicating which fields are missing


@dataclass
class RegisterShapeFeatures:
    """Extracted register-shape features."""
    low8_bucket: int
    high8_bucket: int
    low16_alignment: int
    sign_flag: bool
    zero_flag: bool
    small_int_bucket: int
    pointer_likeness: float
    canonical_address_bits: int
    power_of_two_like: bool
    delta_bucket: int
    xor_bucket: int
    hamming_weight_bucket: int
    hamming_distance_bucket: int


@dataclass
class AddressShapeFeatures:
    """Extracted address-shape features."""
    page_hash_bucket: int
    page_offset_bucket: int
    cache_line_offset: int
    alignment_class: int
    region_relative_offset_bucket: int
    stride_bucket: int
    stride_sign: int  # -1, 0, 1
    same_page_flag: bool
    neighbor_page_flag: bool
    same_region_flag: bool
    far_jump_flag: bool
    reuse_distance_bucket: int
    locality_cluster_id: int
    address_entropy_bucket: int


@dataclass
class MapStateFeatures:
    """Page-table and mapping state features."""
    present: bool
    writable: bool
    executable: bool
    user: bool
    accessed: bool
    dirty: bool
    hugepage: bool
    shared: bool
    anonymous: bool
    file_backed: bool
    cow_candidate: bool
    vma_class: int  # Maps to VMAClass enum
    protection_domain_bucket: int


@dataclass
class TrapFaultFeatures:
    """Trap, fault, and syscall features."""
    trap_code_bucket: int
    fault_type_bucket: int
    syscall_class: int
    return_result_class: int
    repeated_trap_count_bucket: int
    mode_transition_type: int
    pc_bucket: int
    fault_address_page_bucket: int
    fault_address_offset_bucket: int
    trap_burstiness_bucket: int


@dataclass
class ByteWindowSketch:
    """Local byte-window sketch for temporal context."""
    xor_sketch: NDArray[np.uint64]
    rolling_hash: int
    locality_signature: int
    repeated_lower_byte_count: int
    entropy_estimate: float
    pointer_like_ratio: float
    numeric_like_ratio: float
    address_reuse_score: float


@dataclass
class EntitySummary:
    """Summary statistics for an entity (page/region/process)."""
    read_count: int
    write_count: int
    fault_count: int
    cow_count: int
    recency_bucket: int
    volatility: float
    neighbor_density: float
    bandwidth_pressure: float
    reclaim_pressure: float
    queue_depth: float
    numa_pressure: float
    kernel_entry_pressure: float
    kv_pressure: float


@dataclass
class PageState:
    """Inferred page state."""
    cold: float
    recently_reused: float
    burst_hot: float
    likely_write_hot_soon: float
    reclaimable: float
    fault_prone: float
    cow_sensitive: float
    hugepage_friendly: float
    unstable: float


@dataclass
class RegionState:
    """Inferred region state."""
    streaming: float
    clustered_reuse: float
    sparse_random: float
    expanding_heap: float
    shared_object_stable: float
    fragmentation_prone: float
    volatile: float
    reclaim_safe: float
    growing_write_pressure: float


@dataclass
class ProcessPhase:
    """Inferred process phase."""
    compute_heavy: float
    syscall_heavy: float
    allocator_growth: float
    fork_transition: float
    kernel_bound_burst: float
    io_wait_entry: float
    contention_lock: float
    boundary_thrashing: float


@dataclass
class PressureState:
    """System pressure state."""
    memory_pressure: float
    bandwidth_pressure: float
    reclaim_pressure: float
    queue_pressure: float
    remote_numa_pressure: float
    kv_residency_pressure: float
    kernel_crossing_pressure: float


@dataclass
class HazardState:
    """Hazard and uncertainty state."""
    low_ambiguity: float
    poor_observability: float
    stale_observation: float
    route_instability: float
    likely_ood: float
    tocttou_volatility: float
    local_volatility_spike: float


@dataclass
class UncertaintyVector:
    """Decomposed uncertainty outputs."""
    calibration: float  # How likely is confidence score wrong?
    selective_prediction: float  # Should we abstain?
    ranking: float  # How unstable is action ordering?
    ood: float  # Unlike known regimes?
    observability: float  # Missing important info?


@dataclass
class InferredState:
    """Complete inferred hidden state tuple."""
    page_state: PageState
    region_state: RegionState
    process_phase: ProcessPhase
    pressure_state: PressureState
    hazard_state: HazardState
    uncertainty: UncertaintyVector
    confidence: float


# Action recommendation types

class BatchAction(Enum):
    """Batch scheduler actions."""
    GROW_BATCH = auto()
    KEEP_CURRENT = auto()
    SHRINK_BATCH = auto()
    DEFER_LOW_PRIORITY = auto()
    PRIORITIZE_VERIFICATION = auto()
    PRIORITIZE_MEMORY_STABILITY = auto()


class KVAction(Enum):
    """KV policy actions."""
    KEEP_LOCAL = auto()
    COMPRESS = auto()
    PRESERVE_FULL_PRECISION = auto()
    LOWER_PRECISION = auto()
    SHARE_PREFIX = auto()
    PAGE_OUT = auto()
    PREFETCH_IN = auto()
    DEFER_SPILL = auto()


class NUMAAction(Enum):
    """NUMA placement actions."""
    PRESERVE_LOCALITY = auto()
    MIGRATE = auto()
    PIN_WORKER = auto()
    SHIFT_CORE_CLASS = auto()
    HUGEPAGE_CANDIDATE = auto()
    AVOID_PROMOTION = auto()
    REDUCE_CROSS_NODE = auto()


class BoundaryAction(Enum):
    """Kernel boundary control actions."""
    KEEP_USER_PATH = auto()
    BATCH_CROSSINGS = auto()
    USE_IO_URING_PATH = auto()
    COALESCE_OPERATIONS = auto()
    PINNED_BUFFER_PATH = auto()
    DEEPEN_VALIDATION = auto()
    NO_OVERRIDE = auto()


class PageAction(Enum):
    """Page policy actions."""
    PRESERVE = auto()
    RECLAIM_CANDIDATE = auto()
    PRE_COW_PREPARE = auto()
    HUGEPAGE_CANDIDATE = auto()
    KEEP_HEURISTIC = auto()


@dataclass
class ActionScores:
    """Scored action recommendations."""
    batch_scores: Dict[BatchAction, float] = field(default_factory=dict)
    kv_scores: Dict[KVAction, float] = field(default_factory=dict)
    numa_scores: Dict[NUMAAction, float] = field(default_factory=dict)
    boundary_scores: Dict[BoundaryAction, float] = field(default_factory=dict)
    page_scores: Dict[PageAction, float] = field(default_factory=dict)


@dataclass
class Recommendation:
    """Final recommendation output."""
    inferred_state: InferredState
    action_scores: ActionScores
    should_override_heuristic: bool
    abstain: bool
    expert_used: Optional[str]
    prototype_match: Optional[int]
    support_density: float
    drift_score: float
    action_margin: Optional[float] = None  # v1.3: Top1 - Top2 margin for override decisions
