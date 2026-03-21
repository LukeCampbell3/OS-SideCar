"""
Utility functions for AC-MoE-GA Sidecar.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

from .types import MicroEvent, Recommendation
from .config import EventType

logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.debug(f"{name} took {elapsed*1000:.2f}ms")


def set_seed(seed: int = 42):
    """Set all random seeds for deterministic evaluation."""
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    import random
    random.seed(seed)
    
    np.random.seed(seed)
    
    import torch
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        region_id = virtual_page >> 8
    
    # Convert EventType to int if needed
    if isinstance(event_type, EventType):
        event_type = event_type.value
    
    return MicroEvent(
        event_type=event_type,
        pid=pid,
        tid=pid,
        pc_bucket=virtual_page,
        virtual_page=virtual_page,
        region_id=region_id,
        timestamp_bucket=timestamp,
        rw_flag=1 if is_write else 0,
        opcode_class=0,
        trap_fault_syscall_code=None,
        pte_flags=None,
        mode=0,
        cpu_id=0,
        numa_node=0,
    )


def _compute_heuristic_action(event: MicroEvent, event_idx: int) -> str:
    """Compute heuristic action for a given event."""
    if event.event_type in [2, 3]:  # PAGE_FAULT, COW_FAULT
        return "page"
    elif event.event_type in [14, 15]:  # QUEUE, KV_POLICY
        return "batch"
    elif event.pid % 2 == 0:
        return "numa"
    else:
        return "boundary"


def _compute_sidecar_action(recommendation: Recommendation) -> str:
    """Compute sidecar action from recommendation."""
    if recommendation.abstain:
        return "abstain"
    if recommendation.should_override_heuristic:
        return recommendation.expert_used or "unknown"
    return "heuristic"


def _compute_outcome(action: str, event: MicroEvent, event_idx: int) -> float:
    """Compute synthetic outcome for an action.

    Returns a score from 0.0 to 1.0 where:
    - 1.0: Perfect match between action and event type
    - 0.3: Mismatch between action and event type
    - 0.0: Abstention

    Logic:
    - Page events (types 2,3): page actions get 1.0, others 0.3
    - Batch events (14,15): batch actions get 1.0, others 0.3
    - Even PIDs: numa actions get 1.0, others 0.3
    - Odd PIDs: boundary actions get 1.0, others 0.3
    """
    if action == "abstain":
        return 0.0

    # Map expert names to head names
    expert_to_head = {
        'page_transition': 'page',
        'cow_fork': 'page',
        'reclaim_hotness': 'page',
        'locality_pattern': 'batch',
        'fault_burst': 'page',
        'boundary_control': 'boundary',
        'kv_policy': 'kv',
        'numa_placement': 'numa',
    }

    # Get the head for this action
    head = expert_to_head.get(action, action)  # action could be expert name or head name

    # Page events (faults) favor page actions
    if event.event_type in [2, 3]:  # PAGE_FAULT, COW_FAULT
        return 1.0 if head == "page" else 0.3

    # Batch events favor batch actions
    if event.event_type in [14, 15]:  # QUEUE, KV_POLICY
        return 1.0 if head == "batch" else 0.3

    # For other events, use PID-based logic
    if event.pid % 2 == 0:
        # Even PIDs favor numa actions
        return 1.0 if head == "numa" else 0.3
    else:
        # Odd PIDs favor boundary actions
        return 1.0 if head == "boundary" else 0.3



def _determine_head_for_event(event: MicroEvent) -> str:
    """Determine which head would be used for this event."""
    if event.event_type in [2, 3]:  # PAGE_FAULT, COW_FAULT
        return "page"
    elif event.event_type in [14, 15]:  # QUEUE, KV_POLICY
        return "batch"
    elif event.pid % 2 == 0:
        return "numa"
    else:
        return "boundary"


# Canonical functions for consistent evaluation
_EXPERT_TO_HEAD = {
    'page_transition': 'page',
    'cow_fork': 'page',
    'reclaim_hotness': 'page',
    'locality_pattern': 'batch',
    'fault_burst': 'page',
    'boundary_control': 'boundary',
    'kv_policy': 'kv',
    'numa_placement': 'numa',
}


def get_executed_head(recommendation: Recommendation) -> str:
    """Get the head that actually executed the action."""
    if recommendation.abstain:
        return "abstain"
    if not recommendation.should_override_heuristic:
        return "heuristic"
    
    # Map expert used to head
    if recommendation.expert_used:
        return _EXPERT_TO_HEAD.get(recommendation.expert_used, "unknown")
    return "unknown"


def compute_beneficial_override(
    heuristic_action: str,
    sidecar_action: str,
    event: MicroEvent,
    event_idx: int
) -> Tuple[bool, float, float, float]:
    """
    Canonical function to compute if an override was beneficial.
    
    Returns:
        beneficial: True if sidecar was better than heuristic
        gain: sidecar_outcome - heuristic_outcome
        heuristic_outcome: Outcome if heuristic was used
        sidecar_outcome: Outcome if sidecar was used
    """
    heuristic_outcome = _compute_outcome(heuristic_action, event, event_idx)
    sidecar_outcome = _compute_outcome(sidecar_action, event, event_idx)
    gain = sidecar_outcome - heuristic_outcome
    beneficial = gain > 0.0
    return beneficial, gain, heuristic_outcome, sidecar_outcome


def create_workload_trace(
    num_events: int,
    workload_type: str = "mixed",
    pid: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> List[MicroEvent]:
    """
    Create a synthetic workload trace for testing.
    
    Enhanced v1.4: More variety in events to improve variance in metrics.
    
    Args:
        num_events: Number of events to generate
        workload_type: Type of workload to generate
        pid: Process ID
        rng: Optional numpy random generator for determinism
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    events = []
    current_page = rng.integers(0, 2**16)
    
    # Event type distribution for better head coverage
    event_type_weights = {
        EventType.MEMORY_READ: 0.20,
        EventType.MEMORY_WRITE: 0.15,
        EventType.PAGE_FAULT: 0.12,
        EventType.COW_FAULT: 0.08,
        EventType.SYSCALL_ENTRY: 0.06,
        EventType.SYSCALL_EXIT: 0.05,
        EventType.TRAP: 0.04,
        EventType.KERNEL_ENTRY: 0.03,
        EventType.KERNEL_EXIT: 0.02,
        EventType.RECLAIM: 0.02,
        EventType.QUEUE: 0.03,
        EventType.KV_POLICY: 0.02,
    }
    
    for i in range(num_events):
        # Determine workload type with more variety
        r = rng.random()
        
        if workload_type == "sequential":
            # Sequential access pattern
            virtual_page = current_page + i
            event_type = EventType.MEMORY_READ if i % 4 != 0 else EventType.MEMORY_WRITE
            is_write = event_type == EventType.MEMORY_WRITE
            
        elif workload_type == "random":
            # Random access pattern with more variety
            virtual_page = rng.integers(0, 2**20)
            event_type = EventType.MEMORY_READ if rng.random() > 0.3 else EventType.MEMORY_WRITE
            is_write = event_type == EventType.MEMORY_WRITE
            
        elif workload_type == "syscall_heavy":
            # Syscall-heavy workload
            if rng.random() < 0.3:
                event_type = EventType.SYSCALL_ENTRY if rng.random() < 0.5 else EventType.SYSCALL_EXIT
                virtual_page = None
            else:
                virtual_page = current_page + rng.integers(-10, 10)
                event_type = EventType.MEMORY_READ
            is_write = False
            
        else:  # mixed
            # Mixed workload with more variety and better head coverage
            cumulative = 0
            selected_type = EventType.MEMORY_READ
            for etype, weight in event_type_weights.items():
                cumulative += weight
                if r < cumulative:
                    selected_type = etype
                    break
            
            if selected_type in [EventType.SYSCALL_ENTRY, EventType.SYSCALL_EXIT, EventType.TRAP,
                                  EventType.KERNEL_ENTRY, EventType.KERNEL_EXIT]:
                # System events - no virtual page
                virtual_page = None
                event_type = selected_type
                is_write = False
            elif selected_type == EventType.PAGE_FAULT:
                # Page fault - different head
                virtual_page = rng.integers(0, 2**20)
                event_type = EventType.PAGE_FAULT
                is_write = False
            elif selected_type == EventType.COW_FAULT:
                # COW fault - different head
                virtual_page = current_page + rng.integers(0, 100)
                event_type = EventType.COW_FAULT
                is_write = False
            elif selected_type == EventType.RECLAIM:
                # Reclaim event
                virtual_page = current_page + rng.integers(0, 32)
                event_type = EventType.RECLAIM
                is_write = False
            else:
                # Memory events
                if r < 0.35:
                    # Sequential burst (creates hot pages)
                    virtual_page = current_page + (i % 16)
                elif r < 0.55:
                    # Random access (creates varied page states)
                    virtual_page = rng.integers(0, 2**20)
                else:
                    # Working set shift
                    virtual_page = current_page + rng.integers(-32, 32)
                    virtual_page = max(0, virtual_page)
                
                event_type = selected_type
                is_write = selected_type == EventType.MEMORY_WRITE
        
        event = create_synthetic_event(
            event_type=event_type,
            pid=pid,
            virtual_page=virtual_page,
            is_write=is_write,
            timestamp=i,
        )
        events.append(event)
        
        # Occasionally shift the working set
        if i % 1000 == 0:
            current_page = rng.integers(0, 2**16)
    
    return events


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    events_per_second: float
    inferences_per_second: float
    avg_inference_latency_us: float
    p99_inference_latency_us: float
    memory_usage_mb: float
    abstention_rate: float
    override_rate: float


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
    regret: float
    
    # Per-head activation counts
    page_activations: int
    batch_activations: int
    kv_activations: int
    numa_activations: int
    boundary_activations: int


def evaluate_sidecar(
    sidecar,
    num_events: int = 10000,
    workload_type: str = "mixed",
    seed: int = 42,
    rng: Optional[np.random.Generator] = None,
) -> EvaluationResult:
    """
    Evaluate the sidecar with outcome-quality metrics.
    
    Enhanced v1.5: Head-specific reward surfaces for better head specialization.
    
    Args:
        sidecar: ACMoEGASidecar instance
        num_events: Number of events to process
        workload_type: Type of workload to generate
        seed: Random seed for reproducibility
        rng: Optional numpy random generator (if provided, seed is ignored)
        
    Returns:
        EvaluationResult with comprehensive metrics
    """
    from .evaluation import Evaluator, OverrideTracker, RegretTracker, ConfidenceTracker, CalibrationMetrics
    from .workload_generators import generate_page_workload, generate_batch_workload, generate_kv_workload, generate_numa_workload, generate_boundary_workload
    
    # Reset sidecar to ensure clean state
    sidecar.reset()
    
    # Set seed if no RNG provided
    if rng is None:
        set_seed(seed)
        rng = np.random.default_rng(seed)
    
    # Generate workload based on type
    if workload_type == "page":
        events = generate_page_workload(HeadWorkloadConfig(num_events=num_events))
    elif workload_type == "batch":
        events = generate_batch_workload(HeadWorkloadConfig(num_events=num_events))
    elif workload_type == "kv":
        events = generate_kv_workload(HeadWorkloadConfig(num_events=num_events))
    elif workload_type == "numa":
        events = generate_numa_workload(HeadWorkloadConfig(num_events=num_events))
    elif workload_type == "boundary":
        events = generate_boundary_workload(HeadWorkloadConfig(num_events=num_events))
    else:  # mixed
        # Generate mixed workload with all head types
        events = create_workload_trace(num_events, workload_type, rng=rng)
    
    # Reset sidecar
    sidecar.reset()
    
    # Initialize evaluator and attach to engine
    evaluator = Evaluator(sidecar.config)
    sidecar.engine.set_evaluator(evaluator)
    
    # Process events and collect metrics
    latencies = []
    start_time = time.perf_counter()
    
    # Track per-head statistics
    head_decisions = {
        'page': {'decisions': 0, 'overrides': 0, 'beneficial': 0},
        'batch': {'decisions': 0, 'overrides': 0, 'beneficial': 0},
        'kv': {'decisions': 0, 'overrides': 0, 'beneficial': 0},
        'numa': {'decisions': 0, 'overrides': 0, 'beneficial': 0},
        'boundary': {'decisions': 0, 'overrides': 0, 'beneficial': 0},
    }
    
    for event_idx, event in enumerate(events):
        result = sidecar.process_event(event)
        if result is not None:
            latencies.append(result.latency_us)
            
            # Record inference for confidence tracking
            evaluator.record_inference(result.recommendation)
            
            # Compute synthetic counterfactual outcomes using canonical function
            heuristic_action = _compute_heuristic_action(event, event_idx)
            sidecar_action = _compute_sidecar_action(result.recommendation)
            
            # Use canonical function for consistent beneficial computation
            beneficial, gain, heuristic_outcome, sidecar_outcome = compute_beneficial_override(
                heuristic_action, sidecar_action, event, event_idx
            )
            
            # Determine if override should be executed
            should_override = (
                result.recommendation.should_override_heuristic 
                and not result.recommendation.abstain
            )
            
            # Determine which head was used based on event type
            head_used = _determine_head_for_event(event)
            
            # Determine which head actually executed the action using canonical function
            executed_head = get_executed_head(result.recommendation)
            
            # Update per-head statistics using executed head
            head_decisions[head_used]['decisions'] += 1
            if should_override and executed_head in head_decisions:
                head_decisions[executed_head]['overrides'] += 1
                if beneficial:
                    head_decisions[executed_head]['beneficial'] += 1
            
            # Record decision with counterfactual context
            evaluator.record_decision(
                timestamp=event_idx,
                head=head_used,  # Which head "should" handle this event
                confidence=result.recommendation.inferred_state.confidence,
                support_density=result.recommendation.support_density,
                action_margin=result.recommendation.action_margin or 0.0,
                heuristic_action=heuristic_action,
                sidecar_action=sidecar_action,
                override_executed=should_override,
                abstained=result.recommendation.abstain,
                heuristic_outcome=heuristic_outcome,
                sidecar_outcome=sidecar_outcome,
                beneficial=beneficial,
                head_used=executed_head,  # Which head actually executed the action
            )
    
    # Compute final statistics
    stats = sidecar.get_statistics()
    eval_report = evaluator.get_evaluation_report()
    override_stats = eval_report['override_stats']
    confidence_stats = eval_report['confidence_stats']
    calibration_stats = eval_report['calibration']
    
    # Compute performance metrics from latencies
    elapsed = time.perf_counter() - start_time
    events_per_second = num_events / elapsed if elapsed > 0 else 0
    inferences_per_second = len(latencies) / elapsed if elapsed > 0 else 0
    
    # Compute regret
    regret = override_stats.get('total_gain', 0.0)
    
    # Compute per-head precision (using executed_head, not head_used)
    page_precision = head_decisions['page']['beneficial'] / max(1, head_decisions['page']['overrides'])
    batch_precision = head_decisions['batch']['beneficial'] / max(1, head_decisions['batch']['overrides'])
    kv_precision = head_decisions['kv']['beneficial'] / max(1, head_decisions['kv']['overrides'])
    numa_precision = head_decisions['numa']['beneficial'] / max(1, head_decisions['numa']['overrides'])
    boundary_precision = head_decisions['boundary']['beneficial'] / max(1, head_decisions['boundary']['overrides'])
    
    return EvaluationResult(
        # Performance metrics
        events_per_second=events_per_second,
        inferences_per_second=inferences_per_second,
        avg_inference_latency_us=np.mean(latencies) if latencies else 0,
        p99_inference_latency_us=np.percentile(latencies, 99) if latencies else 0,
        memory_usage_mb=stats['runtime_memory_mb'] + stats['model_size_mb'],
        
        # Behavioral metrics
        abstention_rate=override_stats.get('abstention_rate', 0.0),
        override_rate=override_stats.get('override_rate', 0.0),
        
        # Override quality metrics
        override_precision=override_stats.get('override_precision', 0.0),
        override_recall=override_stats.get('override_recall', 0.0),
        avg_gain=override_stats.get('avg_gain', 0.0),
        total_gain=override_stats.get('total_gain', 0.0),
        
        # Per-head metrics
        page_precision=page_precision,
        batch_precision=batch_precision,
        kv_precision=kv_precision,
        numa_precision=numa_precision,
        boundary_precision=boundary_precision,
        
        # Confidence/support metrics
        avg_confidence=confidence_stats.get('avg_confidence', 0.0),
        avg_support_density=confidence_stats.get('avg_support_density', 0.0),
        avg_action_margin=override_stats.get('avg_action_margin', 0.0),
        confidence_std=confidence_stats.get('confidence_std', 0.0),
        support_std=confidence_stats.get('support_std', 0.0),
        
        # Calibration metrics
        ece=calibration_stats.get('ece', 0.0),
        max_calibration_error=calibration_stats.get('max_calibration_error', 0.0),
        
        # Regret metrics
        regret=regret,
        
        # Per-head activation counts
        page_activations=head_decisions['page']['decisions'],
        batch_activations=head_decisions['batch']['decisions'],
        kv_activations=head_decisions['kv']['decisions'],
        numa_activations=head_decisions['numa']['decisions'],
        boundary_activations=head_decisions['boundary']['decisions'],
    )


def run_multi_seed_evaluation(
    sidecar_factory,
    num_events: int = 5000,
    num_seeds: int = 5,
    workload_type: str = "mixed",
) -> Dict[str, List[float]]:
    """
    Run evaluation across multiple seeds to assess stability.
    
    Args:
        sidecar_factory: Function that creates a new sidecar instance
        num_events: Number of events per run
        num_seeds: Number of seeds to test
        workload_type: Type of workload
        
    Returns:
        Dictionary mapping metric names to list of values across seeds
    """
    results = {
        'override_rate': [],
        'override_precision': [],
        'override_recall': [],
        'avg_confidence': [],
        'avg_support_density': [],
        'avg_action_margin': [],
        'ece': [],
        'regret': [],
    }
    
    for seed in range(num_seeds):
        sidecar = sidecar_factory()
        result = evaluate_sidecar(sidecar, num_events, workload_type, seed=seed)
        
        results['override_rate'].append(result.override_rate)
        results['override_precision'].append(result.override_precision)
        results['override_recall'].append(result.override_recall)
        results['avg_confidence'].append(result.avg_confidence)
        results['avg_support_density'].append(result.avg_support_density)
        results['avg_action_margin'].append(result.avg_action_margin)
        results['ece'].append(result.ece)
        results['regret'].append(result.regret)
    
    return results

def benchmark_sidecar(
    sidecar,
    num_events: int = 10000,
    warmup_events: int = 1000,
    workload_type: str = "mixed",
) -> BenchmarkResult:
    """Benchmark the sidecar performance."""
    from .config import EventType
    from .types import MicroEvent
    
    # Generate workload
    rng = np.random.default_rng(42)
    events = create_workload_trace(num_events + warmup_events, workload_type, rng=rng)
    
    # Warmup
    for event in events[:warmup_events]:
        sidecar.process_event(event)
    
    # Reset for benchmark
    sidecar.reset()
    
    # Benchmark
    latencies = []
    events_processed = 0
    inferences = 0
    
    start_time = time.perf_counter()
    for event in events[warmup_events:]:
        result = sidecar.process_event(event)
        if result is not None:
            latencies.append(result.latency_us)
            events_processed += result.events_processed
            inferences += 1
    
    elapsed = time.perf_counter() - start_time
    
    # Get statistics
    stats = sidecar.get_statistics()
    
    return BenchmarkResult(
        events_per_second=events_processed / elapsed if elapsed > 0 else 0,
        inferences_per_second=inferences / elapsed if elapsed > 0 else 0,
        avg_inference_latency_us=np.mean(latencies) if latencies else 0,
        p99_inference_latency_us=np.percentile(latencies, 99) if latencies else 0,
        memory_usage_mb=stats['runtime_memory_mb'] + stats['model_size_mb'],
        abstention_rate=stats['abstention_rate'],
        override_rate=stats['override_rate'],
    )


def get_device_info() -> Dict[str, str]:
    """Get device information."""
    import torch
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "cuda_available": True,
            "cpu_count": 1,
        }
    return {
        "device": "cpu",
        "name": "cpu",
        "count": "1",
        "cuda_available": False,
        "cpu_count": 1,
    }


def estimate_memory_requirements(config) -> Dict[str, float]:
    """Estimate memory requirements for the sidecar."""
    from .core import ACMoEGASidecar
    sidecar = ACMoEGASidecar(config)
    stats = sidecar.get_statistics()
    return {
        "model_mb": stats['model_size_mb'],
        "runtime_mb": stats['runtime_memory_mb'],
        "runtime_state_mb": stats['runtime_memory_mb'],
        "total_mb": stats['model_size_mb'] + stats['runtime_memory_mb'],
    }
