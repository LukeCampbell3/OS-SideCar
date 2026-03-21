# AC-MoE-GA Systems Sidecar v1.3.2

A dense-first, byte-to-state runtime optimization model for systems and OS-level decision support.

## Overview

AC-MoE-GA Systems Sidecar is a compact, living model that compresses raw byte-level machine and runtime evidence into a stable hidden-state representation. It uses that representation to recommend better memory, locality, scheduling, and runtime control decisions than simple heuristics alone.

**Key Features:**
- Dense-first architecture with typed residual experts
- Byte-to-state inference pipeline
- Multi-head output for various optimization domains
- Uncertainty-aware recommendations with calibrated abstention
- Advisory-only design (never overrides correctness-critical logic)

## v1.3.2 Improvements

- **Variance in confidence/support metrics**: std: 0.036/0.021 (was 0.001/0.006)
- **Per-head precision**: Page 8.1%, Batch 51.6% (was mostly zeros)
- **Realistic override metrics**: precision 70%, recall 5.6%
- **Non-zero ECE**: 0.295 with actual correctness labels
- **Regret tracking**: Heuristic vs sidecar comparison with actual outcomes

## v1.3.1 Improvements (Preserved)

- **Counterfactual evaluation**: Proper separation of recommendation from override execution
- **Realistic override metrics**: Override precision/recall based on actual gain computation
- **Calibration tracking**: ECE computed with actual correctness labels
- **Per-head precision**: Page, batch, KV, NUMA, boundary precision with head-specific workloads
- **Regret tracking**: Heuristic vs sidecar comparison with actual outcomes
- **Confidence/support statistics**: Mean, std, min, max with variance visibility

## v1.3 Improvements (Preserved)

- **Outcome-quality metrics**: Override precision, recall, gain tracking
- **Calibration tracking**: ECE, calibration error, confidence calibration
- **Per-head precision**: Page, batch, KV, NUMA, boundary precision
- **Confidence/support statistics**: Mean, std, min, max
- **Regret tracking**: Heuristic vs sidecar comparison
- **Comprehensive evaluation**: `evaluate_sidecar()` function

## v1.2 Improvements (Preserved)

- **Optimized feature extraction**: Pre-allocated arrays with zero-copy tensor conversion
- **Vectorized preprocessing**: Batch dimension support for efficient inference
- **Reduced allocations**: Incremental state updates with rolling buffers
- **Better edge case handling**: Robust handling of None values and missing data
- **~2.5x latency improvement**: 42.5ms → 17.0ms
- **~4.6x throughput improvement**: 99 → 460 events/sec

## v1.1 Improvements (Preserved)

- **Better support/prototype formation**: Improved prototype bank with quality-weighted support tracking
- **Calibrated abstention**: Dedicated AbstentionController with balanced decision logic
- **Sharper action separation**: RankingActionHead with explicit margin computation
- **Balanced-tiny capacity**: ~1.43 MB model, 375K parameters (up from 0.83 MB, 217K)
- **Score-based override**: Multi-factor scoring for override decisions

**Target v1.1 Metrics:**
- Confidence: 0.60-0.80 on in-domain traces
- Support density: 0.30-0.55 on recurring regimes
- Abstention rate: 5-20% depending on workload ambiguity
- Override rate: 10-35% depending on thresholds

## Installation

```bash
pip install ac-moe-ga-sidecar
```

Or install from source:

```bash
git clone https://github.com/ac-moe-ga/sidecar.git
cd sidecar
pip install -e ".[dev]"
```

## Quick Start

```python
from ac_moe_ga_sidecar import ACMoEGASidecar
from ac_moe_ga_sidecar.types import MicroEvent

# Initialize the sidecar
sidecar = ACMoEGASidecar()

# Create and process events
event = MicroEvent(
    timestamp_bucket=1000,
    cpu_id=0,
    numa_node=0,
    pid=1234,
    tid=1234,
    pc_bucket=0,
    event_type=0,  # MEMORY_READ
    opcode_class=0,
    virtual_page=0x7fff0000,
    region_id=1,
    rw_flag=False,
)

result = sidecar.process_event(event)

# Get recommendation when needed
recommendation = sidecar.get_recommendation()

if recommendation.should_override_heuristic and not recommendation.abstain:
    # Use the sidecar's recommendation
    print(f"Page action scores: {recommendation.action_scores.page_scores}")
    print(f"Confidence: {recommendation.inferred_state.confidence}")
    print(f"Support density: {recommendation.support_density}")
```

## Evaluation

```python
from ac_moe_ga_sidecar import ACMoEGASidecar, evaluate_sidecar

sidecar = ACMoEGASidecar()
results = evaluate_sidecar(sidecar, num_events=10000)

print(f"Events/sec: {results.events_per_second:.0f}")
print(f"Avg latency: {results.avg_inference_latency_us:.1f}μs")
print(f"Override precision: {results.override_precision*100:.1f}%")
print(f"Override recall: {results.override_recall*100:.1f}%")
print(f"Avg gain: {results.avg_gain:.3f}")
print(f"Confidence: {results.avg_confidence:.3f}")
print(f"Support density: {results.avg_support_density:.3f}")
print(f"ECE: {results.ece:.3f}")
```

## Architecture

The sidecar is organized into three planes:

### Plane A: Byte-Plane
Handles low-level ingestion and compression of machine evidence:
- Register-shape features
- Address-shape features
- PTE and mapping features
- Trap, fault, and syscall features
- Local byte-window sketches

### Plane B: State-Plane
Handles hidden-state inference:
- Observability refinement
- Belief state tracking
- Predictive bottleneck
- Slow-state memory
- Prototype matching with support density
- Abstention controller (v1.1)

### Plane C: Runtime-Control Plane
Handles action ranking:
- Page policy recommendations with margins
- Batch scheduling recommendations
- KV-cache policy recommendations
- NUMA placement recommendations
- Kernel boundary control recommendations

## Configuration

The balanced build configuration is recommended for most use cases:

```python
from ac_moe_ga_sidecar import BalancedBuildConfig, ACMoEGASidecar

config = BalancedBuildConfig()
sidecar = ACMoEGASidecar(config=config)

print(f"Model size: {sidecar.model.get_model_size_mb():.2f} MB")
print(f"Parameters: {sidecar.model.get_parameter_count():,}")
```

Key v1.3 configuration parameters:
- Fused observation width: 80 (up from 64)
- Belief hidden size: 48 (up from 32)
- Predictive hidden size: 80 (up from 64)
- Bottleneck width: 32 (up from 24)
- Prototype width: 24 (up from 16)
- 96 prototypes (up from 64)
- 8 typed experts
- ~1.43 MB model footprint
- ~19 MB runtime state budget

## Output Heads

The model provides multi-head outputs for:

### State Inference
- Page state (cold, hot, reclaimable, COW-sensitive, etc.)
- Region state (streaming, clustered, volatile, etc.)
- Process phase (compute-heavy, syscall-heavy, fork-transition, etc.)
- Hazard state (observability, OOD, volatility)

### Action Recommendations (with margins)
- **Page Policy**: preserve, reclaim, pre-COW, hugepage candidate
- **Batch Scheduling**: grow/shrink batch, defer, prioritize
- **KV Policy**: keep local, compress, page out, prefetch
- **NUMA Placement**: preserve locality, migrate, pin worker
- **Boundary Control**: batch crossings, coalesce, io_uring path

## Safety and Deployment

The sidecar is **advisory only**. It:
- Scores and ranks recommendations with explicit margins
- Provides calibrated confidence estimates
- Supports abstention when multiple uncertainty signals are present
- Never overrides correctness-critical invariants

Override criteria (v1.1 score-based):
- Calibrated confidence contribution
- Support density contribution
- Familiarity contribution
- Low ranking uncertainty contribution
- Action margin contribution
- Combined score must exceed threshold

## Testing

```bash
pytest tests/ -v
```

90 tests covering config, model, inference, runtime state, and integration.

## License

MIT License

## Citation

If you use this work, please cite:

```bibtex
@software{ac_moe_ga_sidecar,
  title = {AC-MoE-GA Systems Sidecar},
  version = {1.3.2},
  description = {Dense-first, byte-to-state runtime optimization model}
}
```
