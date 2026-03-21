"""Debug beneficial computation."""
import random
import numpy as np
import torch

# Set all seeds for deterministic evaluation
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from ac_moe_ga_sidecar import ACMoEGASidecar
from ac_moe_ga_sidecar.config import SidecarConfig, BalancedBuildConfig
from ac_moe_ga_sidecar.utils import set_seed, create_workload_trace, _compute_heuristic_action, _compute_sidecar_action, _compute_outcome
from ac_moe_ga_sidecar.types import MicroEvent

# Create config
config = BalancedBuildConfig()
config.uncertainty.override_base_threshold = 0.522
config.uncertainty.override_threshold_slope = 0.04

# Create sidecar
sidecar = ACMoEGASidecar(config=config)

# Reset sidecar
sidecar.reset()

# Set seed
set_seed(SEED)
rng = np.random.default_rng(SEED)

# Generate workload
events = create_workload_trace(2000, "mixed", rng=rng)

# Process events and debug beneficial computation
for event_idx, event in enumerate(events):
    result = sidecar.process_event(event)
    if result is not None and result.recommendation.should_override_heuristic:
        # Compute heuristic action
        heuristic_action = _compute_heuristic_action(event, event_idx)
        
        # Compute sidecar action
        sidecar_action = _compute_sidecar_action(result.recommendation)
        
        # Compute outcomes
        heuristic_outcome = _compute_outcome(heuristic_action, event, event_idx)
        sidecar_outcome = _compute_outcome(sidecar_action, event, event_idx)
        
        # Compute beneficial
        beneficial = sidecar_outcome > heuristic_outcome
        gain = sidecar_outcome - heuristic_outcome
        
        print(f"\nEvent {event_idx}:")
        print(f"  Event type: {event.event_type}, PID: {event.pid}")
        print(f"  Heuristic action: {heuristic_action}, outcome: {heuristic_outcome:.3f}")
        print(f"  Sidecar action: {sidecar_action}, outcome: {sidecar_outcome:.3f}")
        print(f"  Beneficial: {beneficial}, gain: {gain:.3f}")
        print(f"  Expert used: {result.recommendation.expert_used}")
        
        # Check mapping
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
        sidecar_head = expert_to_head.get(sidecar_action, sidecar_action)
        print(f"  Sidecar head: {sidecar_head}")
        
        # Only show first 10
        if event_idx >= 10:
            print("\nStopping after 10 events...")
            break