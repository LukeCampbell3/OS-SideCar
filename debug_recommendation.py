"""Debug recommendation fields."""
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
from ac_moe_ga_sidecar.utils import set_seed, create_workload_trace
from ac_moe_ga_sidecar.types import MicroEvent

# Create config
config = BalancedBuildConfig()
config.uncertainty.override_base_threshold = 0.525
config.uncertainty.override_threshold_slope = 0.04

# Create sidecar
sidecar = ACMoEGASidecar(config=config)

# Reset sidecar
sidecar.reset()

# Set seed
set_seed(SEED)
rng = np.random.default_rng(SEED)

# Generate workload
events = create_workload_trace(50, "mixed", rng=rng)

# Process events
for event_idx, event in enumerate(events):
    result = sidecar.process_event(event)
    if result is not None:
        print(f"\nInference at event {event_idx}:")
        rec = result.recommendation
        print(f"  should_override_heuristic: {rec.should_override_heuristic}")
        print(f"  abstain: {rec.abstain}")
        print(f"  expert_used: {rec.expert_used}")
        print(f"  confidence: {rec.inferred_state.confidence:.3f}")
        print(f"  support_density: {rec.support_density:.3f}")
        print(f"  action_margin: {rec.action_margin}")
        
        # Check model output
        print(f"  Model output should_abstain: {result.model_output.should_abstain[0].item()}")
        print(f"  Model output uncertainty: {result.model_output.uncertainty[0].cpu().numpy()}")
        
        # Check if override would be triggered
        if rec.should_override_heuristic:
            print("  *** WOULD OVERRIDE ***")
        elif rec.abstain:
            print("  *** WOULD ABSTAIN ***")
        else:
            print("  *** WOULD USE HEURISTIC ***")