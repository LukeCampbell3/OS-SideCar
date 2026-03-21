"""Threshold sweep to find optimal operating point."""
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
from ac_moe_ga_sidecar.utils import evaluate_sidecar, set_seed

# Thresholds to sweep - these are the dynamic_threshold base values
# dynamic_threshold = base_threshold - 0.04 * support_factor
base_thresholds = [0.522, 0.524, 0.526, 0.528, 0.530]

print("Threshold Sweep Results")
print("=" * 80)

for base_threshold in base_thresholds:
    # Create a fresh config with the new threshold
    config = BalancedBuildConfig()
    config.uncertainty.override_base_threshold = base_threshold
    config.uncertainty.override_threshold_slope = 0.04
    
    # Create a fresh sidecar with the config
    sidecar = ACMoEGASidecar(config=config)
    
    # Run evaluation
    result = evaluate_sidecar(sidecar, num_events=5000, workload_type='mixed', seed=SEED)
    
    print(f"\nBase threshold: {base_threshold:.3f}")
    print(f"  Override rate: {result.override_rate:.1%}")
    print(f"  Override precision: {result.override_precision:.1%}")
    print(f"  Override recall: {result.override_recall:.1%}")
    print(f"  Action margin: {result.avg_action_margin:.3f}")
    print(f"  ECE: {result.ece:.3f}")
    print(f"  Page precision: {result.page_precision:.1%}")
    print(f"  Batch precision: {result.batch_precision:.1%}")
    print(f"  KV precision: {result.kv_precision:.1%}")
    print(f"  NUMA precision: {result.numa_precision:.1%}")
    print(f"  Boundary precision: {result.boundary_precision:.1%}")

print("\n" + "=" * 80)
print("Sweep complete")
