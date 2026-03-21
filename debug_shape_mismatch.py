"""Debug shape mismatch in BitNet model."""
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

from ac_moe_ga_sidecar import ACMoEGASidecar
from ac_moe_ga_sidecar.config import BalancedBuildConfig
from ac_moe_ga_sidecar.feature_extraction import FeatureExtractor
from ac_moe_ga_sidecar.types import MicroEvent
from ac_moe_ga_sidecar.config import EventType

# Create config and feature extractor
config = BalancedBuildConfig()
feature_extractor = FeatureExtractor(config)

# Create a simple event
event = MicroEvent(
    event_type=EventType.MEMORY_READ.value,
    pid=1000,
    tid=1000,
    pc_bucket=100,
    virtual_page=1000,
    region_id=3,
    timestamp_bucket=0,
    rw_flag=0,
    opcode_class=0,
    trap_fault_syscall_code=None,
    pte_flags=None,
    mode=0,
    cpu_id=0,
    numa_node=0,
)

# Extract features
page_summary = None
region_summary = None
process_summary = None
features = feature_extractor.extract(event, page_summary, region_summary, process_summary)

# Convert to tensors
device = torch.device('cpu')
input_tensors = feature_extractor.to_tensors(features, device)

# Create BitNet model
from ac_moe_ga_sidecar.bitnet_model import BitNetACMoEGAModel
from ac_moe_ga_sidecar.bitnet_config import BitNetBuildConfig

bitnet_config = BitNetBuildConfig()
bitnet_model = BitNetACMoEGAModel(bitnet_config)

# Add debugging to linear layers
def add_debug_hooks(model):
    """Add hooks to print tensor shapes before linear layers."""
    hooks = []
    
    def hook_fn(module, input):
        if hasattr(module, 'weight'):
            print(f"{module.__class__.__name__}: input shape = {input[0].shape}, weight shape = {module.weight.shape}")
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, TernaryLinear)):
            hook = module.register_forward_pre_hook(hook_fn)
            hooks.append(hook)
    
    return hooks

import torch.nn as nn
from ac_moe_ga_sidecar.bitnet_layers import TernaryLinear

hooks = add_debug_hooks(bitnet_model)

# Try forward pass
print("Attempting forward pass with debugging...")
try:
    output = bitnet_model(input_tensors)
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")
finally:
    # Remove hooks
    for hook in hooks:
        hook.remove()
