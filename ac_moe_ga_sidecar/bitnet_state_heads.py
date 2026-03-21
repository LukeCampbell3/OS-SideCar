"""State heads for BitNet model."""
import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass

from .config import LatentDimensions
from .bitnet_layers import TernaryLinear
from .types import PageState, RegionState, ProcessPhase, HazardState


class BitNetPageStateHead(nn.Module):
    """Page state head matching original interface."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        self.dims = dims
        
        input_dim = dims.bottleneck + dims.belief_hidden
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.GELU(),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        
        self.page_state = nn.Linear(32, 8)
        self.page_confidence = nn.Linear(32, 1)
    
    def forward(self, z_pred: torch.Tensor, belief: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([z_pred, belief], dim=-1)
        x = self.net(x)
        
        page_state = self.page_state(x)
        page_confidence = self.page_confidence(x)
        
        return {
            'page_state': page_state,
            'page_confidence': page_confidence,
        }
    
    def to_page_state(self, outputs: Dict[str, torch.Tensor], idx: int = 0) -> PageState:
        """Convert head outputs to PageState."""
        return PageState(
            cold=1.0 - outputs['page_state'][idx, 0].item(),
            recently_reused=outputs['page_state'][idx, 1].item(),
            burst_hot=outputs['page_state'][idx, 2].item(),
            likely_write_hot_soon=outputs['page_state'][idx, 3].item(),
            reclaimable=outputs['page_state'][idx, 4].item(),
            fault_prone=outputs['page_state'][idx, 5].item(),
            cow_sensitive=outputs['page_state'][idx, 6].item(),
            hugepage_friendly=outputs['page_state'][idx, 7].item(),
            unstable=1.0 - outputs['page_confidence'][idx].item(),
        )


class BitNetRegionStateHead(nn.Module):
    """Region state head matching original interface."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        self.dims = dims
        
        input_dim = dims.bottleneck + dims.belief_hidden + dims.slow_state
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.GELU(),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        
        self.region_state = nn.Linear(32, 8)
        self.region_confidence = nn.Linear(32, 1)
    
    def forward(self, z_pred: torch.Tensor, belief: torch.Tensor, z_slow: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([z_pred, belief, z_slow], dim=-1)
        x = self.net(x)
        
        region_state = self.region_state(x)
        region_confidence = self.region_confidence(x)
        
        return {
            'region_state': region_state,
            'region_confidence': region_confidence,
        }
    
    def to_region_state(self, outputs: Dict[str, torch.Tensor], idx: int = 0) -> RegionState:
        """Convert head outputs to RegionState."""
        return RegionState(
            streaming=outputs['region_state'][idx, 0].item(),
            clustered_reuse=outputs['region_state'][idx, 1].item(),
            sparse_random=outputs['region_state'][idx, 2].item(),
            expanding_heap=outputs['region_state'][idx, 3].item(),
            shared_object_stable=outputs['region_state'][idx, 4].item(),
            fragmentation_prone=outputs['region_state'][idx, 5].item(),
            volatile=outputs['region_state'][idx, 6].item(),
            reclaim_safe=1.0 - outputs['region_state'][idx, 6].item(),
            growing_write_pressure=outputs['region_state'][idx, 7].item(),
        )


class BitNetProcessPhaseHead(nn.Module):
    """Process phase head matching original interface."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        self.dims = dims
        
        input_dim = dims.bottleneck + dims.slow_state
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.GELU(),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        
        self.process_phase = nn.Linear(32, 8)
        self.process_confidence = nn.Linear(32, 1)
    
    def forward(self, z_pred: torch.Tensor, z_slow: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([z_pred, z_slow], dim=-1)
        x = self.net(x)
        
        process_phase = self.process_phase(x)
        process_confidence = self.process_confidence(x)
        
        return {
            'process_phase': process_phase,
            'process_confidence': process_confidence,
        }
    
    def to_process_phase(self, outputs: Dict[str, torch.Tensor], idx: int = 0) -> ProcessPhase:
        """Convert head outputs to ProcessPhase."""
        return ProcessPhase(
            compute_heavy=outputs['process_phase'][idx, 0].item(),
            syscall_heavy=outputs['process_phase'][idx, 1].item(),
            allocator_growth=outputs['process_phase'][idx, 2].item(),
            fork_transition=outputs['process_phase'][idx, 3].item(),
            kernel_bound_burst=outputs['process_phase'][idx, 4].item(),
            io_wait_entry=outputs['process_phase'][idx, 5].item(),
            contention_lock=outputs['process_phase'][idx, 6].item(),
            boundary_thrashing=outputs['process_phase'][idx, 1].item() * outputs['process_phase'][idx, 4].item(),
        )


class BitNetHazardHead(nn.Module):
    """Hazard head matching original interface."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        self.dims = dims
        
        input_dim = dims.bottleneck + 5  # + uncertainty vector
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.GELU(),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        
        self.heads = nn.ModuleDict({
            'low_observability': nn.Linear(32, 1),
            'stale_evidence': nn.Linear(32, 1),
            'ood_regime': nn.Linear(32, 1),
            'ranking_instability': nn.Linear(32, 1),
            'route_instability': nn.Linear(32, 1),
            'tocttou_volatility': nn.Linear(32, 1),
        })
        
        self.temperature = 1.5
    
    def forward(self, z_pred: torch.Tensor, uncertainty: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([z_pred, uncertainty], dim=-1)
        x = self.shared(x)
        
        hazard = torch.cat([
            self.heads['low_observability'](x),
            self.heads['stale_evidence'](x),
            self.heads['ood_regime'](x),
            self.heads['ranking_instability'](x),
            self.heads['route_instability'](x),
            self.heads['tocttou_volatility'](x),
        ], dim=-1)
        
        return {
            'hazard': hazard,
        }
    
    def to_hazard_state(self, outputs: Dict[str, torch.Tensor], idx: int = 0) -> HazardState:
        """Convert head outputs to HazardState."""
        hazard = outputs['hazard'][idx]
        return HazardState(
            low_ambiguity=1.0 - hazard[3].item(),
            poor_observability=hazard[0].item(),
            stale_observation=hazard[1].item(),
            route_instability=hazard[4].item(),
            likely_ood=hazard[2].item(),
            tocttou_volatility=hazard[5].item(),
            local_volatility_spike=hazard[4].item(),
        )
