"""
Output heads for AC-MoE-GA Sidecar v1.1.

Enhanced for:
- Sharper action separation
- Better ranking margins
- Calibrated confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass

from .config import LatentDimensions
from .types import (
    PageState, RegionState, ProcessPhase, PressureState, HazardState,
    UncertaintyVector, BatchAction, KVAction, NUMAAction, BoundaryAction, PageAction
)


class StateHead(nn.Module):
    """Base class for state inference heads - enhanced v1.1."""
    
    def __init__(self, input_dim: int, num_outputs: int, hidden_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_outputs),
        )
        # Temperature for sharper outputs
        self.temperature = nn.Parameter(torch.tensor(1.2))
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z) * self.temperature)


class PageStateHead(nn.Module):
    """Infers page-level state with sharper separation."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        input_dim = dims.bottleneck + dims.belief_hidden
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 48),
            nn.GELU(),
        )
        
        self.heads = nn.ModuleDict({
            'touch_soon': nn.Linear(48, 1),
            'write_soon': nn.Linear(48, 1),
            'fault_soon': nn.Linear(48, 1),
            'cow_risk': nn.Linear(48, 1),
            'reclaim_safe': nn.Linear(48, 1),
            'hotness_up': nn.Linear(48, 1),
            'hotness_down': nn.Linear(48, 1),
            'hugepage_friendly': nn.Linear(48, 1),
        })
        
        self.temperature = 1.5  # Sharper outputs
        
    def forward(self, z_pred: torch.Tensor, belief: torch.Tensor) -> Dict[str, torch.Tensor]:
        combined = torch.cat([z_pred, belief], dim=-1)
        shared_features = self.shared(combined)
        return {name: torch.sigmoid(head(shared_features) * self.temperature).squeeze(-1) 
                for name, head in self.heads.items()}
    
    def to_page_state(self, outputs: Dict[str, torch.Tensor], idx: int = 0) -> PageState:
        return PageState(
            cold=1.0 - outputs['touch_soon'][idx].item(),
            recently_reused=outputs['touch_soon'][idx].item(),
            burst_hot=outputs['hotness_up'][idx].item(),
            likely_write_hot_soon=outputs['write_soon'][idx].item(),
            reclaimable=outputs['reclaim_safe'][idx].item(),
            fault_prone=outputs['fault_soon'][idx].item(),
            cow_sensitive=outputs['cow_risk'][idx].item(),
            hugepage_friendly=outputs['hugepage_friendly'][idx].item(),
            unstable=1.0 - outputs['reclaim_safe'][idx].item(),
        )


class RegionStateHead(nn.Module):
    """Infers region-level state."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        input_dim = dims.bottleneck + dims.belief_hidden + dims.slow_state
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 48),
            nn.GELU(),
        )
        
        self.heads = nn.ModuleDict({
            'streaming': nn.Linear(48, 1),
            'clustered_reuse': nn.Linear(48, 1),
            'sparse_random': nn.Linear(48, 1),
            'expanding': nn.Linear(48, 1),
            'stable_shared': nn.Linear(48, 1),
            'fragmenting': nn.Linear(48, 1),
            'volatile': nn.Linear(48, 1),
            'hugepage_friendly': nn.Linear(48, 1),
        })
        
        self.temperature = 1.5
        
    def forward(self, z_pred: torch.Tensor, belief: torch.Tensor, z_slow: torch.Tensor) -> Dict[str, torch.Tensor]:
        combined = torch.cat([z_pred, belief, z_slow], dim=-1)
        shared_features = self.shared(combined)
        return {name: torch.sigmoid(head(shared_features) * self.temperature).squeeze(-1) 
                for name, head in self.heads.items()}
    
    def to_region_state(self, outputs: Dict[str, torch.Tensor], idx: int = 0) -> RegionState:
        return RegionState(
            streaming=outputs['streaming'][idx].item(),
            clustered_reuse=outputs['clustered_reuse'][idx].item(),
            sparse_random=outputs['sparse_random'][idx].item(),
            expanding_heap=outputs['expanding'][idx].item(),
            shared_object_stable=outputs['stable_shared'][idx].item(),
            fragmentation_prone=outputs['fragmenting'][idx].item(),
            volatile=outputs['volatile'][idx].item(),
            reclaim_safe=1.0 - outputs['volatile'][idx].item(),
            growing_write_pressure=outputs['expanding'][idx].item(),
        )


class ProcessPhaseHead(nn.Module):
    """Infers process phase."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        input_dim = dims.bottleneck + dims.slow_state
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.GELU(),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        
        self.heads = nn.ModuleDict({
            'compute_heavy': nn.Linear(32, 1),
            'syscall_heavy': nn.Linear(32, 1),
            'fork_transition': nn.Linear(32, 1),
            'allocator_growth': nn.Linear(32, 1),
            'kernel_bound_burst': nn.Linear(32, 1),
            'io_wait_entry': nn.Linear(32, 1),
            'contention_phase': nn.Linear(32, 1),
        })
        
        self.temperature = 1.5
        
    def forward(self, z_pred: torch.Tensor, z_slow: torch.Tensor) -> Dict[str, torch.Tensor]:
        combined = torch.cat([z_pred, z_slow], dim=-1)
        shared_features = self.shared(combined)
        return {name: torch.sigmoid(head(shared_features) * self.temperature).squeeze(-1) 
                for name, head in self.heads.items()}
    
    def to_process_phase(self, outputs: Dict[str, torch.Tensor], idx: int = 0) -> ProcessPhase:
        return ProcessPhase(
            compute_heavy=outputs['compute_heavy'][idx].item(),
            syscall_heavy=outputs['syscall_heavy'][idx].item(),
            allocator_growth=outputs['allocator_growth'][idx].item(),
            fork_transition=outputs['fork_transition'][idx].item(),
            kernel_bound_burst=outputs['kernel_bound_burst'][idx].item(),
            io_wait_entry=outputs['io_wait_entry'][idx].item(),
            contention_lock=outputs['contention_phase'][idx].item(),
            boundary_thrashing=outputs['syscall_heavy'][idx].item() * outputs['kernel_bound_burst'][idx].item(),
        )


class HazardHead(nn.Module):
    """Infers hazard and uncertainty state."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
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
        combined = torch.cat([z_pred, uncertainty], dim=-1)
        shared_features = self.shared(combined)
        return {name: torch.sigmoid(head(shared_features) * self.temperature).squeeze(-1) 
                for name, head in self.heads.items()}
    
    def to_hazard_state(self, outputs: Dict[str, torch.Tensor], idx: int = 0) -> HazardState:
        return HazardState(
            low_ambiguity=1.0 - outputs['ranking_instability'][idx].item(),
            poor_observability=outputs['low_observability'][idx].item(),
            stale_observation=outputs['stale_evidence'][idx].item(),
            route_instability=outputs['route_instability'][idx].item(),
            likely_ood=outputs['ood_regime'][idx].item(),
            tocttou_volatility=outputs['tocttou_volatility'][idx].item(),
            local_volatility_spike=outputs['route_instability'][idx].item(),
        )


class UncertaintyHead(nn.Module):
    """Computes decomposed uncertainty vector with calibration."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        input_dim = dims.bottleneck + dims.belief_hidden
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.GELU(),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        
        # Individual uncertainty components
        self.calibration = nn.Linear(32, 1)
        self.selective = nn.Linear(32, 1)
        self.ranking = nn.Linear(32, 1)
        self.ood = nn.Linear(32, 1)
        self.observability = nn.Linear(32, 1)
        
        # Learnable temperatures for calibration
        self.temps = nn.Parameter(torch.ones(5) * 1.3)
        
    def forward(self, z_pred: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z_pred, belief], dim=-1)
        shared = self.shared(combined)
        
        outputs = torch.cat([
            torch.sigmoid(self.calibration(shared) * self.temps[0]),
            torch.sigmoid(self.selective(shared) * self.temps[1]),
            torch.sigmoid(self.ranking(shared) * self.temps[2]),
            torch.sigmoid(self.ood(shared) * self.temps[3]),
            torch.sigmoid(self.observability(shared) * self.temps[4]),
        ], dim=-1)
        
        return outputs
    
    def to_uncertainty_vector(self, output: torch.Tensor, idx: int = 0) -> UncertaintyVector:
        return UncertaintyVector(
            calibration=output[idx, 0].item(),
            selective_prediction=output[idx, 1].item(),
            ranking=output[idx, 2].item(),
            ood=output[idx, 3].item(),
            observability=output[idx, 4].item(),
        )


class RankingActionHead(nn.Module):
    """
    Enhanced action head with margin-aware ranking.
    
    v1.1: Produces sharper action separation with explicit margin.
    """
    
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 48):
        super().__init__()
        
        # Deeper network for better separation
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )
        
        # Learnable temperature for sharpness - increased for better separation
        self.temperature = nn.Parameter(torch.tensor(2.5))
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: Raw action logits
            margin: Margin between top-2 actions
        """
        logits = self.net(z) * self.temperature
        
        # Compute margin
        sorted_logits, _ = logits.sort(dim=-1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        return logits, margin
    
    def get_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities."""
        return F.softmax(logits, dim=-1)


class BatchSchedulerHead(nn.Module):
    """Scores batch scheduling actions with margin."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        input_dim = dims.bottleneck + dims.slow_state + 8
        self.head = RankingActionHead(input_dim, len(BatchAction))
        
    def forward(self, z_pred: torch.Tensor, z_slow: torch.Tensor, pressure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([z_pred, z_slow, pressure], dim=-1)
        return self.head(combined)
    
    def to_scores(self, logits: torch.Tensor, idx: int = 0) -> Dict[BatchAction, float]:
        probs = F.softmax(logits[idx], dim=-1)
        return {action: probs[i].item() for i, action in enumerate(BatchAction)}


class KVPolicyHead(nn.Module):
    """Scores KV-cache policy actions with margin."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        input_dim = dims.bottleneck + dims.slow_state + 8
        self.head = RankingActionHead(input_dim, len(KVAction))
        
    def forward(self, z_pred: torch.Tensor, z_slow: torch.Tensor, kv_pressure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([z_pred, z_slow, kv_pressure], dim=-1)
        return self.head(combined)
    
    def to_scores(self, logits: torch.Tensor, idx: int = 0) -> Dict[KVAction, float]:
        probs = F.softmax(logits[idx], dim=-1)
        return {action: probs[i].item() for i, action in enumerate(KVAction)}


class NUMAPlacementHead(nn.Module):
    """Scores NUMA placement actions with margin."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        input_dim = dims.bottleneck + dims.slow_state + 8
        self.head = RankingActionHead(input_dim, len(NUMAAction))
        
    def forward(self, z_pred: torch.Tensor, z_slow: torch.Tensor, numa_pressure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([z_pred, z_slow, numa_pressure], dim=-1)
        return self.head(combined)
    
    def to_scores(self, logits: torch.Tensor, idx: int = 0) -> Dict[NUMAAction, float]:
        probs = F.softmax(logits[idx], dim=-1)
        return {action: probs[i].item() for i, action in enumerate(NUMAAction)}


class BoundaryControlHead(nn.Module):
    """Scores kernel boundary control actions with margin."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        input_dim = dims.bottleneck + dims.belief_hidden + 8
        self.head = RankingActionHead(input_dim, len(BoundaryAction))
        
    def forward(self, z_pred: torch.Tensor, belief: torch.Tensor, boundary_pressure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([z_pred, belief, boundary_pressure], dim=-1)
        return self.head(combined)
    
    def to_scores(self, logits: torch.Tensor, idx: int = 0) -> Dict[BoundaryAction, float]:
        probs = F.softmax(logits[idx], dim=-1)
        return {action: probs[i].item() for i, action in enumerate(BoundaryAction)}


class PagePolicyHead(nn.Module):
    """Scores page policy actions with margin."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        input_dim = dims.bottleneck + dims.belief_hidden + 8
        self.head = RankingActionHead(input_dim, len(PageAction))
        
    def forward(self, z_pred: torch.Tensor, belief: torch.Tensor, page_pressure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([z_pred, belief, page_pressure], dim=-1)
        return self.head(combined)
    
    def to_scores(self, logits: torch.Tensor, idx: int = 0) -> Dict[PageAction, float]:
        probs = F.softmax(logits[idx], dim=-1)
        return {action: probs[i].item() for i, action in enumerate(PageAction)}


class OutputHeads(nn.Module):
    """Container for all output heads v1.1."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        
        # State inference heads
        self.page_state = PageStateHead(dims)
        self.region_state = RegionStateHead(dims)
        self.process_phase = ProcessPhaseHead(dims)
        self.hazard = HazardHead(dims)
        self.uncertainty = UncertaintyHead(dims)
        
        # Action heads with margin
        self.batch_scheduler = BatchSchedulerHead(dims)
        self.kv_policy = KVPolicyHead(dims)
        self.numa_placement = NUMAPlacementHead(dims)
        self.boundary_control = BoundaryControlHead(dims)
        self.page_policy = PagePolicyHead(dims)
        
    def forward(
        self,
        z_pred: torch.Tensor,
        belief: torch.Tensor,
        z_slow: torch.Tensor,
        pressure_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all head outputs with margins."""
        # Compute uncertainty first
        uncertainty = self.uncertainty(z_pred, belief)
        
        # Ensure pressure features have right size
        if pressure_features.shape[-1] < 8:
            pressure_features = F.pad(pressure_features, (0, 8 - pressure_features.shape[-1]))
        pressure = pressure_features[:, :8]
        
        # Action heads with margins
        batch_logits, batch_margin = self.batch_scheduler(z_pred, z_slow, pressure)
        kv_logits, kv_margin = self.kv_policy(z_pred, z_slow, pressure)
        numa_logits, numa_margin = self.numa_placement(z_pred, z_slow, pressure)
        boundary_logits, boundary_margin = self.boundary_control(z_pred, belief, pressure)
        page_logits, page_margin = self.page_policy(z_pred, belief, pressure)
        
        # Aggregate margin (minimum across action heads)
        min_margin = torch.stack([batch_margin, kv_margin, numa_margin, boundary_margin, page_margin], dim=-1).min(dim=-1)[0]
        
        return {
            'page_state': self.page_state(z_pred, belief),
            'region_state': self.region_state(z_pred, belief, z_slow),
            'process_phase': self.process_phase(z_pred, z_slow),
            'hazard': self.hazard(z_pred, uncertainty),
            'uncertainty': uncertainty,
            'batch_actions': batch_logits,
            'kv_actions': kv_logits,
            'numa_actions': numa_logits,
            'boundary_actions': boundary_logits,
            'page_actions': page_logits,
            'action_margin': min_margin,
            'page_margin': page_margin,
        }
