"""
Typed Expert System for AC-MoE-GA Sidecar.

Small set of typed residual experts that provide specialized corrections
when justified by support and uncertainty conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from .config import LatentDimensions, RoutingConfig, ExpertType


class BaseExpert(nn.Module, ABC):
    """Base class for typed experts."""
    
    def __init__(self, dims: LatentDimensions, expert_type: ExpertType):
        super().__init__()
        self.expert_type = expert_type
        self.input_dim = dims.bottleneck
        self.output_dim = dims.expert_residual
        
        # Small residual network
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        
        # Output projection back to bottleneck dim
        self.out_proj = nn.Linear(self.output_dim, dims.bottleneck)
        
    def forward(self, z_pred: torch.Tensor) -> torch.Tensor:
        """Compute residual correction."""
        residual = self.net(z_pred)
        return self.out_proj(residual)


class PageTransitionExpert(BaseExpert):
    """Expert for page state transitions."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__(dims, ExpertType.PAGE_TRANSITION)
        
        # Additional page-specific features
        self.page_context = nn.Linear(8, self.output_dim // 2)
        self.fusion = nn.Linear(self.output_dim + self.output_dim // 2, self.output_dim)
        
    def forward(self, z_pred: torch.Tensor, page_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_residual = self.net(z_pred)
        
        if page_features is not None:
            page_ctx = F.gelu(self.page_context(page_features))
            combined = torch.cat([base_residual, page_ctx], dim=-1)
            base_residual = self.fusion(combined)
            
        return self.out_proj(base_residual)


class COWForkExpert(BaseExpert):
    """Expert for Copy-on-Write and fork behavior."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__(dims, ExpertType.COW_FORK)
        
        # COW-specific context
        self.cow_context = nn.Linear(6, self.output_dim // 2)
        self.fusion = nn.Linear(self.output_dim + self.output_dim // 2, self.output_dim)
        
    def forward(self, z_pred: torch.Tensor, cow_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_residual = self.net(z_pred)
        
        if cow_features is not None:
            cow_ctx = F.gelu(self.cow_context(cow_features))
            combined = torch.cat([base_residual, cow_ctx], dim=-1)
            base_residual = self.fusion(combined)
            
        return self.out_proj(base_residual)


class ReclaimHotnessExpert(BaseExpert):
    """Expert for reclaim decisions and hotness tracking."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__(dims, ExpertType.RECLAIM_HOTNESS)
        
        # Reclaim-specific context
        self.reclaim_context = nn.Linear(10, self.output_dim // 2)
        self.fusion = nn.Linear(self.output_dim + self.output_dim // 2, self.output_dim)
        
    def forward(self, z_pred: torch.Tensor, reclaim_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_residual = self.net(z_pred)
        
        if reclaim_features is not None:
            reclaim_ctx = F.gelu(self.reclaim_context(reclaim_features))
            combined = torch.cat([base_residual, reclaim_ctx], dim=-1)
            base_residual = self.fusion(combined)
            
        return self.out_proj(base_residual)


class LocalityPatternExpert(BaseExpert):
    """Expert for locality and access pattern recognition."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__(dims, ExpertType.LOCALITY_PATTERN)
        
        # Locality-specific context
        self.locality_context = nn.Linear(12, self.output_dim // 2)
        self.fusion = nn.Linear(self.output_dim + self.output_dim // 2, self.output_dim)
        
    def forward(self, z_pred: torch.Tensor, locality_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_residual = self.net(z_pred)
        
        if locality_features is not None:
            locality_ctx = F.gelu(self.locality_context(locality_features))
            combined = torch.cat([base_residual, locality_ctx], dim=-1)
            base_residual = self.fusion(combined)
            
        return self.out_proj(base_residual)


class FaultBurstExpert(BaseExpert):
    """Expert for fault burst detection and handling."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__(dims, ExpertType.FAULT_BURST)
        
        # Fault-specific context
        self.fault_context = nn.Linear(8, self.output_dim // 2)
        self.fusion = nn.Linear(self.output_dim + self.output_dim // 2, self.output_dim)
        
    def forward(self, z_pred: torch.Tensor, fault_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_residual = self.net(z_pred)
        
        if fault_features is not None:
            fault_ctx = F.gelu(self.fault_context(fault_features))
            combined = torch.cat([base_residual, fault_ctx], dim=-1)
            base_residual = self.fusion(combined)
            
        return self.out_proj(base_residual)


class BoundaryControlExpert(BaseExpert):
    """Expert for kernel boundary optimization."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__(dims, ExpertType.BOUNDARY_CONTROL)
        
        # Boundary-specific context
        self.boundary_context = nn.Linear(8, self.output_dim // 2)
        self.fusion = nn.Linear(self.output_dim + self.output_dim // 2, self.output_dim)
        
    def forward(self, z_pred: torch.Tensor, boundary_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_residual = self.net(z_pred)
        
        if boundary_features is not None:
            boundary_ctx = F.gelu(self.boundary_context(boundary_features))
            combined = torch.cat([base_residual, boundary_ctx], dim=-1)
            base_residual = self.fusion(combined)
            
        return self.out_proj(base_residual)


class KVPolicyExpert(BaseExpert):
    """Expert for KV-cache policy decisions."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__(dims, ExpertType.KV_POLICY)
        
        # KV-specific context
        self.kv_context = nn.Linear(10, self.output_dim // 2)
        self.fusion = nn.Linear(self.output_dim + self.output_dim // 2, self.output_dim)
        
    def forward(self, z_pred: torch.Tensor, kv_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_residual = self.net(z_pred)
        
        if kv_features is not None:
            kv_ctx = F.gelu(self.kv_context(kv_features))
            combined = torch.cat([base_residual, kv_ctx], dim=-1)
            base_residual = self.fusion(combined)
            
        return self.out_proj(base_residual)


class NUMAPlacementExpert(BaseExpert):
    """Expert for NUMA placement decisions."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__(dims, ExpertType.NUMA_PLACEMENT)
        
        # NUMA-specific context
        self.numa_context = nn.Linear(8, self.output_dim // 2)
        self.fusion = nn.Linear(self.output_dim + self.output_dim // 2, self.output_dim)
        
    def forward(self, z_pred: torch.Tensor, numa_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_residual = self.net(z_pred)
        
        if numa_features is not None:
            numa_ctx = F.gelu(self.numa_context(numa_features))
            combined = torch.cat([base_residual, numa_ctx], dim=-1)
            base_residual = self.fusion(combined)
            
        return self.out_proj(base_residual)


class ExpertRouter(nn.Module):
    """
    Routes inputs to appropriate experts based on probing controller output.
    
    Implements sparse routing with support and uncertainty gating.
    """
    
    def __init__(self, dims: LatentDimensions, config: RoutingConfig):
        super().__init__()
        self.config = config
        self.num_experts = 8
        
        # Create all experts
        self.experts = nn.ModuleDict({
            'page_transition': PageTransitionExpert(dims),
            'cow_fork': COWForkExpert(dims),
            'reclaim_hotness': ReclaimHotnessExpert(dims),
            'locality_pattern': LocalityPatternExpert(dims),
            'fault_burst': FaultBurstExpert(dims),
            'boundary_control': BoundaryControlExpert(dims),
            'kv_policy': KVPolicyExpert(dims),
            'numa_placement': NUMAPlacementExpert(dims),
        })
        
        self.expert_names = list(self.experts.keys())
        
        # Track expert gains for routing decisions
        self.register_buffer('expert_gains', torch.zeros(self.num_experts))
        self.register_buffer('expert_usage', torch.zeros(self.num_experts))
        
    def forward(
        self,
        z_pred: torch.Tensor,
        expert_eligibility: torch.Tensor,
        support_density: torch.Tensor,
        probing_value: torch.Tensor,
        drift_score: torch.Tensor,
        observability: torch.Tensor,
        expert_context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Route to experts and compute fused output.
        
        Returns:
            z_enhanced: Enhanced predictive latent
            used_experts: Names of experts that were used
            routing_weights: Weights assigned to each expert
        """
        batch_size = z_pred.shape[0]
        device = z_pred.device
        
        # Check routing conditions
        should_route = (
            (support_density > self.config.min_support_count / 1000) &
            (probing_value > self.config.min_probing_value) &
            (drift_score < self.config.max_drift_penalty) &
            (observability > self.config.min_observability)
        )
        
        # If no routing justified, return original
        if not should_route.any():
            return z_pred, [], torch.zeros(batch_size, self.num_experts, device=device)
        
        # Compute routing weights
        # Mask out ineligible experts
        routing_logits = expert_eligibility.clone()
        
        # Apply expert gain bonus
        gain_bonus = self.expert_gains.unsqueeze(0).expand(batch_size, -1)
        routing_logits = routing_logits + 0.1 * gain_bonus
        
        # Top-k selection
        top_k = self.config.default_top_k
        if probing_value.mean() > 0.7:  # High probing value allows top-2
            top_k = min(2, self.config.max_experts_per_inference)
        
        top_values, top_indices = routing_logits.topk(top_k, dim=-1)
        
        # Normalize weights
        routing_weights = torch.zeros(batch_size, self.num_experts, device=device)
        routing_weights.scatter_(1, top_indices, F.softmax(top_values, dim=-1))
        
        # Apply routing mask
        routing_weights = routing_weights * should_route.unsqueeze(-1).float()
        
        # Compute expert outputs
        expert_outputs = []
        used_experts = set()
        
        for i, name in enumerate(self.expert_names):
            weight = routing_weights[:, i:i+1]
            if weight.sum() > 0:
                used_experts.add(name)
                context = expert_context.get(name) if expert_context else None
                expert_out = self.experts[name](z_pred, context)
                expert_outputs.append(weight * expert_out)
        
        # Fuse expert outputs (residual style)
        if expert_outputs:
            total_residual = sum(expert_outputs)
            z_enhanced = z_pred + total_residual
        else:
            z_enhanced = z_pred
        
        # Update usage statistics
        self.expert_usage += routing_weights.sum(dim=0).detach()
        
        return z_enhanced, list(used_experts), routing_weights
    
    def update_expert_gains(self, expert_idx: int, gain: float, decay: float = 0.99):
        """Update expert gain tracking."""
        self.expert_gains[expert_idx] = decay * self.expert_gains[expert_idx] + (1 - decay) * gain
