"""
State-Plane modules (Plane B) for hidden-state inference v1.1.

Enhanced for:
- Stronger belief state tracking
- Better prototype/support formation
- Improved regime separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from .config import LatentDimensions, PrototypeConfig


class ObservabilityModule(nn.Module):
    """
    Models evidence quality and observability.
    
    Enhanced v1.1: Better uncertainty decomposition.
    """
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        self.obs_dim = dims.fused_observation
        
        # Observability inputs: missingness, freshness, source quality, conflicts
        self.quality_proj = nn.Linear(16, 24)  # Increased
        
        # Refinement network - deeper
        self.refine = nn.Sequential(
            nn.Linear(self.obs_dim + 24, self.obs_dim),
            nn.GELU(),
            nn.Linear(self.obs_dim, self.obs_dim),
            nn.GELU(),
            nn.Linear(self.obs_dim, self.obs_dim),
        )
        
        # Confidence/uncertainty heads - separate pathways
        self.confidence_net = nn.Sequential(
            nn.Linear(self.obs_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.uncertainty_net = nn.Sequential(
            nn.Linear(self.obs_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        self.norm = nn.LayerNorm(self.obs_dim)
        
    def forward(
        self,
        z_obs0: torch.Tensor,
        missingness_mask: torch.Tensor,
        freshness_ages: torch.Tensor,
        source_quality: torch.Tensor,
        conflict_score: torch.Tensor,
        consistency_score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Combine quality metadata
        quality_input = torch.cat([
            missingness_mask, freshness_ages, source_quality,
            conflict_score.unsqueeze(-1), consistency_score.unsqueeze(-1)
        ], dim=-1)
        quality_features = self.quality_proj(quality_input)
        
        # Refine observation with quality context
        combined = torch.cat([z_obs0, quality_features], dim=-1)
        z_obs = self.norm(z_obs0 + self.refine(combined))
        
        # Compute confidence scores with temperature
        obs_confidence = torch.sigmoid(self.confidence_net(z_obs) * 1.5)  # Sharper
        obs_uncertainty = torch.sigmoid(self.uncertainty_net(z_obs) * 1.5)
        
        return z_obs, obs_confidence.squeeze(-1), obs_uncertainty.squeeze(-1)


class BeliefStateTracker(nn.Module):
    """
    Converts transient observations into persistent beliefs.
    
    Enhanced v1.1: Stronger gating, better regime tracking.
    """
    
    def __init__(self, dims: LatentDimensions, gate_bias: float = 0.15):
        super().__init__()
        self.belief_dim = dims.belief_hidden
        self.obs_dim = dims.fused_observation
        
        # Larger GRU-style update
        hidden = self.belief_dim + 16
        self.update_gate = nn.Sequential(
            nn.Linear(self.obs_dim + self.belief_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.belief_dim),
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(self.obs_dim + self.belief_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.belief_dim),
        )
        self.candidate = nn.Sequential(
            nn.Linear(self.obs_dim + self.belief_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.belief_dim),
        )
        
        # Adaptive alpha computation - more expressive
        self.alpha_net = nn.Sequential(
            nn.Linear(self.obs_dim + self.belief_dim + 4, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        
        self.gate_bias = gate_bias
        self.norm = nn.LayerNorm(self.belief_dim)
        
    def forward(
        self,
        z_obs: torch.Tensor,
        belief_prev: torch.Tensor,
        freshness: torch.Tensor,
        obs_quality: torch.Tensor,
        activity_level: torch.Tensor,
        transition_strength: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([z_obs, belief_prev], dim=-1)
        
        # GRU-style gates with sharper activation
        z = torch.sigmoid(self.update_gate(combined) * 1.2 + self.gate_bias)
        r = torch.sigmoid(self.reset_gate(combined) * 1.2)
        
        # Candidate belief
        reset_belief = r * belief_prev
        candidate_input = torch.cat([z_obs, reset_belief], dim=-1)
        belief_candidate = torch.tanh(self.candidate(candidate_input))
        
        # Compute adaptive alpha with sharper response
        alpha_input = torch.cat([
            z_obs, belief_prev,
            freshness.unsqueeze(-1), obs_quality.unsqueeze(-1),
            activity_level.unsqueeze(-1), transition_strength.unsqueeze(-1)
        ], dim=-1)
        alpha = torch.sigmoid(self.alpha_net(alpha_input) * 1.5)  # Sharper
        
        # Interpolate between old and new belief
        belief_new = z * belief_candidate + (1 - z) * belief_prev
        belief_final = alpha * belief_new + (1 - alpha) * belief_prev
        
        return self.norm(belief_final)
    
    def init_belief(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.belief_dim, device=device)


class PredictiveBottleneck(nn.Module):
    """
    The dense-first brain of the system.
    
    Enhanced v1.1: Wider bottleneck, better compression.
    """
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        self.hidden_dim = dims.predictive_hidden
        self.bottleneck_dim = dims.bottleneck
        
        # Input projection
        input_dim = dims.fused_observation + dims.belief_hidden + dims.slow_state + 16
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # Low-rank recurrent block - larger rank
        self.rank = self.bottleneck_dim
        self.down_proj = nn.Linear(self.hidden_dim, self.rank)
        self.up_proj = nn.Linear(self.rank, self.hidden_dim)
        
        # Recurrent state with gating
        self.state_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.state_transform = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        # Output bottleneck with residual
        self.bottleneck_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(self.bottleneck_dim * 2, self.bottleneck_dim),
        )
        
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.bottleneck_dim)
        
    def forward(
        self,
        z_obs: torch.Tensor,
        belief: torch.Tensor,
        z_slow: torch.Tensor,
        pressure_summary: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Combine inputs
        combined = torch.cat([z_obs, belief, z_slow, pressure_summary], dim=-1)
        h = self.input_proj(combined)
        
        # Low-rank transformation
        low_rank = self.down_proj(h)
        h_transformed = self.up_proj(F.gelu(low_rank))
        h = self.norm1(h + h_transformed)
        
        # Recurrent update
        if hidden_state is not None:
            combined_state = torch.cat([h, hidden_state], dim=-1)
            gate = torch.sigmoid(self.state_gate(combined_state))
            transform = torch.tanh(self.state_transform(combined_state))
            h = gate * transform + (1 - gate) * hidden_state
        
        # Project to bottleneck
        z_pred = self.norm2(self.bottleneck_proj(h))
        
        return z_pred, h
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


class SlowStateMemory(nn.Module):
    """
    Tracks slower and more persistent structure.
    
    Enhanced v1.1: Better long-horizon pattern capture.
    """
    
    def __init__(self, dims: LatentDimensions, decay: float = 0.985):
        super().__init__()
        self.slow_dim = dims.slow_state
        self.decay = decay
        
        # Update network - deeper
        input_dim = dims.fused_observation + dims.belief_hidden
        self.update_net = nn.Sequential(
            nn.Linear(input_dim, self.slow_dim * 2),
            nn.GELU(),
            nn.Linear(self.slow_dim * 2, self.slow_dim * 2),
            nn.GELU(),
            nn.Linear(self.slow_dim * 2, self.slow_dim),
        )
        
        # Gate for selective update - more expressive
        self.update_gate = nn.Sequential(
            nn.Linear(input_dim + self.slow_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        self.norm = nn.LayerNorm(self.slow_dim)
        
    def forward(
        self,
        z_obs: torch.Tensor,
        belief: torch.Tensor,
        slow_prev: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([z_obs, belief], dim=-1)
        
        # Compute update
        update = self.update_net(combined)
        
        # Compute gate with sharper response
        gate_input = torch.cat([combined, slow_prev], dim=-1)
        gate = torch.sigmoid(self.update_gate(gate_input) * 1.5) * (1 - self.decay)
        
        # EMA update
        slow_new = gate * update + (1 - gate) * slow_prev
        
        return self.norm(slow_new)
    
    def init_slow(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.slow_dim, device=device)


class PrototypeBank(nn.Module):
    """
    Maintains a compact bank of recurring regimes.
    
    Enhanced v1.1: Better support formation, prototype consolidation.
    Fixed: Support density now properly normalized to [0, 1] range.
    """
    
    def __init__(self, dims: LatentDimensions, config: PrototypeConfig):
        super().__init__()
        self.num_prototypes = config.num_prototypes
        self.proto_dim = dims.prototype
        self.pred_dim = dims.bottleneck
        self.config = config
        
        # Learnable prototypes with better initialization
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim) * 0.1)
        
        # Projection from predictive latent to prototype space - deeper
        self.proj = nn.Sequential(
            nn.Linear(self.pred_dim, self.proto_dim * 2),
            nn.GELU(),
            nn.Linear(self.proto_dim * 2, self.proto_dim),
        )
        
        # Support counts - start with meaningful base support
        self.register_buffer('support_counts', torch.ones(self.num_prototypes) * 50)
        self.register_buffer('recent_assignments', torch.ones(self.num_prototypes) * 10)
        self.register_buffer('prototype_quality', torch.ones(self.num_prototypes) * 0.7)
        self.register_buffer('total_observations', torch.tensor(50.0 * self.num_prototypes))
        
        # Temperature for similarity
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(
        self,
        z_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = z_pred.shape[0]
        device = z_pred.device
        
        # Project to prototype space
        z_proto = self.proj(z_pred)
        
        # Compute similarities (cosine) with temperature
        z_norm = F.normalize(z_proto, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        similarity = torch.matmul(z_norm, proto_norm.t()) / (self.temperature.abs() + 0.1)
        
        # Soft assignment probabilities
        soft_assign = F.softmax(similarity * 2, dim=-1)  # Sharper
        
        # Best match
        best_sim, best_match = similarity.max(dim=-1)
        
        # Support density: proportion of support for matched prototypes
        # Normalized to [0, 1] based on how much support the best-matching prototypes have
        matched_support = self.support_counts[best_match]
        max_possible_support = self.support_counts.max() + 1e-8
        
        # Support density combines:
        # 1. How much support the matched prototype has (relative to max)
        # 2. Quality of the match (similarity)
        # 3. Quality of the prototype
        matched_quality = self.prototype_quality[best_match]
        support_density = (matched_support / max_possible_support) * matched_quality * torch.sigmoid(best_sim)
        
        # Clamp to valid range and boost for better behavior
        support_density = support_density.clamp(0.1, 1.0)  # Floor at 0.1 for stability
        
        # Familiarity (based on similarity and support)
        familiarity = torch.sigmoid((best_sim - 0.3) * 3) * support_density
        
        # Drift score
        recent_total = self.recent_assignments.sum() + 1e-8
        recent_dist = self.recent_assignments / recent_total
        drift_score = F.kl_div(
            soft_assign.log(), 
            recent_dist.unsqueeze(0).expand_as(soft_assign),
            reduction='none'
        ).sum(dim=-1)
        
        return similarity, support_density, familiarity, drift_score, best_match
    
    def update_support(self, best_match: torch.Tensor, similarities: torch.Tensor, decay: float = 0.998):
        """Update support counts with quality weighting."""
        # Slower decay to build up support
        self.support_counts.data *= decay
        self.recent_assignments.data *= decay
        
        # Soft update based on similarity - stronger updates
        soft_assign = F.softmax(similarities * 2, dim=-1)
        assignment_sum = soft_assign.sum(dim=0)
        
        # Stronger support accumulation
        self.support_counts.data += assignment_sum.detach() * 2.0
        self.recent_assignments.data += assignment_sum.detach() * 2.0
        self.total_observations.data += similarities.shape[0]
        
        # Update quality based on consistency
        for idx in best_match.unique():
            mask = best_match == idx
            if mask.sum() > 1:
                # Higher quality if assignments are consistent
                sims = similarities[mask, idx]
                consistency = 1.0 - sims.std() / (sims.mean().abs() + 1e-8)
                self.prototype_quality[idx] = 0.95 * self.prototype_quality[idx] + 0.05 * consistency.clamp(0.3, 1.0)
            else:
                # Single assignment - slight quality boost for being matched
                self.prototype_quality[idx] = 0.99 * self.prototype_quality[idx] + 0.01 * 0.8
    
    def get_support_stats(self) -> Dict[str, float]:
        """Get support statistics for monitoring."""
        return {
            'mean_support': self.support_counts.mean().item(),
            'max_support': self.support_counts.max().item(),
            'min_support': self.support_counts.min().item(),
            'active_prototypes': (self.support_counts > self.config.min_support_for_match).sum().item(),
            'mean_quality': self.prototype_quality.mean().item(),
        }


class ProbingController(nn.Module):
    """
    Determines whether to spend more model capacity.
    
    Enhanced v1.1: Better expert eligibility, sharper decisions.
    """
    
    def __init__(self, dims: LatentDimensions, num_prototypes: int = 96, num_experts: int = 8):
        super().__init__()
        
        # Input dimensions
        input_dim = dims.bottleneck + dims.slow_state + 5 + num_prototypes + num_experts + num_experts + 1
        
        # Deeper network for better decisions
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Linear(64, 48),
            nn.GELU(),
        )
        
        # Separate heads with sharper outputs
        self.probing_head = nn.Sequential(
            nn.Linear(48, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.support_head = nn.Sequential(
            nn.Linear(48, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.expert_eligibility_head = nn.Sequential(
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, num_experts),
        )
        self.route_confidence_head = nn.Sequential(
            nn.Linear(48, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        
    def forward(
        self,
        z_pred: torch.Tensor,
        z_slow: torch.Tensor,
        uncertainty: torch.Tensor,
        proto_similarity: torch.Tensor,
        support_counts: torch.Tensor,
        recent_expert_gain: torch.Tensor,
        drift_score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Ensure drift_score has correct dimensions
        if drift_score.dim() == 0:
            drift_score = drift_score.unsqueeze(0).unsqueeze(-1).expand(z_pred.shape[0], 1)
        elif drift_score.dim() == 1:
            drift_score = drift_score.unsqueeze(-1)
        
        combined = torch.cat([
            z_pred, z_slow, uncertainty, proto_similarity,
            support_counts, recent_expert_gain, drift_score
        ], dim=-1)
        
        features = self.net(combined)
        
        # Sharper outputs with temperature
        probing_value = torch.sigmoid(self.probing_head(features) * 1.5)
        support_density = torch.sigmoid(self.support_head(features) * 1.5)
        expert_eligibility = torch.sigmoid(self.expert_eligibility_head(features) * 1.3)
        route_confidence = torch.sigmoid(self.route_confidence_head(features) * 1.5)
        
        return probing_value.squeeze(-1), support_density.squeeze(-1), expert_eligibility, route_confidence.squeeze(-1)


class AbstentionController(nn.Module):
    """
    New v1.1: Dedicated abstention decision module.

    Learns when to abstain based on:
    - Uncertainty decomposition
    - Support density
    - Action margin
    - Observability quality

    v1.4: Temperature scaling, support-conditioned calibration, OOD detection.
    """

    def __init__(self, dims: LatentDimensions):
        super().__init__()

        # Input: bottleneck + uncertainty(5) + support + action_margin + observability
        input_dim = dims.bottleneck + 5 + 3

        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.GELU(),
            nn.Linear(48, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )

        # Abstention probability - initialized to be less conservative
        self.abstain_head = nn.Linear(16, 1)
        # Initialize bias to produce moderate abstention probability
        nn.init.constant_(self.abstain_head.bias, -0.3)

        # Temperature scaling for calibration (v1.4)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Support-conditioned calibration (v1.4)
        self.support_calibration = nn.Sequential(
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

        # Confidence calibration - initialized to produce moderate confidence
        self.calibration_head = nn.Linear(16, 1)
        nn.init.constant_(self.calibration_head.bias, 0.3)

    def forward(
        self,
        z_pred: torch.Tensor,
        uncertainty: torch.Tensor,
        support_density: torch.Tensor,
        action_margin: torch.Tensor,
        observability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            abstain_prob: Probability that we should abstain (calibrated to be lower)
            calibrated_confidence: Calibrated confidence score
        """
        # Normalize action margin to [0, 1] for input
        normalized_margin = torch.sigmoid(action_margin)

        combined = torch.cat([
            z_pred, uncertainty,
            support_density.unsqueeze(-1),
            normalized_margin.unsqueeze(-1),
            observability.unsqueeze(-1),
        ], dim=-1)

        features = self.net(combined)

        # Temperature-scaled abstention probability
        abstain_prob = torch.sigmoid(self.abstain_head(features) * self.temperature.abs())

        # Support-conditioned confidence calibration (v1.4)
        base_confidence = torch.sigmoid(self.calibration_head(features))
        support_boost = torch.sigmoid(self.support_calibration(features)) * support_density.unsqueeze(-1)
        calibrated_confidence = (base_confidence + 0.2 * support_boost).clamp(0, 1)

        return abstain_prob.squeeze(-1), calibrated_confidence.squeeze(-1)
