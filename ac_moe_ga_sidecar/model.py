"""
Main neural model for AC-MoE-GA Sidecar v1.1.

Enhanced for:
- Better support/prototype formation
- Calibrated abstention
- Sharper action separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config import SidecarConfig, LatentDimensions
from .encoders import (
    BytePlaneEncoder, AddressShapeEncoder, EventSemanticEncoder,
    MapStateEncoder, SummaryEncoder, FusedObservationBlock
)
from .state_plane import (
    ObservabilityModule, BeliefStateTracker, PredictiveBottleneck,
    SlowStateMemory, PrototypeBank, ProbingController, AbstentionController
)
from .experts import ExpertRouter
from .output_heads import OutputHeads


@dataclass
class ModelState:
    """Container for recurrent model state."""
    belief_page: torch.Tensor
    belief_region: torch.Tensor
    belief_process: torch.Tensor
    slow_state: torch.Tensor
    pred_hidden: torch.Tensor
    
    @classmethod
    def init(cls, batch_size: int, dims: LatentDimensions, device: torch.device) -> 'ModelState':
        return cls(
            belief_page=torch.zeros(batch_size, dims.belief_hidden, device=device),
            belief_region=torch.zeros(batch_size, dims.belief_hidden, device=device),
            belief_process=torch.zeros(batch_size, dims.belief_hidden, device=device),
            slow_state=torch.zeros(batch_size, dims.slow_state, device=device),
            pred_hidden=torch.zeros(batch_size, dims.predictive_hidden, device=device),
        )


@dataclass
class ModelOutput:
    """Container for model outputs v1.1."""
    z_pred: torch.Tensor
    uncertainty: torch.Tensor
    head_outputs: Dict[str, torch.Tensor]
    used_experts: List[str]
    routing_weights: torch.Tensor
    proto_match: torch.Tensor
    support_density: torch.Tensor
    familiarity: torch.Tensor
    drift_score: torch.Tensor
    should_abstain: torch.Tensor
    calibrated_confidence: torch.Tensor
    action_margin: torch.Tensor
    new_state: ModelState


class ACMoEGAModel(nn.Module):
    """
    AC-MoE-GA Systems Sidecar Neural Model v1.1.
    
    Enhanced for:
    - Stronger regime/support formation
    - Calibrated abstention behavior
    - Sharper action separation with margins
    """
    
    def __init__(self, config: SidecarConfig):
        super().__init__()
        self.config = config
        self.dims = config.latent_dims
        
        # Plane A: Byte-plane encoders - pass feature vocab for safe indexing
        self.byte_encoder = BytePlaneEncoder(self.dims, config.feature_vocab)
        self.address_encoder = AddressShapeEncoder(self.dims, config.feature_vocab)
        self.event_encoder = EventSemanticEncoder(self.dims, config.feature_vocab)
        self.map_encoder = MapStateEncoder(self.dims, config.feature_vocab)
        self.summary_encoder = SummaryEncoder(self.dims, config.feature_vocab)
        self.fusion = FusedObservationBlock(self.dims)
        
        # Plane B: State-plane modules
        self.observability = ObservabilityModule(self.dims)
        self.belief_tracker = BeliefStateTracker(self.dims, config.updates.belief_update_gate_bias)
        self.predictive_core = PredictiveBottleneck(self.dims)
        self.slow_memory = SlowStateMemory(self.dims)
        self.prototype_bank = PrototypeBank(self.dims, config.prototypes)
        self.probing_controller = ProbingController(
            self.dims, 
            num_prototypes=config.prototypes.num_prototypes,
            num_experts=config.num_experts
        )
        
        # New v1.1: Abstention controller
        self.abstention_controller = AbstentionController(self.dims)
        
        # Expert system
        self.expert_router = ExpertRouter(self.dims, config.routing)
        
        # Output heads
        self.output_heads = OutputHeads(self.dims)
        
        # Thresholds from config
        self.abstain_threshold = config.uncertainty.abstain_threshold
        self.min_action_margin = config.uncertainty.min_action_margin
        self.min_confidence_for_override = config.uncertainty.min_confidence_for_override
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        state: Optional[ModelState] = None,
        expert_context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> ModelOutput:
        batch_size = inputs['low8_bucket'].shape[0]
        device = inputs['low8_bucket'].device
        
        # Initialize state if needed
        if state is None:
            state = ModelState.init(batch_size, self.dims, device)
        
        # === Plane A: Encode observations ===
        z_byte = self.byte_encoder(
            inputs['low8_bucket'],
            inputs['high8_bucket'],
            inputs['alignment_bucket'],
            inputs['small_int_bucket'],
            inputs['delta_bucket'],
            inputs['hamming_bucket'],
            inputs['continuous_features'],
            inputs['bitfield_features'],
            inputs['sketch_features'],
        )
        
        z_addr = self.address_encoder(
            inputs['page_hash_bucket'],
            inputs['offset_bucket'],
            inputs['cache_line_bucket'],
            inputs['addr_alignment_bucket'],
            inputs['stride_bucket'],
            inputs['reuse_dist_bucket'],
            inputs['locality_cluster'],
            inputs['entropy_bucket'],
            inputs['address_flags'],
        )
        
        z_evt = self.event_encoder(
            inputs['event_type'],
            inputs['fault_class'],
            inputs['syscall_class'],
            inputs['opcode_family'],
            inputs['transition_type'],
            inputs['result_class'],
        )
        
        z_map = self.map_encoder(
            inputs['pte_flags'],
            inputs['vma_class'],
            inputs['protection_domain'],
        )
        
        z_sum = self.summary_encoder(
            inputs['read_count_bucket'],
            inputs['write_count_bucket'],
            inputs['fault_count_bucket'],
            inputs['cow_count_bucket'],
            inputs['recency_bucket'],
            inputs['volatility_features'],
            inputs['pressure_features'],
        )
        
        # Fuse all encodings
        z_obs0 = self.fusion(z_byte, z_addr, z_evt, z_map, z_sum)
        
        # === Observability refinement ===
        z_obs, obs_confidence, obs_uncertainty = self.observability(
            z_obs0,
            inputs['missingness_mask'],
            inputs['freshness_ages'],
            inputs['source_quality'],
            inputs['conflict_score'],
            inputs['consistency_score'],
        )
        
        # === Plane B: State inference ===
        
        # Update belief state
        belief = self.belief_tracker(
            z_obs,
            state.belief_page,
            freshness=torch.ones(batch_size, device=device),
            obs_quality=obs_confidence,
            activity_level=torch.ones(batch_size, device=device),
            transition_strength=torch.ones(batch_size, device=device) * 0.5,
        )
        
        # Update slow state
        z_slow = self.slow_memory(z_obs, belief, state.slow_state)
        
        # Pressure summary
        pressure_summary = inputs['pressure_features'][:, :16]
        if pressure_summary.shape[1] < 16:
            pressure_summary = F.pad(pressure_summary, (0, 16 - pressure_summary.shape[1]))
        
        # Predictive bottleneck
        z_pred, pred_hidden = self.predictive_core(
            z_obs, belief, z_slow, pressure_summary, state.pred_hidden
        )
        
        # Prototype matching
        proto_sim, support_density, familiarity, drift_score, proto_match = self.prototype_bank(z_pred)
        
        # Compute uncertainty
        uncertainty = self.output_heads.uncertainty(z_pred, belief)
        
        # Probing controller
        probing_value, support_score, expert_eligibility, route_confidence = self.probing_controller(
            z_pred, z_slow, uncertainty, proto_sim,
            support_density.unsqueeze(-1).expand(-1, 8),
            self.expert_router.expert_gains.unsqueeze(0).expand(batch_size, -1),
            drift_score,
        )
        
        # === Expert routing ===
        z_enhanced, used_experts, routing_weights = self.expert_router(
            z_pred,
            expert_eligibility,
            support_score,
            probing_value,
            drift_score,
            obs_confidence,
            expert_context,
        )
        
        # === Output heads ===
        head_outputs = self.output_heads(
            z_enhanced, belief, z_slow, inputs['pressure_features']
        )
        
        # Get action margin
        action_margin = head_outputs['action_margin']
        
        # === Abstention decision (v1.1 - balanced logic) ===
        abstain_prob, calibrated_confidence = self.abstention_controller(
            z_pred,
            uncertainty,
            support_density,
            action_margin,
            obs_confidence,
        )
        
        # Normalize action margin to [0, 1] range for comparison
        # Raw margin can be negative or large; sigmoid normalizes it
        normalized_margin = torch.sigmoid(action_margin)
        
        # Determine abstention with balanced criteria:
        # Abstain when signals indicate uncertainty
        # Count how many "concern" signals are present
        concern_count = (
            (abstain_prob > self.abstain_threshold).float() +
            (calibrated_confidence < self.min_confidence_for_override).float() +
            (normalized_margin < 0.4).float() +  # Margin below 40% after sigmoid
            (support_density < 0.15).float()
        )
        
        # Abstain when 2+ concerns (more responsive than 3+)
        should_abstain = concern_count >= 2
        
        # Update prototype support
        self.prototype_bank.update_support(proto_match, proto_sim)
        
        # Build new state
        new_state = ModelState(
            belief_page=belief,
            belief_region=state.belief_region,
            belief_process=state.belief_process,
            slow_state=z_slow,
            pred_hidden=pred_hidden,
        )
        
        return ModelOutput(
            z_pred=z_enhanced,
            uncertainty=uncertainty,
            head_outputs=head_outputs,
            used_experts=used_experts,
            routing_weights=routing_weights,
            proto_match=proto_match,
            support_density=support_density,
            familiarity=familiarity,
            drift_score=drift_score,
            should_abstain=should_abstain,
            calibrated_confidence=calibrated_confidence,
            action_margin=action_margin,
            new_state=new_state,
        )
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
