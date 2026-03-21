"""
BitNet-inspired model for AC-MoE-GA Sidecar.

Implements native low-bit design principles:
- Dense core uses ternary/low-bit operations
- Shared projections to reduce memory traffic
- Fragile components (uncertainty, calibration) stay higher precision
- CPU-friendly inference with quantized activations

This model matches the original ACMoEGAModel interface exactly for drop-in compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config import SidecarConfig, LatentDimensions, FeatureVocabConfig
from .bitnet_layers import (
    TernaryLinear, LowBitEncoder, SharedProjection,
    ternary_quantize, dynamic_quantize
)
from .encoders import GatedFusion
from .bitnet_state_heads import (
    BitNetPageStateHead, BitNetRegionStateHead,
    BitNetProcessPhaseHead, BitNetHazardHead
)
from .output_heads import RankingActionHead
from .state_plane import (
    ObservabilityModule, BeliefStateTracker, PredictiveBottleneck,
    SlowStateMemory, PrototypeBank, ProbingController, AbstentionController
)
from .experts import ExpertRouter
from .output_heads import OutputHeads


@dataclass
class BitNetModelState:
    """Container for recurrent model state (BitNet version)."""
    belief_page: torch.Tensor
    belief_region: torch.Tensor
    belief_process: torch.Tensor
    slow_state: torch.Tensor
    pred_hidden: torch.Tensor
    
    @classmethod
    def init(cls, batch_size: int, dims: LatentDimensions, device: torch.device) -> 'BitNetModelState':
        return cls(
            belief_page=torch.zeros(batch_size, dims.belief_hidden, device=device),
            belief_region=torch.zeros(batch_size, dims.belief_hidden, device=device),
            belief_process=torch.zeros(batch_size, dims.belief_hidden, device=device),
            slow_state=torch.zeros(batch_size, dims.slow_state, device=device),
            pred_hidden=torch.zeros(batch_size, dims.predictive_hidden, device=device),
        )


@dataclass
class BitNetModelOutput:
    """Container for model outputs (BitNet version)."""
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
    new_state: BitNetModelState


class BitNetACMoEGAModel(nn.Module):
    """
    BitNet-inspired AC-MoE-GA Sidecar Neural Model.
    
    Implements native low-bit design while matching the original model interface exactly.
    Key differences from original:
    - Dense core uses ternary/low-bit operations
    - Shared projections reduce memory traffic
    - Fragile components (uncertainty, calibration) stay higher precision
    
    Note: Uses the same dimensions as the standard model for interface parity.
    """
    
    def __init__(self, config: SidecarConfig):
        super().__init__()
        self.config = config
        # Use standard dimensions for interface parity
        self.dims = config.latent_dims
        
        # === Plane A: Low-bit encoders ===
        # These use ternary weights and 8-bit quantization
        self.byte_encoder = BitNetBytePlaneEncoder(self.dims, config.feature_vocab)
        self.address_encoder = BitNetAddressShapeEncoder(self.dims, config.feature_vocab)
        self.event_encoder = BitNetEventSemanticEncoder(self.dims, config.feature_vocab)
        self.map_encoder = BitNetMapStateEncoder(self.dims, config.feature_vocab)
        self.summary_encoder = BitNetSummaryEncoder(self.dims, config.feature_vocab)
        self.fusion = BitNetFusedObservationBlock(self.dims)
        
        # === Observability refinement (higher precision) ===
        # Keep this at higher precision for stable confidence/uncertainty
        self.observability = ObservabilityModule(self.dims)
        
        # === Plane B: Low-bit state-plane modules ===
        self.belief_tracker = BitNetBeliefStateTracker(self.dims, config.updates.belief_update_gate_bias)
        self.predictive_core = BitNetPredictiveBottleneck(self.dims)
        self.slow_memory = BitNetSlowStateMemory(self.dims)
        self.prototype_bank = BitNetPrototypeBank(self.dims, config.prototypes)
        self.probing_controller = BitNetProbingController(
            self.dims, 
            num_prototypes=config.prototypes.num_prototypes,
            num_experts=config.num_experts
        )
        self.abstention_controller = AbstentionController(self.dims)
        
        # Expert router (low-bit)
        self.expert_router = ExpertRouter(self.dims, config.routing)
        
        # Precision-preserving projection for action heads
        # This ensures action margins are not flattened by low-bit compression
        # Keep this at higher precision (FP16/FP32)
        self.action_projection = nn.Linear(self.dims.bottleneck, self.dims.bottleneck)
        with torch.no_grad():
            # Initialize as near-identity with slight gain to compensate for compression
            self.action_projection.weight.data.copy_(
                torch.eye(self.dims.bottleneck) * 1.1  # Slight gain
            )
            self.action_projection.bias.data.zero_()
        
        # Output heads (mixed precision)
        self.output_heads = BitNetOutputHeads(self.dims)
        
        # === Configuration for abstention (from original model) ===
        self.abstain_threshold = config.uncertainty.abstain_threshold
        self.min_confidence_for_override = config.uncertainty.min_confidence_for_override
        
        # BitNet-specific override threshold adjustment
        # BitNet with temperature sharpening produces larger margins
        # Need to increase threshold to avoid over-aggressive overrides
        self.override_threshold = 0.55  # Higher than original 0.48
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        state: Optional[BitNetModelState] = None,
        expert_context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> BitNetModelOutput:
        """Forward pass matching original ACMoEGAModel interface exactly."""
        batch_size = inputs['low8_bucket'].shape[0]
        device = inputs['low8_bucket'].device
        
        # Initialize state if needed
        if state is None:
            state = BitNetModelState.init(batch_size, self.dims, device)
        
        # === Plane A: Encode observations (low-bit encoders) ===
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
        
        # === Observability refinement (higher precision) ===
        z_obs, obs_confidence, obs_uncertainty = self.observability(
            z_obs0,
            inputs['missingness_mask'],
            inputs['freshness_ages'],
            inputs['source_quality'],
            inputs['conflict_score'],
            inputs['consistency_score'],
        )
        
        # === Plane B: State inference (low-bit) ===
        
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
        
        # Predictive bottleneck (low-bit)
        z_pred, pred_hidden = self.predictive_core(
            z_obs, belief, z_slow, pressure_summary, state.pred_hidden
        )
        
        # Prototype matching
        proto_sim, support_density, familiarity, drift_score, proto_match = self.prototype_bank(z_pred)
        
        # Compute uncertainty (higher precision)
        uncertainty = self.output_heads.uncertainty(z_pred, belief)
        
        # Probing controller (low-bit)
        probing_value, support_score, expert_eligibility, route_confidence = self.probing_controller(
            z_pred, z_slow, uncertainty, proto_sim,
            support_density.unsqueeze(-1).expand(-1, 8),
            self.expert_router.expert_gains.unsqueeze(0).expand(batch_size, -1),
            drift_score,
        )
        
        # === Expert routing (low-bit) ===
        z_enhanced, used_experts, routing_weights = self.expert_router(
            z_pred,
            expert_eligibility,
            support_score,
            probing_value,
            drift_score,
            obs_confidence,
            expert_context,
        )
        
        # Precision-preserving projection for action heads
        # This ensures action margins are not flattened by low-bit compression
        z_for_actions = self.action_projection(z_enhanced)
        
        # === Output heads (mixed precision) ===
        head_outputs = self.output_heads(
            z_for_actions, belief, z_slow, inputs['pressure_features']
        )
        
        # Get action margin
        action_margin = head_outputs['action_margin']
        
        # === Abstention decision (higher precision) ===
        abstain_prob, calibrated_confidence = self.abstention_controller(
            z_pred,
            uncertainty,
            support_density,
            action_margin,
            obs_confidence,
        )
        
        # Normalize action margin to [0, 1] range for comparison
        normalized_margin = torch.sigmoid(action_margin)
        
        # Quality-aware override decision (v1.2)
        # Override only if ALL conditions are met:
        # 1. Margin shows preference (margin > threshold1)
        # 2. Confidence is reasonable (confidence > threshold2)
        # 3. Drift is not too high (drift < threshold3)
        # 4. Support density is sufficient (support > threshold4)
        
        margin_threshold = 0.02  # Minimum margin for any preference
        confidence_threshold = 0.50  # Minimum confidence
        drift_threshold = 0.05  # Maximum drift
        support_threshold = 0.10  # Minimum support
        
        quality_gates = (
            (normalized_margin > margin_threshold).float() +
            (calibrated_confidence > confidence_threshold).float() +
            (drift_score < drift_threshold).float() +
            (support_density > support_threshold).float()
        )
        
        # Override if 3+ quality gates pass (out of 4)
        should_override = quality_gates >= 3
        
        # Determine abstention with balanced criteria
        concern_count = (
            (abstain_prob > self.abstain_threshold).float() +
            (calibrated_confidence < self.min_confidence_for_override).float() +
            (normalized_margin < 0.4).float() +
            (support_density < 0.15).float()
        )
        
        should_abstain = concern_count >= 2
        
        # Update prototype support
        self.prototype_bank.update_support(proto_match, proto_sim)
        
        # Build new state
        new_state = BitNetModelState(
            belief_page=belief,
            belief_region=state.belief_region,
            belief_process=state.belief_process,
            slow_state=z_slow,
            pred_hidden=pred_hidden,
        )
        
        return BitNetModelOutput(
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
        """Get parameter count."""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB (BitNet version).
        
        BitNet-style ternary weights use ~1.5 bits per parameter.
        """
        total_params = self.get_parameter_count()
        
        # Estimate size with ternary compression
        # Ternary: ~1.5 bits per param, plus scaling factors
        ternary_ratio = 0.7  # 70% of weights are ternary
        full_precision_ratio = 1.0 - ternary_ratio
        
        # Full precision: 4 bytes per param
        # Ternary: ~1.5 bits per param = 0.1875 bytes per param
        size_mb = (total_params * (ternary_ratio * 0.1875 + full_precision_ratio * 4)) / (1024 * 1024)
        
        return size_mb
    
    def enable_ternary_inference(self):
        """Enable ternary inference mode."""
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.use_ternary = True
    
    def disable_ternary_inference(self):
        """Disable ternary inference mode."""
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.use_ternary = False
    
    def get_margin_scale(self) -> float:
        """Get the margin scale for this model (for threshold recalibration)."""
        return 1.0  # Default scale
    
    def get_adjusted_override_threshold(self, original_threshold: float = 0.48) -> float:
        """Get the override threshold adjusted for BitNet's margin distribution."""
        # BitNet with action projection has higher margins
        # Increase threshold to avoid over-aggressive overrides
        return original_threshold + 0.1  # Adjust for higher margins
    
    def get_override_threshold(self) -> float:
        """Get the override threshold for this model."""
        return self.override_threshold


# Low-bit encoder implementations

class BitNetBytePlaneEncoder(nn.Module):
    """Low-bit byte plane encoder matching original interface."""
    
    def __init__(self, dims: LatentDimensions, feature_vocab: Optional[Dict] = None):
        super().__init__()
        self.dims = dims
        self.feature_vocab = feature_vocab or FeatureVocabConfig()
        
        # Embeddings for categorical features (matching original dimensions)
        self.low8_embed = nn.Embedding(self.feature_vocab.page_bucket_vocab, 8)
        self.high8_embed = nn.Embedding(self.feature_vocab.page_bucket_vocab, 8)
        self.alignment_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        self.small_int_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        self.delta_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 6)
        self.hamming_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        
        # Low-bit projections for continuous features
        self.continuous_proj = TernaryLinear(6, 8)
        self.bitfield_proj = TernaryLinear(16, 8)
        self.sketch_proj = TernaryLinear(32, 12)
        
        # Fusion - use GatedFusion like original
        total_embed = 8 + 8 + 4 + 4 + 6 + 4 + 8 + 8 + 12
        self.fusion = GatedFusion(total_embed, dims.byte_encoder)
        self.norm = nn.LayerNorm(dims.byte_encoder)
    
    def forward(self, low8, high8, alignment, small_int, delta, hamming,
                continuous, bitfield, sketch) -> torch.Tensor:
        """Forward pass matching original interface."""
        # Convert bucket indices to long if needed (embeddings require int)
        low8 = low8.long()
        high8 = high8.long()
        alignment = alignment.long()
        small_int = small_int.long()
        delta = delta.long()
        hamming = hamming.long()
        
        e_low8 = self.low8_embed(low8)
        e_high8 = self.high8_embed(high8)
        e_align = self.alignment_embed(alignment)
        e_small = self.small_int_embed(small_int)
        e_delta = self.delta_embed(delta)
        e_hamming = self.hamming_embed(hamming)
        
        p_cont = self.continuous_proj(continuous)
        p_bits = self.bitfield_proj(bitfield)
        p_sketch = self.sketch_proj(sketch)
        
        combined = torch.cat([
            e_low8, e_high8, e_align, e_small, e_delta, e_hamming,
            p_cont, p_bits, p_sketch
        ], dim=-1)
        
        z_byte = self.fusion(combined)
        return self.norm(z_byte)


class BitNetAddressShapeEncoder(nn.Module):
    """Low-bit address shape encoder matching original interface."""
    
    def __init__(self, dims: LatentDimensions, feature_vocab: Optional[Dict] = None):
        super().__init__()
        self.dims = dims
        self.feature_vocab = feature_vocab or FeatureVocabConfig()
        
        # Bucket embeddings - use feature vocab sizes (matching original)
        self.page_hash_embed = nn.Embedding(self.feature_vocab.page_bucket_vocab, 8)
        self.offset_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        self.cache_line_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        self.addr_alignment_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 3)
        self.stride_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 6)
        self.reuse_dist_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 5)
        self.locality_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 6)
        self.entropy_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 3)
        
        # Low-bit projection for address flags
        self.flags_proj = TernaryLinear(5, 5)
        
        # Fusion - use GatedFusion like original
        total_embed = 8 + 4 + 4 + 3 + 6 + 5 + 6 + 3 + 5
        self.fusion = GatedFusion(total_embed, dims.address_encoder)
        self.norm = nn.LayerNorm(dims.address_encoder)
    
    def forward(self, page_hash, offset, cache_line, addr_alignment,
                stride, reuse_dist, locality_cluster, entropy, address_flags) -> torch.Tensor:
        # Convert bucket indices to long
        page_hash = page_hash.long()
        offset = offset.long()
        cache_line = cache_line.long()
        addr_alignment = addr_alignment.long()
        stride = stride.long()
        reuse_dist = reuse_dist.long()
        locality_cluster = locality_cluster.long()
        entropy = entropy.long()
        
        e_ph = self.page_hash_embed(page_hash)
        e_off = self.offset_embed(offset)
        e_cl = self.cache_line_embed(cache_line)
        e_aa = self.addr_alignment_embed(addr_alignment)
        e_str = self.stride_embed(stride)
        e_rd = self.reuse_dist_embed(reuse_dist)
        e_lc = self.locality_embed(locality_cluster)
        e_ent = self.entropy_embed(entropy)
        
        p_flags = self.flags_proj(address_flags)
        
        combined = torch.cat([
            e_ph, e_off, e_cl, e_aa, e_str, e_rd, e_lc, e_ent, p_flags
        ], dim=-1)
        
        z_addr = self.fusion(combined)
        return self.norm(z_addr)


class BitNetEventSemanticEncoder(nn.Module):
    """Low-bit event semantic encoder matching original interface."""
    
    def __init__(self, dims: LatentDimensions, feature_vocab: Optional[Dict] = None):
        super().__init__()
        self.dims = dims
        self.feature_vocab = feature_vocab or FeatureVocabConfig()
        
        # Embeddings for categorical features (matching original vocab sizes)
        self.event_type_embed = nn.Embedding(self.feature_vocab.event_type_vocab, 16)
        self.fault_class_embed = nn.Embedding(self.feature_vocab.fault_code_vocab, 16)
        self.syscall_class_embed = nn.Embedding(self.feature_vocab.syscall_class_vocab, 16)
        self.opcode_family_embed = nn.Embedding(self.feature_vocab.opcode_class_vocab, 16)
        self.transition_type_embed = nn.Embedding(self.feature_vocab.event_type_vocab, 16)
        self.result_class_embed = nn.Embedding(4, 16)  # 0=UNK, 1..3 valid
        
        # Fusion
        self.fusion = TernaryLinear(96, dims.event_encoder)
        self.norm = nn.LayerNorm(dims.event_encoder)
    
    def forward(self, event_type, fault_class, syscall_class, opcode_family,
                transition_type, result_class) -> torch.Tensor:
        # Convert bucket indices to long
        event_type = event_type.long()
        fault_class = fault_class.long()
        syscall_class = syscall_class.long()
        opcode_family = opcode_family.long()
        transition_type = transition_type.long()
        result_class = result_class.long()
        
        e_et = self.event_type_embed(event_type)
        e_fc = self.fault_class_embed(fault_class)
        e_sc = self.syscall_class_embed(syscall_class)
        e_of = self.opcode_family_embed(opcode_family)
        e_tt = self.transition_type_embed(transition_type)
        e_rc = self.result_class_embed(result_class)
        
        combined = torch.cat([e_et, e_fc, e_sc, e_of, e_tt, e_rc], dim=-1)
        
        z_evt = self.fusion(combined)
        return self.norm(z_evt)


class BitNetMapStateEncoder(nn.Module):
    """Low-bit map state encoder matching original interface."""
    
    def __init__(self, dims: LatentDimensions, feature_vocab: Optional[Dict] = None):
        super().__init__()
        self.dims = dims
        self.feature_vocab = feature_vocab or FeatureVocabConfig()
        
        # Embeddings for categorical features (matching original)
        self.vma_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        self.protection_embed = nn.Embedding(self.feature_vocab.mode_vocab, 3)
        self.pte_proj = TernaryLinear(11, 8)
        
        # Fusion
        total_embed = 4 + 3 + 8
        self.fusion = GatedFusion(total_embed, dims.map_encoder)
        self.norm = nn.LayerNorm(dims.map_encoder)
    
    def forward(self, pte_flags, vma_class, protection_domain) -> torch.Tensor:
        # Convert bucket indices to long
        vma_class = vma_class.long()
        protection_domain = protection_domain.long()
        
        e_vma = self.vma_embed(vma_class)
        e_prot = self.protection_embed(protection_domain)
        p_pte = self.pte_proj(pte_flags.float())
        
        combined = torch.cat([e_vma, e_prot, p_pte], dim=-1)
        
        z_map = self.fusion(combined)
        return self.norm(z_map)


class BitNetSummaryEncoder(nn.Module):
    """Low-bit summary encoder matching original interface."""
    
    def __init__(self, dims: LatentDimensions, feature_vocab: Optional[Dict] = None):
        super().__init__()
        self.dims = dims
        self.feature_vocab = feature_vocab or FeatureVocabConfig()
        
        # Count buckets - use feature vocab sizes (matching original)
        self.count_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 5)
        
        # Continuous pressure features - deeper projection
        self.pressure_proj = nn.Sequential(
            TernaryLinear(12, 24),
            nn.GELU(),
            TernaryLinear(24, 20),
        )
        
        # Recency and volatility
        self.recency_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 5)
        self.volatility_proj = nn.Sequential(
            TernaryLinear(4, 8),
            nn.GELU(),
            TernaryLinear(8, 6),
        )
        
        total_embed = 5 * 4 + 20 + 5 + 6  # 51
        self.fusion = GatedFusion(total_embed, dims.summary_encoder)
        self.norm = nn.LayerNorm(dims.summary_encoder)
    
    def forward(self, read_count, write_count, fault_count, cow_count,
                recency, volatility, pressure) -> torch.Tensor:
        # Convert bucket indices to long
        read_count = read_count.long()
        write_count = write_count.long()
        fault_count = fault_count.long()
        cow_count = cow_count.long()
        recency = recency.long()
        
        e_rc = self.count_embed(read_count)
        e_wc = self.count_embed(write_count)
        e_fc = self.count_embed(fault_count)
        e_cc = self.count_embed(cow_count)
        e_r = self.recency_embed(recency)
        
        p_vol = self.volatility_proj(volatility)
        p_pres = self.pressure_proj(pressure)
        
        combined = torch.cat([e_rc, e_wc, e_fc, e_cc, e_r, p_vol, p_pres], dim=-1)
        
        z_sum = self.fusion(combined)
        return self.norm(z_sum)


class BitNetFusedObservationBlock(nn.Module):
    """Fused observation block matching original interface."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        self.dims = dims
        
        input_dim = (dims.byte_encoder + dims.address_encoder + 
                     dims.event_encoder + dims.map_encoder + dims.summary_encoder)
        
        # Deeper gated fusion (matching original)
        self.gate = nn.Sequential(
            TernaryLinear(input_dim, dims.fused_observation),
            nn.GELU(),
            TernaryLinear(dims.fused_observation, dims.fused_observation),
        )
        self.transform = nn.Sequential(
            TernaryLinear(input_dim, dims.fused_observation),
            nn.GELU(),
            TernaryLinear(dims.fused_observation, dims.fused_observation),
        )
        
        # Residual MLP (matching original)
        self.mlp = nn.Sequential(
            nn.Linear(dims.fused_observation, dims.fused_observation * 2),
            nn.GELU(),
            nn.Linear(dims.fused_observation * 2, dims.fused_observation),
        )
        
        self.norm1 = nn.LayerNorm(dims.fused_observation)
        self.norm2 = nn.LayerNorm(dims.fused_observation)
    
    def forward(self, z_byte, z_addr, z_evt, z_map, z_sum) -> torch.Tensor:
        """Forward pass matching original interface."""
        combined = torch.cat([z_byte, z_addr, z_evt, z_map, z_sum], dim=-1)
        
        gate = torch.sigmoid(self.gate(combined) * 1.2)
        fused = gate * self.transform(combined)
        fused = self.norm1(fused)
        
        z_obs0 = fused + self.mlp(fused)
        return self.norm2(z_obs0)


class BitNetBeliefStateTracker(BeliefStateTracker):
    """Belief state tracker with low-bit projections."""
    
    def __init__(self, dims: LatentDimensions, gate_bias: float = 0.15):
        super().__init__(dims, gate_bias)
        
        # Replace linear layers with ternary versions
        self.belief_proj = TernaryLinear(dims.fused_observation, dims.belief_hidden)


class BitNetPredictiveBottleneck(PredictiveBottleneck):
    """Predictive bottleneck with low-bit dense core."""
    
    def __init__(self, dims: LatentDimensions):
        super().__init__(dims)
        
        # Low-bit MLP for bottleneck
        self.bottleneck_mlp = nn.Sequential(
            TernaryLinear(dims.fused_observation + dims.belief_hidden, dims.predictive_hidden),
            nn.GELU(),
            TernaryLinear(dims.predictive_hidden, dims.bottleneck),
        )
        self.norm = nn.LayerNorm(dims.bottleneck)


class BitNetSlowStateMemory(SlowStateMemory):
    """Slow state memory with low-bit updates."""
    
    def __init__(self, dims: LatentDimensions, decay: float = 0.985):
        super().__init__(dims, decay)
        
        # Low-bit projection
        self.update_proj = TernaryLinear(dims.bottleneck, dims.slow_state)


class BitNetPrototypeBank(PrototypeBank):
    """Prototype bank with efficient similarity computation."""
    
    def __init__(self, dims: LatentDimensions, config):
        super().__init__(dims, config)
        
        # Low-bit prototype projections
        self.proto_proj = TernaryLinear(dims.bottleneck, dims.prototype)


class BitNetProbingController(ProbingController):
    """Probing controller matching original interface."""
    
    def __init__(self, dims: LatentDimensions, num_prototypes: int = 96, num_experts: int = 8):
        super().__init__(dims, num_prototypes, num_experts)
        
        # Input dimensions (matching original)
        input_dim = dims.bottleneck + dims.slow_state + 5 + num_prototypes + num_experts + num_experts + 1
        
        # Deeper network for better decisions (matching original)
        self.net = nn.Sequential(
            TernaryLinear(input_dim, 96),
            nn.GELU(),
            TernaryLinear(96, 64),
            nn.GELU(),
            TernaryLinear(64, 48),
            nn.GELU(),
        )
        
        # Separate heads with sharper outputs (matching original)
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


class BitNetOutputHeads(OutputHeads):
    """Output heads matching original interface with mixed precision.
    
    Key design: Dense core uses low-bit, action heads use higher precision.
    This preserves action margin sharpness while keeping model size small.
    """
    
    def __init__(self, dims: LatentDimensions):
        super().__init__(dims)
        
        # State inference heads (matching original)
        self.page_state = BitNetPageStateHead(dims)
        self.region_state = BitNetRegionStateHead(dims)
        self.process_phase = BitNetProcessPhaseHead(dims)
        self.hazard = BitNetHazardHead(dims)
        self.uncertainty = BitNetUncertaintyHead(dims, precision=16)
        
        # Action heads with margin - use higher precision for margin preservation
        # use_ternary=False keeps these at standard precision
        self.batch_scheduler = BitNetBatchSchedulerHead(dims, use_ternary=False)
        self.kv_policy = BitNetKVPolicyHead(dims, use_ternary=False)
        self.numa_placement = BitNetNUMAPlacementHead(dims, use_ternary=False)
        self.boundary_control = BitNetBoundaryControlHead(dims, use_ternary=False)
        self.page_policy = BitNetPagePolicyHead(dims, use_ternary=False)


class BitNetPagePolicyHead(nn.Module):
    """Page policy head matching original interface."""
    
    def __init__(self, dims: LatentDimensions, use_ternary: bool = True):
        super().__init__()
        self.dims = dims
        self.use_ternary = use_ternary
        
        from .types import PageAction
        input_dim = dims.bottleneck + dims.belief_hidden + 8
        self.head = RankingActionHead(input_dim, len(PageAction))
        
        # Temperature sharpening for BitNet
        self.logit_temperature = nn.Parameter(torch.tensor(0.6))
    
    def forward(self, z_pred: torch.Tensor, belief: torch.Tensor,
                page_pressure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_pred, belief, page_pressure[:, :8]], dim=-1)
        logits, margin = self.head(x)
        
        # Apply temperature sharpening
        sharpened_logits = logits / self.logit_temperature
        
        # Recompute margin with sharpened logits
        sorted_logits, _ = sharpened_logits.sort(dim=-1, descending=True)
        sharpened_margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        return sharpened_logits, sharpened_margin
    
    def to_scores(self, logits: torch.Tensor, idx: int = 0) -> Dict:
        """Convert logits to action scores."""
        from .types import PageAction
        import torch.nn.functional as F
        probs = F.softmax(logits[idx], dim=-1)
        return {action: probs[i].item() for i, action in enumerate(PageAction)}


class BitNetBatchSchedulerHead(nn.Module):
    """Batch scheduler head matching original interface."""
    
    def __init__(self, dims: LatentDimensions, use_ternary: bool = True):
        super().__init__()
        self.dims = dims
        self.use_ternary = use_ternary
        
        from .types import BatchAction
        input_dim = dims.bottleneck + dims.slow_state + 8
        self.head = RankingActionHead(input_dim, len(BatchAction))
        
        # Temperature sharpening for BitNet to restore logit separation
        # τ < 1 sharpens the distribution, increasing margin separation
        self.logit_temperature = nn.Parameter(torch.tensor(0.6))
    
    def forward(self, z_pred: torch.Tensor, z_slow: torch.Tensor,
                pressure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_pred, z_slow, pressure[:, :8]], dim=-1)
        logits, margin = self.head(x)
        
        # Apply temperature sharpening to restore logit separation
        sharpened_logits = logits / self.logit_temperature
        
        # Recompute margin with sharpened logits
        sorted_logits, _ = sharpened_logits.sort(dim=-1, descending=True)
        sharpened_margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        return sharpened_logits, sharpened_margin
    
    def to_scores(self, logits: torch.Tensor, idx: int = 0) -> Dict:
        """Convert logits to action scores."""
        from .types import BatchAction
        import torch.nn.functional as F
        probs = F.softmax(logits[idx], dim=-1)
        return {action: probs[i].item() for i, action in enumerate(BatchAction)}


class BitNetKVPolicyHead(nn.Module):
    """KV policy head matching original interface."""
    
    def __init__(self, dims: LatentDimensions, use_ternary: bool = True):
        super().__init__()
        self.dims = dims
        self.use_ternary = use_ternary
        
        from .types import KVAction
        input_dim = dims.bottleneck + dims.slow_state + 8
        self.head = RankingActionHead(input_dim, len(KVAction))
        
        # Temperature sharpening for BitNet
        self.logit_temperature = nn.Parameter(torch.tensor(0.6))
    
    def forward(self, z_pred: torch.Tensor, z_slow: torch.Tensor,
                kv_pressure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_pred, z_slow, kv_pressure[:, :8]], dim=-1)
        logits, margin = self.head(x)
        
        # Apply temperature sharpening
        sharpened_logits = logits / self.logit_temperature
        
        # Recompute margin with sharpened logits
        sorted_logits, _ = sharpened_logits.sort(dim=-1, descending=True)
        sharpened_margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        return sharpened_logits, sharpened_margin
    
    def to_scores(self, logits: torch.Tensor, idx: int = 0) -> Dict:
        """Convert logits to action scores."""
        from .types import KVAction
        import torch.nn.functional as F
        probs = F.softmax(logits[idx], dim=-1)
        return {action: probs[i].item() for i, action in enumerate(KVAction)}


class BitNetNUMAPlacementHead(nn.Module):
    """NUMA placement head matching original interface."""
    
    def __init__(self, dims: LatentDimensions, use_ternary: bool = True):
        super().__init__()
        self.dims = dims
        self.use_ternary = use_ternary
        
        from .types import NUMAAction
        input_dim = dims.bottleneck + dims.slow_state + 8
        self.head = RankingActionHead(input_dim, len(NUMAAction))
        
        # Temperature sharpening for BitNet
        self.logit_temperature = nn.Parameter(torch.tensor(0.6))
    
    def forward(self, z_pred: torch.Tensor, z_slow: torch.Tensor,
                numa_pressure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_pred, z_slow, numa_pressure[:, :8]], dim=-1)
        logits, margin = self.head(x)
        
        # Apply temperature sharpening
        sharpened_logits = logits / self.logit_temperature
        
        # Recompute margin with sharpened logits
        sorted_logits, _ = sharpened_logits.sort(dim=-1, descending=True)
        sharpened_margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        return sharpened_logits, sharpened_margin
    
    def to_scores(self, logits: torch.Tensor, idx: int = 0) -> Dict:
        """Convert logits to action scores."""
        from .types import NUMAAction
        import torch.nn.functional as F
        probs = F.softmax(logits[idx], dim=-1)
        return {action: probs[i].item() for i, action in enumerate(NUMAAction)}


class BitNetBoundaryControlHead(nn.Module):
    """Boundary control head matching original interface."""
    
    def __init__(self, dims: LatentDimensions, use_ternary: bool = True):
        super().__init__()
        self.dims = dims
        self.use_ternary = use_ternary
        
        from .types import BoundaryAction
        input_dim = dims.bottleneck + dims.belief_hidden + 8
        self.head = RankingActionHead(input_dim, len(BoundaryAction))
        
        # Temperature sharpening for BitNet
        self.logit_temperature = nn.Parameter(torch.tensor(0.6))
    
    def forward(self, z_pred: torch.Tensor, belief: torch.Tensor,
                boundary_pressure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_pred, belief, boundary_pressure[:, :8]], dim=-1)
        logits, margin = self.head(x)
        
        # Apply temperature sharpening
        sharpened_logits = logits / self.logit_temperature
        
        # Recompute margin with sharpened logits
        sorted_logits, _ = sharpened_logits.sort(dim=-1, descending=True)
        sharpened_margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        return sharpened_logits, sharpened_margin
    
    def to_scores(self, logits: torch.Tensor, idx: int = 0) -> Dict:
        """Convert logits to action scores."""
        from .types import BoundaryAction
        import torch.nn.functional as F
        probs = F.softmax(logits[idx], dim=-1)
        return {action: probs[i].item() for i, action in enumerate(BoundaryAction)}


class BitNetUncertaintyHead(nn.Module):
    """Uncertainty head matching original interface."""
    
    def __init__(self, dims: LatentDimensions, precision: int = 16):
        super().__init__()
        self.dims = dims
        self.precision = precision
        
        input_dim = dims.bottleneck + dims.belief_hidden
        
        # Shared features (matching original)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.GELU(),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        
        # Individual uncertainty components (matching original)
        self.calibration = nn.Linear(32, 1)
        self.selective = nn.Linear(32, 1)
        self.ranking = nn.Linear(32, 1)
        self.ood = nn.Linear(32, 1)
        self.observability = nn.Linear(32, 1)
        
        # Learnable temperatures for calibration
        self.temps = nn.Parameter(torch.ones(5) * 1.3)
    
    def forward(self, z_pred: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_pred, belief], dim=-1)
        x = self.shared(x)
        
        calibration = self.calibration(x)
        selective = self.selective(x)
        ranking = self.ranking(x)
        ood = self.ood(x)
        observability = self.observability(x)
        
        uncertainty = torch.cat([calibration, selective, ranking, ood, observability], dim=-1)
        
        return uncertainty
    
    def to_uncertainty_vector(self, output: torch.Tensor, idx: int = 0) -> 'UncertaintyVector':
        """Convert head outputs to UncertaintyVector."""
        from .types import UncertaintyVector
        return UncertaintyVector(
            calibration=output[idx, 0].item(),
            selective_prediction=output[idx, 1].item(),
            ranking=output[idx, 2].item(),
            ood=output[idx, 3].item(),
            observability=output[idx, 4].item(),
        )


class BitNetCalibrationHead(nn.Module):
    """Calibration head with higher precision."""
    
    def __init__(self, dims: LatentDimensions, precision: int = 16):
        super().__init__()
        self.dims = dims
        self.precision = precision
        
        # Higher precision for calibration
        self.proj = nn.Linear(dims.bottleneck, 16)
        self.output_proj = nn.Linear(16, 1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z)
        x = F.relu(x)
        calibration = self.output_proj(x)
        return calibration
