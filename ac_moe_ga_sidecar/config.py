"""
Configuration classes for AC-MoE-GA Systems Sidecar v1.1.

Balanced-tiny profile with improved capacity distribution for:
- Stronger support/prototype formation
- Better policy head separation
- Calibrated abstention behavior
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum, auto


class ExpertType(Enum):
    """Typed expert categories for specialized processing."""
    PAGE_TRANSITION = auto()
    COW_FORK = auto()
    RECLAIM_HOTNESS = auto()
    LOCALITY_PATTERN = auto()
    FAULT_BURST = auto()
    BOUNDARY_CONTROL = auto()
    KV_POLICY = auto()
    NUMA_PLACEMENT = auto()


class EventType(Enum):
    """Supported event classes for micro-event processing."""
    MEMORY_READ = auto()
    MEMORY_WRITE = auto()
    PAGE_FAULT = auto()
    COW_FAULT = auto()
    TRAP = auto()
    SYSCALL_ENTRY = auto()
    SYSCALL_EXIT = auto()
    KERNEL_ENTRY = auto()
    KERNEL_EXIT = auto()
    RECLAIM = auto()
    PROMOTION = auto()
    DEMOTION = auto()
    MIGRATION = auto()
    QUEUE = auto()
    CACHE = auto()
    KV_POLICY = auto()


@dataclass
class FeatureVocabConfig:
    """Vocabulary sizes for categorical features used by embeddings.
    
    All categorical features should map to indices in [0, vocab_size-1].
    Index 0 is reserved for UNK/unknown/invalid values.
    Valid values map to indices 1..vocab_size-1.
    """
    event_type_vocab: int = 17      # 0=UNK, 1..16 valid (16 event types)
    opcode_class_vocab: int = 33    # 0=UNK, 1..32 valid
    fault_code_vocab: int = 33      # 0=UNK, 1..32 valid
    syscall_class_vocab: int = 65   # 0=UNK, 1..64 valid
    mode_vocab: int = 5             # 0=UNK, 1..4 valid
    rw_flag_vocab: int = 4          # 0=UNK, 1..3 valid
    pte_flags_vocab: int = 256      # 0=UNK, 1..255 valid
    page_bucket_vocab: int = 4097   # 0=UNK, 1..4096 valid
    region_bucket_vocab: int = 1025 # 0=UNK, 1..1024 valid
    cpu_id_vocab: int = 129         # 0=UNK, 1..128 valid
    numa_node_vocab: int = 17       # 0=UNK, 1..16 valid
    trap_code_vocab: int = 65       # 0=UNK, 1..64 valid


@dataclass
class LatentDimensions:
    """
    Latent space dimensions for balanced-tiny v1.1.
    
    Increased capacity in:
    - Summary encoder (24 -> 32): Better pressure/context encoding
    - Belief hidden (32 -> 48): Stronger regime tracking
    - Predictive bottleneck (24 -> 32): More expressive core
    - Prototype width (16 -> 24): Better regime separation
    """
    fused_observation: int = 80  # Up from 64
    belief_hidden: int = 48      # Up from 32 - key for regime tracking
    predictive_hidden: int = 80  # Up from 64
    bottleneck: int = 32         # Up from 24 - core expressivity
    slow_state: int = 32         # Up from 24
    prototype: int = 24          # Up from 16 - better regime separation
    uncertainty: int = 8
    expert_residual: int = 32    # Up from 24
    
    # Encoder output dimensions - summary encoder enlarged
    byte_encoder: int = 24
    address_encoder: int = 20    # Slight increase
    event_encoder: int = 16      # Slight increase
    map_encoder: int = 12
    summary_encoder: int = 32    # Up from 24 - key for pressure encoding


@dataclass
class RuntimeMemoryBudget:
    """Memory budget allocation for runtime state."""
    deterministic_state_mb: float = 12.0
    active_latent_cache_mb: float = 6.0   # Increased for better regime tracking
    prototype_tables_mb: float = 1.0      # Increased for better support
    
    @property
    def total_mb(self) -> float:
        return (self.deterministic_state_mb + 
                self.active_latent_cache_mb + 
                self.prototype_tables_mb)


@dataclass
class RoutingConfig:
    """Expert routing configuration."""
    max_experts_per_inference: int = 2
    default_top_k: int = 1
    min_support_count: int = 50          # Lowered to allow earlier specialization
    min_probing_value: float = 0.25      # Lowered
    max_drift_penalty: float = 0.6       # Relaxed
    min_observability: float = 0.35      # Relaxed
    expert_gain_threshold: float = 0.005 # More sensitive


@dataclass
class InferenceCadence:
    """Controls when learned inference runs."""
    min_events_between_inference: int = 8
    max_events_between_inference: int = 64
    fault_burst_trigger: bool = True
    cow_fault_trigger: bool = True
    syscall_burst_trigger: bool = True
    pressure_change_trigger: bool = True
    batch_decision_trigger: bool = True


@dataclass
class UpdateConfig:
    """Configuration for runtime updates."""
    # Per-event update settings
    ema_decay_fast: float = 0.92         # Slightly faster adaptation
    ema_decay_slow: float = 0.985        # Slightly faster
    counter_saturation: int = 65535
    
    # Belief state update
    belief_update_gate_bias: float = 0.15  # Stronger updates
    min_freshness_for_update: float = 0.25
    
    # Deferred adaptation interval (in events)
    adaptation_interval: int = 5000       # More frequent
    
    # Slow background learning interval (in events)
    background_learning_interval: int = 50000


@dataclass
class UncertaintyThresholds:
    """
    Thresholds for uncertainty-based decisions.
    
    Calibrated for realistic abstention behavior (v1.1 tuned).
    v1.4: More aggressive abstention to achieve 5-20% target.
    """
    abstain_threshold: float = 0.20       # Lowered from 0.25 - more aggressive abstention for untrained models
    override_max_uncertainty: float = 0.55
    ood_threshold: float = 0.65           # Higher - don't flag OOD too easily
    ranking_stability_min: float = 0.3    # Lower - allow some ranking uncertainty
    calibration_warning: float = 0.4
    
    # Margin requirements for override (v1.1 tuned)
    min_action_margin: float = 0.05       # Lower - allow smaller margins initially
    min_confidence_for_override: float = 0.40  # Lower - allow override with moderate confidence
    
    # Support density thresholds
    min_support_for_action: float = 0.05  # Minimum support to take action
    min_support_for_override: float = 0.10  # Higher bar for override
    
    # Override gate threshold (v1.4.1: support-dependent threshold)
    override_base_threshold: float = 0.525  # Base threshold for override gate
    override_threshold_slope: float = 0.04  # Slope for support-dependent threshold


@dataclass
class PrototypeConfig:
    """
    Configuration for prototype bank.
    
    Enhanced for better support density formation.
    """
    num_prototypes: int = 96              # Up from 64
    similarity_threshold: float = 0.65    # Slightly lower for more matches
    support_decay: float = 0.995          # Slower decay
    min_support_for_match: int = 30       # Lowered
    drift_window_size: int = 2000         # Larger window
    
    # New: prototype consolidation
    consolidation_threshold: float = 0.85  # Merge similar prototypes
    min_prototype_usage: int = 10          # Prune unused prototypes


@dataclass
class TrainingConfig:
    """
    Training configuration with enhanced objectives.
    
    Added ranking-margin loss and calibration improvements.
    """
    sequence_length: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss weights - rebalanced for v1.1
    predictive_weight: float = 1.0
    medium_horizon_weight: float = 0.6    # Increased
    policy_outcome_weight: float = 0.5    # Increased
    belief_consistency_weight: float = 0.15
    routing_sparsity_weight: float = 0.08
    uncertainty_calibration_weight: float = 0.35  # Increased significantly
    distillation_weight: float = 0.1
    
    # New v1.1 loss weights
    ranking_margin_weight: float = 0.6    # Increased from 0.4 - stronger action separation
    abstention_weight: float = 0.3        # New: abstention calibration
    support_density_weight: float = 0.2   # New: prototype formation
    wrong_high_conf_penalty: float = 0.5  # New: calibration penalty
    
    # Replay emphasis
    hard_case_replay_factor: float = 4.0  # Increased
    
    # Margin loss parameters
    ranking_margin_target: float = 0.25   # Increased from 0.15 - target margin between actions


@dataclass
class SidecarConfig:
    """Main configuration container for the sidecar v1.4.1."""
    latent_dims: LatentDimensions = field(default_factory=LatentDimensions)
    feature_vocab: FeatureVocabConfig = field(default_factory=FeatureVocabConfig)
    memory_budget: RuntimeMemoryBudget = field(default_factory=RuntimeMemoryBudget)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    cadence: InferenceCadence = field(default_factory=InferenceCadence)
    updates: UpdateConfig = field(default_factory=UpdateConfig)
    uncertainty: UncertaintyThresholds = field(default_factory=UncertaintyThresholds)
    prototypes: PrototypeConfig = field(default_factory=PrototypeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Entity limits
    max_active_pages: int = 100000
    max_active_regions: int = 10000
    max_active_processes: int = 1000
    
    # Expert configuration
    num_experts: int = 8
    expert_types: List[ExpertType] = field(
        default_factory=lambda: list(ExpertType)
    )
    
    # Version
    version: str = "1.4.1"
    
    def validate(self) -> bool:
        """Validate configuration consistency."""
        assert self.latent_dims.fused_observation > 0
        assert self.memory_budget.total_mb <= 30.0
        assert self.routing.max_experts_per_inference <= self.num_experts
        assert 0.0 <= self.uncertainty.abstain_threshold <= 1.0
        assert self.uncertainty.min_action_margin > 0
        return True


def BalancedBuildConfig() -> SidecarConfig:
    """Factory for the balanced-tiny v1.1 configuration."""
    return SidecarConfig()


# Legacy v1.0 config for comparison
def LegacyV10Config() -> SidecarConfig:
    """Factory for v1.0 configuration (for comparison)."""
    config = SidecarConfig()
    config.version = "1.0"
    config.latent_dims = LatentDimensions(
        fused_observation=64,
        belief_hidden=32,
        predictive_hidden=64,
        bottleneck=24,
        slow_state=24,
        prototype=16,
        uncertainty=8,
        expert_residual=24,
        byte_encoder=24,
        address_encoder=16,
        event_encoder=12,
        map_encoder=12,
        summary_encoder=24,
    )
    config.prototypes.num_prototypes = 64
    config.uncertainty.abstain_threshold = 0.7
    return config
