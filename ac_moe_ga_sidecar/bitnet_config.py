"""
BitNet-inspired configuration for AC-MoE-GA Sidecar.

This config implements native low-bit design principles from BitNet:
- Dense core uses ternary/low-bit operations
- Repetitive compute paths optimized for CPU/edge
- Fragile decision logic stays higher precision
- Shared projections to reduce memory traffic
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .config import (
    LatentDimensions, SidecarConfig, FeatureVocabConfig, RuntimeMemoryBudget,
    RoutingConfig, InferenceCadence, UpdateConfig, UncertaintyThresholds,
    PrototypeConfig, TrainingConfig, ExpertType
)


@dataclass
class BitNetLatentDimensions(LatentDimensions):
    """BitNet-optimized latent dimensions.
    
    Key changes from standard dimensions:
    - Same dimensions as standard for interface parity
    - Dense core uses ternary/low-bit operations
    - Shared projections reduce memory traffic
    """
    fused_observation: int = 80   # Same as standard
    belief_hidden: int = 48       # Same as standard
    predictive_hidden: int = 80   # Same as standard
    bottleneck: int = 32          # Same as standard
    slow_state: int = 32          # Same as standard
    prototype: int = 24           # Same as standard
    uncertainty: int = 8          # Same as standard
    expert_residual: int = 32     # Same as standard
    
    # Encoder output dimensions - same as standard
    byte_encoder: int = 24
    address_encoder: int = 20
    event_encoder: int = 16
    map_encoder: int = 12
    summary_encoder: int = 32


@dataclass
class BitNetTrainingConfig:
    """Training configuration for BitNet-style low-bit model."""
    # Quantization-aware training
    quantization_aware_training: bool = True
    freeze_scaling_factors: bool = False
    
    # Low-bit specific settings
    ternary_sparsity_target: float = 0.3  # Target 30% sparsity in ternary weights
    quantization_noise: float = 0.01      # Add noise during training
    
    # Loss weights - adjusted for low-bit
    predictive_weight: float = 1.0
    medium_horizon_weight: float = 0.6
    policy_outcome_weight: float = 0.5
    belief_consistency_weight: float = 0.15
    routing_sparsity_weight: float = 0.08
    uncertainty_calibration_weight: float = 0.35
    distillation_weight: float = 0.1
    
    # New BitNet-style loss weights
    ranking_margin_weight: float = 0.6
    abstention_weight: float = 0.3
    support_density_weight: float = 0.2
    wrong_high_conf_penalty: float = 0.5
    
    # Ternary regularization
    ternary_regularization_weight: float = 0.01  # Encourage sparsity
    
    # Replay emphasis
    hard_case_replay_factor: float = 4.0


@dataclass
class BitNetRuntimeConfig:
    """Runtime configuration for BitNet-style low-bit model."""
    # Inference settings
    use_ternary_inference: bool = True
    use_quantized_activations: bool = True
    
    # Memory optimization
    share_encoder_projections: bool = True
    use_fused_operations: bool = True
    
    # CPU-friendly settings
    batch_inference: bool = True
    enable_cpu_optimizations: bool = True


@dataclass
class BitNetSidecarConfig:
    """Main BitNet-inspired configuration for the sidecar."""
    latent_dims: BitNetLatentDimensions = field(default_factory=BitNetLatentDimensions)
    
    # BitNet-specific settings
    bitnet_training: BitNetTrainingConfig = field(default_factory=BitNetTrainingConfig)
    bitnet_runtime: BitNetRuntimeConfig = field(default_factory=BitNetRuntimeConfig)
    
    # Standard sidecar config - use defaults
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
    expert_types: List[ExpertType] = field(default_factory=lambda: list(ExpertType))
    
    # Version
    version: str = "1.5.0-bitnet"
    
    def to_standard_config(self) -> SidecarConfig:
        """Convert to standard SidecarConfig for compatibility."""
        # This would convert BitNet config to standard config
        # For now, return a placeholder
        from .config import SidecarConfig, BalancedBuildConfig
        return BalancedBuildConfig()


def BitNetBuildConfig() -> BitNetSidecarConfig:
    """Factory for BitNet-inspired configuration."""
    return BitNetSidecarConfig()


def BitNetTinyBuildConfig() -> BitNetSidecarConfig:
    """Factory for ultra-tiny BitNet configuration."""
    config = BitNetSidecarConfig()
    config.latent_dims = BitNetLatentDimensions(
        fused_observation=48,
        belief_hidden=32,
        predictive_hidden=48,
        bottleneck=16,
        slow_state=16,
        prototype=12,
        uncertainty=8,
        expert_residual=16,
        byte_encoder=12,
        address_encoder=12,
        event_encoder=10,
        map_encoder=8,
        summary_encoder=16,
    )
    config.num_experts = 4  # Fewer experts for tiny model
    return config
