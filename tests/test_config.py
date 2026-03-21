"""Tests for configuration module v1.1."""

import pytest
from ac_moe_ga_sidecar.config import (
    SidecarConfig, BalancedBuildConfig, LatentDimensions,
    RuntimeMemoryBudget, RoutingConfig, ExpertType
)


class TestLatentDimensions:
    def test_default_values(self):
        """Test v1.1 default latent dimensions (increased capacity)."""
        dims = LatentDimensions()
        assert dims.fused_observation == 80  # Up from 64
        assert dims.belief_hidden == 48      # Up from 32
        assert dims.predictive_hidden == 80  # Up from 64
        assert dims.bottleneck == 32         # Up from 24
        assert dims.slow_state == 32         # Up from 24
        assert dims.prototype == 24          # Up from 16
        assert dims.uncertainty == 8
        assert dims.expert_residual == 32    # Up from 24

    def test_encoder_dimensions(self):
        """Test v1.1 encoder dimensions."""
        dims = LatentDimensions()
        assert dims.byte_encoder == 24
        assert dims.address_encoder == 20   # Up from 16
        assert dims.event_encoder == 16     # Up from 12
        assert dims.map_encoder == 12
        assert dims.summary_encoder == 32   # Up from 24


class TestRuntimeMemoryBudget:
    def test_default_values(self):
        """Test v1.1 memory budget (increased for better regime tracking)."""
        budget = RuntimeMemoryBudget()
        assert budget.deterministic_state_mb == 12.0
        assert budget.active_latent_cache_mb == 6.0  # Up from 4.0
        assert budget.prototype_tables_mb == 1.0     # Up from 0.5

    def test_total_calculation(self):
        budget = RuntimeMemoryBudget()
        assert budget.total_mb == 19.0  # 12 + 6 + 1

    def test_custom_values(self):
        budget = RuntimeMemoryBudget(
            deterministic_state_mb=8.0,
            active_latent_cache_mb=2.0,
            prototype_tables_mb=0.25
        )
        assert budget.total_mb == 10.25


class TestRoutingConfig:
    def test_default_values(self):
        """Test v1.1 routing config (relaxed for earlier specialization)."""
        config = RoutingConfig()
        assert config.max_experts_per_inference == 2
        assert config.default_top_k == 1
        assert config.min_support_count == 50   # Down from 100
        assert config.min_probing_value == 0.25 # Down from 0.3
        assert config.max_drift_penalty == 0.6  # Up from 0.5


class TestSidecarConfig:
    def test_default_initialization(self):
        config = SidecarConfig()
        assert config.num_experts == 8
        assert config.max_active_pages == 100000
        assert config.max_active_regions == 10000
        assert config.max_active_processes == 1000
        assert config.version == "1.4.1"

    def test_validation_passes(self):
        config = SidecarConfig()
        assert config.validate() is True

    def test_expert_types(self):
        config = SidecarConfig()
        assert len(config.expert_types) == 8
        assert ExpertType.PAGE_TRANSITION in config.expert_types
        assert ExpertType.COW_FORK in config.expert_types
        assert ExpertType.NUMA_PLACEMENT in config.expert_types


class TestBalancedBuildConfig:
    def test_factory_returns_config(self):
        config = BalancedBuildConfig()
        assert isinstance(config, SidecarConfig)

    def test_balanced_build_validates(self):
        config = BalancedBuildConfig()
        assert config.validate() is True

    def test_memory_budget_within_limits(self):
        config = BalancedBuildConfig()
        assert config.memory_budget.total_mb <= 30.0
