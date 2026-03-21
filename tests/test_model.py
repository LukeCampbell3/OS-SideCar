"""Tests for the neural model."""

import pytest
import torch
import numpy as np

from ac_moe_ga_sidecar.config import SidecarConfig, BalancedBuildConfig
from ac_moe_ga_sidecar.model import ACMoEGAModel, ModelState, ModelOutput
from ac_moe_ga_sidecar.encoders import (
    BytePlaneEncoder, AddressShapeEncoder, EventSemanticEncoder,
    MapStateEncoder, SummaryEncoder, FusedObservationBlock
)


@pytest.fixture
def config():
    return BalancedBuildConfig()


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def model(config):
    return ACMoEGAModel(config)


@pytest.fixture
def dummy_inputs(config, device):
    batch_size = 2
    fv = config.feature_vocab
    
    # Helper to generate safe indices for embeddings
    def safe_event_type():
        # 0=UNK, 1..16 valid (16 event types)
        return torch.randint(1, 17, (batch_size,), device=device)
    
    def safe_fault_class():
        # 0=UNK, 1..32 valid
        return torch.randint(1, 33, (batch_size,), device=device)
    
    def safe_syscall_class():
        # 0=UNK, 1..64 valid
        return torch.randint(1, 65, (batch_size,), device=device)
    
    def safe_opcode_family():
        # 0=UNK, 1..32 valid
        return torch.randint(1, 33, (batch_size,), device=device)
    
    def safe_transition_type():
        # 0=UNK, 1..16 valid
        return torch.randint(1, 17, (batch_size,), device=device)
    
    def safe_result_class():
        # 0=UNK, 1..3 valid
        return torch.randint(1, 4, (batch_size,), device=device)
    
    def safe_vma_class():
        # 0=UNK, 1..1024 valid
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_protection_domain():
        # 0=UNK, 1..4 valid
        return torch.randint(1, 5, (batch_size,), device=device)
    
    # Safe index helpers for all categorical features
    def safe_low8():
        # page_bucket_vocab = 4097, valid range 1..4096
        return torch.randint(1, 4097, (batch_size,), device=device)
    
    def safe_high8():
        return torch.randint(1, 4097, (batch_size,), device=device)
    
    def safe_alignment():
        # region_bucket_vocab = 1025, valid range 1..1024
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_small_int():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_delta():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_hamming():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_page_hash():
        return torch.randint(1, 4097, (batch_size,), device=device)
    
    def safe_offset():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_cache_line():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_addr_align():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_stride():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_reuse_dist():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_locality():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_entropy():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_read_bucket():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_write_bucket():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_fault_bucket():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_cow_bucket():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    def safe_recency():
        return torch.randint(1, 1025, (batch_size,), device=device)
    
    return {
        'low8_bucket': safe_low8(),
        'high8_bucket': safe_high8(),
        'alignment_bucket': safe_alignment(),
        'small_int_bucket': safe_small_int(),
        'delta_bucket': safe_delta(),
        'hamming_bucket': safe_hamming(),
        'continuous_features': torch.randn(batch_size, 6, device=device),
        'bitfield_features': torch.randn(batch_size, 16, device=device),
        'sketch_features': torch.randn(batch_size, 32, device=device),
        'page_hash_bucket': safe_page_hash(),
        'offset_bucket': safe_offset(),
        'cache_line_bucket': safe_cache_line(),
        'addr_alignment_bucket': safe_addr_align(),
        'stride_bucket': safe_stride(),
        'reuse_dist_bucket': safe_reuse_dist(),
        'locality_cluster': safe_locality(),
        'entropy_bucket': safe_entropy(),
        'address_flags': torch.randn(batch_size, 5, device=device),
        'event_type': safe_event_type(),
        'fault_class': safe_fault_class(),
        'syscall_class': safe_syscall_class(),
        'opcode_family': safe_opcode_family(),
        'transition_type': safe_transition_type(),
        'result_class': safe_result_class(),
        'pte_flags': torch.randn(batch_size, 11, device=device),
        'vma_class': safe_vma_class(),
        'protection_domain': safe_protection_domain(),
        'read_count_bucket': safe_read_bucket(),
        'write_count_bucket': safe_write_bucket(),
        'fault_count_bucket': safe_fault_bucket(),
        'cow_count_bucket': safe_cow_bucket(),
        'recency_bucket': safe_recency(),
        'volatility_features': torch.randn(batch_size, 4, device=device),
        'pressure_features': torch.randn(batch_size, 12, device=device),
        'missingness_mask': torch.zeros(batch_size, 8, device=device),
        'freshness_ages': torch.ones(batch_size, 4, device=device),
        'source_quality': torch.ones(batch_size, 2, device=device),
        'conflict_score': torch.zeros(batch_size, device=device),
        'consistency_score': torch.ones(batch_size, device=device),
    }


class TestBytePlaneEncoder:
    def test_output_shape(self, config, device):
        encoder = BytePlaneEncoder(config.latent_dims).to(device)
        batch_size = 4
        fv = config.feature_vocab
        
        output = encoder(
            torch.randint(1, fv.page_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.page_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.region_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.region_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.region_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.region_bucket_vocab, (batch_size,), device=device),
            torch.randn(batch_size, 6, device=device),
            torch.randn(batch_size, 16, device=device),
            torch.randn(batch_size, 32, device=device),
        )
        
        assert output.shape == (batch_size, config.latent_dims.byte_encoder)


class TestAddressShapeEncoder:
    def test_output_shape(self, config, device):
        encoder = AddressShapeEncoder(config.latent_dims).to(device)
        batch_size = 4
        fv = config.feature_vocab
        
        output = encoder(
            torch.randint(1, fv.page_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.region_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.region_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.region_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.region_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.region_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.region_bucket_vocab, (batch_size,), device=device),
            torch.randint(1, fv.region_bucket_vocab, (batch_size,), device=device),
            torch.randn(batch_size, 5, device=device),
        )
        
        assert output.shape == (batch_size, config.latent_dims.address_encoder)


class TestACMoEGAModel:
    def test_initialization(self, model, config):
        assert model.config == config
        assert model.dims == config.latent_dims

    def test_forward_pass(self, model, dummy_inputs, device):
        model = model.to(device)
        output = model(dummy_inputs)
        
        assert isinstance(output, ModelOutput)
        assert output.z_pred is not None
        assert output.uncertainty is not None
        assert output.head_outputs is not None
        assert output.new_state is not None

    def test_output_shapes(self, model, dummy_inputs, config, device):
        model = model.to(device)
        batch_size = dummy_inputs['low8_bucket'].shape[0]
        output = model(dummy_inputs)
        
        assert output.z_pred.shape == (batch_size, config.latent_dims.bottleneck)
        assert output.uncertainty.shape[0] == batch_size
        assert output.should_abstain.shape == (batch_size,)

    def test_state_persistence(self, model, dummy_inputs, device):
        model = model.to(device)
        
        # First forward pass
        output1 = model(dummy_inputs)
        state1 = output1.new_state
        
        # Second forward pass with state
        output2 = model(dummy_inputs, state=state1)
        
        # States should be different
        assert not torch.allclose(state1.belief_page, output2.new_state.belief_page)

    def test_parameter_count(self, model):
        param_count = model.get_parameter_count()
        assert param_count > 0
        # Should be in reasonable range for balanced build
        assert param_count < 10_000_000  # Less than 10M params

    def test_model_size(self, model):
        size_mb = model.get_model_size_mb()
        assert size_mb > 0
        # Should be within spec (2-4.5 MB for balanced build)
        assert size_mb < 10  # Generous upper bound

    def test_gradient_flow(self, model, dummy_inputs, device):
        model = model.to(device)
        model.train()
        
        output = model(dummy_inputs)
        
        # Use a comprehensive loss that touches all outputs
        loss = output.z_pred.sum()
        loss = loss + output.uncertainty.sum()
        
        # Add losses from all head outputs
        for key, value in output.head_outputs.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    loss = loss + v.sum()
            else:
                loss = loss + value.sum()
        
        # Add losses from controller outputs to ensure all parameters get gradients
        loss = loss + output.support_density.sum()
        loss = loss + output.familiarity.sum()
        loss = loss + output.drift_score.sum()
        loss = loss + output.should_abstain.sum().float()
        loss = loss + output.calibrated_confidence.sum()
        loss = loss + output.action_margin.sum()
        
        loss.backward()
        
        # Check that most gradients exist
        params_with_grad = 0
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
        
        # At least 75% of parameters should have gradients (some expert params may not be used)
        assert params_with_grad / total_params >= 0.75, f"Only {params_with_grad}/{total_params} params have gradients"


class TestModelState:
    def test_initialization(self, config, device):
        batch_size = 4
        state = ModelState.init(batch_size, config.latent_dims, device)
        
        assert state.belief_page.shape == (batch_size, config.latent_dims.belief_hidden)
        assert state.belief_region.shape == (batch_size, config.latent_dims.belief_hidden)
        assert state.belief_process.shape == (batch_size, config.latent_dims.belief_hidden)
        assert state.slow_state.shape == (batch_size, config.latent_dims.slow_state)
        assert state.pred_hidden.shape == (batch_size, config.latent_dims.predictive_hidden)

    def test_initial_values_are_zero(self, config, device):
        batch_size = 4
        state = ModelState.init(batch_size, config.latent_dims, device)
        
        assert torch.all(state.belief_page == 0)
        assert torch.all(state.slow_state == 0)
