"""
Encoder modules for the Byte-Plane (Plane A) v1.4.1.

Enhanced with safe categorical index handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .config import LatentDimensions, FeatureVocabConfig


class GatedFusion(nn.Module):
    """Gated fusion layer for combining multiple input streams."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, output_dim)
        self.transform = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x) * 1.2)  # Sharper gating
        return gate * self.transform(x)


class BytePlaneEncoder(nn.Module):
    """Encodes byte-shape and register-shape information - v1.4.1 with safe vocab sizes."""
    
    def __init__(self, dims: LatentDimensions, feature_vocab: Optional[FeatureVocabConfig] = None):
        super().__init__()
        self.output_dim = dims.byte_encoder
        self.feature_vocab = feature_vocab or FeatureVocabConfig()
        
        # Bucket embeddings - use feature vocab sizes
        self.low8_embed = nn.Embedding(self.feature_vocab.page_bucket_vocab, 8)
        self.high8_embed = nn.Embedding(self.feature_vocab.page_bucket_vocab, 8)
        self.alignment_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        self.small_int_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        self.delta_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 6)
        self.hamming_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        
        # Continuous features projection
        self.continuous_proj = nn.Linear(6, 8)
        
        # Bitfield projection
        self.bitfield_proj = nn.Linear(16, 8)
        
        # Byte window sketch projection
        self.sketch_proj = nn.Linear(32, 12)
        
        # Fusion
        total_embed = 8 + 8 + 4 + 4 + 6 + 4 + 8 + 8 + 12
        self.fusion = GatedFusion(total_embed, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)
        
    def forward(
        self,
        low8_bucket: torch.Tensor,
        high8_bucket: torch.Tensor,
        alignment_bucket: torch.Tensor,
        small_int_bucket: torch.Tensor,
        delta_bucket: torch.Tensor,
        hamming_bucket: torch.Tensor,
        continuous_features: torch.Tensor,
        bitfield_features: torch.Tensor,
        sketch_features: torch.Tensor,
    ) -> torch.Tensor:
        e_low8 = self.low8_embed(low8_bucket)
        e_high8 = self.high8_embed(high8_bucket)
        e_align = self.alignment_embed(alignment_bucket)
        e_small = self.small_int_embed(small_int_bucket)
        e_delta = self.delta_embed(delta_bucket)
        e_hamming = self.hamming_embed(hamming_bucket)
        
        p_cont = self.continuous_proj(continuous_features)
        p_bits = self.bitfield_proj(bitfield_features)
        p_sketch = self.sketch_proj(sketch_features)
        
        combined = torch.cat([
            e_low8, e_high8, e_align, e_small, e_delta, e_hamming,
            p_cont, p_bits, p_sketch
        ], dim=-1)
        
        z_byte = self.fusion(combined)
        return self.norm(z_byte)


class AddressShapeEncoder(nn.Module):
    """Encodes locality and access geometry - v1.4.1 enhanced with safe vocab sizes."""
    
    def __init__(self, dims: LatentDimensions, feature_vocab: Optional[FeatureVocabConfig] = None):
        super().__init__()
        self.output_dim = dims.address_encoder
        self.feature_vocab = feature_vocab or FeatureVocabConfig()
        
        # Bucket embeddings - use feature vocab sizes
        self.page_hash_embed = nn.Embedding(self.feature_vocab.page_bucket_vocab, 8)
        self.offset_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        self.cache_line_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        self.alignment_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 3)
        self.stride_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 6)
        self.reuse_dist_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 5)
        self.locality_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 6)
        self.entropy_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 3)
        
        # Flags projection
        self.flags_proj = nn.Linear(5, 5)
        
        # Fusion
        total_embed = 8 + 4 + 4 + 3 + 6 + 5 + 6 + 3 + 5
        self.fusion = GatedFusion(total_embed, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)
        
    def forward(
        self,
        page_hash_bucket: torch.Tensor,
        offset_bucket: torch.Tensor,
        cache_line_bucket: torch.Tensor,
        alignment_bucket: torch.Tensor,
        stride_bucket: torch.Tensor,
        reuse_dist_bucket: torch.Tensor,
        locality_cluster: torch.Tensor,
        entropy_bucket: torch.Tensor,
        flags: torch.Tensor,
    ) -> torch.Tensor:
        e_page = self.page_hash_embed(page_hash_bucket)
        e_offset = self.offset_embed(offset_bucket)
        e_cache = self.cache_line_embed(cache_line_bucket)
        e_align = self.alignment_embed(alignment_bucket)
        e_stride = self.stride_embed(stride_bucket)
        e_reuse = self.reuse_dist_embed(reuse_dist_bucket)
        e_locality = self.locality_embed(locality_cluster)
        e_entropy = self.entropy_embed(entropy_bucket)
        p_flags = self.flags_proj(flags)
        
        combined = torch.cat([
            e_page, e_offset, e_cache, e_align, e_stride,
            e_reuse, e_locality, e_entropy, p_flags
        ], dim=-1)
        
        z_addr = self.fusion(combined)
        return self.norm(z_addr)


class EventSemanticEncoder(nn.Module):
    """Encodes event type and semantic information - v1.4.1 enhanced with safe vocab sizes."""
    
    def __init__(self, dims: LatentDimensions, feature_vocab: Optional[FeatureVocabConfig] = None):
        super().__init__()
        self.output_dim = dims.event_encoder
        self.feature_vocab = feature_vocab or FeatureVocabConfig()
        
        # Use feature vocab sizes for embeddings
        self.event_type_embed = nn.Embedding(self.feature_vocab.event_type_vocab, 6)
        self.fault_class_embed = nn.Embedding(self.feature_vocab.fault_code_vocab, 5)
        self.syscall_class_embed = nn.Embedding(self.feature_vocab.syscall_class_vocab, 6)
        self.opcode_embed = nn.Embedding(self.feature_vocab.opcode_class_vocab, 4)
        self.transition_embed = nn.Embedding(self.feature_vocab.event_type_vocab, 3)
        self.result_class_embed = nn.Embedding(self.feature_vocab.rw_flag_vocab, 3)
        
        total_embed = 6 + 5 + 6 + 4 + 3 + 3
        self.fusion = GatedFusion(total_embed, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)
        
    def forward(
        self,
        event_type: torch.Tensor,
        fault_class: torch.Tensor,
        syscall_class: torch.Tensor,
        opcode_family: torch.Tensor,
        transition_type: torch.Tensor,
        result_class: torch.Tensor,
    ) -> torch.Tensor:
        e_event = self.event_type_embed(event_type)
        e_fault = self.fault_class_embed(fault_class)
        e_syscall = self.syscall_class_embed(syscall_class)
        e_opcode = self.opcode_embed(opcode_family)
        e_trans = self.transition_embed(transition_type)
        e_result = self.result_class_embed(result_class)
        
        combined = torch.cat([
            e_event, e_fault, e_syscall, e_opcode, e_trans, e_result
        ], dim=-1)
        
        z_evt = self.fusion(combined)
        return self.norm(z_evt)


class MapStateEncoder(nn.Module):
    """Encodes page-table and mapping state - v1.4.1 with safe vocab sizes."""
    
    def __init__(self, dims: LatentDimensions, feature_vocab: Optional[FeatureVocabConfig] = None):
        super().__init__()
        self.output_dim = dims.map_encoder
        self.feature_vocab = feature_vocab or FeatureVocabConfig()
        
        self.vma_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 4)
        self.protection_embed = nn.Embedding(self.feature_vocab.mode_vocab, 3)
        self.pte_proj = nn.Linear(11, 8)
        
        total_embed = 4 + 3 + 8
        self.fusion = GatedFusion(total_embed, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)
        
    def forward(
        self,
        pte_flags: torch.Tensor,
        vma_class: torch.Tensor,
        protection_domain: torch.Tensor,
    ) -> torch.Tensor:
        e_vma = self.vma_embed(vma_class)
        e_prot = self.protection_embed(protection_domain)
        p_pte = self.pte_proj(pte_flags.float())
        
        combined = torch.cat([e_vma, e_prot, p_pte], dim=-1)
        z_map = self.fusion(combined)
        return self.norm(z_map)


class SummaryEncoder(nn.Module):
    """
    Encodes windowed counters and pressure summaries.
    
    v1.4.1: Enhanced with safe vocab sizes.
    """
    
    def __init__(self, dims: LatentDimensions, feature_vocab: Optional[FeatureVocabConfig] = None):
        super().__init__()
        self.output_dim = dims.summary_encoder
        self.feature_vocab = feature_vocab or FeatureVocabConfig()
        
        # Count buckets - use feature vocab sizes
        self.count_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 5)
        
        # Continuous pressure features - deeper projection
        self.pressure_proj = nn.Sequential(
            nn.Linear(12, 24),
            nn.GELU(),
            nn.Linear(24, 20),
        )
        
        # Recency and volatility
        self.recency_embed = nn.Embedding(self.feature_vocab.region_bucket_vocab, 5)
        self.volatility_proj = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Linear(8, 6),
        )
        
        total_embed = 5 * 4 + 20 + 5 + 6  # 51
        self.fusion = GatedFusion(total_embed, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)
        
    def forward(
        self,
        read_count_bucket: torch.Tensor,
        write_count_bucket: torch.Tensor,
        fault_count_bucket: torch.Tensor,
        cow_count_bucket: torch.Tensor,
        recency_bucket: torch.Tensor,
        volatility_features: torch.Tensor,
        pressure_features: torch.Tensor,
    ) -> torch.Tensor:
        e_read = self.count_embed(read_count_bucket)
        e_write = self.count_embed(write_count_bucket)
        e_fault = self.count_embed(fault_count_bucket)
        e_cow = self.count_embed(cow_count_bucket)
        e_recency = self.recency_embed(recency_bucket)
        
        p_vol = self.volatility_proj(volatility_features)
        p_pressure = self.pressure_proj(pressure_features)
        
        combined = torch.cat([
            e_read, e_write, e_fault, e_cow, e_recency, p_vol, p_pressure
        ], dim=-1)
        
        z_sum = self.fusion(combined)
        return self.norm(z_sum)


class FusedObservationBlock(nn.Module):
    """
    Fuses all encoded evidence into a compact shared observation state.
    
    v1.1: Deeper fusion for better integration.
    """
    
    def __init__(self, dims: LatentDimensions):
        super().__init__()
        self.output_dim = dims.fused_observation
        
        input_dim = (dims.byte_encoder + dims.address_encoder + 
                     dims.event_encoder + dims.map_encoder + dims.summary_encoder)
        
        # Deeper gated fusion
        self.gate = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.transform = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        
        # Residual MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim * 2),
            nn.GELU(),
            nn.Linear(self.output_dim * 2, self.output_dim),
        )
        
        self.norm1 = nn.LayerNorm(self.output_dim)
        self.norm2 = nn.LayerNorm(self.output_dim)
        
    def forward(
        self,
        z_byte: torch.Tensor,
        z_addr: torch.Tensor,
        z_evt: torch.Tensor,
        z_map: torch.Tensor,
        z_sum: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([z_byte, z_addr, z_evt, z_map, z_sum], dim=-1)
        
        gate = torch.sigmoid(self.gate(combined) * 1.2)
        fused = gate * self.transform(combined)
        fused = self.norm1(fused)
        
        z_obs0 = fused + self.mlp(fused)
        return self.norm2(z_obs0)
