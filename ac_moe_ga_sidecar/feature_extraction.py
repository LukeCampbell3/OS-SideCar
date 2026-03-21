"""
Feature extraction for AC-MoE-GA Sidecar v1.2.

Optimized for:
- Vectorized preprocessing
- Reduced allocations
- Contiguous tensor storage
- Incremental state updates

Key improvements over v1.1:
- Pre-allocated feature arrays
- Batched state updates
- Incremental sketch updates
- Direct tensor output
"""

import numpy as np
from numpy.typing import NDArray
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .types import (
    MicroEvent, RegisterShapeFeatures, AddressShapeFeatures,
    MapStateFeatures, TrapFaultFeatures, ByteWindowSketch, EntitySummary
)
from .config import SidecarConfig


@dataclass
class ExtractedFeatures:
    """Container for all extracted features from a micro-event.
    
    v1.2: Pre-allocated arrays for zero-copy tensor conversion.
    """
    # Byte-plane features (pre-allocated)
    low8_bucket: int
    high8_bucket: int
    alignment_bucket: int
    small_int_bucket: int
    delta_bucket: int
    hamming_bucket: int
    continuous_features: NDArray[np.float32]  # [6]
    bitfield_features: NDArray[np.float32]  # [16]
    sketch_features: NDArray[np.float32]  # [32]
    
    # Address-shape features
    page_hash_bucket: int
    offset_bucket: int
    cache_line_bucket: int
    addr_alignment_bucket: int
    stride_bucket: int
    reuse_dist_bucket: int
    locality_cluster: int
    entropy_bucket: int
    address_flags: NDArray[np.float32]  # [5]
    
    # Event semantic features
    event_type: int
    fault_class: int
    syscall_class: int
    opcode_family: int
    transition_type: int
    result_class: int
    
    # Map state features
    pte_flags: NDArray[np.float32]  # [11]
    vma_class: int
    protection_domain: int
    
    # Summary features
    read_count_bucket: int
    write_count_bucket: int
    fault_count_bucket: int
    cow_count_bucket: int
    recency_bucket: int
    volatility_features: NDArray[np.float32]  # [4]
    pressure_features: NDArray[np.float32]  # [12]
    
    # Observability metadata
    missingness_mask: NDArray[np.float32]  # [8]
    freshness_ages: NDArray[np.float32]  # [4]
    source_quality: NDArray[np.float32]  # [2]
    conflict_score: float
    consistency_score: float


def to_safe_index(value, vocab_size: int, unknown_idx: int = 0) -> int:
    """
    Maps arbitrary input to a safe embedding index in [0, vocab_size-1].
    0 is reserved for unknown / invalid.
    
    Args:
        value: Raw value to map
        vocab_size: Size of the embedding vocabulary
        unknown_idx: Index to use for unknown/invalid values (default: 0)
    
    Returns:
        Safe index in [0, vocab_size-1]
    """
    if vocab_size <= 1:
        return 0
    
    if value is None:
        return unknown_idx
    
    try:
        v = int(value)
    except (TypeError, ValueError):
        return unknown_idx
    
    if v < 0:
        return unknown_idx
    
    # Reserve 0 for UNK, map valid ids into 1..vocab_size-1
    return (v % (vocab_size - 1)) + 1


class FeatureExtractor:
    """
    Extracts structured features from raw micro-events.
    
    v1.4.1: Safe categorical index handling with centralized normalization.
    """
    
    # Pre-computed bucket mappings for speed
    _LOW8_BUCKETS = np.array([i * 32 for i in range(8)], dtype=np.int64)
    _HIGH8_BUCKETS = np.array([i * 8192 for i in range(8)], dtype=np.int64)
    _ALIGNMENT_BUCKETS = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
    _SMALL_INT_BUCKETS = np.array([0, 1, 2, 3, 4, 8, 16, 32], dtype=np.int64)
    _HAMMING_BUCKETS = np.array([i for i in range(9)], dtype=np.int64)
    
    def __init__(self, config: SidecarConfig):
        self.config = config
        
        # Pre-allocated feature arrays for zero-copy
        self._continuous = np.zeros(6, dtype=np.float32)
        self._bitfield = np.zeros(16, dtype=np.float32)
        self._sketch = np.zeros(32, dtype=np.float32)
        self._address_flags = np.zeros(5, dtype=np.float32)
        self._pte_flags = np.zeros(11, dtype=np.float32)
        self._volatility = np.zeros(4, dtype=np.float32)
        self._pressure = np.zeros(12, dtype=np.float32)
        self._missingness = np.zeros(8, dtype=np.float32)
        self._freshness = np.zeros(4, dtype=np.float32)
        self._source_quality = np.zeros(2, dtype=np.float32)
        
        # State for delta encoding
        self.prev_register_value: Optional[int] = None
        self.prev_address: Optional[int] = None
        self.prev_event_type: Optional[int] = None
        
        # Locality tracking
        self.recent_pages: list = []
        self.recent_pages_max = 256
        
        # Reuse distance tracking
        self.page_access_times: Dict[int, int] = {}
        self.access_counter = 0
        
        # Locality clustering
        self.page_clusters: Dict[int, int] = {}
        self.cluster_id = 0
        
    def extract(
        self,
        event: MicroEvent,
        page_summary: EntitySummary,
        region_summary: EntitySummary,
        process_summary: EntitySummary,
    ) -> ExtractedFeatures:
        """Extract all features from a micro-event.
        
        v1.2: Optimized with pre-allocated arrays and incremental updates.
        """
        # Extract register-shape features
        self._extract_register_features(event)
        
        # Extract address-shape features
        self._extract_address_features(event)
        
        # Extract event semantic features
        self._extract_event_features(event)
        
        # Extract map state features
        self._extract_map_features(event)
        
        # Extract summary features
        self._extract_summary_features(
            page_summary, region_summary, process_summary
        )
        
        # Extract observability metadata
        self._extract_observability_features(event)
        
        # Update state for next event
        self._update_state(event)
        
        return ExtractedFeatures(
            # Byte-plane
            low8_bucket=self._last_low8,
            high8_bucket=self._last_high8,
            alignment_bucket=self._last_alignment,
            small_int_bucket=self._last_small_int,
            delta_bucket=self._last_delta,
            hamming_bucket=self._last_hamming,
            continuous_features=self._continuous,
            bitfield_features=self._bitfield,
            sketch_features=self._sketch,
            
            # Address-shape
            page_hash_bucket=self._last_page_hash,
            offset_bucket=self._last_offset,
            cache_line_bucket=self._last_cache_line,
            addr_alignment_bucket=self._last_addr_align,
            stride_bucket=self._last_stride,
            reuse_dist_bucket=self._last_reuse_dist,
            locality_cluster=self._last_locality,
            entropy_bucket=self._last_entropy,
            address_flags=self._address_flags,
            
            # Event semantic
            event_type=self._last_event_type,
            fault_class=self._last_fault_class,
            syscall_class=self._last_syscall_class,
            opcode_family=self._last_opcode_family,
            transition_type=self._last_transition,
            result_class=self._last_result_class,
            
            # Map state
            pte_flags=self._pte_flags,
            vma_class=self._last_vma_class,
            protection_domain=self._last_prot_domain,
            
            # Summary
            read_count_bucket=self._last_read_bucket,
            write_count_bucket=self._last_write_bucket,
            fault_count_bucket=self._last_fault_bucket,
            cow_count_bucket=self._last_cow_bucket,
            recency_bucket=self._last_recency,
            volatility_features=self._volatility,
            pressure_features=self._pressure,
            
            # Observability
            missingness_mask=self._missingness,
            freshness_ages=self._freshness,
            source_quality=self._source_quality,
            conflict_score=self._last_conflict,
            consistency_score=self._last_consistency,
        )
    
    def _extract_register_features(self, event: MicroEvent):
        """Extract register-shape features with safe categorical indexing."""
        # Low 8 bits bucket
        low8 = event.pc_bucket & 0xFF
        self._last_low8 = to_safe_index(
            np.searchsorted(self._LOW8_BUCKETS, low8, side='right') - 1,
            self.config.feature_vocab.page_bucket_vocab
        )
        
        # High 8 bits bucket
        high8 = (event.pc_bucket >> 8) & 0xFF
        self._last_high8 = to_safe_index(
            np.searchsorted(self._HIGH8_BUCKETS, high8, side='right') - 1,
            self.config.feature_vocab.page_bucket_vocab
        )
        
        # Alignment bucket
        alignment = event.pc_bucket & 63
        self._last_alignment = to_safe_index(
            np.searchsorted(self._ALIGNMENT_BUCKETS, alignment, side='right') - 1,
            self.config.feature_vocab.region_bucket_vocab
        )
        
        # Small int bucket
        small_int = min(event.pc_bucket, 32)
        self._last_small_int = to_safe_index(
            np.searchsorted(self._SMALL_INT_BUCKETS, small_int, side='right') - 1,
            self.config.feature_vocab.region_bucket_vocab
        )
        
        # Delta encoding
        if self.prev_register_value is not None:
            delta = abs(event.pc_bucket - self.prev_register_value)
        else:
            delta = 0
        self._last_delta = to_safe_index(
            min(delta // 32, 63), self.config.feature_vocab.region_bucket_vocab
        )
        
        # Hamming distance
        if self.prev_register_value is not None:
            xor_val = event.pc_bucket ^ self.prev_register_value
            hamming = bin(xor_val).count('1')
        else:
            hamming = 0
        self._last_hamming = to_safe_index(
            min(hamming, 8), self.config.feature_vocab.region_bucket_vocab
        )
        
        # Continuous features (pre-allocated)
        self._continuous[0] = np.log1p(event.pc_bucket) / 20.0
        self._continuous[1] = np.log1p(delta) / 15.0
        self._continuous[2] = (event.pid & 255) / 255.0
        self._continuous[3] = (event.tid & 255) / 255.0
        self._continuous[4] = to_safe_index(
            event.cpu_id & 7, self.config.feature_vocab.cpu_id_vocab
        ) / 7.0
        self._continuous[5] = (event.timestamp_bucket & 1023) / 1023.0
        
        # Bitfield features (pre-allocated)
        rw_flag = event.rw_flag if event.rw_flag is not None else 0
        self._bitfield[0] = to_safe_index(
            rw_flag & 1, self.config.feature_vocab.rw_flag_vocab
        ) * 1.0
        self._bitfield[1] = to_safe_index(
            (event.mode & 1) ^ 1, self.config.feature_vocab.mode_vocab
        ) * 1.0  # kernel mode
        self._bitfield[2] = to_safe_index(
            event.mode & 1, self.config.feature_vocab.mode_vocab
        ) * 1.0  # user mode
        self._bitfield[3] = ((event.event_type >> 4) & 1) * 1.0
        self._bitfield[4] = ((event.event_type >> 5) & 1) * 1.0
        self._bitfield[5] = ((event.event_type >> 6) & 1) * 1.0
        self._bitfield[6] = ((event.event_type >> 7) & 1) * 1.0
        self._bitfield[7] = ((event.event_type >> 8) & 1) * 1.0
        self._bitfield[8] = ((event.event_type >> 9) & 1) * 1.0
        self._bitfield[9] = ((event.event_type >> 10) & 1) * 1.0
        self._bitfield[10] = ((event.event_type >> 11) & 1) * 1.0
        self._bitfield[11] = ((event.event_type >> 12) & 1) * 1.0
        self._bitfield[12] = ((event.event_type >> 13) & 1) * 1.0
        self._bitfield[13] = ((event.event_type >> 14) & 1) * 1.0
        self._bitfield[14] = ((event.event_type >> 15) & 1) * 1.0
        self._bitfield[15] = ((event.event_type >> 16) & 1) * 1.0
        
        # Sketch features (incremental update)
        self._update_byte_sketch(event)
    
    def _extract_address_features(self, event: MicroEvent):
        """Extract address-shape features with safe categorical indexing."""
        if event.virtual_page is not None:
            page = event.virtual_page
            offset = page & 255
            cache_line = (page >> 3) & 31
            addr_align = page & 63
            
            # Page hash bucket - safe index
            page_hash = (page * 2654435761) & 0xFFFFFFFF
            self._last_page_hash = to_safe_index(
                (page_hash >> 24) & 255, self.config.feature_vocab.page_bucket_vocab
            )
            
            # Offset bucket - safe index
            self._last_offset = to_safe_index(
                min(offset // 32, 7), self.config.feature_vocab.region_bucket_vocab
            )
            
            # Cache line bucket - safe index
            self._last_cache_line = to_safe_index(
                min(cache_line, 31), self.config.feature_vocab.region_bucket_vocab
            )
            
            # Address alignment bucket - safe index
            self._last_addr_align = to_safe_index(
                min(addr_align // 8, 7), self.config.feature_vocab.region_bucket_vocab
            )
            
            # Stride calculation
            if self.prev_address is not None:
                stride = page - self.prev_address
                self._last_stride = self._encode_stride(stride)
            else:
                self._last_stride = 0
            
            # Reuse distance
            self._last_reuse_dist = self._compute_reuse_distance(page)
            
            # Locality cluster
            self._last_locality = self._compute_locality_cluster(page)
            
            # Address entropy
            self._last_entropy = self._compute_address_entropy(page)
            
            # Address flags (pre-allocated)
            self._address_flags[0] = (page & 1) * 1.0  # odd page
            self._address_flags[1] = ((page >> 8) & 1) * 1.0
            self._address_flags[2] = ((page >> 16) & 1) * 1.0
            self._address_flags[3] = ((page >> 24) & 1) * 1.0
            self._address_flags[4] = (page > 0x100000000) * 1.0  # high memory
        else:
            self._last_page_hash = 0
            self._last_offset = 0
            self._last_cache_line = 0
            self._last_addr_align = 0
            self._last_stride = 0
            self._last_reuse_dist = 63
            self._last_locality = 0
            self._last_entropy = 0
            self._address_flags.fill(0.0)
        
        self.prev_address = event.virtual_page
    
    def _extract_event_features(self, event: MicroEvent):
        """Extract event semantic features with safe categorical indexing."""
        # Use safe index for event_type
        self._last_event_type = to_safe_index(
            event.event_type, self.config.feature_vocab.event_type_vocab
        )
        
        # Fault class - safe index
        if event.event_type in [2, 3]:  # PAGE_FAULT, COW_FAULT
            self._last_fault_class = to_safe_index(
                event.trap_fault_syscall_code, self.config.feature_vocab.fault_code_vocab
            ) if event.trap_fault_syscall_code is not None else 1
        else:
            self._last_fault_class = 0
        
        # Syscall class - safe index
        if event.event_type in [5, 6]:  # SYSCALL_ENTRY, SYSCALL_EXIT
            self._last_syscall_class = to_safe_index(
                event.trap_fault_syscall_code, self.config.feature_vocab.syscall_class_vocab
            ) if event.trap_fault_syscall_code is not None else 1
        else:
            self._last_syscall_class = 0
        
        # Opcode family - safe index
        self._last_opcode_family = to_safe_index(
            event.opcode_class, self.config.feature_vocab.opcode_class_vocab
        )
        
        # Transition type - safe index
        self._last_transition = to_safe_index(
            event.event_type, self.config.feature_vocab.event_type_vocab
        )
        
        # Result class - safe index
        rw_flag = event.rw_flag if event.rw_flag is not None else 0
        self._last_result_class = to_safe_index(
            rw_flag, self.config.feature_vocab.rw_flag_vocab
        )
    
    def _extract_map_features(self, event: MicroEvent):
        """Extract map state features with safe categorical indexing."""
        if event.pte_flags:
            # Decode PTE flags - safe index
            pte_flags_idx = to_safe_index(
                event.pte_flags & 0xFF, self.config.feature_vocab.pte_flags_vocab
            )
            self._pte_flags[0] = ((pte_flags_idx >> 0) & 1) * 1.0  # present
            self._pte_flags[1] = ((pte_flags_idx >> 1) & 1) * 1.0  # writable
            self._pte_flags[2] = ((pte_flags_idx >> 2) & 1) * 1.0  # user
            self._pte_flags[3] = ((pte_flags_idx >> 3) & 1) * 1.0  # accessed
            self._pte_flags[4] = ((pte_flags_idx >> 4) & 1) * 1.0  # dirty
            self._pte_flags[5] = ((pte_flags_idx >> 5) & 1) * 1.0  # huge
            self._pte_flags[6] = ((pte_flags_idx >> 6) & 1) * 1.0  # global
            self._pte_flags[7] = ((pte_flags_idx >> 7) & 1) * 1.0  # software
            self._pte_flags[8] = ((pte_flags_idx >> 8) & 1) * 1.0
            self._pte_flags[9] = ((pte_flags_idx >> 9) & 1) * 1.0
            self._pte_flags[10] = ((pte_flags_idx >> 10) & 1) * 1.0
        else:
            self._pte_flags.fill(0.0)
        
        self._last_vma_class = 0
        self._last_prot_domain = 0
    
    def _extract_summary_features(self, page_summary, region_summary, process_summary):
        """Extract summary features from runtime state with safe categorical indexing."""
        # Read count bucket - safe index
        read_count = page_summary.read_count if page_summary else 0
        self._last_read_bucket = to_safe_index(
            min(int(np.log2(max(read_count, 1))), 15), self.config.feature_vocab.region_bucket_vocab
        )
        
        # Write count bucket - safe index
        write_count = page_summary.write_count if page_summary else 0
        self._last_write_bucket = to_safe_index(
            min(int(np.log2(max(write_count, 1))), 15), self.config.feature_vocab.region_bucket_vocab
        )
        
        # Fault count bucket - safe index
        fault_count = page_summary.fault_count if page_summary else 0
        self._last_fault_bucket = to_safe_index(
            min(int(np.log2(max(fault_count, 1))), 15), self.config.feature_vocab.region_bucket_vocab
        )
        
        # COW count bucket - safe index
        cow_count = page_summary.cow_count if page_summary else 0
        self._last_cow_bucket = to_safe_index(
            min(int(np.log2(max(cow_count, 1))), 15), self.config.feature_vocab.region_bucket_vocab
        )
        
        # Recency bucket (already a bucket, not a raw recency value) - safe index
        recency = page_summary.recency_bucket if page_summary else 31
        self._last_recency = to_safe_index(
            min(recency, 31), self.config.feature_vocab.region_bucket_vocab
        )
        
        # Volatility features (pre-allocated) - use available volatility
        volatility = page_summary.volatility if page_summary else 0.0
        self._volatility[0] = min(volatility, 1.0)
        self._volatility[1] = min(region_summary.volatility if region_summary else 0.0, 1.0)
        self._volatility[2] = min(process_summary.volatility if process_summary else 0.0, 1.0)
        self._volatility[3] = min(page_summary.neighbor_density if page_summary else 0.0, 1.0)
        
        # Pressure features (pre-allocated) - use available pressure fields
        if page_summary:
            self._pressure[0] = page_summary.bandwidth_pressure
            self._pressure[1] = page_summary.reclaim_pressure
            self._pressure[2] = page_summary.queue_depth
            self._pressure[3] = page_summary.numa_pressure
            self._pressure[4] = page_summary.kernel_entry_pressure
            self._pressure[5] = page_summary.kv_pressure
        else:
            self._pressure.fill(0.0)
    
    def _extract_observability_features(self, event: MicroEvent):
        """Extract observability metadata with safe categorical indexing."""
        # Missingness mask (pre-allocated)
        self._missingness[0] = (event.virtual_page is None) * 1.0
        self._missingness[1] = (event.pid == 0) * 1.0
        self._missingness[2] = (event.tid == 0) * 1.0
        self._missingness[3] = (event.pc_bucket == 0) * 1.0
        self._missingness[4] = (event.event_type == 0) * 1.0
        self._missingness[5] = (event.timestamp_bucket == 0) * 1.0
        self._missingness[6] = 0.0
        self._missingness[7] = 0.0
        
        # Freshness ages (pre-allocated)
        self._freshness[0] = to_safe_index(
            event.timestamp_bucket & 63, self.config.feature_vocab.region_bucket_vocab
        ) / 31.0
        self._freshness[1] = 0.0
        self._freshness[2] = 0.0
        self._freshness[3] = 0.0
        
        # Source quality (pre-allocated)
        self._source_quality[0] = 0.5
        self._source_quality[1] = 0.5
        
        self._last_conflict = 0.0
        self._last_consistency = 0.5
    
    def _update_state(self, event: MicroEvent):
        """Update internal state for next event."""
        self.prev_register_value = event.pc_bucket
        self.prev_event_type = event.event_type
        self.access_counter += 1
        
        # Update recent pages
        if event.virtual_page is not None:
            page = event.virtual_page
            if page not in self.recent_pages:
                if len(self.recent_pages) >= self.recent_pages_max:
                    self.recent_pages.pop(0)
                self.recent_pages.append(page)
            
            # Update access time
            self.page_access_times[page] = self.access_counter
    
    def _encode_delta(self, delta: int, num_buckets: int = 64) -> int:
        """Encode delta into bucket."""
        if delta == 0:
            return 0
        bucket = int(np.log2(delta)) + 1
        return min(bucket, num_buckets - 1)
    
    def _encode_stride(self, stride: int) -> int:
        """Encode stride into bucket."""
        if stride == 0:
            return 0
        if stride < 0:
            stride = -stride
        bucket = int(np.log2(stride)) + 1
        return min(bucket, 63)
    
    def _compute_reuse_distance(self, page: int) -> int:
        """Compute reuse distance for a page."""
        if page not in self.page_access_times:
            return 63
        
        recent_pages = set(self.recent_pages[-256:] if len(self.recent_pages) > 256 else self.recent_pages)
        if page not in recent_pages:
            return 63
        
        # Count unique pages accessed since last access
        access_time = self.page_access_times[page]
        recent_times = [self.page_access_times.get(p, 0) for p in recent_pages]
        recent_times = [t for t in recent_times if 0 < t < access_time]
        
        return min(len(set(recent_times)), 63)
    
    def _compute_locality_cluster(self, page: int) -> int:
        """Compute locality cluster for a page."""
        if page in self.page_clusters:
            return self.page_clusters[page]
        
        # Assign new cluster
        cluster = self.cluster_id
        self.cluster_id = (self.cluster_id + 1) & 255
        self.page_clusters[page] = cluster
        
        # Limit cluster ID space
        if len(self.page_clusters) > 256:
            # Merge least used clusters
            self.page_clusters = {p: i & 127 for i, p in enumerate(self.page_clusters.keys())}
        
        return cluster
    
    def _compute_address_entropy(self, page: int) -> float:
        """Compute address entropy estimate."""
        if page not in self.page_access_times:
            return 0.0
        
        # Simple entropy estimate based on access pattern
        access_time = self.page_access_times[page]
        if len(self.recent_pages) < 2:
            return 0.0
        
        # Ratio of recent accesses to this page
        recent_accesses = sum(1 for p in self.recent_pages[-64:] if p == page)
        return min(recent_accesses / 64.0, 1.0)
    
    def _update_byte_sketch(self, event: MicroEvent):
        """Update byte window sketch incrementally."""
        # Simple rolling sketch
        self._sketch = np.roll(self._sketch, 1)
        
        # Add new features
        if event.virtual_page is not None:
            self._sketch[0] = (event.virtual_page & 255) / 255.0
            self._sketch[1] = ((event.virtual_page >> 8) & 255) / 255.0
            self._sketch[2] = ((event.virtual_page >> 16) & 255) / 255.0
            self._sketch[3] = ((event.virtual_page >> 24) & 255) / 255.0
        
        if event.pc_bucket is not None:
            self._sketch[4] = (event.pc_bucket & 255) / 255.0
            self._sketch[5] = ((event.pc_bucket >> 8) & 255) / 255.0
        
        self._sketch[6] = (event.event_type & 15) / 15.0
        rw_flag = event.rw_flag if event.rw_flag is not None else 0
        self._sketch[7] = (rw_flag & 1) * 1.0
        self._sketch[8] = (event.pid & 255) / 255.0
        self._sketch[9] = (event.cpu_id & 7) / 7.0
        
        # Fill rest with zeros
        self._sketch[10:].fill(0.0)
    
    def to_tensors(self, features: ExtractedFeatures, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert features to tensors with minimal copying.
        
        v1.2: Direct numpy-to-tensor conversion with batch dimension for features.
        """
        return {
            'low8_bucket': torch.tensor([features.low8_bucket], device=device, dtype=torch.long),
            'high8_bucket': torch.tensor([features.high8_bucket], device=device, dtype=torch.long),
            'alignment_bucket': torch.tensor([features.alignment_bucket], device=device, dtype=torch.long),
            'small_int_bucket': torch.tensor([features.small_int_bucket], device=device, dtype=torch.long),
            'delta_bucket': torch.tensor([features.delta_bucket], device=device, dtype=torch.long),
            'hamming_bucket': torch.tensor([features.hamming_bucket], device=device, dtype=torch.long),
            'continuous_features': torch.from_numpy(features.continuous_features).unsqueeze(0).to(device),
            'bitfield_features': torch.from_numpy(features.bitfield_features).unsqueeze(0).to(device),
            'sketch_features': torch.from_numpy(features.sketch_features).unsqueeze(0).to(device),
            'page_hash_bucket': torch.tensor([features.page_hash_bucket], device=device, dtype=torch.long),
            'offset_bucket': torch.tensor([features.offset_bucket], device=device, dtype=torch.long),
            'cache_line_bucket': torch.tensor([features.cache_line_bucket], device=device, dtype=torch.long),
            'addr_alignment_bucket': torch.tensor([features.addr_alignment_bucket], device=device, dtype=torch.long),
            'stride_bucket': torch.tensor([features.stride_bucket], device=device, dtype=torch.long),
            'reuse_dist_bucket': torch.tensor([features.reuse_dist_bucket], device=device, dtype=torch.long),
            'locality_cluster': torch.tensor([features.locality_cluster], device=device, dtype=torch.long),
            'entropy_bucket': torch.tensor([features.entropy_bucket], device=device, dtype=torch.long),
            'address_flags': torch.from_numpy(features.address_flags).unsqueeze(0).to(device),
            'event_type': torch.tensor([features.event_type], device=device, dtype=torch.long),
            'fault_class': torch.tensor([features.fault_class], device=device, dtype=torch.long),
            'syscall_class': torch.tensor([features.syscall_class], device=device, dtype=torch.long),
            'opcode_family': torch.tensor([features.opcode_family], device=device, dtype=torch.long),
            'transition_type': torch.tensor([features.transition_type], device=device, dtype=torch.long),
            'result_class': torch.tensor([features.result_class], device=device, dtype=torch.long),
            'pte_flags': torch.from_numpy(features.pte_flags).unsqueeze(0).to(device),
            'vma_class': torch.tensor([features.vma_class], device=device, dtype=torch.long),
            'protection_domain': torch.tensor([features.protection_domain], device=device, dtype=torch.long),
            'read_count_bucket': torch.tensor([features.read_count_bucket], device=device, dtype=torch.long),
            'write_count_bucket': torch.tensor([features.write_count_bucket], device=device, dtype=torch.long),
            'fault_count_bucket': torch.tensor([features.fault_count_bucket], device=device, dtype=torch.long),
            'cow_count_bucket': torch.tensor([features.cow_count_bucket], device=device, dtype=torch.long),
            'recency_bucket': torch.tensor([features.recency_bucket], device=device, dtype=torch.long),
            'volatility_features': torch.from_numpy(features.volatility_features).unsqueeze(0).to(device),
            'pressure_features': torch.from_numpy(features.pressure_features).unsqueeze(0).to(device),
            'missingness_mask': torch.from_numpy(features.missingness_mask).unsqueeze(0).to(device),
            'freshness_ages': torch.from_numpy(features.freshness_ages).unsqueeze(0).to(device),
            'source_quality': torch.from_numpy(features.source_quality).unsqueeze(0).to(device),
            'conflict_score': torch.tensor([features.conflict_score], device=device, dtype=torch.float32),
            'consistency_score': torch.tensor([features.consistency_score], device=device, dtype=torch.float32),
        }
