"""
Core sidecar interface for AC-MoE-GA Systems Sidecar v1.3.

Provides the main entry point for using the sidecar in production.

v1.3 improvements:
- Outcome-quality metrics (override precision, regret, gain)
- Calibration tracking (ECE, calibration error)
- Per-head precision tracking
- Confidence/support statistics
- Regret tracking (heuristic vs sidecar)

v1.2 improvements (preserved):
- Optimized feature extraction with pre-allocated arrays
- Vectorized preprocessing with batch dimension support
- Reduced allocations and improved throughput
- Better handling of edge cases (None values)

v1.1 improvements (preserved):
- Better support/prototype formation
- Calibrated abstention behavior
- Sharper action separation with margins
- Balanced-tiny capacity profile (~1.4 MB)
"""

import torch
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import logging

from .config import SidecarConfig, BalancedBuildConfig
from .types import MicroEvent, Recommendation
from .model import ACMoEGAModel
from .inference import InferenceEngine, InferenceResult
from .runtime_state import RuntimeStateManager

logger = logging.getLogger(__name__)


class ACMoEGASidecar:
    """
    AC-MoE-GA Systems Sidecar v1.3
    
    A dense-first, byte-to-state runtime optimization model for systems
    and OS-level decision support.
    
    v1.3 improvements:
    - Outcome-quality metrics (override precision, regret, gain)
    - Calibration tracking (ECE, calibration error)
    - Per-head precision tracking
    - Confidence/support statistics
    - Regret tracking (heuristic vs sidecar)
    
    v1.2 improvements (preserved):
    - Optimized feature extraction with pre-allocated arrays
    - Vectorized preprocessing with batch dimension support
    - Reduced allocations and improved throughput
    - Better handling of edge cases (None values)
    
    v1.1 improvements (preserved):
    - Better support/prototype formation
    - Calibrated abstention behavior
    - Sharper action separation with margins
    - Balanced-tiny capacity profile (~1.4 MB)
    
    This is the main interface for using the sidecar. It handles:
    - Model initialization and loading
    - Event processing
    - Recommendation generation
    - State management
    
    Example usage:
        ```python
        # Initialize with default balanced build
        sidecar = ACMoEGASidecar()
        
        # Process events
        event = MicroEvent(
            timestamp_bucket=1000,
            cpu_id=0,
            numa_node=0,
            pid=1234,
            tid=1234,
            pc_bucket=0,
            event_type=0,  # MEMORY_READ
            opcode_class=0,
            virtual_page=0x7fff0000,
            region_id=1,
            rw_flag=False,
        )
        
        result = sidecar.process_event(event)
        if result is not None:
            print(f"Recommendation: {result.recommendation}")
        ```
    
    The sidecar is ADVISORY ONLY - it does not replace correctness-critical
    OS logic. All recommendations should be filtered by uncertainty and
    support criteria before being acted upon.
    """
    
    def __init__(
        self,
        config: Optional[SidecarConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the sidecar.
        
        Args:
            config: Configuration object. Uses balanced build if not provided.
            model_path: Path to saved model weights. Random init if not provided.
            device: Device to run on ('cuda', 'cpu', or None for auto-detect).
        """
        self.config = config or BalancedBuildConfig()
        self.config.validate()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing AC-MoE-GA Sidecar on {self.device}")
        
        # Initialize model
        self.model = ACMoEGAModel(self.config)
        
        # Load weights if provided
        if model_path is not None:
            self.load_model(model_path)
        
        # Initialize inference engine
        self.engine = InferenceEngine(self.config, self.model, self.device)
        
        logger.info(f"Model size: {self.model.get_model_size_mb():.2f} MB")
        logger.info(f"Parameter count: {self.model.get_parameter_count():,}")
    
    def process_event(self, event: MicroEvent) -> Optional[InferenceResult]:
        """
        Process a single micro-event.
        
        The sidecar will buffer events and run inference according to
        the configured cadence. Returns InferenceResult when inference
        is triggered, None otherwise.
        
        Args:
            event: The micro-event to process.
            
        Returns:
            InferenceResult if inference was triggered, None otherwise.
        """
        return self.engine.process_event(event)
    
    def process_batch(self, events: List[MicroEvent]) -> List[InferenceResult]:
        """
        Process a batch of micro-events.
        
        Args:
            events: List of micro-events to process.
            
        Returns:
            List of InferenceResults for each triggered inference.
        """
        return self.engine.process_batch(events)
    
    def force_inference(self) -> InferenceResult:
        """
        Force an inference regardless of cadence.
        
        Useful for getting a recommendation at specific decision points.
        
        Returns:
            InferenceResult with current recommendation.
        """
        return self.engine.force_inference()
    
    def get_recommendation(self) -> Recommendation:
        """
        Get the current recommendation without processing new events.
        
        Returns:
            Current recommendation based on buffered state.
        """
        return self.force_inference().recommendation
    
    def should_override_heuristic(self) -> bool:
        """
        Quick check if the sidecar recommends overriding the heuristic.
        
        Returns:
            True if override is recommended with sufficient confidence.
        """
        rec = self.get_recommendation()
        return rec.should_override_heuristic and not rec.abstain
    
    def get_statistics(self) -> Dict:
        """
        Get runtime statistics.
        
        Returns:
            Dictionary of statistics including event counts, inference
            counts, abstention rate, override rate, and memory usage.
        """
        return self.engine.get_statistics()
    
    def reset(self):
        """Reset all state for a fresh start."""
        self.engine.reset_state()
        logger.info("Sidecar state reset")
    
    def save_model(self, path: str):
        """
        Save model weights to file.
        
        Args:
            path: Path to save the model.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model weights from file.
        
        Args:
            path: Path to the saved model.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {path}")
    
    def export_onnx(self, path: str, batch_size: int = 1):
        """
        Export model to ONNX format for deployment.
        
        Args:
            path: Path to save the ONNX model.
            batch_size: Batch size for the exported model.
        """
        self.model.eval()
        
        # Create dummy inputs
        dummy_inputs = self._create_dummy_inputs(batch_size)
        
        torch.onnx.export(
            self.model,
            (dummy_inputs, None, None),
            path,
            input_names=['inputs'],
            output_names=['z_pred', 'uncertainty', 'should_abstain'],
            dynamic_axes={
                'inputs': {0: 'batch_size'},
                'z_pred': {0: 'batch_size'},
                'uncertainty': {0: 'batch_size'},
                'should_abstain': {0: 'batch_size'},
            },
            opset_version=14,
        )
        logger.info(f"Model exported to ONNX: {path}")
    
    def _create_dummy_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Create dummy inputs for export."""
        dims = self.config.latent_dims
        device = self.device
        
        return {
            'low8_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'high8_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'alignment_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'small_int_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'delta_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'hamming_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'continuous_features': torch.zeros(batch_size, 6, device=device),
            'bitfield_features': torch.zeros(batch_size, 16, device=device),
            'sketch_features': torch.zeros(batch_size, 32, device=device),
            'page_hash_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'offset_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'cache_line_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'addr_alignment_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'stride_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'reuse_dist_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'locality_cluster': torch.zeros(batch_size, dtype=torch.long, device=device),
            'entropy_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'address_flags': torch.zeros(batch_size, 5, device=device),
            'event_type': torch.zeros(batch_size, dtype=torch.long, device=device),
            'fault_class': torch.zeros(batch_size, dtype=torch.long, device=device),
            'syscall_class': torch.zeros(batch_size, dtype=torch.long, device=device),
            'opcode_family': torch.zeros(batch_size, dtype=torch.long, device=device),
            'transition_type': torch.zeros(batch_size, dtype=torch.long, device=device),
            'result_class': torch.zeros(batch_size, dtype=torch.long, device=device),
            'pte_flags': torch.zeros(batch_size, 11, device=device),
            'vma_class': torch.zeros(batch_size, dtype=torch.long, device=device),
            'protection_domain': torch.zeros(batch_size, dtype=torch.long, device=device),
            'read_count_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'write_count_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'fault_count_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'cow_count_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'recency_bucket': torch.zeros(batch_size, dtype=torch.long, device=device),
            'volatility_features': torch.zeros(batch_size, 4, device=device),
            'pressure_features': torch.zeros(batch_size, 12, device=device),
            'missingness_mask': torch.zeros(batch_size, 8, device=device),
            'freshness_ages': torch.ones(batch_size, 4, device=device),
            'source_quality': torch.ones(batch_size, 2, device=device),
            'conflict_score': torch.zeros(batch_size, device=device),
            'consistency_score': torch.ones(batch_size, device=device),
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"ACMoEGASidecar(\n"
            f"  device={self.device},\n"
            f"  model_size_mb={stats['model_size_mb']:.2f},\n"
            f"  runtime_memory_mb={stats['runtime_memory_mb']:.2f},\n"
            f"  total_events={stats['total_events']},\n"
            f"  total_inferences={stats['total_inferences']},\n"
            f")"
        )
