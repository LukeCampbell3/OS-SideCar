"""Comprehensive behavioral A/B evaluation comparing original and BitNet models."""

import torch
import numpy as np
from collections import defaultdict
from ac_moe_ga_sidecar.config import BalancedBuildConfig
from ac_moe_ga_sidecar.bitnet_model import BitNetACMoEGAModel
from ac_moe_ga_sidecar.model import ACMoEGAModel
from ac_moe_ga_sidecar.utils import create_workload_trace
from ac_moe_ga_sidecar.inference import InferenceEngine


class TrackingInferenceEngine(InferenceEngine):
    """Inference engine that tracks detailed metrics for A/B comparison."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_metrics = defaultdict(list)
    
    def _run_inference(self):
        """Override to track metrics."""
        result = super()._run_inference()
        if result and result.recommendation:
            rec = result.recommendation
            self.tracked_metrics['override'].append(rec.should_override_heuristic)
            self.tracked_metrics['abstain'].append(rec.abstain)
            self.tracked_metrics['confidence'].append(rec.inferred_state.confidence)
            self.tracked_metrics['support_density'].append(rec.support_density)
            self.tracked_metrics['drift_score'].append(rec.drift_score)
            if rec.action_margin is not None:
                self.tracked_metrics['action_margin'].append(rec.action_margin)
        return result


def run_behavioral_eval(seed: int = 42, num_events: int = 5000, workload_type: str = "mixed"):
    """Run comprehensive behavioral A/B evaluation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = BalancedBuildConfig()

    # Create both models
    original_model = ACMoEGAModel(config)
    bitnet_model = BitNetACMoEGAModel(config)

    print("=== Behavioral A/B Evaluation ===")
    print(f"Workload: {num_events} events ({workload_type})")
    print(f"Seed: {seed}")
    print()

    # Create workload
    workload = create_workload_trace(num_events, workload_type=workload_type)

    # Evaluate original model
    print("=== Original Model ===")
    original_engine = TrackingInferenceEngine(config, device=torch.device("cpu"))
    original_engine.model = original_model

    for event in workload:
        original_engine.process_event(event)
    original_engine.force_inference()

    original_stats = original_engine.get_statistics()
    original_metrics = original_engine.tracked_metrics

    print(f"Override rate: {original_stats['override_rate']*100:.2f}%")
    print(f"Abstention rate: {original_stats['abstention_rate']*100:.2f}%")
    print(f"Total inferences: {original_stats['total_inferences']}")
    print_metrics_summary("Original", original_metrics)

    # Evaluate BitNet model
    print()
    print("=== BitNet Model ===")
    bitnet_engine = TrackingInferenceEngine(config, device=torch.device("cpu"))
    bitnet_engine.model = bitnet_model

    for event in workload:
        bitnet_engine.process_event(event)
    bitnet_engine.force_inference()

    bitnet_stats = bitnet_engine.get_statistics()
    bitnet_metrics = bitnet_engine.tracked_metrics

    print(f"Override rate: {bitnet_stats['override_rate']*100:.2f}%")
    print(f"Abstention rate: {bitnet_stats['abstention_rate']*100:.2f}%")
    print(f"Total inferences: {bitnet_stats['total_inferences']}")
    print_metrics_summary("BitNet", bitnet_metrics)

    # Compare key metrics
    print()
    print("=== Comparison ===")
    print_metric_diff("Override rate", 
                      original_metrics['override'], 
                      bitnet_metrics['override'])
    print_metric_diff("Confidence", 
                      original_metrics['confidence'], 
                      bitnet_metrics['confidence'])
    print_metric_diff("Support density", 
                      original_metrics['support_density'], 
                      bitnet_metrics['support_density'])
    print_metric_diff("Drift score", 
                      original_metrics['drift_score'], 
                      bitnet_metrics['drift_score'])
    if original_metrics['action_margin'] and bitnet_metrics['action_margin']:
        print_metric_diff("Action margin", 
                          original_metrics['action_margin'], 
                          bitnet_metrics['action_margin'])


def print_metrics_summary(prefix: str, metrics: dict):
    """Print summary statistics for tracked metrics."""
    for key, values in metrics.items():
        if values:
            arr = np.array(values)
            print(f"{prefix} {key}: mean={arr.mean():.3f}, std={arr.std():.3f}, min={arr.min():.3f}, max={arr.max():.3f}")


def print_metric_diff(prefix: str, original: list, bitnet: list):
    """Print difference between original and BitNet metrics."""
    if not original or not bitnet:
        print(f"{prefix}: N/A (no data)")
        return
    
    orig_arr = np.array(original)
    bit_arr = np.array(bitnet)
    
    diff_mean = bit_arr.mean() - orig_arr.mean()
    diff_std = np.sqrt(bit_arr.var() + orig_arr.var())
    
    print(f"{prefix} diff: {diff_mean:+.3f} (orig={orig_arr.mean():.3f}, bitnet={bit_arr.mean():.3f})")


if __name__ == "__main__":
    run_behavioral_eval()
