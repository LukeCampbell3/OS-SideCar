"""Margin vs Override curve for BitNet analysis."""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from ac_moe_ga_sidecar.config import BalancedBuildConfig
from ac_moe_ga_sidecar.bitnet_model import BitNetACMoEGAModel
from ac_moe_ga_sidecar.model import ACMoEGAModel
from ac_moe_ga_sidecar.utils import create_workload_trace
from ac_moe_ga_sidecar.inference import InferenceEngine


class TrackingInferenceEngine(InferenceEngine):
    """Inference engine that tracks detailed metrics."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_metrics = []
    
    def _run_inference(self):
        """Override to track metrics."""
        result = super()._run_inference()
        if result and result.recommendation:
            rec = result.recommendation
            self.tracked_metrics.append({
                'override': rec.should_override_heuristic,
                'margin': rec.action_margin if rec.action_margin is not None else 0.0,
                'margin_sigmoid': 1.0 / (1.0 + np.exp(-rec.action_margin)) if rec.action_margin is not None else 0.5,
            })
        return result


def collect_margin_override_data(seed: int = 42, num_events: int = 5000, workload_type: str = "mixed"):
    """Collect margin vs override data for both models."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = BalancedBuildConfig()
    workload = create_workload_trace(num_events, workload_type=workload_type)

    results = {}

    # Original model
    print("Collecting original model data...")
    original_engine = TrackingInferenceEngine(config, device=torch.device("cpu"))
    original_engine.model = ACMoEGAModel(config)

    for event in workload:
        original_engine.process_event(event)
    original_engine.force_inference()

    orig_override_count = sum(m['override'] for m in original_engine.tracked_metrics)
    results['original'] = {
        'margins': [m['margin'] for m in original_engine.tracked_metrics],
        'override': [m['override'] for m in original_engine.tracked_metrics],
        'override_rate': orig_override_count / len(original_engine.tracked_metrics) if original_engine.tracked_metrics else 0,
        'override_count': orig_override_count,
    }

    # BitNet model
    print("Collecting BitNet model data...")
    bitnet_engine = TrackingInferenceEngine(config, device=torch.device("cpu"))
    bitnet_engine.model = BitNetACMoEGAModel(config)

    for event in workload:
        bitnet_engine.process_event(event)
    bitnet_engine.force_inference()

    bitnet_override_count = sum(m['override'] for m in bitnet_engine.tracked_metrics)
    results['bitnet'] = {
        'margins': [m['margin'] for m in bitnet_engine.tracked_metrics],
        'override': [m['override'] for m in bitnet_engine.tracked_metrics],
        'override_rate': bitnet_override_count / len(bitnet_engine.tracked_metrics) if bitnet_engine.tracked_metrics else 0,
        'override_count': bitnet_override_count,
    }

    return results


def plot_margin_curve(results: dict, output_path: str = "margin_curve.png"):
    """Plot margin vs override curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram of margins
    ax1 = axes[0]
    
    if results['original']['margins']:
        orig_margins = np.array(results['original']['margins'])
        ax1.hist(orig_margins, bins=50, alpha=0.7, label=f"Original (n={len(orig_margins)})", color='blue')
    
    if results['bitnet']['margins']:
        bitnet_margins = np.array(results['bitnet']['margins'])
        ax1.hist(bitnet_margins, bins=50, alpha=0.7, label=f"BitNet (n={len(bitnet_margins)})", color='red')
    
    ax1.set_xlabel('Action Margin')
    ax1.set_ylabel('Count')
    ax1.set_title('Action Margin Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Margin vs Override scatter
    ax2 = axes[1]
    
    if results['original']['margins'] and results['original']['override']:
        orig_margins = np.array(results['original']['margins'])
        orig_override = np.array(results['original']['override'])
        ax2.scatter(orig_margins, orig_override, alpha=0.3, s=20, label='Original', color='blue')
    
    if results['bitnet']['margins'] and results['bitnet']['override']:
        bitnet_margins = np.array(results['bitnet']['margins'])
        bitnet_override = np.array(results['bitnet']['override'])
        ax2.scatter(bitnet_margins, bitnet_override, alpha=0.3, s=20, label='BitNet', color='red')
    
    ax2.set_xlabel('Action Margin')
    ax2.set_ylabel('Override (1) / No Override (0)')
    ax2.set_title('Margin vs Override Decision')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved margin curve plot to {output_path}")


if __name__ == "__main__":
    print("=== Margin vs Override Curve Analysis ===")
    print()
    
    results = collect_margin_override_data()
    
    print()
    print("=== Results ===")
    print(f"Original override rate: {results['original']['override_rate']*100:.2f}% ({results['original']['override_count']} overrides)")
    print(f"BitNet override rate: {results['bitnet']['override_rate']*100:.2f}% ({results['bitnet']['override_count']} overrides)")
    
    if results['original']['margins']:
        orig_margins = np.array(results['original']['margins'])
        print(f"Original margin: mean={orig_margins.mean():.4f}, std={orig_margins.std():.4f}")
    
    if results['bitnet']['margins']:
        bitnet_margins = np.array(results['bitnet']['margins'])
        print(f"BitNet margin: mean={bitnet_margins.mean():.4f}, std={bitnet_margins.std():.4f}")
    
    plot_margin_curve(results, output_path="margin_curve.png")
