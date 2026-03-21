"""A/B benchmark comparing original and BitNet models."""

import torch
import time
import numpy as np
from ac_moe_ga_sidecar.config import BalancedBuildConfig
from ac_moe_ga_sidecar.bitnet_model import BitNetACMoEGAModel
from ac_moe_ga_sidecar.model import ACMoEGAModel
from ac_moe_ga_sidecar.utils import create_workload_trace
from ac_moe_ga_sidecar.inference import InferenceEngine


def run_benchmark(seed: int = 42, num_events: int = 500, workload_type: str = "mixed"):
    """Run A/B benchmark with same seed and workload."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = BalancedBuildConfig()

    # Create both models (each with their own initialization)
    original_model = ACMoEGAModel(config)
    bitnet_model = BitNetACMoEGAModel(config)

    print("=== A/B Benchmark Setup ===")
    print(f"Original model size: {original_model.get_model_size_mb():.2f} MB")
    print(f"BitNet model size: {bitnet_model.get_model_size_mb():.2f} MB")
    print(f"Size reduction: {(1 - bitnet_model.get_model_size_mb() / original_model.get_model_size_mb()) * 100:.1f}%")
    print()

    # Create workload
    workload = create_workload_trace(num_events, workload_type=workload_type)
    print(f"Workload: {len(workload)} events ({workload_type})")
    print()

    # Test original model
    print("=== Original Model ===")
    original_engine = InferenceEngine(config, device=torch.device("cpu"))
    original_engine.model = original_model

    start_time = time.time()
    for event in workload:
        original_engine.process_event(event)
    original_engine.force_inference()
    original_wall_time = time.time() - start_time

    original_stats = original_engine.get_statistics()
    print(f"Wall time: {original_wall_time*1000:.1f} ms")
    print(f"Events/sec: {len(workload)/original_wall_time:.1f}")
    print(f"Inferences: {original_stats['total_inferences']}")
    print(f"Override rate: {original_stats['override_rate']*100:.1f}%")
    print(f"Abstention rate: {original_stats['abstention_rate']*100:.1f}%")
    print()

    # Test BitNet model
    print("=== BitNet Model ===")
    bitnet_engine = InferenceEngine(config, device=torch.device("cpu"))
    bitnet_engine.model = bitnet_model

    start_time = time.time()
    for event in workload:
        bitnet_engine.process_event(event)
    bitnet_engine.force_inference()
    bitnet_wall_time = time.time() - start_time

    bitnet_stats = bitnet_engine.get_statistics()
    print(f"Wall time: {bitnet_wall_time*1000:.1f} ms")
    print(f"Events/sec: {len(workload)/bitnet_wall_time:.1f}")
    print(f"Inferences: {bitnet_stats['total_inferences']}")
    print(f"Override rate: {bitnet_stats['override_rate']*100:.1f}%")
    print(f"Abstention rate: {bitnet_stats['abstention_rate']*100:.1f}%")
    print()

    # Compare key metrics
    print("=== Comparison ===")
    speedup = original_wall_time / bitnet_wall_time if bitnet_wall_time > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Override rate diff: {bitnet_stats['override_rate']*100 - original_stats['override_rate']*100:+.1f} pp")
    print(f"Abstention rate diff: {bitnet_stats['abstention_rate']*100 - original_stats['abstention_rate']*100:+.1f} pp")
    print(f"Size reduction: {(1 - bitnet_model.get_model_size_mb() / original_model.get_model_size_mb()) * 100:.1f}%")


if __name__ == "__main__":
    run_benchmark()
