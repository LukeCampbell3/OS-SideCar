#!/usr/bin/env python3
"""
Run ChampSim + AC-MoE-GA Sidecar co-simulation.

Usage:
    # Generate synthetic traces and run simulation
    python run_champsim_sim.py --synthetic

    # Run with a real ChampSim trace
    python run_champsim_sim.py --trace path/to/trace.champsimtrace.xz

    # Run with custom parameters
    python run_champsim_sim.py --synthetic --num-instructions 500000 --warmup 100000

    # Run all synthetic workloads
    python run_champsim_sim.py --synthetic --all-workloads

    # Save results to JSON
    python run_champsim_sim.py --synthetic --output results.json
"""

import argparse
import logging
import sys
from pathlib import Path

from champsim_integration.simulator import ChampSimSidecarSimulator, SimulatorConfig
from champsim_integration.cache_model import CacheHierarchyConfig
from champsim_integration.synthetic_traces import (
    SyntheticTraceConfig,
    generate_streaming_trace,
    generate_hotcold_trace,
    generate_phase_trace,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


SYNTHETIC_GENERATORS = {
    "streaming": ("Streaming sequential scan", generate_streaming_trace),
    "hotcold": ("Hot/cold working set", generate_hotcold_trace),
    "phase": ("Phase-changing workload", generate_phase_trace),
}


def run_synthetic(args):
    """Generate synthetic traces and run simulation."""
    trace_dir = Path("traces")
    trace_dir.mkdir(exist_ok=True)

    trace_cfg = SyntheticTraceConfig(
        num_instructions=args.num_instructions,
        seed=args.seed,
    )

    workloads = list(SYNTHETIC_GENERATORS.keys()) if args.all_workloads else [args.workload]
    all_metrics = []

    for workload_name in workloads:
        desc, generator = SYNTHETIC_GENERATORS[workload_name]
        trace_path = trace_dir / f"{workload_name}.champsimtrace"

        print()
        print(f"{'='*72}")
        print(f"  Workload: {desc}")
        print(f"{'='*72}")

        # Generate trace
        generator(trace_cfg, str(trace_path))

        # Run simulation
        sim_config = SimulatorConfig(
            trace_path=str(trace_path),
            max_instructions=args.num_instructions,
            warmup_instructions=args.warmup,
            fault_injection_rate=args.fault_rate,
            enable_prefetch=not args.no_prefetch,
            prefetch_degree=args.prefetch_degree,
            enable_online_learning=not args.no_online_learning,
            online_lr=args.online_lr,
            online_batch_size=args.online_batch_size,
            online_update_interval=args.online_update_interval,
            report_interval=args.report_interval,
        )

        sim = ChampSimSidecarSimulator(sim_config)
        metrics = sim.run()
        all_metrics.append(metrics)

    # Summary across workloads
    if len(all_metrics) > 1:
        print()
        print("=" * 72)
        print("  Cross-Workload Summary")
        print("=" * 72)
        print()
        print(f"  {'Workload':<15} {'BL LLC HR':>10} {'SC LLC HR':>10} {'Δ Latency':>10} {'Override':>10}")
        print(f"  {'-'*55}")
        for m in all_metrics:
            name = Path(m.trace_name).stem
            print(
                f"  {name:<15} "
                f"{m.baseline_llc_hit_rate:>10.4f} "
                f"{m.sidecar_llc_hit_rate:>10.4f} "
                f"{m.latency_improvement_pct:>+9.2f}% "
                f"{m.override_rate:>10.2%}"
            )
        print()

    # Save results
    if args.output:
        import json
        results = [m.to_dict() for m in all_metrics]
        Path(args.output).write_text(json.dumps(results, indent=2))
        logger.info(f"Results saved to {args.output}")


def run_trace(args):
    """Run simulation with a real ChampSim trace."""
    trace_path = Path(args.trace)
    if not trace_path.exists():
        logger.error(f"Trace file not found: {trace_path}")
        sys.exit(1)

    sim_config = SimulatorConfig(
        trace_path=str(trace_path),
        max_instructions=args.num_instructions,
        warmup_instructions=args.warmup,
        fault_injection_rate=args.fault_rate,
        enable_prefetch=not args.no_prefetch,
        prefetch_degree=args.prefetch_degree,
        enable_online_learning=not args.no_online_learning,
        online_lr=args.online_lr,
        online_batch_size=args.online_batch_size,
        online_update_interval=args.online_update_interval,
        report_interval=args.report_interval,
        output_path=args.output,
    )

    sim = ChampSimSidecarSimulator(sim_config)
    metrics = sim.run()
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="ChampSim + AC-MoE-GA Sidecar Co-Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="Use synthetic traces")
    mode.add_argument("--trace", type=str, help="Path to ChampSim trace file")

    # Synthetic options
    parser.add_argument("--workload", choices=list(SYNTHETIC_GENERATORS.keys()),
                        default="hotcold", help="Synthetic workload type (default: hotcold)")
    parser.add_argument("--all-workloads", action="store_true",
                        help="Run all synthetic workloads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Simulation options
    parser.add_argument("--num-instructions", type=int, default=200_000,
                        help="Number of instructions to simulate (default: 200K)")
    parser.add_argument("--warmup", type=int, default=50_000,
                        help="Warmup instructions (default: 50K)")
    parser.add_argument("--fault-rate", type=float, default=0.05,
                        help="Synthetic page fault injection rate (default: 0.05)")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Disable sidecar-guided prefetch")
    parser.add_argument("--prefetch-degree", type=int, default=2,
                        help="Prefetch degree (default: 2)")

    # Online learning
    parser.add_argument("--no-online-learning", action="store_true",
                        help="Disable online learning (model won't update during simulation)")
    parser.add_argument("--online-lr", type=float, default=1e-4,
                        help="Online learning rate (default: 1e-4)")
    parser.add_argument("--online-batch-size", type=int, default=32,
                        help="Online learning batch size (default: 32)")
    parser.add_argument("--online-update-interval", type=int, default=16,
                        help="Online learning update interval in inferences (default: 16)")

    parser.add_argument("--report-interval", type=int, default=100_000,
                        help="Progress report interval")

    # Output
    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    if args.synthetic:
        run_synthetic(args)
    else:
        run_trace(args)


if __name__ == "__main__":
    main()
