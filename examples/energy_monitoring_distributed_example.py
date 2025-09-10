#!/usr/bin/env python3
r"""
Distributed Energy Monitoring Example for RoseLLM

This example demonstrates distributed energy monitoring across multiple
processes and parallelism dimensions including:
- Multi-process energy aggregation
- Hierarchical energy reporting
- Integration with distributed training
- Cross-parallelism energy tracking

Usage:
    # Single process (simulation)
    python examples/energy_monitoring_distributed_example.py

    # Multi-process distributed
    torchrun --nproc_per_node=4 \
        examples/energy_monitoring_distributed_example.py --distributed

    # With specific parallelism
    torchrun --nproc_per_node=8 \
        examples/energy_monitoring_distributed_example.py \
        --distributed --tp=2 --dp=4
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.monitoring import (
    EnergyMonitor,
    EnergyMonitoringConfig,
    EnergyMonitoringMode,
)
from rosellm.rosetrainer.parallelism import initialize_model_parallel


def setup_distributed():
    """Set up distributed training environment."""
    # Initialize distributed process group
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )

        return local_rank
    else:
        return 0


def create_distributed_model(local_rank):
    """Create a model suitable for distributed training."""
    model = nn.Sequential(
        nn.Linear(2048, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    )

    if torch.cuda.is_available():
        model = model.cuda(local_rank)
        if dist.is_initialized():
            model = DDP(model, device_ids=[local_rank])

    return model


def distributed_training_simulation(model, local_rank, num_steps=10):
    """Simulate distributed training workload."""
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(
        f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] "
        f"Starting distributed training simulation on {device}"
    )

    for step in range(num_steps):
        # Generate synthetic batch
        batch_size = 16
        input_data = torch.randn(batch_size, 2048, device=device)
        target = torch.randn(batch_size, 1024, device=device)

        # Forward pass
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Synchronize gradients if distributed
        if dist.is_initialized():
            # DDP handles gradient synchronization automatically
            pass

        optimizer.step()

        if step % 3 == 0:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank {rank}] Step {step}: Loss = {loss.item():.4f}")

        # Add some computation variety
        time.sleep(0.1 + (step % 3) * 0.05)


def basic_distributed_example():
    """Basic distributed energy monitoring example."""
    print("\n=== Basic Distributed Energy Monitoring ===\n")

    # Set up distributed environment
    local_rank = setup_distributed()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    print(f"[Rank {rank}/{world_size}] Setting up distributed energy monitoring")

    # Create configuration for distributed mode
    config = EnergyMonitoringConfig()
    config.mode = EnergyMonitoringMode.DISTRIBUTED
    config.distributed.aggregation_interval = 2.0  # Aggregate every 2 seconds
    config.distributed.enable_hierarchical_reporting = True
    config.integration.log_interval = 3  # Log every 3 steps

    print(f"[Rank {rank}] Energy monitoring configuration:")
    print(f"  - Mode: {config.mode.value}")
    print(f"  - Aggregation interval: {config.distributed.aggregation_interval}s")
    hier_enabled = config.distributed.enable_hierarchical_reporting
    print(f"  - Hierarchical reporting: {hier_enabled}")

    # Create and configure energy monitor
    with EnergyMonitor(config) as monitor:
        print(f"[Rank {rank}] Energy monitoring started")

        # Create model
        model = create_distributed_model(local_rank)

        # Run distributed training
        distributed_training_simulation(model, local_rank, num_steps=8)

        # Get distributed energy statistics
        if rank == 0:  # Only print from rank 0 to avoid clutter
            print("\n=== Distributed Energy Statistics ===")
            stats = monitor.get_current_statistics()

            if stats.get("monitoring_active"):
                if "distributed_total_power_watts" in stats:
                    print(
                        f"Total power: {stats['distributed_total_power_watts']:.1f} W"
                    )

                if "distributed_total_energy_joules" in stats:
                    print(
                        f"Total energy: {stats['distributed_total_energy_joules']:.1f} J"
                    )

                if "distributed_process_count" in stats:
                    print(f"Number of processes: {stats['distributed_process_count']}")

                if "distributed_average_power_watts" in stats:
                    print(
                        f"Avg power/proc: {stats['distributed_average_power_watts']:.1f} W"
                    )

                # Hierarchical report
                if "hierarchical_report" in stats:
                    print("\nHierarchical Energy Report:")
                    hierarchical = stats["hierarchical_report"]

                    if "global" in hierarchical:
                        global_stats = hierarchical["global"]
                        print(
                            f"  Global: {global_stats.get('total_power_watts', 0):.1f} W, "
                            f"{global_stats.get('total_energy_joules', 0):.1f} J"
                        )

                    for dim in ["dp", "tp", "pp", "cp", "ep"]:
                        if dim in hierarchical:
                            dim_stats = hierarchical[dim]
                            print(
                                f"  {dim.upper()}: {dim_stats.get('power_contribution_watts', 0):.1f} W"
                            )

                # Efficiency metrics
                if "efficiency_metrics" in stats:
                    efficiency = stats["efficiency_metrics"]
                    print(f"\nEfficiency Metrics:")
                    print(
                        f"  Average power: {efficiency.get('average_power_watts', 0):.1f} W"
                    )
                    print(
                        f"  Peak power: {efficiency.get('peak_power_watts', 0):.1f} W"
                    )
                    print(
                        f"  Energy per second: {efficiency.get('energy_per_second_joules', 0):.1f} J/s"
                    )

        # Synchronize all processes before finishing
        if dist.is_initialized():
            dist.barrier()

        print(f"[Rank {rank}] Distributed training and monitoring completed")


def parallelism_aware_example(tp_size=1, dp_size=None):
    """Example with explicit parallelism dimensions."""
    print(f"\n=== Parallelism-Aware Energy Monitoring (TP={tp_size}) ===\n")

    local_rank = setup_distributed()

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Set up model parallelism if requested
        if tp_size > 1:
            try:
                # Calculate parallelism dimensions
                if dp_size is None:
                    dp_size = world_size // tp_size

                print(
                    f"[Rank {rank}] Initializing model parallel: TP={tp_size}, DP={dp_size}"
                )

                # Initialize model parallelism
                initialize_model_parallel(
                    tensor_model_parallel_size=tp_size,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=dp_size,
                )

                print(f"[Rank {rank}] Model parallelism initialized successfully")

            except Exception as e:
                print(f"[Rank {rank}] Could not initialize model parallelism: {e}")
                print("Falling back to data parallelism only")
    else:
        rank = 0
        print("Running in single-process mode")

    # Create hierarchical monitoring configuration
    config = EnergyMonitoringConfig()
    config.mode = EnergyMonitoringMode.HIERARCHICAL
    config.distributed.aggregation_interval = 1.5
    config.distributed.enable_hierarchical_reporting = True
    config.gpu_tracker.enable_detailed_metrics = True
    config.integration.log_interval = 2

    with EnergyMonitor(config) as monitor:
        print(f"[Rank {rank}] Hierarchical energy monitoring started")

        # Create model
        model = create_distributed_model(local_rank)

        # Run training workload
        distributed_training_simulation(model, local_rank, num_steps=6)

        # Generate comprehensive energy report
        if rank == 0:
            print("\n=== Comprehensive Energy Report ===")
            report = monitor.get_energy_report()

            print(
                f"Monitoring duration: {time.time() - report['timestamp']:.1f} seconds"
            )
            print(f"Total steps: {report.get('step_count', 0)}")
            print(f"Error count: {report.get('error_count', 0)}")

            # Current statistics
            current_stats = report.get("current_statistics", {})
            if "distributed_total_power_watts" in current_stats:
                print(
                    f"Current total power: {current_stats['distributed_total_power_watts']:.1f} W"
                )

            # Device information
            device_info = report.get("device_information", {})
            if device_info:
                print(f"\nDevice Information:")
                for device_id, info in device_info.items():
                    print(f"  Device {device_id}: {info}")

            # Recent measurements trend
            recent_summary = report.get("recent_measurements_summary", {})
            if recent_summary:
                print(f"\nRecent Measurements:")
                print(f"  Count: {recent_summary.get('measurement_count', 0)}")
                print(f"  Time span: {recent_summary.get('time_span_seconds', 0):.1f}s")

                power_trend = recent_summary.get("power_trend", [])
                if power_trend:
                    print(
                        f"  Power trend: {' -> '.join(f'{p:.1f}W' for p in power_trend)}"
                    )

        if dist.is_initialized():
            dist.barrier()


def energy_efficiency_analysis():
    """Demonstrate energy efficiency analysis."""
    print("\n=== Energy Efficiency Analysis ===\n")

    local_rank = setup_distributed()
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Configuration optimized for efficiency analysis
    config = EnergyMonitoringConfig.create_debug()  # Detailed monitoring
    config.distributed.aggregation_interval = 1.0  # Frequent aggregation

    with EnergyMonitor(config) as monitor:
        print(f"[Rank {rank}] Starting energy efficiency analysis")

        model = create_distributed_model(local_rank)

        # Run multiple phases with different workloads
        phases = [
            ("Warmup", 3, 0.05),
            ("Intensive", 4, 0.02),
            ("Cool-down", 3, 0.1),
        ]

        phase_energies = []

        for phase_name, steps, sleep_time in phases:
            print(f"[Rank {rank}] Starting phase: {phase_name}")

            # Get initial energy
            initial_stats = monitor.get_current_statistics()
            initial_energy = initial_stats.get("local_total_energy_joules", {})

            # Run phase workload
            for step in range(steps):
                batch_size = 16
                device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

                input_data = torch.randn(batch_size, 2048, device=device)
                with torch.no_grad():
                    _ = model(input_data)  # Forward pass for measurement

                time.sleep(sleep_time)

            # Get final energy for this phase
            final_stats = monitor.get_current_statistics()
            final_energy = final_stats.get("local_total_energy_joules", {})

            # Calculate phase energy delta
            if isinstance(initial_energy, dict) and isinstance(final_energy, dict):
                phase_energy = sum(final_energy.values()) - sum(initial_energy.values())
            else:
                phase_energy = (
                    final_energy - initial_energy if initial_energy else final_energy
                )

            phase_energies.append((phase_name, phase_energy))

            if rank == 0:
                print(f"  Phase {phase_name} energy: {phase_energy:.1f} J")

        # Final efficiency analysis
        if rank == 0:
            print("\n=== Efficiency Analysis Results ===")

            # Get efficiency metrics from current statistics
            current_stats = monitor.get_current_statistics()
            efficiency_metrics = current_stats.get("efficiency_metrics", {})
            if efficiency_metrics:
                print(
                    f"Average power: {efficiency_metrics.get('average_power_watts', 0):.1f} W"
                )
                print(
                    f"Peak power: {efficiency_metrics.get('peak_power_watts', 0):.1f} W"
                )
                print(
                    f"Power variance: {efficiency_metrics.get('power_variance', 0):.2f}"
                )
                print(
                    f"Energy per second: {efficiency_metrics.get('energy_per_second_joules', 0):.1f} J/s"
                )

            print("\nPhase-by-phase energy consumption:")
            for phase_name, energy in phase_energies:
                print(f"  {phase_name}: {energy:.1f} J")

            total_phase_energy = sum(energy for _, energy in phase_energies)
            print(f"Total phase energy: {total_phase_energy:.1f} J")


def main():
    """Main function to run examples based on arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed Energy Monitoring Examples"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run in distributed mode (use with torchrun)",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument(
        "--dp",
        type=int,
        default=None,
        help="Data parallelism size (auto-calculated if not specified)",
    )
    parser.add_argument(
        "--example",
        choices=["basic", "parallelism", "efficiency", "all"],
        default="all",
        help="Which example to run",
    )

    args = parser.parse_args()

    print("Distributed Energy Monitoring Examples")
    print("=" * 50)

    try:
        if args.example in ["basic", "all"]:
            basic_distributed_example()

        if args.example in ["parallelism", "all"]:
            parallelism_aware_example(tp_size=args.tp, dp_size=args.dp)

        if args.example in ["efficiency", "all"]:
            energy_efficiency_analysis()

        # Final cleanup
        if dist.is_initialized():
            rank = dist.get_rank()
            print(f"\n[Rank {rank}] All distributed examples completed!")
            dist.destroy_process_group()
        else:
            print("\nAll examples completed!")

        print("\nKey Takeaways:")
        print("- Distributed energy monitoring aggregates across all processes")
        print("- Hierarchical reporting shows energy by parallelism dimension")
        print("- Energy efficiency can be analyzed across different workload phases")
        print("- The system handles various parallelism configurations automatically")

    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] Example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
