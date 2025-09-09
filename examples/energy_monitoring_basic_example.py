#!/usr/bin/env python3
"""
Basic Energy Monitoring Example for RoseLLM

This example demonstrates basic usage of the energy monitoring system
including:
- Simple energy monitoring setup
- Context manager usage
- Getting current statistics
- Basic configuration options

Usage:
    python examples/energy_monitoring_basic_example.py
"""

import time

import torch
import torch.nn as nn

from rosellm.rosetrainer.monitoring import (
    EnergyMonitor,
    EnergyMonitoringConfig,
    EnergyMonitoringMode,
)


def create_simple_model():
    """Create a simple model for demonstration."""
    return nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
    )


def simple_training_loop(model, device, num_steps=10):
    """Simple training loop for demonstration."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    print(f"Running {num_steps} training steps on {device}...")

    for step in range(num_steps):
        # Generate random data
        batch_size = 32
        input_data = torch.randn(batch_size, 1024, device=device)
        target = torch.randn(batch_size, 512, device=device)

        # Forward pass
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        if step % 2 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")

        # Small delay to see energy changes
        time.sleep(0.1)


def basic_energy_monitoring_example():
    """Demonstrate basic energy monitoring."""
    print("=== Basic Energy Monitoring Example ===\n")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cpu":
        print("Note: Running on CPU - energy monitoring will use fallback estimates")

    # Create a simple model
    model = create_simple_model()
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Basic energy monitoring with default configuration
    print("\n1. Basic energy monitoring with default settings:")

    with EnergyMonitor() as monitor:
        print("   Energy monitoring started automatically (context manager)")

        # Run some computation
        simple_training_loop(model, device, num_steps=5)

        # Get current statistics
        stats = monitor.get_current_statistics()
        print(f"\n   Current energy statistics:")

        if stats["monitoring_active"]:
            if "local_current_power_watts" in stats:
                power = stats["local_current_power_watts"]
                if isinstance(power, dict):
                    total_power = sum(power.values())
                    print(f"   - Total power consumption: {total_power:.1f} watts")
                    for device_id, device_power in power.items():
                        print(
                            f"   - Device {device_id} power: {device_power:.1f} watts"
                        )
                else:
                    print(f"   - Power consumption: {power:.1f} watts")

            if "local_total_energy_joules" in stats:
                energy = stats["local_total_energy_joules"]
                if isinstance(energy, dict):
                    total_energy = sum(energy.values())
                    print(f"   - Total energy consumed: {total_energy:.1f} joules")
                else:
                    print(f"   - Energy consumed: {energy:.1f} joules")
        else:
            print("   - Monitoring not active")

    print("   Energy monitoring stopped automatically (context manager)")


def custom_configuration_example():
    """Demonstrate custom configuration options."""
    print("\n\n=== Custom Configuration Example ===\n")

    # Create custom configuration for faster sampling
    config = EnergyMonitoringConfig()
    config.mode = EnergyMonitoringMode.LOCAL_ONLY
    config.gpu_tracker.sampling_interval = 0.5  # Sample every 0.5 seconds
    config.gpu_tracker.enable_detailed_metrics = True
    config.integration.log_interval = 2  # Log every 2 steps

    print("Custom configuration:")
    print(f"- Mode: {config.mode.value}")
    print(f"- Sampling interval: {config.gpu_tracker.sampling_interval}s")
    print(f"- Detailed metrics: {config.gpu_tracker.enable_detailed_metrics}")
    print(f"- Log interval: {config.integration.log_interval} steps")

    # Create monitor with custom config
    monitor = EnergyMonitor(config)

    # Manual start/stop instead of context manager
    print("\nStarting energy monitoring manually...")
    success = monitor.start_monitoring()

    if success:
        print("Energy monitoring started successfully")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_simple_model()

        # Run training with step-based logging
        print("\nRunning training with energy logging:")
        simple_training_loop(model, device, num_steps=6)

        # Get detailed statistics
        print("\nDetailed energy statistics:")
        stats = monitor.get_current_statistics()

        for key, value in stats.items():
            if key.startswith("device_") and key.endswith("_stats"):
                device_id = key.split("_")[1]
                print(f"\nDevice {device_id} statistics:")
                for stat_key, stat_value in value.items():
                    print(f"  {stat_key}: {stat_value}")

        # Generate comprehensive report
        print("\nGenerating energy report...")
        report = monitor.get_energy_report()
        print(f"Report timestamp: {report['timestamp']}")
        print(f"Total steps monitored: {report['step_count']}")
        print(f"Error count: {report['error_count']}")

        # Stop monitoring
        final_results = monitor.stop_monitoring()
        print(f"\nFinal energy results: {final_results}")
    else:
        print("Failed to start energy monitoring")


def pause_resume_example():
    """Demonstrate pause/resume functionality."""
    print("\n\n=== Pause/Resume Example ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_simple_model()

    # Create monitor
    monitor = EnergyMonitor()

    with monitor:
        print("Phase 1: Training with energy monitoring")
        simple_training_loop(model, device, num_steps=3)

        stats_after_phase1 = monitor.get_current_statistics()
        energy_after_phase1 = stats_after_phase1.get("local_total_energy_joules", {})
        print(f"Energy after phase 1: {energy_after_phase1}")

        # Pause monitoring during some operation
        print("\nPausing energy monitoring for data loading simulation...")
        with monitor.paused():
            print("  Simulating data loading (monitoring paused)")
            time.sleep(1.0)  # Simulate long data loading
            print("  Data loading complete")

        print("Resumed energy monitoring")

        print("\nPhase 2: More training with resumed monitoring")
        simple_training_loop(model, device, num_steps=3)

        final_stats = monitor.get_current_statistics()
        final_energy = final_stats.get("local_total_energy_joules", {})
        print(f"Final energy consumption: {final_energy}")


def error_handling_example():
    """Demonstrate error handling."""
    print("\n\n=== Error Handling Example ===\n")

    # Create configuration with aggressive error handling
    config = EnergyMonitoringConfig()
    config.gpu_tracker.max_consecutive_errors = 3
    config.gpu_tracker.error_recovery_delay = 1.0
    config.gpu_tracker.fallback_power_estimate = 200.0  # Conservative estimate

    print("Configuration with error handling:")
    print(f"- Max consecutive errors: {config.gpu_tracker.max_consecutive_errors}")
    print(f"- Error recovery delay: {config.gpu_tracker.error_recovery_delay}s")
    print(f"- Fallback power estimate: {config.gpu_tracker.fallback_power_estimate}W")

    monitor = EnergyMonitor(config)

    print("\nStarting monitoring (will handle errors gracefully):")
    success = monitor.start_monitoring()

    if success:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_simple_model()

        try:
            # This should work even if NVML has issues
            simple_training_loop(model, device, num_steps=5)

            # Get statistics (should work with fallback if needed)
            stats = monitor.get_current_statistics()
            print(f"\nMonitoring active: {stats.get('monitoring_active', False)}")

            if "local_current_power_watts" in stats:
                power = stats["local_current_power_watts"]
                print(f"Current power (may be estimated): {power}")

        except Exception as e:
            print(f"Error occurred but monitoring continued: {e}")

        finally:
            results = monitor.stop_monitoring()
            print(f"Monitoring completed. Error count: {results.get('error_count', 0)}")
    else:
        print("Could not start monitoring")


def main():
    """Run all examples."""
    print("Energy Monitoring System Examples")
    print("=" * 50)

    try:
        # Run examples
        basic_energy_monitoring_example()
        custom_configuration_example()
        pause_resume_example()
        error_handling_example()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nTips:")
        print("- Use context managers for automatic start/stop")
        print("- Customize sampling intervals based on your needs")
        print("- Enable detailed metrics for development, disable for production")
        print("- Use pause/resume during non-computational phases")
        print("- The system gracefully handles NVML unavailability")

    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
