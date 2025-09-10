#!/usr/bin/env python3
"""
Energy Monitoring Integration Example for RoseLLM

This example demonstrates integration of energy monitoring with:
- RoseTrainer integration
- Configuration management
- Report generation and export
- Production deployment patterns
- Advanced monitoring scenarios

Usage:
    python examples/energy_monitoring_integration_example.py
"""

import json
import os
import tempfile
import time

import torch
import torch.nn as nn

from rosellm.rosetrainer.monitoring import (
    EnergyMonitor,
    EnergyMonitoringConfig,
    EnergyMonitoringMode,
    FallbackStrategy,
    create_debug_monitor,
    create_production_monitor,
)


class MockTrainer:
    """Mock trainer class to demonstrate integration."""

    def __init__(self):
        self.hooks = {}
        self.step_hooks = []
        self.current_step = 0

    def add_hook(self, event, callback):
        """Add event hook."""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(callback)

    def add_step_hook(self, callback):
        """Add step hook."""
        self.step_hooks.append(callback)

    def trigger_hook(self, event):
        """Trigger event hooks."""
        if event in self.hooks:
            for callback in self.hooks[event]:
                callback()

    def step(self):
        """Simulate training step."""
        self.current_step += 1
        for callback in self.step_hooks:
            callback(self.current_step)

    def train(self, num_steps):
        """Simulate training process."""
        print(f"Starting training for {num_steps} steps...")

        self.trigger_hook("before_train")

        try:
            for _ in range(num_steps):
                # Simulate forward/backward pass
                time.sleep(0.1)
                self.step()

                # Simulate occasional checkpointing
                if self.current_step % 5 == 0:
                    self.trigger_hook("before_checkpoint")
                    time.sleep(0.2)  # Simulate checkpoint saving
                    self.trigger_hook("after_checkpoint")

                # Simulate occasional evaluation
                if self.current_step % 7 == 0:
                    self.trigger_hook("before_eval")
                    time.sleep(0.15)  # Simulate evaluation
                    self.trigger_hook("after_eval")

        finally:
            self.trigger_hook("after_train")

        print("Training completed!")


def configuration_management_example():
    """Demonstrate comprehensive configuration management."""
    print("=== Configuration Management Example ===\n")

    # 1. Default configuration
    print("1. Default Configuration:")
    default_config = EnergyMonitoringConfig.create_default()
    print(f"   Mode: {default_config.mode.value}")
    print(f"   Sampling interval: {default_config.gpu_tracker.sampling_interval}s")
    print(f"   Auto-start: {default_config.auto_start}")

    # 2. Production configuration
    print("\n2. Production Configuration:")
    prod_config = EnergyMonitoringConfig.create_production()
    print(f"   Sampling interval: {prod_config.gpu_tracker.sampling_interval}s")
    print(f"   Detailed metrics: {prod_config.gpu_tracker.enable_detailed_metrics}")
    print(f"   Save measurements: {prod_config.integration.save_measurements}")
    print(f"   Log interval: {prod_config.integration.log_interval}")

    # 3. Debug configuration
    print("\n3. Debug Configuration:")
    debug_config = EnergyMonitoringConfig.create_debug()
    print(f"   Sampling interval: {debug_config.gpu_tracker.sampling_interval}s")
    print(f"   Log level: {debug_config.integration.log_level}")
    print(f"   Log interval: {debug_config.integration.log_interval}")

    # 4. Custom configuration with validation
    print("\n4. Custom Configuration:")
    try:
        custom_config = EnergyMonitoringConfig()
        custom_config.mode = EnergyMonitoringMode.LOCAL_ONLY
        custom_config.gpu_tracker.sampling_interval = 0.5
        custom_config.gpu_tracker.fallback_strategy = FallbackStrategy.ZERO
        custom_config.gpu_tracker.devices = (
            [0, 1] if torch.cuda.device_count() > 1 else [0]
        )
        custom_config.distributed.aggregation_interval = 3.0
        custom_config.integration.log_interval = 25
        custom_config.integration.output_directory = "/tmp/energy_monitoring"

        print(f"   Custom mode: {custom_config.mode.value}")
        print(f"   Custom devices: {custom_config.gpu_tracker.devices}")
        print(
            f"   Custom fallback: {custom_config.gpu_tracker.fallback_strategy.value}"
        )

        # Validate configuration
        custom_config.validate()
        print("   ✓ Configuration validation passed")

    except Exception as e:
        print(f"   ✗ Configuration validation failed: {e}")

    # 5. Configuration serialization
    print("\n5. Configuration Serialization:")
    config_dict = custom_config.to_dict()
    print(f"   Serialized to dict with {len(config_dict)} top-level keys")

    # Round-trip test
    restored_config = EnergyMonitoringConfig.from_dict(config_dict)
    print(f"   Restored mode: {restored_config.mode.value}")
    print(
        f"   Restored sampling interval: {restored_config.gpu_tracker.sampling_interval}s"
    )

    # Clone test
    cloned_config = custom_config.clone()
    cloned_config.gpu_tracker.sampling_interval = 2.0
    print(
        f"   Original interval after cloning: {custom_config.gpu_tracker.sampling_interval}s"
    )
    print(f"   Cloned interval: {cloned_config.gpu_tracker.sampling_interval}s")


def trainer_integration_example():
    """Demonstrate integration with trainer."""
    print("\n\n=== Trainer Integration Example ===\n")

    # Create configuration for trainer integration
    config = EnergyMonitoringConfig()
    config.integration.integrate_with_trainer = True
    config.integration.log_interval = 3
    config.integration.pause_during_checkpointing = True
    config.integration.pause_during_evaluation = True
    config.gpu_tracker.sampling_interval = 0.5

    print("Configuration:")
    print(f"- Trainer integration: {config.integration.integrate_with_trainer}")
    print(f"- Log interval: {config.integration.log_interval} steps")
    print(
        f"- Pause during checkpointing: {config.integration.pause_during_checkpointing}"
    )
    print(f"- Pause during evaluation: {config.integration.pause_during_evaluation}")

    # Create energy monitor and mock trainer
    monitor = EnergyMonitor(config)
    trainer = MockTrainer()

    # Integrate monitor with trainer
    print("\nIntegrating energy monitor with trainer...")
    integration_success = monitor.integrate_with_trainer(trainer)

    if integration_success:
        print("✓ Integration successful")

        # Verify hooks were added
        print(f"Added hooks: {list(trainer.hooks.keys())}")
        print(f"Step hooks: {len(trainer.step_hooks)} registered")

        # Run simulated training
        print("\nRunning training with integrated energy monitoring:")
        trainer.train(num_steps=10)

        # Get final statistics
        print("\nFinal energy statistics:")
        stats = monitor.get_current_statistics()
        if stats.get("monitoring_active"):
            if "local_current_power_watts" in stats:
                power = stats["local_current_power_watts"]
                print(f"- Current power: {power}")

            if "local_total_energy_joules" in stats:
                energy = stats["local_total_energy_joules"]
                print(f"- Total energy: {energy}")

    else:
        print("✗ Integration failed")


def report_generation_example():
    """Demonstrate report generation and export."""
    print("\n\n=== Report Generation and Export Example ===\n")

    # Create monitor with save enabled
    config = EnergyMonitoringConfig()
    config.integration.save_measurements = True
    config.gpu_tracker.sampling_interval = 0.3
    config.integration.log_interval = 2

    with tempfile.TemporaryDirectory() as temp_dir:
        config.integration.output_directory = temp_dir

        print(f"Output directory: {temp_dir}")

        with EnergyMonitor(config) as monitor:
            # Generate some activity for monitoring
            print("Generating workload for energy monitoring...")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = nn.Sequential(
                nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 512)
            ).to(device)

            # Run workload
            for step in range(6):
                batch = torch.randn(32, 1024, device=device)
                with torch.no_grad():
                    _ = model(batch)  # Forward pass for measurement

                # Log at intervals
                monitor.log_energy_statistics(step=step)
                time.sleep(0.2)

            print("Workload completed. Generating reports...")

            # 1. Get comprehensive report
            report = monitor.get_energy_report()
            print(f"\nGenerated report with {len(report)} sections:")
            for key in report.keys():
                print(f"- {key}")

            # 2. Save reports in different formats
            formats = ["json", "csv"]  # Skip parquet if pandas not available

            for fmt in formats:
                try:
                    filename = f"energy_report.{fmt}"
                    filepath = os.path.join(temp_dir, filename)

                    success = monitor.save_energy_report(filepath, format=fmt)
                    if success:
                        file_size = os.path.getsize(filepath)
                        print(
                            f"✓ Saved {fmt.upper()} report: {filename} ({file_size} bytes)"
                        )

                        # Show sample of JSON content
                        if fmt == "json":
                            with open(filepath, "r") as f:
                                data = json.load(f)
                            print(f"  Sample fields: {list(data.keys())[:5]}")
                    else:
                        print(f"✗ Failed to save {fmt.upper()} report")

                except Exception as e:
                    print(f"✗ Error saving {fmt.upper()} report: {e}")

            # 3. Auto-generated timestamped report
            auto_success = monitor.save_energy_report()  # Auto-generated filename
            if auto_success:
                print("✓ Saved auto-timestamped report")

        # List all generated files
        print(f"\nGenerated files in {temp_dir}:")
        for file in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, file)
            size = os.path.getsize(filepath)
            print(f"- {file} ({size} bytes)")


def production_patterns_example():
    """Demonstrate production deployment patterns."""
    print("\n\n=== Production Deployment Patterns ===\n")

    # 1. Using factory functions
    print("1. Factory Functions:")

    # Production monitor
    prod_monitor = create_production_monitor()
    print(
        f"   Production monitor: {prod_monitor.config.gpu_tracker.sampling_interval}s sampling"
    )
    print(
        f"   Detailed metrics: {prod_monitor.config.gpu_tracker.enable_detailed_metrics}"
    )

    # Debug monitor
    debug_monitor = create_debug_monitor()
    print(
        f"   Debug monitor: {debug_monitor.config.gpu_tracker.sampling_interval}s sampling"
    )
    print(f"   Log level: {debug_monitor.config.integration.log_level}")

    # 2. Environment-based configuration
    print("\n2. Environment-Based Configuration:")

    # Simulate environment variables
    env_vars = {
        "ROSE_ENERGY_MODE": "local",
        "ROSE_ENERGY_SAMPLING_INTERVAL": "2.0",
        "ROSE_ENERGY_LOG_INTERVAL": "100",
        "ROSE_ENERGY_FALLBACK_POWER": "200.0",
    }

    print("   Simulated environment variables:")
    for key, value in env_vars.items():
        print(f"     {key}={value}")

    # Apply environment overrides (simulation)
    env_config = EnergyMonitoringConfig()
    for key, value in env_vars.items():
        if key == "ROSE_ENERGY_MODE":
            env_config.mode = EnergyMonitoringMode.LOCAL_ONLY
        elif key == "ROSE_ENERGY_SAMPLING_INTERVAL":
            env_config.gpu_tracker.sampling_interval = float(value)
        elif key == "ROSE_ENERGY_LOG_INTERVAL":
            env_config.integration.log_interval = int(value)
        elif key == "ROSE_ENERGY_FALLBACK_POWER":
            env_config.gpu_tracker.fallback_power_estimate = float(value)

    print(f"   Applied config - Mode: {env_config.mode.value}")
    print(f"   Applied config - Sampling: {env_config.gpu_tracker.sampling_interval}s")

    # 3. Conditional monitoring
    print("\n3. Conditional Monitoring:")

    # Check system capabilities
    has_cuda = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if has_cuda else 0

    print(f"   CUDA available: {has_cuda}")
    print(f"   GPU device count: {device_count}")

    # Configure based on system
    if has_cuda and device_count > 0:
        print("   → Using GPU monitoring with NVML")
        system_config = EnergyMonitoringConfig()
        system_config.gpu_tracker.devices = list(
            range(min(device_count, 4))
        )  # Limit to 4 GPUs
    else:
        print("   → Using CPU fallback mode")
        system_config = EnergyMonitoringConfig()
        system_config.mode = EnergyMonitoringMode.LOCAL_ONLY
        system_config.gpu_tracker.fallback_strategy = FallbackStrategy.ESTIMATE
        system_config.gpu_tracker.fallback_power_estimate = 100.0  # Lower for CPU

    # 4. Error handling and recovery
    print("\n4. Error Handling and Recovery:")

    robust_config = EnergyMonitoringConfig()
    robust_config.gpu_tracker.max_consecutive_errors = 5
    robust_config.gpu_tracker.error_recovery_delay = 2.0
    robust_config.gpu_tracker.fallback_strategy = FallbackStrategy.ESTIMATE

    print(
        f"   Max consecutive errors: {robust_config.gpu_tracker.max_consecutive_errors}"
    )
    print(f"   Recovery delay: {robust_config.gpu_tracker.error_recovery_delay}s")
    print(f"   Fallback strategy: {robust_config.gpu_tracker.fallback_strategy.value}")

    with EnergyMonitor(robust_config) as monitor:
        print("   ✓ Robust monitoring started successfully")

        # Simulate some work
        time.sleep(0.5)

        # Check if monitoring is active
        if monitor.is_monitoring():
            print("   ✓ Monitoring is active and healthy")
        else:
            print("   ⚠ Monitoring is not active (may be paused or failed)")


def advanced_scenarios_example():
    """Demonstrate advanced monitoring scenarios."""
    print("\n\n=== Advanced Monitoring Scenarios ===\n")

    # 1. Multi-phase training with different monitoring needs
    print("1. Multi-Phase Training:")

    config = EnergyMonitoringConfig()
    config.gpu_tracker.sampling_interval = 0.5

    with EnergyMonitor(config) as monitor:
        phases = [
            ("Data Loading", 0.3, True),  # Pause during data loading
            ("Forward Pass", 0.1, False),  # Monitor during computation
            ("Backward Pass", 0.1, False),  # Monitor during computation
            ("Optimizer Step", 0.2, False),  # Monitor during optimization
        ]

        for phase_name, duration, should_pause in phases:
            print(
                f"   Phase: {phase_name} ({'paused' if should_pause else 'monitored'})"
            )

            if should_pause:
                with monitor.paused():
                    time.sleep(duration)
            else:
                time.sleep(duration)
                # Log energy for active phases
                stats = monitor.get_current_statistics()
                if "local_current_power_watts" in stats:
                    power = stats["local_current_power_watts"]
                    if isinstance(power, dict):
                        total_power = sum(power.values())
                    else:
                        total_power = power
                    print(f"     Power during {phase_name}: {total_power:.1f}W")

    # 2. Conditional monitoring based on workload
    print("\n2. Conditional Monitoring:")

    workloads = [
        ("Light", 0.1, True),  # Enable monitoring for light workload
        ("Heavy", 0.05, True),  # Enable monitoring for heavy workload
        ("Idle", 0.5, False),  # Disable monitoring during idle
    ]

    for workload_name, compute_time, enable_monitoring in workloads:
        print(
            f"   Workload: {workload_name} ({'monitored' if enable_monitoring else 'not monitored'})"
        )

        if enable_monitoring:
            config = EnergyMonitoringConfig()
            config.gpu_tracker.sampling_interval = compute_time * 2  # Adaptive sampling

            with EnergyMonitor(config) as monitor:
                # Simulate workload
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x = torch.randn(64, 1024, device=device)

                for _ in range(5):
                    with torch.no_grad():
                        _ = torch.mm(x, x.T)  # Matrix multiply for load
                    time.sleep(compute_time)

                # Get final statistics
                stats = monitor.get_current_statistics()
                energy = stats.get("local_total_energy_joules", {})
                print(f"     Energy consumed: {energy}")
        else:
            # Just simulate idle time without monitoring
            time.sleep(0.2)

    print("\n✓ All advanced scenarios completed")


def main():
    """Run all integration examples."""
    print("Energy Monitoring Integration Examples")
    print("=" * 60)

    try:
        configuration_management_example()
        trainer_integration_example()
        report_generation_example()
        production_patterns_example()
        advanced_scenarios_example()

        print("\n" + "=" * 60)
        print("Integration Examples Summary:")
        print("✓ Configuration management and validation")
        print("✓ Trainer integration with hooks")
        print("✓ Report generation and export")
        print("✓ Production deployment patterns")
        print("✓ Advanced monitoring scenarios")

        print("\nBest Practices:")
        print("- Use factory functions for common configurations")
        print("- Leverage environment variables for deployment flexibility")
        print("- Integrate with trainer lifecycle for automatic management")
        print("- Use pause/resume for non-computational phases")
        print("- Configure appropriate error handling for production")
        print("- Export reports for post-training analysis")

    except Exception as e:
        print(f"\nIntegration example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
