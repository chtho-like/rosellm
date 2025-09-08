# Distributed Optimizer Module

A production-ready distributed optimizer implementation with advanced memory optimization techniques for large-scale model training.

## Features

### Core Capabilities
- **Parameter Partitioning**: Distribute model parameters across data parallel ranks
- **Gradient Partitioning**: Shard gradients to reduce memory footprint
- **Optimizer State Partitioning**: Distribute optimizer states (momentum, variance) across ranks
- **Mixed Precision Training**: FP16/BF16 training with automatic loss scaling
- **Memory Profiling**: Real-time memory usage tracking and optimization recommendations
- **Factory Pattern**: Simplified optimizer creation with presets

### Advanced Features
- Thread-safe distributed operations with proper locking
- Gradient overflow detection and recovery
- Adaptive loss scaling for mixed precision
- Hierarchical communication patterns
- CPU offloading support
- Contiguous gradient buffers for efficient all-reduce

## Quick Start

### Basic Usage

```python
from rosellm.rosetrainer.optimizer import OptimizerFactory

# Create optimizer with preset configuration
optimizer = OptimizerFactory.create_from_model(
    model,
    optimizer_name="AdamW",
    lr=1e-4,
    preset="memory_efficient",  # Use memory-efficient configuration
    process_group=process_group
)

# Training loop
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Available Presets

- **`baseline`**: No optimizations (for comparison)
- **`memory_efficient`**: Maximum memory savings with parameter/gradient/state partitioning
- **`speed_optimized`**: Optimized for training speed with gradient partitioning only
- **`mixed_precision`**: FP16 training with automatic mixed precision
- **`cpu_offload`**: Offload optimizer states to CPU memory

### Custom Configuration

```python
from rosellm.rosetrainer.optimizer import (
    DistributedOptimizer,
    DistributedOptimizerConfig,
    PartitioningStrategy
)

# Create custom configuration
config = DistributedOptimizerConfig(
    partition_parameters=True,
    partition_gradients=True,
    partition_optimizer_states=True,
    partitioning_strategy=PartitioningStrategy.BALANCED,
    mixed_precision=True,
    grad_clip_value=1.0,
    check_gradients=True,
    verbose=True
)

# Create optimizer with custom config
optimizer = DistributedOptimizer(
    params=model.parameters(),
    optimizer_class=torch.optim.AdamW,
    optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.01},
    config=config,
    process_group=process_group
)
```

## Memory Profiling

Monitor and optimize memory usage during training:

```python
from rosellm.rosetrainer.optimizer import MemoryProfiler

# Initialize profiler
profiler = MemoryProfiler()
profiler.set_baseline()

# Training loop with profiling
for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    optimizer.step()
    
    # Profile memory every N steps
    if step % 100 == 0:
        profiler.record_snapshot()
        print(profiler.get_memory_summary())
        
        # Get optimization recommendations
        recommendations = profiler.optimize_memory()
        for rec_type, rec_text in recommendations.items():
            print(f"[{rec_type}]: {rec_text}")
```

## Architecture

### Class Hierarchy

```
DistributedOptimizer
├── DistributedOptimizerConfig  # Configuration
├── ParameterPartitioner        # Parameter distribution logic
│   └── ParameterRange          # Range mapping for each rank
├── OptimizerFactory            # Factory for creating optimizers
└── MemoryProfiler             # Memory tracking and optimization
```

### Communication Patterns

1. **Gradient Reduction**: All-reduce gradients across data parallel ranks
2. **Parameter Allgather**: Gather updated parameters after optimization
3. **Bucketed Communication**: Group small tensors for efficient all-reduce

## Performance Considerations

### Memory Savings

With full partitioning across N ranks:
- Parameters: 1/N memory per rank
- Gradients: 1/N memory per rank  
- Optimizer States: 1/N memory per rank
- Total Savings: ~67% for Adam/AdamW with 2 ranks

### Communication Overhead

- Gradient all-reduce: O(P) where P is parameter count
- Parameter allgather: O(P/N) per rank with partitioning
- Can overlap communication with computation when `overlap_grad_reduce=True`

## Best Practices

1. **Start with presets**: Use factory presets before custom configuration
2. **Profile first**: Use MemoryProfiler to understand memory bottlenecks
3. **Gradient accumulation**: Combine with gradient accumulation for larger effective batch sizes
4. **Mixed precision**: Enable for 2x memory savings with minimal accuracy loss
5. **Monitor overflows**: Check `optimizer.overflow_count` for training stability

## Troubleshooting

### High Memory Usage
- Enable parameter partitioning: `partition_parameters=True`
- Use mixed precision: `mixed_precision=True`
- Reduce batch size or enable gradient accumulation
- Enable CPU offloading: `cpu_offload=True`

### Gradient Overflows
- Reduce learning rate
- Increase loss scale growth interval
- Enable gradient clipping: `grad_clip_value=1.0`
- Check for numerical instabilities in model

### Slow Training
- Disable parameter partitioning if communication-bound
- Use larger gradient buckets: `bucket_size_mb=50`
- Enable communication overlap: `overlap_grad_reduce=True`
- Use hierarchical all-reduce for large clusters

## API Reference

See individual class docstrings for detailed API documentation:
- `DistributedOptimizer`: Main optimizer class
- `DistributedOptimizerConfig`: Configuration options
- `OptimizerFactory`: Factory methods for optimizer creation
- `MemoryProfiler`: Memory profiling utilities
- `ParameterPartitioner`: Parameter distribution logic