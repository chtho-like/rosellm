# RoseLLM Distributed Optimizer

A production-ready distributed optimizer implementation with advanced features for efficient large-scale training.

## Features

### Core Capabilities
- **Gradient Bucketing**: Efficient gradient aggregation with configurable bucket sizes
- **Asynchronous Communication**: Overlapped gradient reduction with backward computation
- **Parameter Partitioning**: Multiple strategies for optimizer state sharding
- **Performance Monitoring**: Comprehensive metrics collection and analysis
- **Error Handling**: Robust exception handling with specific error types
- **Thread Safety**: Proper locking mechanisms for concurrent operations

### Partitioning Strategies
- **Round Robin**: Balanced parameter distribution across ranks
- **Size Balanced**: Memory-optimized distribution based on parameter sizes
- **Layer Wise**: Consecutive parameter grouping for model parallelism

### Performance Optimizations
- Batched parameter broadcasting for reduced communication overhead
- Memory-efficient gradient buffer management
- Dynamic bucket size optimization
- Communication-computation overlap metrics

## Usage

### Basic Example
```python
from rosellm.rosetrainer.optimizer import DistributedOptimizer
import torch.optim as optim

# Create base optimizer
base_optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Wrap with distributed optimizer
optimizer = DistributedOptimizer(
    base_optimizer,
    models=model,
    bucket_size_mb=25.0,
    overlap_grad_reduce=True,
    partition_optimizer_states=True,
    partitioning_strategy="size_balanced",
    gradient_accumulation_steps=4,
    clip_grad_norm=1.0,
    enable_metrics=True,
)

# Training loop
for batch in dataloader:
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Get performance metrics
    stats = optimizer.get_statistics()
    print(f"Step metrics: {stats['performance']}")
```

### Advanced Configuration
```python
from rosellm.rosetrainer.optimizer import (
    DistributedOptimizer,
    PartitioningStrategyFactory,
    validate_bucket_configuration,
)

# Validate bucket configuration
params = list(model.parameters())
validation = validate_bucket_configuration(params, bucket_size_mb=50.0)
print(f"Bucket efficiency: {validation['efficiency']:.2%}")

# Custom partitioning strategy
strategy = PartitioningStrategyFactory.create("layer_wise")
partitions = strategy.partition(params, world_size=4)

# Create optimizer with custom configuration
optimizer = DistributedOptimizer(
    base_optimizer,
    models=model,
    bucket_size_mb=validation['target_bucket_size_mb'],
    partitioning_strategy="layer_wise",
    enable_metrics=True,
)
```

## Architecture

### Module Structure
```
optimizer/
├── distributed_optimizer.py    # Main optimizer wrapper
├── gradient_buffer.py         # Gradient bucketing and reduction
├── partitioning_strategies.py # Parameter partitioning strategies
├── metrics.py                 # Performance monitoring
├── optimizer_utils.py         # Utility functions
└── exceptions.py              # Custom exception types
```

### Design Patterns
- **Strategy Pattern**: Flexible parameter partitioning strategies
- **Factory Pattern**: Strategy creation and registration
- **Observer Pattern**: Performance metrics collection
- **Thread Safety**: Lock-based synchronization for concurrent access

## Performance Considerations

### Memory Optimization
- Gradient bucketing reduces memory fragmentation
- Parameter partitioning enables optimizer state sharding
- Configurable bucket sizes for memory-bandwidth tradeoff

### Communication Efficiency
- Asynchronous all-reduce operations
- Batched parameter broadcasting
- Communication-computation overlap metrics

### Monitoring and Debugging
- Comprehensive performance metrics
- Moving average statistics
- Memory usage tracking
- Communication efficiency analysis

## Error Handling

The module provides specific exception types for better debugging:
- `ConfigurationError`: Invalid optimizer configuration
- `GradientBufferError`: Gradient buffer operations
- `CommunicationError`: Distributed communication failures
- `SynchronizationError`: Rank synchronization issues
- `PartitioningError`: Parameter partitioning errors

## Testing

Comprehensive test coverage including:
- Unit tests for all components
- Integration tests with PyTorch optimizers
- Mock-based distributed testing
- Performance benchmarking
- Error handling validation

## Requirements

- PyTorch >= 1.10.0
- Python >= 3.8
- NCCL (for GPU communication)
- Gloo (for CPU communication)