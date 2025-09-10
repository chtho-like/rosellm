# Shared Weight Gradient Reduction

This module provides advanced gradient synchronization for models with shared weights, particularly for tied embeddings between input and output layers. It follows the Megatron-LM pattern for efficient distributed training of large language models.

## Key Features

### 1. Tied Weight Synchronization
- **Input/Output Embedding Sharing**: Automatically detects and synchronizes gradients for tied embeddings
- **Position Embedding Sharing**: Supports shared position embeddings across encoder/decoder
- **Custom Parameter Sharing**: Flexible API for arbitrary shared parameters

### 2. Advanced Error Handling
- **NaN/Inf Detection**: Automatic detection and handling of gradient overflow
- **Graceful Degradation**: Continues training even when errors occur
- **Comprehensive Logging**: Detailed error messages and warnings

### 3. Performance Optimizations
- **Gradient Bucketing**: Coalesces small gradients for efficient communication
- **Adaptive Strategies**: Dynamically selects optimal reduction strategy
- **Async Operations**: Support for non-blocking gradient reduction
- **Memory Efficiency**: Minimizes memory overhead through careful tensor management

### 4. Design Patterns
- **Strategy Pattern**: Pluggable reduction strategies (AllReduce, ReduceScatter, Hierarchical)
- **Factory Pattern**: Easy strategy creation and configuration
- **Observer Pattern**: Metrics collection and monitoring

## Architecture

```
SharedWeightGradientReducer
├── Configuration (SharedWeightConfig)
│   ├── Safety settings (NaN checks, gradient clipping)
│   ├── Performance settings (bucketing, async ops)
│   └── Strategy selection
├── Reduction Strategies
│   ├── AllReduceStrategy
│   ├── ReduceScatterStrategy
│   ├── HierarchicalReductionStrategy
│   └── AdaptiveReductionStrategy
├── Gradient Bucketing
│   ├── GradientBucketer
│   └── SmartGradientBucketer (adaptive sizing)
└── Metrics & Monitoring
    ├── ReductionMetrics
    └── Performance tracking
```

## Usage Examples

### Basic Usage with Tied Embeddings

```python
from rosellm.rosetrainer.gradient import (
    SharedWeightConfig,
    SharedWeightGradientReducer,
)

# Configure shared weight reduction
config = SharedWeightConfig(
    share_embeddings_and_output_weights=True,
    share_position_embeddings=True,
    max_gradient_norm=1000.0,
    check_for_nan=True,
    reduction_strategy="all_reduce",
)

# Create reducer
reducer = SharedWeightGradientReducer(config)

# During training loop
reducer.allreduce_word_embedding_grads([model])
reducer.allreduce_position_embedding_grads([model])
```

### Advanced Configuration

```python
# Configure with performance optimizations
config = SharedWeightConfig(
    share_embeddings_and_output_weights=True,
    
    # Safety features
    max_gradient_norm=100.0,
    check_for_nan=True,
    check_for_inf=True,
    skip_on_error=True,
    
    # Performance optimizations
    reduction_strategy="adaptive",
    coalesce_gradients=True,
    hierarchical_reduction=True,
    async_reduction=True,
    bucket_size_mb=50,
    
    # Gradient scaling
    gradient_predivide_factor=1.0,
    gradient_postdivide_factor=8.0,  # For 8-way data parallel
)
```

### Custom Shared Parameters

```python
# Define custom shared parameters
shared_params = [
    ("layer_norm.weight", model.layer_norm.weight),
    ("layer_norm.bias", model.layer_norm.bias),
]

# Reduce custom shared parameters
reducer.allreduce_shared_params(
    model=[model],
    shared_params=shared_params,
    reduce_group=custom_process_group,
)
```

### Integration with GradientFinalizer

```python
from rosellm.rosetrainer.gradient import GradientFinalizer

# The GradientFinalizer automatically uses SharedWeightGradientReducer
config = GradientFinalizationConfig(
    share_embeddings_and_output_weights=True,
    share_position_embeddings=True,
)

finalizer = GradientFinalizer(model, config)

# During training
finalizer.finalize_gradients(
    clip_gradients=True,
    check_finite=True,
    collect_stats=True,
)
```

## Reduction Strategies

### AllReduceStrategy
- **Best for**: Small to medium scale training
- **Pros**: Simple, well-tested, good performance for small tensors
- **Cons**: Can be inefficient for very large models

### ReduceScatterStrategy
- **Best for**: Memory-constrained training
- **Pros**: Lower memory usage per rank
- **Cons**: More complex, requires careful partitioning

### HierarchicalReductionStrategy
- **Best for**: Large-scale multi-node training
- **Pros**: Optimized for network topology, reduces inter-node traffic
- **Cons**: More complex setup, requires node-aware configuration

### AdaptiveReductionStrategy
- **Best for**: Dynamic workloads
- **Pros**: Automatically selects best strategy
- **Cons**: Small overhead for decision making

## Performance Considerations

### Gradient Bucketing
- Default bucket size: 25 MB
- Adjust based on network bandwidth and latency
- Larger buckets = fewer communications but higher latency
- Smaller buckets = more communications but lower latency

### Async Operations
- Enable for overlapping computation and communication
- Requires careful synchronization to avoid race conditions
- Best for models with independent parameter groups

### Memory Optimization
- Use FP16 compression for large models
- Enable gradient accumulation for batch size scaling
- Consider gradient checkpointing for very deep models

## Troubleshooting

### Common Issues

1. **NaN Gradients**
   - Enable `check_for_nan=True`
   - Reduce learning rate
   - Check for numerical instability in model

2. **Slow Communication**
   - Increase bucket size
   - Enable hierarchical reduction for multi-node
   - Check network configuration

3. **Memory Issues**
   - Enable gradient accumulation
   - Use smaller bucket sizes
   - Consider gradient checkpointing

### Debugging

Enable verbose logging:
```python
import logging
logging.getLogger("rosellm.rosetrainer.gradient").setLevel(logging.DEBUG)
```

Monitor reduction metrics:
```python
metrics = reducer.get_reduction_metrics()
print(f"Bytes reduced: {metrics.total_bytes_reduced}")
print(f"Time spent: {metrics.reduction_time_ms} ms")
print(f"Overflow detected: {metrics.overflow_detected}")
```

## Testing

Run tests with:
```bash
pytest tests/rosetrainer/gradient/test_shared_weight_reducer.py -v
```

For distributed tests:
```bash
torchrun --nproc_per_node=2 tests/test_distributed_shared_weights.py
```

## References

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)