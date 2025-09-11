# ChainedOptimizer Technical Specification

## Overview
ChainedOptimizer manages multiple optimizers for different parameter groups in distributed training scenarios, particularly for MoE models and multi-model training.

## Core Components

### 1. ChainedOptimizer Class
```python
class ChainedOptimizer(MegatronOptimizer):
    """
    Manages multiple optimizers for different model components.
    
    Key features:
    - Sequential execution of multiple optimizers
    - Unified interface for optimizer operations
    - State dict splitting and merging
    - Gradient statistics aggregation
    - Loss scale synchronization
    """
```

### 2. ProxyDict for State Management
```python
class ProxyDict:
    """
    Proxy dictionary that aggregates states from multiple optimizers.
    Maps (optimizer_idx, param_id) -> state for unified access.
    """
```

### 3. Integration Points

#### With Existing RoseLLM Components:
- **RoseTrainer**: Modified to support ChainedOptimizer initialization
- **Parallelism**: Integration with EP (Expert Parallel) groups
- **Gradient Utilities**: Unified gradient norm calculation across optimizers
- **Checkpointing**: State dict splitting/merging for distributed checkpoints

#### Memory Layout:
```
ChainedOptimizer
├── Dense Parameters Optimizer (DP+TP+PP)
│   ├── Model Chunk 0 (overlapped param gather)
│   └── Model Chunks 1-N (standard)
└── Expert Parameters Optimizer (EP+TP)
    └── All expert parameters
```

## API Design

### Initialization
```python
optimizer = ChainedOptimizer(
    optimizers=[dense_optimizer, expert_optimizer],
    config=optimizer_config
)
```

### Key Methods
- `zero_grad()`: Zeros gradients across all optimizers
- `step()`: Sequential step through all optimizers
- `get_loss_scale()`: Returns unified loss scale
- `state_dict()`: Merges states from all optimizers
- `load_state_dict()`: Splits and loads states to respective optimizers

## Performance Characteristics

### Memory Impact
- **Overhead**: ~2-5% additional memory for state management
- **Optimization**: Shared gradient buffers across optimizers
- **Scaling**: Linear with number of optimizers

### Communication Pattern
```
Expert Optimizer:     [EP Group All-Reduce]
Dense Optimizer:      [DP Group All-Reduce]
Synchronization:      [TP Group Broadcast]
```

## Compatibility Matrix

| Feature | Megatron-LM | RoseLLM Implementation |
|---------|-------------|----------------------|
| Multi-optimizer support | ✓ | ✓ |
| State dict splitting | ✓ | ✓ |
| Gradient norm aggregation | ✓ | ✓ |
| Expert parallel integration | ✓ | ✓ |
| Overlapped param gather | ✓ | ✓ |
| Dynamic optimizer creation | ✓ | ✓ |

## Testing Requirements

1. **Unit Tests**:
   - State dict save/load consistency
   - Gradient norm calculation accuracy
   - Zero grad functionality
   
2. **Integration Tests**:
   - Multi-GPU training convergence
   - Expert parallel integration
   - Memory efficiency validation

3. **Bit-to-bit Validation**:
   - Parameter updates match Megatron-LM
   - State dict format compatibility
   - Loss scale synchronization accuracy