# Next Mini-Feature Recommendation: Custom Gradient Scaler

## Feature Selection Rationale

After analyzing the Megatron-LM reference implementation and RoseLLM's current state, I recommend implementing **Custom Gradient Scalers with Dynamic Loss Scaling** as the next mini-feature.

### Why This Feature?

1. **High Impact**: Critical for stable FP16 training of large models
2. **Right Size**: ~400 lines of core implementation (fits PR size constraints)
3. **Clear Patterns**: Follows established Megatron-LM design with clear API
4. **Testable**: Enables bit-to-bit validation against reference implementation
5. **Foundation**: Unlocks future features like precision-aware optimizers

## Implementation Overview

### Core Components

```python
# File: rosellm/rosetrainer/optimizer/grad_scaler.py

class AbstractGradScaler(ABC):
    """Base class matching Megatron-LM interface"""
    - scale property
    - inv_scale property  
    - update(found_inf) method
    - state_dict/load_state_dict

class ConstantGradScaler(AbstractGradScaler):
    """Fixed loss scale (no adjustments)"""
    
class DynamicGradScaler(AbstractGradScaler):
    """Adaptive scaling with hysteresis"""
    - Backs off on gradient overflow
    - Grows scale after successful iterations
    - Configurable hysteresis and growth intervals
```

### Key Implementation Details

1. **Dynamic Scaling Algorithm** (from Megatron-LM):
   - On overflow: Decrement hysteresis counter, scale down if depleted
   - On success: Increment growth counter, scale up at interval
   - Scale bounds: Clamp between min_scale and max_scale

2. **Integration Points**:
   - Replace PyTorch's GradScaler in MixedPrecisionManager
   - Add to gradient_utils.py overflow checking
   - Support in RoseTrainer engine configuration

3. **Distributed Compatibility**:
   - Scale synchronization across data parallel ranks
   - Works with gradient accumulation
   - Compatible with all parallelism dimensions

## Validation Strategy

### 1. Unit Tests
```python
# tests/rosetrainer/optimizer/test_grad_scaler.py
- Test constant scaler immutability
- Test dynamic scaler backoff with hysteresis
- Test dynamic scaler growth intervals
- Test state persistence
```

### 2. Bit-to-Bit Validation
```python
# tests/rosetrainer/optimizer/test_megatron_parity.py
- Compare scale evolution over 10,000 iterations
- Verify identical behavior on overflow sequences
- Validate state dict compatibility
```

### 3. End-to-End Example
```python
# examples/gradient_scaler_example.py
- Train small transformer with FP16
- Demonstrate overflow recovery
- Show scale adaptation over time
- Compare convergence with/without custom scaler
```

## Testing on Limited Hardware

With 2 GPUs available:

```bash
# CPU Testing (most coverage)
python -m pytest tests/rosetrainer/optimizer/test_grad_scaler.py -v

# GPU Testing (2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    examples/gradient_scaler_example.py

# Distributed CPU simulation (32 processes)
torchrun --nproc_per_node=32 examples/gradient_scaler_example.py \
    --backend=gloo --device=cpu
```

## Implementation Checklist

### Phase 1: Core (4-6 hours)
- [ ] Create optimizer/ directory structure
- [ ] Implement AbstractGradScaler base class
- [ ] Implement ConstantGradScaler
- [ ] Implement DynamicGradScaler with hysteresis
- [ ] Add GradScalerConfig dataclass

### Phase 2: Integration (2-3 hours)
- [ ] Update gradient_utils.py for overflow checking
- [ ] Integrate with MixedPrecisionManager
- [ ] Add configuration to RoseTrainer
- [ ] Handle distributed synchronization

### Phase 3: Testing (3-4 hours)
- [ ] Write comprehensive unit tests
- [ ] Implement Megatron parity tests
- [ ] Create end-to-end training example
- [ ] Validate on GPU and CPU backends

### Phase 4: Documentation (1-2 hours)
- [ ] Add detailed docstrings
- [ ] Update CLAUDE.md with new feature
- [ ] Create usage examples
- [ ] Document migration from PyTorch GradScaler

## Expected Outcomes

1. **Improved Training Stability**: Better FP16 convergence for large models
2. **Megatron-LM Compatibility**: Matching behavior enables model migration
3. **Performance**: < 0.1% overhead with potential training speedup
4. **Foundation**: Enables future precision-aware optimizer implementation

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Device compatibility | Test on both CPU and GPU backends |
| Distributed deadlocks | Use timeout in distributed tests |
| Numerical divergence | Extensive validation against reference |
| API breaking changes | Maintain backward compatibility layer |

## Code Quality Standards

- Type hints on all public methods
- 100% test coverage for core logic
- Docstrings following NumPy style
- Pass all linting (black, isort, mypy)
- No IDE warnings or errors

## Next Steps After This Feature

Once gradient scalers are implemented, the following features become feasible:
1. **Precision-Aware Optimizers**: Lower precision optimizer states
2. **Per-Layer Scaling**: Different scales for different layers
3. **Zero Gradient Counting**: Statistics for debugging
4. **Advanced FP8 Support**: Delayed scaling strategies

## Files to Create/Modify

### New Files
- `/data/projects/rosellm/rosellm/rosetrainer/optimizer/__init__.py`
- `/data/projects/rosellm/rosellm/rosetrainer/optimizer/grad_scaler.py`
- `/data/projects/rosellm/rosellm/rosetrainer/optimizer/scaler_config.py`
- `/data/projects/rosellm/tests/rosetrainer/optimizer/test_grad_scaler.py`
- `/data/projects/rosellm/tests/rosetrainer/optimizer/test_megatron_parity.py`
- `/data/projects/rosellm/examples/gradient_scaler_example.py`

### Modified Files
- `/data/projects/rosellm/rosellm/rosetrainer/utils/gradient_utils.py`
- `/data/projects/rosellm/rosellm/rosetrainer/memory/mixed_precision.py`
- `/data/projects/rosellm/rosellm/rosetrainer/engine.py`
- `/data/projects/rosellm/rosellm/rosetrainer/config.py`

This feature provides excellent value: manageable scope, high impact, and clear validation path against Megatron-LM reference implementation.