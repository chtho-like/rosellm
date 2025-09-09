# Microbatch Calculator Implementation Plan for RoseLLM

## Executive Summary

The Microbatch Calculator is a critical component for efficient distributed training that dynamically manages the number of microbatches based on global batch size, data parallel size, and pipeline parallel configuration. This feature is essential for:
- Pipeline parallelism efficiency
- Gradient accumulation optimization
- Memory management with dynamic batch sizing
- Training stability with batch size rampup

## Feature Overview

### What is a Microbatch Calculator?

The Microbatch Calculator determines how to split a global batch into smaller microbatches that can be processed sequentially through pipeline stages or accumulated for gradient updates. It handles:

1. **Constant Batch Size Mode**: Fixed number of microbatches throughout training
2. **Rampup Mode**: Gradually increase batch size from a smaller value to target
3. **Dynamic Adjustment**: Ensure divisibility constraints are met
4. **Memory Optimization**: Adjust microbatch size based on available memory

### Why This Feature?

- **Megatron-LM Parity**: Core feature in Megatron-LM for efficient training
- **Pipeline Parallelism**: Essential for efficient pipeline scheduling
- **Memory Management**: Enables training larger models with gradient accumulation
- **Training Stability**: Batch size rampup improves convergence for large batch training
- **Small Scope**: ~400-500 lines of core implementation

## Technical Specification

### Architecture Overview

```
rosellm/rosetrainer/
├── parallelism/
│   ├── microbatch_calculator.py  # Main implementation
│   └── __init__.py               # Export calculator functions
├── config/
│   └── training_config.py        # Add microbatch configuration
└── engine.py                     # Integration point
```

### Core Components

#### 1. Base Calculator Class
```python
class MicrobatchCalculator(ABC):
    """Base class for microbatch calculation strategies."""
    
    def __init__(self):
        self.num_micro_batches: Optional[int] = None
        self.current_global_batch_size: Optional[int] = None
        self.micro_batch_size: Optional[int] = None
        self.current_running_global_batch_size: Optional[int] = None
    
    @abstractmethod
    def update(self, consumed_samples: int, 
               consistency_check: bool = True,
               verbose: bool = False) -> None:
        """Update microbatch count based on training progress."""
        pass
    
    def get(self) -> int:
        """Get current number of microbatches."""
        return self.num_micro_batches
    
    def get_micro_batch_size(self) -> int:
        """Get current micro batch size."""
        return self.micro_batch_size
```

#### 2. Constant Batch Size Calculator
```python
class ConstantMicrobatchCalculator(MicrobatchCalculator):
    """Calculator for constant global batch size."""
    
    def __init__(self,
                 global_batch_size: int,
                 micro_batch_size: int,
                 data_parallel_size: int,
                 pipeline_parallel_size: int = 1,
                 decrease_batch_size_if_needed: bool = False):
        # Validate and compute microbatch count
        # Handle divisibility constraints
        # Support automatic adjustment
```

#### 3. Rampup Batch Size Calculator
```python
class RampupBatchsizeCalculator(MicrobatchCalculator):
    """Calculator with batch size rampup schedule."""
    
    def __init__(self,
                 target_global_batch_size: int,
                 micro_batch_size: int,
                 data_parallel_size: int,
                 pipeline_parallel_size: int,
                 start_global_batch_size: int,
                 batch_size_increment: int,
                 rampup_samples: int,
                 decrease_batch_size_if_needed: bool = False):
        # Implement gradual batch size increase
        # Track consumed samples
        # Update microbatch count dynamically
```

#### 4. Global State Management
```python
# Global calculator instance
_GLOBAL_MICROBATCH_CALCULATOR: Optional[MicrobatchCalculator] = None

def init_microbatch_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    pipeline_parallel_size: int = 1,
    decrease_batch_size_if_needed: bool = False
) -> None:
    """Initialize the global microbatch calculator."""
    global _GLOBAL_MICROBATCH_CALCULATOR
    # Create appropriate calculator based on configuration
    # Log initialization details on rank 0

def get_num_microbatches() -> int:
    """Get current number of microbatches."""
    return _GLOBAL_MICROBATCH_CALCULATOR.get()

def update_num_microbatches(consumed_samples: int) -> None:
    """Update microbatch count based on training progress."""
    _GLOBAL_MICROBATCH_CALCULATOR.update(consumed_samples)
```

### Integration Points

#### 1. Training Engine Integration
```python
# In rosellm/rosetrainer/engine.py
class RoseTrainer:
    def __init__(self, ...):
        # Initialize microbatch calculator
        if self.config.use_microbatch_calculator:
            init_microbatch_calculator(
                rank=self.rank,
                rampup_batch_size=self.config.rampup_batch_size,
                global_batch_size=self.config.global_batch_size,
                micro_batch_size=self.config.micro_batch_size,
                data_parallel_size=get_data_parallel_world_size(),
                pipeline_parallel_size=get_pipeline_model_parallel_world_size()
            )
    
    def train_step(self, batch):
        # Get current microbatch configuration
        num_microbatches = get_num_microbatches()
        micro_batch_size = get_micro_batch_size()
        
        # Split batch into microbatches
        microbatches = self._split_batch(batch, num_microbatches)
        
        # Process microbatches with gradient accumulation
        for i, microbatch in enumerate(microbatches):
            # Forward/backward with gradient scaling
            loss = self.forward_backward(microbatch)
            
        # Update microbatch calculator
        self.consumed_samples += get_current_global_batch_size()
        update_num_microbatches(self.consumed_samples)
```

#### 2. Pipeline Parallelism Integration
```python
# In rosellm/rosetrainer/parallelism/pipeline_parallel.py
class PipelineParallel:
    def schedule_microbatches(self):
        """Schedule microbatches through pipeline stages."""
        num_microbatches = get_num_microbatches()
        # Use calculator's microbatch count for scheduling
```

#### 3. Data Parallel Integration
```python
# In rosellm/rosetrainer/parallelism/data_parallel.py
class DataParallelWrapper:
    def configure_gradient_accumulation(self):
        """Configure gradient accumulation steps."""
        num_microbatches = get_num_microbatches()
        # Set accumulation steps based on microbatches
```

## Memory and Performance Analysis

### Memory Impact
- **Constant Mode**: No additional memory overhead
- **Rampup Mode**: Minimal state tracking (~100 bytes)
- **Benefits**: Enables larger effective batch sizes through accumulation

### Performance Characteristics
- **Computation**: O(1) for all operations
- **Communication**: No additional communication required
- **Synchronization**: Optional barrier only at initialization

### Comparison with Megatron-LM
| Feature | Megatron-LM | RoseLLM (Proposed) |
|---------|-------------|-------------------|
| Constant batch size | ✓ | ✓ |
| Batch size rampup | ✓ | ✓ |
| Auto-adjustment | ✓ | ✓ |
| Pipeline integration | ✓ | ✓ |
| Global state management | ✓ | ✓ |
| Multi-dimensional parallel | ✓ | ✓ (Enhanced) |

## Implementation Milestones

### Phase 1: Core Implementation (Day 1)
1. Create `microbatch_calculator.py` with base classes
2. Implement `ConstantMicrobatchCalculator`
3. Add global state management functions
4. Write unit tests for constant mode

### Phase 2: Advanced Features (Day 2)
1. Implement `RampupBatchsizeCalculator`
2. Add divisibility constraint handling
3. Implement auto-adjustment logic
4. Write unit tests for rampup mode

### Phase 3: Integration (Day 3)
1. Integrate with `RoseTrainer` engine
2. Update pipeline parallel scheduling
3. Modify gradient accumulation logic
4. Create integration tests

### Phase 4: Validation (Day 4)
1. Create end-to-end example
2. Implement bit-to-bit validation against Megatron-LM
3. Performance benchmarking
4. Documentation and examples

## Testing Strategy

### Unit Tests
```python
# tests/rosetrainer/parallelism/test_microbatch_calculator.py

def test_constant_calculator_initialization():
    """Test constant calculator with various configurations."""
    calc = ConstantMicrobatchCalculator(
        global_batch_size=512,
        micro_batch_size=8,
        data_parallel_size=8
    )
    assert calc.get() == 8  # 512 / (8 * 8)

def test_rampup_calculator_progression():
    """Test batch size rampup over training steps."""
    calc = RampupBatchsizeCalculator(
        target_global_batch_size=512,
        micro_batch_size=8,
        data_parallel_size=8,
        pipeline_parallel_size=1,
        start_global_batch_size=128,
        batch_size_increment=128,
        rampup_samples=1000
    )
    # Test progression at different sample counts
    calc.update(0)
    assert calc.get_current_global_batch_size() == 128
    calc.update(500)
    assert calc.get_current_global_batch_size() == 256
    calc.update(1000)
    assert calc.get_current_global_batch_size() == 512

def test_divisibility_constraints():
    """Test automatic batch size adjustment."""
    calc = ConstantMicrobatchCalculator(
        global_batch_size=500,  # Not divisible by 8*8
        micro_batch_size=8,
        data_parallel_size=8,
        decrease_batch_size_if_needed=True
    )
    assert calc.get_current_running_global_batch_size() == 448  # Rounded down
```

### Integration Tests
```python
# tests/integration/test_microbatch_integration.py

def test_pipeline_with_microbatches():
    """Test pipeline parallelism with microbatch calculator."""
    # Initialize parallel state
    initialize_model_parallel(tp=1, pp=2, dp=2)
    
    # Setup calculator
    init_microbatch_calculator(
        rank=get_rank(),
        rampup_batch_size=None,
        global_batch_size=64,
        micro_batch_size=8,
        data_parallel_size=2,
        pipeline_parallel_size=2
    )
    
    # Run pipeline schedule
    pipeline = PipelineParallel(...)
    pipeline.run_forward_backward()
    
    # Verify microbatch processing
    assert get_num_microbatches() == 4

def test_gradient_accumulation_with_calculator():
    """Test gradient accumulation with dynamic microbatches."""
    # Test that gradients are properly accumulated
    # across the calculated number of microbatches
```

### End-to-End Example
```python
# examples/microbatch_calculator_example.py

import torch
from rosellm.rosetrainer import RoseTrainer
from rosellm.rosetrainer.parallelism import (
    init_microbatch_calculator,
    get_num_microbatches,
    update_num_microbatches
)

def main():
    # Configuration with batch size rampup
    config = TrainingConfig(
        global_batch_size=512,
        micro_batch_size=8,
        rampup_batch_size=[128, 128, 10000],  # Start at 128, increment by 128, over 10k samples
        use_microbatch_calculator=True
    )
    
    # Initialize trainer
    trainer = RoseTrainer(model, optimizer, config)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Calculator automatically determines microbatch count
            loss = trainer.train_step(batch)
            
            # Log current configuration
            print(f"Microbatches: {get_num_microbatches()}")
            print(f"Global batch: {get_current_global_batch_size()}")
    
    print("Training completed with dynamic microbatch calculation!")

if __name__ == "__main__":
    main()
```

### Bit-to-Bit Validation
```python
# validation/validate_against_megatron.py

def validate_microbatch_calculator():
    """Validate RoseLLM calculator against Megatron-LM."""
    
    # Test configurations from Megatron-LM
    test_cases = [
        # (global_batch, micro_batch, dp_size, expected_microbatches)
        (512, 8, 8, 8),
        (1024, 4, 16, 16),
        (256, 2, 32, 4),
    ]
    
    for global_bs, micro_bs, dp_size, expected in test_cases:
        # RoseLLM calculator
        rosellm_calc = ConstantMicrobatchCalculator(
            global_batch_size=global_bs,
            micro_batch_size=micro_bs,
            data_parallel_size=dp_size
        )
        
        # Compare results
        assert rosellm_calc.get() == expected
        print(f"✓ Test passed: {global_bs}/{micro_bs}/{dp_size} = {expected}")
    
    # Test rampup behavior
    validate_rampup_behavior()
    
    print("All validation tests passed!")
```

## Success Metrics

1. **Functional Correctness**
   - All unit tests pass
   - Integration tests with pipeline/data parallel pass
   - Bit-to-bit match with Megatron-LM behavior

2. **Performance**
   - Zero overhead for calculator operations (<1μs per call)
   - No additional memory allocation during training
   - Seamless integration with existing parallelism

3. **Usability**
   - Simple API matching Megatron-LM patterns
   - Clear documentation and examples
   - Automatic configuration validation

4. **Robustness**
   - Handles edge cases (batch size not divisible)
   - Graceful degradation with warnings
   - Thread-safe global state management

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Global state conflicts | Use proper locking and initialization checks |
| Divisibility constraints | Implement auto-adjustment with clear logging |
| Pipeline scheduling complexity | Start with simple 1F1B schedule |
| Backward compatibility | Make feature optional with config flag |

## Code Quality Requirements

- Type hints for all public functions
- Comprehensive docstrings with examples
- Unit test coverage >95%
- Integration with existing linting/formatting
- Performance profiling for critical paths

## Documentation Deliverables

1. API documentation in docstrings
2. Integration guide for trainer
3. Migration guide from fixed batch size
4. Performance tuning guide
5. Troubleshooting common issues

## Conclusion

The Microbatch Calculator is an ideal next feature for RoseLLM because it:
- Has clear boundaries and implementation patterns from Megatron-LM
- Provides immediate value for pipeline parallelism and gradient accumulation
- Can be implemented incrementally without breaking existing functionality
- Enables future features like dynamic loss scaling and memory optimization
- Strengthens RoseLLM's position as a Megatron-LM alternative

The implementation is straightforward, well-tested in production systems, and will significantly enhance RoseLLM's training efficiency and flexibility.