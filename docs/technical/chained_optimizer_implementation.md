# ChainedOptimizer Implementation Plan

## Phase 1: Core Infrastructure (200 lines)

### File: `/data/projects/rosellm/rosellm/rosetrainer/optimizer/chained_optimizer.py`

```python
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..optimizer.base import MegatronOptimizer

class ProxyDict:
    """Proxy dictionary for aggregating optimizer states."""
    
    def __init__(self, state_dicts: List[dict]):
        self.state_dicts = state_dicts
        self._cache = {}
    
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            opt_idx, param_id = key
            if opt_idx < len(self.state_dicts):
                return self.state_dicts[opt_idx].get(param_id, {})
        return {}
    
    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            opt_idx, param_id = key
            if opt_idx < len(self.state_dicts):
                self.state_dicts[opt_idx][param_id] = value
    
    def items(self):
        for opt_idx, state_dict in enumerate(self.state_dicts):
            for param_id, state in state_dict.items():
                yield (opt_idx, param_id), state

class ChainedOptimizer(MegatronOptimizer):
    """
    Chains multiple optimizers for different parameter groups.
    
    Supports:
    - Expert parallel parameters with separate optimizer
    - Dense parameters with different optimization strategies
    - Model chunks with varying learning rates
    """
    
    def __init__(
        self, 
        chained_optimizers: List[MegatronOptimizer],
        config: Optional[Any] = None
    ):
        self.chained_optimizers = chained_optimizers
        self.model_chunks = []
        self.is_stub_optimizer = False
        
        # Validate and extract configuration
        if chained_optimizers:
            self.config = config or getattr(chained_optimizers[0], 'config', None)
            
            # Collect model chunks from all optimizers
            for optimizer in chained_optimizers:
                if hasattr(optimizer, 'model_chunks'):
                    for chunk in optimizer.model_chunks:
                        if chunk not in self.model_chunks:
                            self.model_chunks.append(chunk)
                
                # Ensure config consistency
                opt_config = getattr(optimizer, 'config', None)
                if self.config and opt_config:
                    assert self.config == opt_config, "Inconsistent configs"
            
            # Check if all are stub optimizers
            self.is_stub_optimizer = all(
                getattr(opt, 'is_stub_optimizer', False) 
                for opt in chained_optimizers
            )
        else:
            self.is_stub_optimizer = True
            self.config = config
    
    @property
    def param_groups(self) -> List[dict]:
        """Aggregate param groups from all optimizers."""
        groups = []
        for opt in self.chained_optimizers:
            groups.extend(opt.param_groups)
        return groups
    
    @property
    def state(self) -> ProxyDict:
        """Return aggregated state with (opt_idx, param) keys."""
        states = [opt.state for opt in self.chained_optimizers]
        return ProxyDict(states)
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients across all optimizers."""
        for opt in self.chained_optimizers:
            opt.zero_grad(set_to_none)
    
    def step(self, closure=None):
        """Execute step for all optimizers sequentially."""
        loss = None
        for opt_idx, opt in enumerate(self.chained_optimizers):
            # Handle overlapped param gather for first optimizer
            if hasattr(self.config, 'overlap_param_gather_with_optimizer_step'):
                if self.config.overlap_param_gather_with_optimizer_step and opt_idx == 0:
                    # Special handling for overlapped execution
                    pass
            
            current_loss = opt.step(closure)
            if current_loss is not None:
                loss = current_loss
        return loss
    
    def get_loss_scale(self) -> torch.Tensor:
        """Get unified loss scale from first optimizer."""
        if self.chained_optimizers:
            return self.chained_optimizers[0].get_loss_scale()
        return torch.tensor([1.0], dtype=torch.float32, device='cuda')
    
    def state_dict(self) -> dict:
        """Merge state dicts from all optimizers."""
        if len(self.chained_optimizers) == 1:
            return self.chained_optimizers[0].state_dict()
        
        merged = {}
        prefix = "model"
        offset = 0
        
        for opt in self.chained_optimizers:
            if hasattr(opt, 'model_chunks'):
                state = opt.state_dict()
                for chunk_idx in range(len(opt.model_chunks)):
                    key = f"{prefix}{chunk_idx}"
                    if key in state:
                        merged[f"{prefix}{offset}"] = state[key]
                        offset += 1
        
        return merged
    
    def load_state_dict(self, state_dict: dict):
        """Split and load state dict to respective optimizers."""
        split_dicts = self._split_state_dict(state_dict)
        for opt, split_dict in zip(self.chained_optimizers, split_dicts):
            if split_dict is not None:
                opt.load_state_dict(split_dict)
    
    def _split_state_dict(self, state_dict: dict) -> List[Optional[dict]]:
        """Split unified state dict into per-optimizer dicts."""
        if state_dict is None:
            return [None] * len(self.chained_optimizers)
        
        if len(self.model_chunks) == 1:
            return [state_dict] + [None] * (len(self.chained_optimizers) - 1)
        
        # Detect prefix format
        prefix = "model" if "model0" in state_dict else "model_"
        split_dicts = []
        offset = 0
        
        for opt in self.chained_optimizers:
            if hasattr(opt, 'model_chunks'):
                opt_dict = {}
                for chunk_idx in range(len(opt.model_chunks)):
                    global_key = f"{prefix}{offset}"
                    local_key = f"{prefix}{chunk_idx}"
                    
                    if global_key in state_dict:
                        opt_dict[local_key] = state_dict[global_key]
                    offset += 1
                
                split_dicts.append(opt_dict if opt_dict else None)
            else:
                split_dicts.append(None)
        
        return split_dicts
```

## Phase 2: Integration Layer (150 lines)

### File: `/data/projects/rosellm/rosellm/rosetrainer/optimizer/optimizer_factory.py`

```python
def create_chained_optimizer(
    model_chunks: List[MegatronModule],
    config: OptimizerConfig,
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
) -> ChainedOptimizer:
    """
    Create ChainedOptimizer for multi-model training.
    
    Handles:
    - Expert parallel parameters
    - Dense model parameters  
    - Overlapped parameter gathering
    """
    
    optimizers = []
    
    # Separate expert and dense parameters
    expert_params = []
    dense_params = []
    
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            
            # Check if expert parallel
            is_expert = not getattr(param, 'allreduce', True)
            
            if is_expert:
                expert_params.append(param)
            else:
                dense_params.append(param)
    
    # Create dense optimizer
    if dense_params:
        dense_opt = create_optimizer(
            dense_params, 
            config,
            param_type='dense'
        )
        optimizers.append(dense_opt)
    
    # Create expert optimizer with different config
    if expert_params:
        expert_config = replace(
            config,
            lr=config.lr * config.expert_lr_multiplier
        )
        expert_opt = create_optimizer(
            expert_params,
            expert_config, 
            param_type='expert'
        )
        optimizers.append(expert_opt)
    
    return ChainedOptimizer(optimizers, config)
```

## Phase 3: Testing Infrastructure (150 lines)

### File: `/data/projects/rosellm/tests/rosetrainer/optimizer/test_chained_optimizer.py`

```python
import pytest
import torch
from rosellm.rosetrainer.optimizer import ChainedOptimizer

class TestChainedOptimizer:
    
    def test_state_dict_consistency(self):
        """Verify state dict save/load preserves optimizer state."""
        # Create model with expert and dense params
        model = create_test_model_with_experts()
        
        # Create chained optimizer
        optimizer = create_chained_optimizer(model)
        
        # Run training steps
        for _ in range(5):
            loss = model(torch.randn(8, 512))
            loss.backward()
            optimizer.step()
        
        # Save state
        state1 = optimizer.state_dict()
        
        # Create new optimizer and load state
        optimizer2 = create_chained_optimizer(model)
        optimizer2.load_state_dict(state1)
        
        # Verify states match
        state2 = optimizer2.state_dict()
        assert_states_equal(state1, state2)
    
    def test_gradient_norm_aggregation(self):
        """Test gradient norm calculation across optimizers."""
        # Implementation
        pass
    
    def test_expert_parallel_integration(self):
        """Verify expert parallel parameter handling."""
        # Implementation
        pass
```

## Implementation Timeline

### Day 1-2: Core Infrastructure
- Implement ChainedOptimizer class
- Add ProxyDict for state management
- Create basic unit tests

### Day 3: Integration
- Modify optimizer factory
- Integrate with RoseTrainer
- Add configuration support

### Day 4: Testing & Validation
- Comprehensive test suite
- Bit-to-bit validation with Megatron-LM
- Performance benchmarking

### Day 5: Documentation & Examples
- API documentation
- Usage examples
- Migration guide

## Success Metrics

1. **Functional Correctness**:
   - All unit tests pass
   - State dict save/load preserves training state
   - Gradient norms match single optimizer baseline

2. **Performance**:
   - < 5% overhead vs single optimizer
   - Memory usage within 2% of theoretical minimum
   - Communication patterns optimized

3. **Compatibility**:
   - State dicts loadable in Megatron-LM
   - API matches Megatron-LM patterns
   - Supports all parallelism dimensions