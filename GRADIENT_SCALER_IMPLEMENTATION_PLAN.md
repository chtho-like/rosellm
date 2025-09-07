# Custom Gradient Scaler Implementation Plan for RoseLLM

## Executive Summary

Implement a custom gradient scaler with dynamic loss scaling capabilities following Megatron-LM's design patterns while maintaining compatibility with RoseLLM's architecture. This feature will enhance mixed precision training stability and provide foundation for future precision-aware optimizations.

## 1. Feature Overview

### Core Components
- **MegatronGradScaler**: Abstract base class for gradient scaling
- **ConstantGradScaler**: Fixed loss scale implementation
- **DynamicGradScaler**: Adaptive loss scaling with hysteresis
- **Integration with existing gradient utilities**

### Key Benefits
- Improved FP16 training stability
- Better convergence for large models
- Foundation for precision-aware optimizers
- Bit-to-bit validation against Megatron-LM

## 2. Technical Specification

### 2.1 File Structure
```
rosellm/rosetrainer/optimizer/
├── __init__.py
├── grad_scaler.py          # Main implementation
└── scaler_config.py        # Configuration dataclass

rosellm/rosetrainer/utils/
└── gradient_utils.py       # Integration updates

tests/rosetrainer/optimizer/
├── __init__.py
├── test_grad_scaler.py     # Unit tests
└── test_megatron_parity.py # Bit-to-bit validation

examples/
└── gradient_scaler_example.py  # End-to-end usage example
```

### 2.2 Core Classes

#### AbstractGradScaler (Base Class)
```python
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch

class AbstractGradScaler(ABC):
    """Abstract base class for gradient scalers."""
    
    def __init__(self, initial_scale: float, device: str = 'cuda'):
        assert initial_scale > 0.0
        self._scale = torch.tensor([initial_scale], dtype=torch.float, device=device)
        self._device = device
    
    @property
    def scale(self) -> torch.Tensor:
        return self._scale
    
    @property
    def inv_scale(self) -> torch.Tensor:
        return self._scale.double().reciprocal().float()
    
    @abstractmethod
    def update(self, found_inf: bool) -> None:
        """Update scale based on gradient overflow status."""
        pass
    
    @abstractmethod
    def state_dict(self) -> Dict:
        """Get state for checkpointing."""
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state from checkpoint."""
        pass
```

#### ConstantGradScaler
```python
class ConstantGradScaler(AbstractGradScaler):
    """Constant loss scale (never adjusted)."""
    
    def update(self, found_inf: bool) -> None:
        # No-op for constant scaler
        pass
    
    def state_dict(self) -> Dict:
        return {'scale': self._scale}
    
    def load_state_dict(self, state_dict: Dict) -> None:
        self._scale = state_dict['scale'].to(self._device)
```

#### DynamicGradScaler
```python
class DynamicGradScaler(AbstractGradScaler):
    """Dynamic loss scaling with hysteresis."""
    
    def __init__(
        self,
        initial_scale: float,
        min_scale: float = 1.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        hysteresis: int = 2,
        device: str = 'cuda'
    ):
        super().__init__(initial_scale, device)
        
        # Validation
        assert min_scale > 0.0 and min_scale <= initial_scale
        assert growth_factor > 1.0
        assert 0.0 < backoff_factor < 1.0
        assert growth_interval > 0
        assert hysteresis > 0
        
        # Scale bounds and factors
        self.min_scale = torch.tensor([min_scale], dtype=torch.float, device=device)
        self.growth_factor = torch.tensor([growth_factor], dtype=torch.float, device=device)
        self.backoff_factor = torch.tensor([backoff_factor], dtype=torch.float, device=device)
        
        # Hysteresis parameters
        self.growth_interval = growth_interval
        self.hysteresis = hysteresis
        
        # Trackers
        self._growth_tracker = 0
        self._hysteresis_tracker = hysteresis
    
    def update(self, found_inf: bool) -> None:
        """Update scale based on gradient overflow."""
        if found_inf:
            self._growth_tracker = 0
            self._hysteresis_tracker -= 1
            
            if self._hysteresis_tracker <= 0:
                # Scale down
                self._scale = torch.max(
                    self._scale * self.backoff_factor,
                    self.min_scale
                )
                self._hysteresis_tracker = self.hysteresis
        else:
            self._growth_tracker += 1
            
            if self._growth_tracker >= self.growth_interval:
                # Scale up
                self._scale = self._scale * self.growth_factor
                self._growth_tracker = 0
                self._hysteresis_tracker = self.hysteresis
    
    def state_dict(self) -> Dict:
        return {
            'scale': self._scale,
            'growth_tracker': self._growth_tracker,
            'hysteresis_tracker': self._hysteresis_tracker
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        self._scale = state_dict['scale'].to(self._device)
        self._growth_tracker = state_dict['growth_tracker']
        self._hysteresis_tracker = state_dict['hysteresis_tracker']
```

### 2.3 Integration with Gradient Utilities

Update `gradient_utils.py` to support custom scalers:

```python
def check_for_inf_and_nan(
    parameters: Union[List[torch.Tensor], nn.Module],
    scaler: Optional[AbstractGradScaler] = None
) -> bool:
    """Check for inf/nan in gradients."""
    found_inf = False
    
    if isinstance(parameters, nn.Module):
        param_list = list(parameters.parameters())
    else:
        param_list = parameters
    
    for param in param_list:
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                found_inf = True
                break
    
    # Update scaler if provided
    if scaler is not None:
        scaler.update(found_inf)
    
    return found_inf
```

### 2.4 Configuration

```python
@dataclass
class GradScalerConfig:
    """Configuration for gradient scaler."""
    
    scaler_type: str = "dynamic"  # "constant", "dynamic", "none"
    initial_scale: float = 2**16
    min_scale: float = 1.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    hysteresis: int = 2
    
    def create_scaler(self, device: str = 'cuda') -> Optional[AbstractGradScaler]:
        """Factory method to create scaler."""
        if self.scaler_type == "constant":
            return ConstantGradScaler(self.initial_scale, device)
        elif self.scaler_type == "dynamic":
            return DynamicGradScaler(
                self.initial_scale,
                self.min_scale,
                self.growth_factor,
                self.backoff_factor,
                self.growth_interval,
                self.hysteresis,
                device
            )
        elif self.scaler_type == "none":
            return None
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
```

## 3. Integration Points

### 3.1 With Mixed Precision Manager

Update `mixed_precision.py`:

```python
class MixedPrecisionManager:
    def __init__(self, ...):
        # Replace PyTorch GradScaler with custom implementation
        if use_custom_scaler:
            self.scaler = scaler_config.create_scaler(device)
        else:
            # Fallback to PyTorch native
            self.scaler = GradScaler(...)
```

### 3.2 With RoseTrainer Engine

Update `engine.py`:

```python
class RoseTrainer:
    def __init__(self, ...):
        # Initialize custom scaler if configured
        if config.use_custom_grad_scaler:
            self.grad_scaler = config.grad_scaler_config.create_scaler()
```

### 3.3 With Parallelism

Ensure scaler works with all parallelism dimensions:
- Synchronize scale across data parallel ranks
- Handle tensor parallel gradient reductions
- Support pipeline parallel gradient accumulation

## 4. Memory and Performance Implications

### Memory Impact
- Minimal: ~3 scalar tensors per scaler
- No additional gradient buffers required
- State dict adds < 100 bytes for checkpointing

### Performance Impact
- Negligible compute overhead (< 0.1% training time)
- Improved convergence can reduce total training time
- Better numerical stability for large models

## 5. Testing Strategy

### 5.1 Unit Tests

```python
class TestGradScaler(unittest.TestCase):
    """Test gradient scaler functionality."""
    
    def test_constant_scaler(self):
        """Test constant scaler maintains scale."""
        scaler = ConstantGradScaler(1024.0)
        initial_scale = scaler.scale.item()
        
        # Should not change on update
        scaler.update(found_inf=True)
        assert scaler.scale.item() == initial_scale
        
        scaler.update(found_inf=False)
        assert scaler.scale.item() == initial_scale
    
    def test_dynamic_scaler_backoff(self):
        """Test dynamic scaler reduces on overflow."""
        scaler = DynamicGradScaler(
            initial_scale=1024.0,
            backoff_factor=0.5,
            hysteresis=2
        )
        
        # Need hysteresis overflows to trigger backoff
        scaler.update(found_inf=True)
        assert scaler.scale.item() == 1024.0  # No change yet
        
        scaler.update(found_inf=True)
        assert scaler.scale.item() == 512.0  # Backed off
    
    def test_dynamic_scaler_growth(self):
        """Test dynamic scaler increases without overflow."""
        scaler = DynamicGradScaler(
            initial_scale=1024.0,
            growth_factor=2.0,
            growth_interval=100
        )
        
        for _ in range(100):
            scaler.update(found_inf=False)
        
        assert scaler.scale.item() == 2048.0  # Grew by factor
```

### 5.2 Bit-to-Bit Validation

```python
class TestMegatronParity(unittest.TestCase):
    """Validate bit-to-bit accuracy with Megatron-LM."""
    
    def test_dynamic_scaler_parity(self):
        """Compare with Megatron-LM DynamicGradScaler."""
        # Create identical configurations
        rosellm_scaler = DynamicGradScaler(
            initial_scale=65536.0,
            min_scale=1.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1000,
            hysteresis=2
        )
        
        # Simulate Megatron-LM behavior
        megatron_scale = 65536.0
        megatron_growth_tracker = 0
        megatron_hysteresis_tracker = 2
        
        # Test sequence of updates
        test_sequence = [False] * 500 + [True, True] + [False] * 1000
        
        for found_inf in test_sequence:
            rosellm_scaler.update(found_inf)
            
            # Simulate Megatron-LM logic
            if found_inf:
                megatron_growth_tracker = 0
                megatron_hysteresis_tracker -= 1
                if megatron_hysteresis_tracker <= 0:
                    megatron_scale = max(megatron_scale * 0.5, 1.0)
                    megatron_hysteresis_tracker = 2
            else:
                megatron_growth_tracker += 1
                if megatron_growth_tracker == 1000:
                    megatron_scale = megatron_scale * 2.0
                    megatron_growth_tracker = 0
                    megatron_hysteresis_tracker = 2
            
            # Verify bit-to-bit accuracy
            assert abs(rosellm_scaler.scale.item() - megatron_scale) < 1e-6
```

### 5.3 End-to-End Example

```python
# examples/gradient_scaler_example.py
import torch
import torch.nn as nn
from rosellm.rosetrainer.optimizer import DynamicGradScaler, GradScalerConfig
from rosellm.rosetrainer.utils import check_for_inf_and_nan

def train_with_custom_scaler():
    """Demonstrate custom gradient scaler usage."""
    
    # Model and optimizer
    model = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Configure custom scaler
    scaler_config = GradScalerConfig(
        scaler_type="dynamic",
        initial_scale=2**16,
        growth_interval=500
    )
    scaler = scaler_config.create_scaler()
    
    # Training loop
    for epoch in range(10):
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Forward with mixed precision
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(batch)
                loss = compute_loss(output)
            
            # Scale loss and backward
            scaled_loss = loss * scaler.scale
            scaled_loss.backward()
            
            # Check for overflow and update scaler
            found_inf = check_for_inf_and_nan(model, scaler)
            
            if not found_inf:
                # Unscale gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(scaler.inv_scale)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                optimizer.step()
            
            print(f"Scale: {scaler.scale.item():.1f}, Loss: {loss.item():.4f}")
```

## 6. Validation Requirements

### Functional Correctness
- [ ] Constant scaler maintains fixed scale
- [ ] Dynamic scaler backs off on overflow with hysteresis
- [ ] Dynamic scaler grows on successful iterations
- [ ] State dict save/load preserves exact state
- [ ] CPU fallback works correctly

### Numerical Accuracy
- [ ] Bit-to-bit match with Megatron-LM for identical sequences
- [ ] Scale calculations use same precision (float32)
- [ ] Overflow detection matches Megatron-LM behavior

### Performance
- [ ] < 0.1% overhead on training step time
- [ ] Memory usage < 1KB per scaler instance
- [ ] No performance regression in distributed training

### Integration
- [ ] Works with all parallelism dimensions (TP, PP, DP, CP, EP)
- [ ] Compatible with gradient accumulation
- [ ] Integrates with existing mixed precision manager
- [ ] Checkpoint compatibility maintained

## 7. Implementation Milestones

### Phase 1: Core Implementation (Day 1)
- [ ] Implement AbstractGradScaler base class
- [ ] Implement ConstantGradScaler
- [ ] Implement DynamicGradScaler
- [ ] Add configuration dataclass

### Phase 2: Integration (Day 2)
- [ ] Update gradient_utils.py for scaler support
- [ ] Integrate with MixedPrecisionManager
- [ ] Add RoseTrainer engine support
- [ ] Handle distributed training cases

### Phase 3: Testing (Day 3)
- [ ] Write comprehensive unit tests
- [ ] Implement Megatron-LM parity tests
- [ ] Create end-to-end example
- [ ] Validate on 2-GPU setup

### Phase 4: Documentation & Polish (Day 4)
- [ ] Add docstrings and type hints
- [ ] Write usage documentation
- [ ] Performance profiling
- [ ] Final validation and PR preparation

## 8. Success Metrics

- **Code Quality**: 100% test coverage, passes all linting
- **Accuracy**: Bit-to-bit match with Megatron-LM reference
- **Performance**: < 0.1% training overhead
- **Usability**: Clear API with comprehensive examples
- **Compatibility**: Works with all existing RoseLLM features

## 9. Potential Pitfalls & Solutions

### Pitfall 1: Device Mismatch
**Issue**: Scale tensor on wrong device
**Solution**: Always specify device in constructor, use `.to(device)` in load_state_dict

### Pitfall 2: Distributed Synchronization
**Issue**: Different ranks have different scales
**Solution**: Broadcast scale from rank 0 after updates

### Pitfall 3: Numerical Precision
**Issue**: Float32 vs Float16 scale calculations
**Solution**: Always use Float32 for scale computations

### Pitfall 4: State Migration
**Issue**: Incompatible state dicts between versions
**Solution**: Version state dict format, provide migration utilities

## 10. Future Extensions

Once this feature is implemented, it enables:
- Precision-aware optimizers
- Per-layer loss scaling
- Adaptive precision switching
- Advanced FP8 training support
- Gradient statistics tracking

## Appendix: Reference Implementation Snippets

### Megatron-LM DynamicGradScaler Logic
```python
# From megatron/core/optimizer/grad_scaler.py
if found_inf:
    self._growth_tracker = 0
    self._hysteresis_tracker -= 1
    if self._hysteresis_tracker <= 0:
        self._scale = torch.max(self._scale * self.backoff_factor, self.min_scale)
else:
    self._growth_tracker += 1
    if self._growth_tracker == self.growth_interval:
        self._growth_tracker = 0
        self._hysteresis_tracker = self.hysteresis
        self._scale = self._scale * self.growth_factor
```

This implementation plan provides a clear roadmap for implementing custom gradient scalers in RoseLLM while maintaining compatibility with Megatron-LM patterns and ensuring thorough testing and validation.