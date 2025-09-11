# ChainedOptimizer Comprehensive Testing Strategy

## 1. Unit Test Suite

### Core Functionality Tests
```python
def test_chained_optimizer_initialization():
    """Test optimizer creation with various configurations."""
    # Single optimizer chain
    opt1 = create_optimizer(params1)
    chained = ChainedOptimizer([opt1])
    assert len(chained.chained_optimizers) == 1
    assert not chained.is_stub_optimizer
    
    # Multiple optimizer chain
    opt2 = create_optimizer(params2)
    chained = ChainedOptimizer([opt1, opt2])
    assert len(chained.param_groups) == len(opt1.param_groups) + len(opt2.param_groups)

def test_zero_grad_propagation():
    """Verify zero_grad affects all sub-optimizers."""
    model = create_test_model()
    chained = create_chained_optimizer(model)
    
    # Create gradients
    loss = model(torch.randn(4, 512)).sum()
    loss.backward()
    
    # Verify gradients exist
    assert any(p.grad is not None for p in model.parameters())
    
    # Zero gradients
    chained.zero_grad()
    
    # Verify all gradients cleared
    assert all(p.grad is None or p.grad.sum() == 0 for p in model.parameters())

def test_state_dict_splitting():
    """Test state dict split/merge operations."""
    # Create unified state dict
    unified = {
        "model0": {"param": torch.randn(100)},
        "model1": {"param": torch.randn(200)},
        "model2": {"param": torch.randn(300)}
    }
    
    # Split for two optimizers
    chained = ChainedOptimizer([opt1, opt2])
    split = chained._split_state_dict(unified)
    
    assert "model0" in split[0]
    assert "model0" in split[1]  # Renamed from model1
    assert "model1" in split[1]  # Renamed from model2
```

## 2. Integration Tests

### Multi-GPU Distributed Training
```python
@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_distributed_training_convergence(world_size):
    """Verify training convergence with ChainedOptimizer."""
    
    def run_worker(rank, world_size):
        # Initialize distributed
        init_process_group(rank, world_size)
        
        # Create model with expert and dense params
        model = create_moe_model()
        model = DistributedDataParallel(model)
        
        # Create chained optimizer
        optimizer = create_chained_optimizer_for_moe(model)
        
        # Training loop
        losses = []
        for epoch in range(10):
            data = generate_synthetic_data(rank)
            loss = model(data).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        
        # Verify convergence
        assert losses[-1] < losses[0] * 0.5  # 50% loss reduction
        
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size)
```

### Expert Parallel Integration
```python
def test_expert_parallel_optimizer_separation():
    """Test separate optimization of expert and dense parameters."""
    
    # Initialize EP groups
    initialize_model_parallel(ep_size=2, dp_size=2)
    
    # Create model with experts
    model = MoEModel(num_experts=8)
    
    # Create chained optimizer
    optimizer = create_chained_optimizer(
        model,
        config=OptimizerConfig(
            lr=1e-3,
            expert_lr_multiplier=0.1  # Different LR for experts
        )
    )
    
    # Verify separate param groups
    expert_groups = [g for g in optimizer.param_groups if g['is_expert_parallel']]
    dense_groups = [g for g in optimizer.param_groups if not g['is_expert_parallel']]
    
    assert len(expert_groups) > 0
    assert len(dense_groups) > 0
    assert expert_groups[0]['lr'] == 1e-4  # Expert LR
    assert dense_groups[0]['lr'] == 1e-3   # Dense LR
```

## 3. Bit-to-Bit Validation

### Megatron-LM Compatibility Test
```python
def test_megatron_compatibility():
    """Validate against Megatron-LM reference implementation."""
    
    # Set deterministic mode
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    
    # Create identical models
    rosellm_model = create_test_model()
    megatron_model = create_test_model()
    
    # Synchronize initial parameters
    for (rp, mp) in zip(rosellm_model.parameters(), megatron_model.parameters()):
        mp.data.copy_(rp.data)
    
    # Create optimizers
    rosellm_opt = ChainedOptimizer([
        Adam(rosellm_model.parameters(), lr=1e-3)
    ])
    
    megatron_opt = MegatronChainedOptimizer([
        MegatronAdam(megatron_model.parameters(), lr=1e-3)
    ])
    
    # Run identical training steps
    for step in range(100):
        # Same input
        input_data = torch.randn(8, 512)
        
        # RoseLLM forward/backward
        rosellm_loss = rosellm_model(input_data).sum()
        rosellm_loss.backward()
        rosellm_opt.step()
        rosellm_opt.zero_grad()
        
        # Megatron forward/backward
        megatron_loss = megatron_model(input_data).sum()
        megatron_loss.backward()
        megatron_opt.step()
        megatron_opt.zero_grad()
        
        # Compare parameters (bit-to-bit)
        for (rp, mp) in zip(rosellm_model.parameters(), megatron_model.parameters()):
            torch.testing.assert_close(rp, mp, rtol=0, atol=0)
        
        # Compare optimizer states
        rosellm_state = rosellm_opt.state_dict()
        megatron_state = megatron_opt.state_dict()
        assert_states_identical(rosellm_state, megatron_state)
```

### Numerical Precision Test
```python
def test_gradient_accumulation_precision():
    """Test numerical precision with gradient accumulation."""
    
    model = create_test_model()
    optimizer = ChainedOptimizer([Adam(model.parameters())])
    
    # Accumulate gradients
    accumulated_grads = {}
    for micro_step in range(4):
        loss = model(torch.randn(2, 512)).sum() / 4
        loss.backward()
        
        # Store accumulated gradients
        if micro_step == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    accumulated_grads[name] = param.grad.clone()
        else:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    accumulated_grads[name] += param.grad
    
    # Single step with full batch
    model.zero_grad()
    full_loss = model(torch.randn(8, 512)).sum()
    full_loss.backward()
    
    # Compare gradients (should be identical)
    for name, param in model.named_parameters():
        if param.grad is not None:
            torch.testing.assert_close(
                accumulated_grads[name], 
                param.grad,
                rtol=1e-5, 
                atol=1e-7
            )
```

## 4. Performance Benchmarks

### Memory Efficiency Test
```python
def benchmark_memory_usage():
    """Compare memory usage vs single optimizer."""
    
    import torch.cuda.memory as memory
    
    model = create_large_model(param_count=1_000_000_000)
    
    # Baseline: Single optimizer
    memory.reset_peak_memory_stats()
    single_opt = Adam(model.parameters())
    single_memory = memory.max_memory_allocated()
    
    # ChainedOptimizer with 2 optimizers
    del single_opt
    memory.reset_peak_memory_stats()
    
    expert_params = [p for p in model.parameters() if is_expert(p)]
    dense_params = [p for p in model.parameters() if not is_expert(p)]
    
    chained_opt = ChainedOptimizer([
        Adam(dense_params),
        Adam(expert_params)
    ])
    chained_memory = memory.max_memory_allocated()
    
    # Verify overhead < 5%
    overhead = (chained_memory - single_memory) / single_memory
    assert overhead < 0.05, f"Memory overhead {overhead:.2%} exceeds 5%"
```

### Communication Efficiency Test
```python
@pytest.mark.distributed
def test_communication_patterns():
    """Verify optimized communication patterns."""
    
    from torch.profiler import profile, ProfilerActivity
    
    model = create_moe_model()
    optimizer = create_chained_optimizer(model)
    
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            loss = model(torch.randn(8, 512)).mean()
            loss.backward()
            optimizer.step()
    
    # Analyze communication events
    nccl_events = [e for e in prof.key_averages() if 'nccl' in e.key]
    
    # Verify separate all-reduces for expert and dense
    expert_allreduce = [e for e in nccl_events if 'expert' in e.key]
    dense_allreduce = [e for e in nccl_events if 'dense' in e.key]
    
    assert len(expert_allreduce) > 0
    assert len(dense_allreduce) > 0
    
    # Verify no redundant communication
    total_comm_time = sum(e.cuda_time_total for e in nccl_events)
    assert total_comm_time < threshold_ms
```

## 5. Edge Cases and Error Handling

```python
def test_empty_optimizer_chain():
    """Test handling of empty optimizer chain."""
    chained = ChainedOptimizer([])
    assert chained.is_stub_optimizer
    assert len(chained.param_groups) == 0
    
    # Should not error
    chained.zero_grad()
    chained.step()
    state = chained.state_dict()
    assert state == {}

def test_mismatched_configs():
    """Test error handling for mismatched configs."""
    opt1 = Adam(params1, lr=1e-3)
    opt1.config = OptimizerConfig(lr=1e-3)
    
    opt2 = Adam(params2, lr=1e-4)
    opt2.config = OptimizerConfig(lr=1e-4)
    
    with pytest.raises(AssertionError, match="Inconsistent configs"):
        ChainedOptimizer([opt1, opt2])

def test_partial_state_dict_load():
    """Test loading partial state dicts."""
    optimizer = create_chained_optimizer(model)
    
    # Save full state
    full_state = optimizer.state_dict()
    
    # Create partial state (missing model1)
    partial_state = {k: v for k, v in full_state.items() if 'model1' not in k}
    
    # Should handle gracefully
    optimizer.load_state_dict(partial_state)
    
    # Verify partial load
    new_state = optimizer.state_dict()
    assert 'model0' in new_state
    # model1 should be reinitialized
```

## Test Execution Plan

### Local Testing (2 GPUs)
```bash
# Unit tests
pytest tests/optimizer/test_chained_optimizer.py -v

# Integration tests with 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 pytest tests/optimizer/test_chained_optimizer.py::test_distributed -v

# Bit-to-bit validation
python tests/validation/compare_with_megatron.py
```

### CPU Simulation (32 cores)
```bash
# Simulate 8-way parallelism
CUDA_VISIBLE_DEVICES="" torchrun --nproc_per_node=8 \
    tests/optimizer/test_chained_distributed.py
```

### Continuous Integration
```yaml
test-chained-optimizer:
  runs-on: [self-hosted, gpu]
  steps:
    - name: Unit Tests
      run: pytest tests/optimizer/test_chained_optimizer.py
    
    - name: Integration Tests
      run: |
        torchrun --nproc_per_node=2 tests/distributed/test_chained.py
    
    - name: Megatron Compatibility
      run: python tests/validation/megatron_compat.py
    
    - name: Performance Benchmarks
      run: python benchmarks/chained_optimizer_perf.py
```

## Success Criteria

1. **100% test coverage** of ChainedOptimizer class
2. **Bit-to-bit accuracy** with Megatron-LM for 1000 training steps
3. **< 5% memory overhead** compared to single optimizer
4. **< 2% performance overhead** in training throughput
5. **Zero failures** in 24-hour stress test with random configs