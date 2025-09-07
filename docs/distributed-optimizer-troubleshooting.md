# Distributed Optimizer Troubleshooting & Optimization Guide

## Common Issues and Solutions

### 1. Performance Issues

#### Issue: Gradient Reduction Taking Longer Than Computation

**Symptoms**:
- Communication efficiency < 50%
- Training slows down with more GPUs
- GPU utilization drops during gradient synchronization

**Diagnosis**:
```python
# Add profiling to identify bottleneck
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    schedule=profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(5):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prof.step()

# Analyze timeline in TensorBoard
# Look for gaps between computation and communication
```

**Solutions**:

1. **Increase Bucket Size**:
```python
# Reduce number of all-reduce operations
optimizer = DistributedOptimizer(
    base_optimizer,
    bucket_size_mb=50,  # Increased from default 25MB
)
```

2. **Enable Gradient Compression**:
```python
class CompressedDistributedOptimizer(DistributedOptimizer):
    def compress_gradients(self, bucket):
        # Top-K sparsification
        k = int(0.01 * bucket.grad_buffer.numel())  # Keep top 1%
        values, indices = torch.topk(
            bucket.grad_buffer.abs(), k
        )
        sparse_grad = torch.zeros_like(bucket.grad_buffer)
        sparse_grad[indices] = bucket.grad_buffer[indices]
        return sparse_grad, indices
```

3. **Use Hierarchical Reduction**:
```python
def setup_hierarchical_groups(world_size, local_size):
    """
    Setup two-level hierarchy for large clusters.
    First reduce within node, then across nodes.
    """
    # Intra-node groups
    intra_node_groups = []
    for node in range(world_size // local_size):
        ranks = list(range(node * local_size, (node + 1) * local_size))
        intra_node_groups.append(
            dist.new_group(ranks, backend='nccl')
        )
    
    # Inter-node group (one rank per node)
    inter_node_ranks = [i * local_size for i in range(world_size // local_size)]
    inter_node_group = dist.new_group(inter_node_ranks, backend='nccl')
    
    return intra_node_groups, inter_node_group
```

#### Issue: Memory Fragmentation

**Symptoms**:
- OOM errors despite sufficient total memory
- Increasing memory usage over time
- `torch.cuda.memory_allocated()` much less than `torch.cuda.memory_reserved()`

**Diagnosis**:
```python
def diagnose_memory_fragmentation():
    """Check for memory fragmentation issues."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    fragmentation_ratio = 1 - (allocated / reserved)
    
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
    print(f"Fragmentation: {fragmentation_ratio:.2%}")
    
    if fragmentation_ratio > 0.3:
        print("WARNING: High memory fragmentation detected")
        
    # Get detailed memory stats
    print("\nMemory Statistics:")
    for key, value in torch.cuda.memory_stats().items():
        if 'allocated' in key or 'reserved' in key:
            print(f"{key}: {value / 1024**2:.2f} MB")
```

**Solutions**:

1. **Use Memory Pool**:
```python
# Pre-allocate buffers to avoid fragmentation
class MemoryEfficientGradientBuffer(GradientBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_memory_pool()
    
    def _setup_memory_pool(self):
        # Pre-allocate all buffers at once
        total_size = sum(b.size for b in self.buckets)
        self.memory_pool = torch.zeros(
            total_size, dtype=self.dtype, device=self.device
        )
        
        # Assign slices to buckets
        offset = 0
        for bucket in self.buckets:
            bucket.grad_buffer = self.memory_pool[
                offset:offset + bucket.size
            ]
            offset += bucket.size
```

2. **Periodic Memory Defragmentation**:
```python
def defragment_memory(optimizer, frequency=100):
    """Periodically clear cache to defragment memory."""
    if optimizer.step_count % frequency == 0:
        # Save current state
        state_dict = optimizer.state_dict()
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Restore state
        optimizer.load_state_dict(state_dict)
```

### 2. Convergence Issues

#### Issue: Model Not Converging with Distributed Training

**Symptoms**:
- Loss not decreasing or oscillating
- Different results with different number of GPUs
- Gradient norms exploding or vanishing

**Diagnosis**:
```python
class ConvergenceDiagnostics:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.gradient_history = []
        self.loss_history = []
        
    def record_step(self, loss):
        # Record gradient statistics
        grad_norms = []
        for param in self.optimizer.all_params:
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        self.gradient_history.append({
            'min': min(grad_norms),
            'max': max(grad_norms),
            'mean': sum(grad_norms) / len(grad_norms),
        })
        self.loss_history.append(loss.item())
        
    def diagnose(self):
        # Check for gradient explosion
        recent_grads = self.gradient_history[-10:]
        if any(g['max'] > 100 for g in recent_grads):
            print("WARNING: Gradient explosion detected")
            
        # Check for gradient vanishing
        if all(g['mean'] < 1e-6 for g in recent_grads):
            print("WARNING: Gradient vanishing detected")
            
        # Check for loss oscillation
        recent_loss = self.loss_history[-10:]
        if len(recent_loss) > 5:
            variance = torch.var(torch.tensor(recent_loss))
            if variance > 0.1 * abs(recent_loss[-1]):
                print("WARNING: Loss oscillation detected")
```

**Solutions**:

1. **Gradient Clipping and Scaling**:
```python
# Ensure consistent gradient clipping across ranks
optimizer = DistributedOptimizer(
    base_optimizer,
    clip_grad_norm=1.0,  # Clip before reduction
)

# Scale learning rate with world size
scaled_lr = base_lr * math.sqrt(world_size)
```

2. **Verify Gradient Synchronization**:
```python
def verify_gradient_sync(model, optimizer):
    """Ensure all ranks have identical gradients."""
    for param in model.parameters():
        if param.grad is not None:
            # Compute gradient checksum
            checksum = param.grad.sum().item()
            
            # Gather checksums from all ranks
            checksums = [None] * dist.get_world_size()
            dist.all_gather_object(checksums, checksum)
            
            # Verify all are identical
            if not all(abs(c - checksums[0]) < 1e-5 for c in checksums):
                raise ValueError(f"Gradient mismatch detected: {checksums}")
```

3. **Fix Random Seed Consistency**:
```python
def setup_reproducible_training(seed=42):
    """Ensure reproducible training across ranks."""
    # Python random
    import random
    random.seed(seed)
    
    # NumPy
    import numpy as np
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDA algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Distributed sampler seed
    # Each rank gets different but deterministic data
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.manual_seed(seed + rank)
```

### 3. Deadlocks and Hangs

#### Issue: Training Hangs During Gradient Reduction

**Symptoms**:
- Training freezes at `optimizer.step()`
- No error messages
- GPUs at 0% utilization

**Diagnosis**:
```python
import signal
import traceback

def timeout_handler(signum, frame):
    """Print stack trace on timeout."""
    print("Operation timed out! Stack trace:")
    traceback.print_stack()
    raise TimeoutError("Operation timed out")

# Set timeout for debugging
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)  # 60 second timeout

try:
    optimizer.step()
finally:
    signal.alarm(0)  # Cancel alarm
```

**Solutions**:

1. **Add Timeout to Communications**:
```python
class TimeoutDistributedOptimizer(DistributedOptimizer):
    def _reduce_gradients(self):
        """Reduce with timeout to detect hangs."""
        if self.dp_size <= 1:
            return
            
        # Set timeout for all-reduce
        timeout = timedelta(seconds=30)
        
        try:
            # Create monitored all-reduce
            work = dist.all_reduce(
                tensor, 
                group=self.dp_process_group,
                async_op=True
            )
            
            # Wait with timeout
            if not work.wait(timeout):
                raise TimeoutError(
                    f"All-reduce timed out after {timeout.seconds}s"
                )
                
        except TimeoutError as e:
            # Log debugging information
            print(f"Rank {self.dp_rank}: {e}")
            print(f"Waiting for ranks: {self.get_waiting_ranks()}")
            raise
```

2. **Implement Deadlock Detection**:
```python
class DeadlockDetector:
    def __init__(self, timeout=60):
        self.timeout = timeout
        self.last_progress = time.time()
        self.step_count = 0
        
    def check_progress(self):
        """Check if training is making progress."""
        current_time = time.time()
        
        if current_time - self.last_progress > self.timeout:
            # Collect state from all ranks
            states = [None] * dist.get_world_size()
            local_state = {
                'rank': dist.get_rank(),
                'step': self.step_count,
                'waiting_for': self.get_waiting_operations(),
            }
            dist.all_gather_object(states, local_state)
            
            # Analyze for deadlock
            self.analyze_deadlock(states)
            
    def analyze_deadlock(self, states):
        """Analyze states to identify deadlock."""
        # Check if ranks are at different steps
        steps = [s['step'] for s in states]
        if len(set(steps)) > 1:
            print(f"Ranks out of sync: {steps}")
            print("Possible deadlock due to mismatched operations")
```

### 4. Integration Issues

#### Issue: Incompatibility with Mixed Precision Training

**Symptoms**:
- Gradient overflow/underflow with FP16
- Loss becomes NaN
- Gradient scaler conflicts

**Solution**:
```python
class MixedPrecisionDistributedOptimizer(DistributedOptimizer):
    def __init__(self, *args, grad_scaler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_scaler = grad_scaler
        
    def step(self, closure=None):
        # Unscale gradients before reduction
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.base_optimizer)
            
        # Check for inf/nan
        found_inf = sum(
            torch.isinf(p.grad).any().item() if p.grad is not None else 0
            for p in self.all_params
        )
        
        # Synchronize inf check across ranks
        found_inf = torch.tensor([found_inf], device='cuda')
        dist.all_reduce(found_inf, op=dist.ReduceOp.MAX)
        
        if found_inf.item() > 0:
            # Skip step if inf/nan found
            return None
            
        # Proceed with normal step
        return super().step(closure)
```

#### Issue: Memory Leaks with Dynamic Graphs

**Symptoms**:
- Steadily increasing memory usage
- OOM after many iterations
- Retained gradients from previous iterations

**Solution**:
```python
class DynamicGraphOptimizer(DistributedOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_cache = {}
        
    def zero_grad(self, set_to_none=True):
        """Properly clear gradients and caches."""
        # Clear base gradients
        super().zero_grad(set_to_none)
        
        # Clear gradient cache
        self.gradient_cache.clear()
        
        # Clear bucket buffers if dynamic
        if self.gradient_buffer:
            for bucket in self.gradient_buffer.buckets:
                if bucket.grad_buffer is not None:
                    bucket.grad_buffer.zero_()
                    
        # Force garbage collection periodically
        if self.step_count % 100 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
```

## Performance Optimization Strategies

### 1. Network-Aware Optimization

```python
class NetworkAwareOptimizer:
    """Optimizer that adapts to network conditions."""
    
    def __init__(self, base_optimizer):
        self.base_optimizer = base_optimizer
        self.network_profiler = NetworkProfiler()
        
    def profile_network(self):
        """Measure network bandwidth and latency."""
        test_sizes = [1, 10, 100]  # MB
        results = {}
        
        for size_mb in test_sizes:
            size_bytes = size_mb * 1024 * 1024
            tensor = torch.randn(
                size_bytes // 4, device='cuda'
            )
            
            start = time.time()
            dist.all_reduce(tensor)
            duration = time.time() - start
            
            bandwidth = size_bytes / duration / 1024**3  # GB/s
            results[size_mb] = {
                'bandwidth': bandwidth,
                'latency': duration - size_bytes / (bandwidth * 1024**3)
            }
            
        return results
    
    def optimize_for_network(self, profile_results):
        """Adjust settings based on network profile."""
        avg_bandwidth = sum(
            r['bandwidth'] for r in profile_results.values()
        ) / len(profile_results)
        
        if avg_bandwidth < 10:  # GB/s
            # Low bandwidth: larger buckets, compression
            self.base_optimizer.bucket_size_mb = 100
            self.enable_compression = True
        elif avg_bandwidth < 50:
            # Medium bandwidth: standard settings
            self.base_optimizer.bucket_size_mb = 25
            self.enable_compression = False
        else:
            # High bandwidth: smaller buckets for overlap
            self.base_optimizer.bucket_size_mb = 10
            self.enable_compression = False
```

### 2. Compute-Communication Overlap Analysis

```python
def analyze_overlap_potential(model, batch_size, profile_iterations=10):
    """
    Analyze potential for compute-communication overlap.
    Returns recommendations for optimizer configuration.
    """
    # Profile backward pass
    backward_times = []
    for _ in range(profile_iterations):
        dummy_input = torch.randn(batch_size, *input_shape)
        output = model(dummy_input)
        loss = output.sum()
        
        start = time.time()
        loss.backward()
        backward_times.append(time.time() - start)
    
    avg_backward_time = sum(backward_times) / len(backward_times)
    
    # Profile communication
    param_sizes = [p.numel() * p.element_size() for p in model.parameters()]
    total_param_size = sum(param_sizes)
    
    # Estimate communication time
    bandwidth = 50e9  # 50 GB/s typical for NVLink
    latency = 1e-6    # 1 microsecond
    num_buckets = max(1, total_param_size // (25 * 1024 * 1024))
    comm_time = num_buckets * latency + total_param_size / bandwidth
    
    # Compute overlap ratio
    overlap_ratio = min(1.0, avg_backward_time / comm_time)
    
    recommendations = {
        'overlap_ratio': overlap_ratio,
        'recommended_bucket_size_mb': 25 if overlap_ratio > 0.5 else 50,
        'enable_overlap': overlap_ratio > 0.3,
        'explanation': f"Backward time: {avg_backward_time:.3f}s, "
                      f"Communication time: {comm_time:.3f}s"
    }
    
    return recommendations
```

### 3. Adaptive Performance Tuning

```python
class AdaptiveOptimizer(DistributedOptimizer):
    """
    Optimizer that automatically tunes parameters
    based on runtime performance.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tuning_history = []
        self.best_config = None
        self.tuning_phase = True
        
    def tune_step(self):
        """Execute one tuning iteration."""
        configs = [
            {'bucket_size_mb': 10, 'overlap': True},
            {'bucket_size_mb': 25, 'overlap': True},
            {'bucket_size_mb': 50, 'overlap': True},
            {'bucket_size_mb': 25, 'overlap': False},
        ]
        
        for config in configs:
            # Apply configuration
            self.bucket_size_mb = config['bucket_size_mb']
            self.overlap_grad_reduce = config['overlap']
            self._init_gradient_buffer()
            
            # Measure performance
            times = []
            for _ in range(5):
                start = time.time()
                self.step()
                times.append(time.time() - start)
                
            avg_time = sum(times) / len(times)
            self.tuning_history.append({
                'config': config,
                'time': avg_time
            })
        
        # Select best configuration
        self.best_config = min(
            self.tuning_history, 
            key=lambda x: x['time']
        )['config']
        
        # Apply best configuration
        self.bucket_size_mb = self.best_config['bucket_size_mb']
        self.overlap_grad_reduce = self.best_config['overlap']
        self._init_gradient_buffer()
        
        self.tuning_phase = False
        print(f"Tuning complete. Best config: {self.best_config}")
```

## Debugging Tools and Techniques

### 1. Gradient Verification Tool

```python
class GradientVerifier:
    """Tool for verifying gradient correctness in distributed training."""
    
    @staticmethod
    def verify_gradients(model, optimizer, input_batch, target_batch):
        """
        Verify that distributed gradients match single-GPU reference.
        """
        # Save current state
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        
        # Compute reference gradients (single GPU)
        model.load_state_dict(model_state)
        output = model(input_batch)
        loss = F.cross_entropy(output, target_batch)
        loss.backward()
        
        reference_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                reference_grads[name] = param.grad.clone()
        
        # Reset and compute distributed gradients
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        optimizer.zero_grad()
        
        output = model(input_batch)
        loss = F.cross_entropy(output, target_batch)
        loss.backward()
        optimizer._reduce_gradients()
        
        # Compare gradients
        mismatches = []
        for name, param in model.named_parameters():
            if name in reference_grads:
                ref_grad = reference_grads[name]
                dist_grad = param.grad
                
                if not torch.allclose(ref_grad, dist_grad, rtol=1e-5):
                    rel_error = (ref_grad - dist_grad).abs().max() / ref_grad.abs().max()
                    mismatches.append({
                        'param': name,
                        'rel_error': rel_error.item(),
                        'ref_norm': ref_grad.norm().item(),
                        'dist_norm': dist_grad.norm().item(),
                    })
        
        return mismatches
```

### 2. Communication Profiler

```python
class CommunicationProfiler:
    """Profile communication patterns in distributed optimizer."""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.comm_events = []
        
    def profile_step(self):
        """Profile a single optimizer step."""
        # Hook into communication operations
        original_all_reduce = dist.all_reduce
        
        def profiled_all_reduce(tensor, **kwargs):
            start = time.time()
            handle = original_all_reduce(tensor, **kwargs)
            duration = time.time() - start
            
            self.comm_events.append({
                'op': 'all_reduce',
                'size_mb': tensor.numel() * tensor.element_size() / 1024**2,
                'duration': duration,
                'timestamp': start,
            })
            return handle
        
        # Temporarily replace all_reduce
        dist.all_reduce = profiled_all_reduce
        
        try:
            self.optimizer.step()
        finally:
            dist.all_reduce = original_all_reduce
    
    def analyze(self):
        """Analyze communication patterns."""
        if not self.comm_events:
            return {}
            
        total_time = sum(e['duration'] for e in self.comm_events)
        total_data = sum(e['size_mb'] for e in self.comm_events)
        
        # Identify overlapped operations
        overlapped = []
        for i in range(len(self.comm_events) - 1):
            e1, e2 = self.comm_events[i], self.comm_events[i+1]
            if e1['timestamp'] + e1['duration'] > e2['timestamp']:
                overlapped.append((i, i+1))
        
        return {
            'num_operations': len(self.comm_events),
            'total_time': total_time,
            'total_data_mb': total_data,
            'avg_bandwidth_gbps': total_data / total_time / 1024,
            'overlapped_ops': len(overlapped),
            'overlap_efficiency': len(overlapped) / max(1, len(self.comm_events) - 1),
        }
```

### 3. Memory Leak Detector

```python
class MemoryLeakDetector:
    """Detect memory leaks in distributed training."""
    
    def __init__(self, threshold_mb=100):
        self.threshold_mb = threshold_mb
        self.memory_history = []
        
    def check_memory(self, step):
        """Record memory usage at each step."""
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        self.memory_history.append({
            'step': step,
            'allocated_mb': allocated,
            'reserved_mb': reserved,
        })
        
        # Check for leak
        if len(self.memory_history) > 10:
            recent = self.memory_history[-10:]
            memory_growth = recent[-1]['allocated_mb'] - recent[0]['allocated_mb']
            
            if memory_growth > self.threshold_mb:
                self.report_leak(memory_growth)
    
    def report_leak(self, growth_mb):
        """Report suspected memory leak."""
        print(f"WARNING: Possible memory leak detected!")
        print(f"Memory growth: {growth_mb:.2f} MB over last 10 steps")
        
        # Analyze tensors
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                print(f"  Tensor: shape={obj.shape}, grad_fn={obj.grad_fn}")
```

## Best Practices Checklist

### Pre-Training Checklist

- [ ] Profile network bandwidth and latency
- [ ] Verify gradient synchronization with small test
- [ ] Check memory requirements with full batch size
- [ ] Test checkpoint/resume functionality
- [ ] Validate mixed precision settings if used
- [ ] Run single-step verification against reference

### During Training Monitoring

- [ ] Monitor communication efficiency (target > 70%)
- [ ] Track gradient norms for explosion/vanishing
- [ ] Check memory fragmentation ratio (< 30%)
- [ ] Verify all ranks are synchronized
- [ ] Monitor for memory leaks
- [ ] Track bucket utilization

### Post-Training Analysis

- [ ] Analyze communication patterns
- [ ] Identify performance bottlenecks
- [ ] Compare against theoretical scaling
- [ ] Document optimal configuration
- [ ] Create performance report

## Conclusion

Effective troubleshooting and optimization of the distributed optimizer requires understanding both the algorithmic design and practical system constraints. The tools and techniques presented here provide a comprehensive framework for diagnosing issues, optimizing performance, and ensuring robust distributed training at scale.

Key principles:
1. Always measure before optimizing
2. Understand the trade-offs between memory, computation, and communication
3. Use profiling tools to identify bottlenecks
4. Implement defensive programming for production systems
5. Maintain comprehensive logging and monitoring

With these practices, the RoseLLM Distributed Optimizer can achieve near-linear scaling efficiency while maintaining numerical stability and convergence guarantees.