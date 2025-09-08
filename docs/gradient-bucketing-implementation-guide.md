# Gradient Bucketing Implementation Guide: Code Examples & Best Practices

## Table of Contents

1. [Quick Start & Basic Usage](#quick-start--basic-usage)
2. [Advanced Configuration Patterns](#advanced-configuration-patterns)  
3. [Custom Strategy Implementation](#custom-strategy-implementation)
4. [Performance Optimization Techniques](#performance-optimization-techniques)
5. [Debugging & Troubleshooting](#debugging--troubleshooting)
6. [Production Deployment Patterns](#production-deployment-patterns)
7. [Integration with Training Loops](#integration-with-training-loops)

## Quick Start & Basic Usage

### Basic Bucket Manager Setup

```python
import torch
from rosellm.rosetrainer.communication import (
    BucketConfig, BucketManager, BucketStrategy
)

# Basic configuration
config = BucketConfig(
    strategy=BucketStrategy.SIZE_BASED,
    max_bucket_size_mb=25.0,
    min_bucket_size_mb=1.0,
    overlap_communication=True,
    gradient_predivision=True
)

# Initialize manager
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bucket_manager = BucketManager(config, device, dtype=torch.float32)

# Example: Assign gradients from a model
model = YourTransformerModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training step with bucketing
def training_step(batch):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(batch)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Collect gradients for bucketing
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad
    
    # Assign to buckets
    assignments = bucket_manager.assign_gradients_bulk(gradients)
    
    # Synchronize all buckets
    sync_stats = bucket_manager.synchronize_buckets()
    
    # Get updated gradients
    updated_gradients = bucket_manager.get_bucket_assignments()
    
    # Apply gradients back to model
    for name, param in model.named_parameters():
        if name in updated_gradients:
            param.grad = updated_gradients[name]
    
    # Optimizer step
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'bucketing_stats': sync_stats
    }
```

### Integration with Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_bucketing():
    # Initialize distributed process group
    dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    
    # Create model and wrap with DDP
    model = YourModel().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    
    # Configure bucketing for distributed environment
    config = BucketConfig(
        strategy=BucketStrategy.MIXED,
        max_bucket_size_mb=50.0,  # Larger buckets for distributed
        backend=CommunicationBackend.NCCL,
        overlap_communication=True,
        gradient_predivision=True,
        communication_timeout_ms=60000  # 1 minute timeout
    )
    
    bucket_manager = BucketManager(config, device)
    
    return ddp_model, bucket_manager

def distributed_training_step(model, bucket_manager, batch, optimizer):
    """Training step with distributed bucketing."""
    optimizer.zero_grad()
    
    # Forward and backward pass
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    
    # Extract gradients before DDP all-reduce
    gradients = {}
    for name, param in model.module.named_parameters():  # Note: model.module for DDP
        if param.grad is not None:
            gradients[name] = param.grad.clone()  # Clone to avoid DDP interference
    
    # Use bucketing instead of DDP's built-in all-reduce
    # (Disable DDP's gradient synchronization if using custom bucketing)
    assignments = bucket_manager.assign_gradients_bulk(gradients)
    
    # Get process group for communication
    process_group = dist.group.WORLD  # or custom process group
    sync_stats = bucket_manager.synchronize_buckets(process_group=process_group)
    
    # Apply synchronized gradients
    updated_gradients = bucket_manager.get_bucket_assignments()
    for name, param in model.module.named_parameters():
        if name in updated_gradients:
            param.grad = updated_gradients[name]
    
    optimizer.step()
    
    return loss.item(), sync_stats
```

## Advanced Configuration Patterns

### Layer-Based Strategy with Custom Groups

```python
# Define custom layer groupings for your model architecture
layer_groups = {
    "embedding": ["embed", "position_embed", "token_embed"],
    "transformer_attention": [
        "attn", "attention", "self_attn", "cross_attn", 
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    "transformer_mlp": ["mlp", "ffn", "fc1", "fc2", "gate_proj", "down_proj"],
    "normalization": ["norm", "layer_norm", "rmsnorm", "ln"],
    "output": ["lm_head", "output", "classifier", "head"]
}

config = BucketConfig(
    strategy=BucketStrategy.LAYER_BASED,
    max_bucket_size_mb=30.0,
    layer_groups=layer_groups,
    dynamic_bucketing=True,  # Enable adaptive optimization
    gradient_predivision=True
)

# Advanced: Mixed strategy with size thresholds
mixed_config = BucketConfig(
    strategy=BucketStrategy.MIXED,
    max_bucket_size_mb=25.0,
    min_bucket_size_mb=2.0,
    layer_groups=layer_groups,
    # Advanced features
    dynamic_bucketing=True,
    compress_gradients=False,  # Enable if bandwidth-limited
    bucket_cap_mb=100.0,  # Hard limit for very large gradients
)
```

### Hierarchical Bucket Groups

```python
from rosellm.rosetrainer.communication.bucket_groups import (
    BucketGroupConfig, BucketGroupManager, GroupStrategy, PriorityLevel
)

# Configure hierarchical grouping
group_config = BucketGroupConfig(
    group_strategy=GroupStrategy.HIERARCHICAL,
    max_groups=8,
    min_buckets_per_group=2,
    max_buckets_per_group=10,
    enable_prioritization=True,
    priority_threshold_mb=15.0,  # High priority for large buckets
    overlap_groups=True,
    pipeline_communication=True,
    max_concurrent_groups=4
)

# Create group manager
group_manager = BucketGroupManager(group_config, bucket_manager)

def training_with_groups(model, bucket_manager, group_manager, batch, optimizer):
    """Training with hierarchical bucket groups."""
    optimizer.zero_grad()
    
    # Standard forward/backward
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    
    # Collect and assign gradients to buckets
    gradients = {name: param.grad for name, param in model.named_parameters() 
                if param.grad is not None}
    bucket_manager.assign_gradients_bulk(gradients)
    
    # Assign buckets to groups based on strategy
    assignment_stats = group_manager.assign_buckets_to_groups()
    
    # Synchronize with priority-based communication
    sync_stats = group_manager.synchronize_groups()
    
    # Apply gradients and step
    updated_gradients = bucket_manager.get_bucket_assignments()
    for name, param in model.named_parameters():
        if name in updated_gradients:
            param.grad = updated_gradients[name]
    
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'assignment_stats': assignment_stats,
        'sync_stats': sync_stats
    }
```

## Custom Strategy Implementation

### Creating a Custom Bucketing Strategy

```python
def custom_bucket_strategy(param_name: str, gradient: torch.Tensor) -> str:
    """
    Custom bucketing function based on parameter characteristics.
    
    Example: Group by parameter role and gradient magnitude.
    """
    grad_norm = gradient.norm().item()
    param_size = gradient.numel()
    
    # Categorize by gradient magnitude
    if grad_norm > 1.0:
        magnitude_category = "high_grad"
    elif grad_norm > 0.1:
        magnitude_category = "med_grad"  
    else:
        magnitude_category = "low_grad"
    
    # Categorize by parameter size
    if param_size > 1000000:  # 1M parameters
        size_category = "large"
    elif param_size > 10000:  # 10K parameters
        size_category = "medium"
    else:
        size_category = "small"
    
    # Combine categories
    return f"{magnitude_category}_{size_category}"

# Use custom strategy
custom_config = BucketConfig(
    strategy=BucketStrategy.CUSTOM,
    custom_bucket_fn=custom_bucket_strategy,
    max_bucket_size_mb=20.0,
    dynamic_bucketing=True
)
```

### Advanced Custom Strategy with Learning

```python
class AdaptiveBucketStrategy:
    """Learning-based bucket strategy that adapts over time."""
    
    def __init__(self):
        self.parameter_history = {}
        self.bucket_performance = {}
        self.step_count = 0
    
    def __call__(self, param_name: str, gradient: torch.Tensor) -> str:
        """Adaptive bucket assignment based on historical performance."""
        self.step_count += 1
        
        # Record gradient statistics
        grad_stats = {
            'norm': gradient.norm().item(),
            'size': gradient.numel(),
            'sparsity': (gradient == 0).float().mean().item()
        }
        
        if param_name not in self.parameter_history:
            self.parameter_history[param_name] = []
        
        self.parameter_history[param_name].append(grad_stats)
        
        # Keep only recent history
        if len(self.parameter_history[param_name]) > 100:
            self.parameter_history[param_name] = self.parameter_history[param_name][-100:]
        
        # Determine bucket based on learned patterns
        if self.step_count < 10:
            # Bootstrap phase: use simple heuristics
            return self._bootstrap_assignment(param_name, gradient)
        else:
            # Learned phase: use historical data
            return self._learned_assignment(param_name, gradient)
    
    def _bootstrap_assignment(self, param_name: str, gradient: torch.Tensor) -> str:
        """Simple assignment during bootstrap phase."""
        size = gradient.numel()
        if size > 100000:
            return "bootstrap_large"
        elif size > 1000:
            return "bootstrap_medium"
        else:
            return "bootstrap_small"
    
    def _learned_assignment(self, param_name: str, gradient: torch.Tensor) -> str:
        """Assignment based on learned patterns."""
        history = self.parameter_history[param_name]
        
        # Calculate average characteristics
        avg_norm = sum(h['norm'] for h in history) / len(history)
        avg_sparsity = sum(h['sparsity'] for h in history) / len(history)
        
        # Current gradient characteristics
        current_norm = gradient.norm().item()
        current_sparsity = (gradient == 0).float().mean().item()
        
        # Adaptive assignment based on stability
        if abs(current_norm - avg_norm) / avg_norm > 0.5:
            # Unstable gradient - separate bucket
            return f"unstable_{param_name.split('.')[0]}"
        elif current_sparsity > 0.8:
            # Sparse gradient - special handling
            return f"sparse_{param_name.split('.')[0]}"
        else:
            # Stable gradient - group by layer type
            layer_type = self._extract_layer_type(param_name)
            return f"stable_{layer_type}"
    
    def _extract_layer_type(self, param_name: str) -> str:
        """Extract layer type from parameter name."""
        if "attn" in param_name:
            return "attention"
        elif "mlp" in param_name or "ffn" in param_name:
            return "feedforward"
        elif "norm" in param_name:
            return "normalization"
        elif "embed" in param_name:
            return "embedding"
        else:
            return "other"

# Use adaptive strategy
adaptive_strategy = AdaptiveBucketStrategy()
adaptive_config = BucketConfig(
    strategy=BucketStrategy.CUSTOM,
    custom_bucket_fn=adaptive_strategy,
    max_bucket_size_mb=30.0,
    dynamic_bucketing=True
)
```

## Performance Optimization Techniques

### Memory Pool Tuning

```python
def optimize_memory_pools(bucket_manager: BucketManager, 
                         expected_tensor_sizes: List[int]):
    """Pre-warm memory pools with expected tensor sizes."""
    
    # Access memory pools for pre-warming
    for bucket in bucket_manager.buckets:
        pool = bucket._memory_pool
        
        # Pre-allocate tensors for expected sizes
        for size in expected_tensor_sizes:
            temp_tensor = pool.get_tensor(size)
            temp_tensor.zero_()
            pool.return_tensor(temp_tensor)
    
    print("Memory pools pre-warmed with expected tensor sizes")

def monitor_memory_pool_efficiency(bucket_manager: BucketManager):
    """Monitor memory pool hit rates and efficiency."""
    
    pool_stats = {}
    for i, bucket in enumerate(bucket_manager.buckets):
        pool = bucket._memory_pool
        
        # Analyze pool contents
        total_cached = sum(len(tensors) for tensors in pool._pool.values())
        unique_sizes = len(pool._pool)
        
        pool_stats[f"bucket_{i}"] = {
            'total_cached_tensors': total_cached,
            'unique_sizes': unique_sizes,
            'pool_sizes': list(pool._pool.keys())
        }
    
    return pool_stats

# Usage in training loop
def optimized_training_step(model, bucket_manager, batch, optimizer):
    """Training step with performance monitoring."""
    
    # Pre-warm pools on first step
    if not hasattr(optimized_training_step, 'pools_warmed'):
        # Estimate tensor sizes from model
        expected_sizes = []
        for param in model.parameters():
            expected_sizes.append(param.numel())
        
        optimize_memory_pools(bucket_manager, expected_sizes)
        optimized_training_step.pools_warmed = True
    
    # Standard training step
    start_time = time.time()
    
    optimizer.zero_grad()
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    
    # Efficient bulk assignment
    gradients = {name: param.grad for name, param in model.named_parameters() 
                if param.grad is not None}
    
    assignment_time = time.time()
    assignments = bucket_manager.assign_gradients_bulk(gradients, batch_size=20)
    assignment_duration = time.time() - assignment_time
    
    # Synchronize with timing
    sync_time = time.time()
    sync_stats = bucket_manager.synchronize_buckets()
    sync_duration = time.time() - sync_time
    
    # Apply gradients
    updated_gradients = bucket_manager.get_bucket_assignments()
    for name, param in model.named_parameters():
        if name in updated_gradients:
            param.grad = updated_gradients[name]
    
    optimizer.step()
    
    total_duration = time.time() - start_time
    
    # Performance metrics
    perf_stats = {
        'total_time': total_duration,
        'assignment_time': assignment_duration,
        'sync_time': sync_duration,
        'sync_efficiency': sync_stats.get('overlap_efficiency', 0),
        'num_buckets': len(assignments)
    }
    
    # Periodic pool monitoring
    if hasattr(optimized_training_step, 'step_count'):
        optimized_training_step.step_count += 1
    else:
        optimized_training_step.step_count = 1
    
    if optimized_training_step.step_count % 100 == 0:
        pool_stats = monitor_memory_pool_efficiency(bucket_manager)
        print(f"Step {optimized_training_step.step_count} pool stats: {pool_stats}")
    
    return loss.item(), perf_stats
```

### Communication Overlap Optimization

```python
import threading
from concurrent.futures import ThreadPoolExecutor

class OverlappedBucketManager:
    """Bucket manager with advanced communication overlap."""
    
    def __init__(self, base_manager: BucketManager, max_workers: int = 4):
        self.base_manager = base_manager
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.communication_futures = []
    
    def async_bucket_communication(self, bucket_ids: List[int], 
                                 process_group=None) -> List[Future]:
        """Start asynchronous communication for specified buckets."""
        
        futures = []
        for bucket_id in bucket_ids:
            if bucket_id < len(self.base_manager.buckets):
                bucket = self.base_manager.buckets[bucket_id]
                
                # Submit communication task to thread pool
                future = self.executor.submit(
                    self._communicate_bucket,
                    bucket,
                    process_group
                )
                futures.append(future)
        
        return futures
    
    def _communicate_bucket(self, bucket: GradientBucket, 
                          process_group=None) -> Dict[str, Any]:
        """Communicate a single bucket (runs in thread pool)."""
        
        start_time = time.time()
        
        # Start communication
        handle = bucket.start_communication(process_group=process_group)
        
        if handle is not None:
            # Wait for completion
            comm_time = bucket.wait_communication()
            
            return {
                'bucket_id': bucket.bucket_id,
                'communication_time': comm_time,
                'total_time': time.time() - start_time,
                'success': True
            }
        else:
            return {
                'bucket_id': bucket.bucket_id,
                'success': False,
                'error': 'Failed to start communication'
            }
    
    def overlapped_synchronization(self, process_group=None) -> Dict[str, Any]:
        """Synchronize all buckets with maximum overlap."""
        
        # Group buckets by priority/size for optimal scheduling
        bucket_groups = self._prioritize_buckets()
        
        all_futures = []
        group_stats = []
        
        # Start high-priority buckets first
        for group_name, bucket_ids in bucket_groups.items():
            group_start = time.time()
            
            futures = self.async_bucket_communication(bucket_ids, process_group)
            all_futures.extend(futures)
            
            # Don't wait for completion yet - allow overlap
            group_stats.append({
                'group': group_name,
                'bucket_count': len(bucket_ids),
                'start_time': group_start
            })
        
        # Wait for all communications to complete
        results = []
        for future in all_futures:
            try:
                result = future.result(timeout=60)  # 1 minute timeout
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e)
                })
        
        # Compile statistics
        successful_comms = [r for r in results if r.get('success', False)]
        total_time = max(r.get('total_time', 0) for r in results) if results else 0
        avg_comm_time = (sum(r.get('communication_time', 0) for r in successful_comms) / 
                        len(successful_comms)) if successful_comms else 0
        
        return {
            'total_buckets': len(results),
            'successful_buckets': len(successful_comms),
            'failed_buckets': len(results) - len(successful_comms),
            'total_time': total_time,
            'avg_communication_time': avg_comm_time,
            'overlap_efficiency': avg_comm_time / total_time if total_time > 0 else 0
        }
    
    def _prioritize_buckets(self) -> Dict[str, List[int]]:
        """Group buckets by priority for scheduling."""
        
        groups = {
            'large': [],      # Large buckets - start first
            'medium': [],     # Medium buckets 
            'small': []       # Small buckets - batch together
        }
        
        for i, bucket in enumerate(self.base_manager.buckets):
            size_mb = bucket.current_size_bytes / (1024 * 1024)
            
            if size_mb > 20:
                groups['large'].append(i)
            elif size_mb > 5:
                groups['medium'].append(i) 
            else:
                groups['small'].append(i)
        
        return groups
    
    def cleanup(self):
        """Clean up thread pool resources."""
        self.executor.shutdown(wait=True)

# Usage with overlap optimization
def overlapped_training_step(model, bucket_manager, batch, optimizer):
    """Training step with communication overlap."""
    
    # Wrap manager for overlap optimization
    if not hasattr(overlapped_training_step, 'overlap_manager'):
        overlapped_training_step.overlap_manager = OverlappedBucketManager(
            bucket_manager, max_workers=4
        )
    
    overlap_manager = overlapped_training_step.overlap_manager
    
    optimizer.zero_grad()
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    
    # Assign gradients to buckets
    gradients = {name: param.grad for name, param in model.named_parameters() 
                if param.grad is not None}
    bucket_manager.assign_gradients_bulk(gradients)
    
    # Use overlapped synchronization  
    sync_stats = overlap_manager.overlapped_synchronization()
    
    # Apply gradients
    updated_gradients = bucket_manager.get_bucket_assignments()
    for name, param in model.named_parameters():
        if name in updated_gradients:
            param.grad = updated_gradients[name]
    
    optimizer.step()
    
    return loss.item(), sync_stats
```

## Debugging & Troubleshooting

### Comprehensive Bucket Analysis

```python
def analyze_bucket_performance(bucket_manager: BucketManager) -> Dict[str, Any]:
    """Comprehensive analysis of bucket performance and issues."""
    
    analysis = {
        'bucket_distribution': {},
        'communication_patterns': {},
        'memory_usage': {},
        'potential_issues': []
    }
    
    # Analyze bucket size distribution
    bucket_sizes = [b.current_size_bytes / (1024 * 1024) for b in bucket_manager.buckets]
    if bucket_sizes:
        analysis['bucket_distribution'] = {
            'num_buckets': len(bucket_sizes),
            'total_size_mb': sum(bucket_sizes),
            'avg_size_mb': sum(bucket_sizes) / len(bucket_sizes),
            'min_size_mb': min(bucket_sizes),
            'max_size_mb': max(bucket_sizes),
            'size_std': np.std(bucket_sizes) if len(bucket_sizes) > 1 else 0
        }
    
    # Analyze communication patterns
    all_comm_times = []
    bucket_utilizations = []
    
    for bucket in bucket_manager.buckets:
        if bucket.communication_times:
            all_comm_times.extend(bucket.communication_times)
            
        utilization = bucket.current_size_bytes / bucket.max_size_bytes
        bucket_utilizations.append(utilization)
    
    if all_comm_times:
        analysis['communication_patterns'] = {
            'total_communications': len(all_comm_times),
            'avg_time': sum(all_comm_times) / len(all_comm_times),
            'min_time': min(all_comm_times),
            'max_time': max(all_comm_times),
            'time_std': np.std(all_comm_times)
        }
    
    if bucket_utilizations:
        analysis['memory_usage'] = {
            'avg_utilization': sum(bucket_utilizations) / len(bucket_utilizations),
            'min_utilization': min(bucket_utilizations),
            'max_utilization': max(bucket_utilizations),
            'underutilized_buckets': sum(1 for u in bucket_utilizations if u < 0.3),
            'overutilized_buckets': sum(1 for u in bucket_utilizations if u > 0.9)
        }
    
    # Identify potential issues
    issues = []
    
    # Too many small buckets
    small_buckets = sum(1 for size in bucket_sizes if size < 1.0)
    if small_buckets > len(bucket_sizes) * 0.5:
        issues.append(f"Many small buckets ({small_buckets}/{len(bucket_sizes)}). Consider increasing min_bucket_size_mb.")
    
    # Communication time outliers  
    if all_comm_times and len(all_comm_times) > 10:
        mean_time = sum(all_comm_times) / len(all_comm_times)
        outliers = [t for t in all_comm_times if t > mean_time * 3]
        if outliers:
            issues.append(f"Communication outliers detected: {len(outliers)} times > 3x average ({mean_time:.3f}s)")
    
    # Memory pool efficiency
    total_pool_tensors = 0
    pool_sizes = set()
    for bucket in bucket_manager.buckets:
        pool = bucket._memory_pool
        for size, tensors in pool._pool.items():
            total_pool_tensors += len(tensors)
            pool_sizes.add(size)
    
    if total_pool_tensors > len(pool_sizes) * 20:
        issues.append(f"Memory pool may be oversized: {total_pool_tensors} cached tensors for {len(pool_sizes)} unique sizes")
    
    analysis['potential_issues'] = issues
    
    return analysis

def debug_bucket_assignment(bucket_manager: BucketManager, 
                          gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Debug gradient assignment process."""
    
    debug_info = {
        'assignment_details': {},
        'strategy_effectiveness': {},
        'capacity_analysis': {}
    }
    
    # Analyze assignment strategy effectiveness
    bucket_keys = {}
    for param_name, gradient in gradients.items():
        bucket_key = bucket_manager._get_bucket_key(param_name, gradient)
        if bucket_key not in bucket_keys:
            bucket_keys[bucket_key] = []
        bucket_keys[bucket_key].append(param_name)
    
    debug_info['strategy_effectiveness'] = {
        'unique_keys': len(bucket_keys),
        'key_distribution': {k: len(v) for k, v in bucket_keys.items()},
        'avg_params_per_key': sum(len(v) for v in bucket_keys.values()) / len(bucket_keys)
    }
    
    # Analyze capacity constraints
    capacity_stats = []
    for i, bucket in enumerate(bucket_manager.buckets):
        available_bytes = bucket.max_size_bytes - bucket.current_size_bytes
        capacity_stats.append({
            'bucket_id': i,
            'used_mb': bucket.current_size_bytes / (1024 * 1024),
            'available_mb': available_bytes / (1024 * 1024),
            'utilization': bucket.current_size_bytes / bucket.max_size_bytes
        })
    
    debug_info['capacity_analysis'] = capacity_stats
    
    # Test assignment feasibility
    assignment_simulation = {}
    for param_name, gradient in gradients.items():
        grad_size = gradient.numel() * gradient.element_size()
        
        # Find buckets that could accommodate this gradient
        compatible_buckets = []
        for i, bucket in enumerate(bucket_manager.buckets):
            if bucket.can_add_gradient(gradient):
                compatible_buckets.append(i)
        
        assignment_simulation[param_name] = {
            'gradient_size_mb': grad_size / (1024 * 1024),
            'compatible_buckets': compatible_buckets,
            'would_create_new': len(compatible_buckets) == 0
        }
    
    debug_info['assignment_details'] = assignment_simulation
    
    return debug_info

# Usage in debugging session
def debug_training_step(model, bucket_manager, batch, optimizer):
    """Training step with comprehensive debugging."""
    
    optimizer.zero_grad()
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    
    # Collect gradients
    gradients = {name: param.grad for name, param in model.named_parameters() 
                if param.grad is not None}
    
    print(f"=== Gradient Collection ===")
    print(f"Total gradients: {len(gradients)}")
    print(f"Total gradient memory: {sum(g.numel() * g.element_size() for g in gradients.values()) / (1024*1024):.2f}MB")
    
    # Debug assignment process
    print(f"\n=== Assignment Analysis ===")
    assignment_debug = debug_bucket_assignment(bucket_manager, gradients)
    print(f"Strategy effectiveness: {assignment_debug['strategy_effectiveness']}")
    
    # Perform assignment
    start_time = time.time()
    assignments = bucket_manager.assign_gradients_bulk(gradients)
    assignment_time = time.time() - start_time
    
    print(f"Assignment completed in {assignment_time:.4f}s")
    print(f"Assignments: {len(assignments)} gradients across {len(set(assignments.values()))} buckets")
    
    # Analyze bucket state after assignment
    print(f"\n=== Bucket Performance Analysis ===")
    perf_analysis = analyze_bucket_performance(bucket_manager)
    print(f"Bucket distribution: {perf_analysis['bucket_distribution']}")
    
    if perf_analysis['potential_issues']:
        print(f"\n=== Potential Issues ===")
        for issue in perf_analysis['potential_issues']:
            print(f"- {issue}")
    
    # Synchronize with timing
    sync_start = time.time()
    sync_stats = bucket_manager.synchronize_buckets()
    sync_time = time.time() - sync_start
    
    print(f"\n=== Communication Results ===")
    print(f"Sync time: {sync_time:.4f}s")
    print(f"Sync stats: {sync_stats}")
    
    # Apply gradients and finish
    updated_gradients = bucket_manager.get_bucket_assignments()
    for name, param in model.named_parameters():
        if name in updated_gradients:
            param.grad = updated_gradients[name]
    
    optimizer.step()
    
    return loss.item(), {
        'assignment_time': assignment_time,
        'sync_time': sync_time,
        'sync_stats': sync_stats,
        'perf_analysis': perf_analysis
    }
```

### Error Recovery Patterns

```python
class RobustBucketManager:
    """Wrapper around BucketManager with enhanced error recovery."""
    
    def __init__(self, config: BucketConfig, device: torch.device, 
                 fallback_strategy: BucketStrategy = BucketStrategy.SIZE_BASED):
        self.primary_manager = BucketManager(config, device)
        self.fallback_config = BucketConfig(
            strategy=fallback_strategy,
            max_bucket_size_mb=10.0,  # Conservative fallback
            min_bucket_size_mb=1.0
        )
        self.fallback_manager = None
        self.error_count = 0
        self.max_errors = 10
        
    def safe_assign_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Gradient assignment with error recovery."""
        
        try:
            return self.primary_manager.assign_gradients_bulk(gradients)
        
        except Exception as e:
            self.error_count += 1
            print(f"Primary manager error ({self.error_count}/{self.max_errors}): {e}")
            
            if self.error_count >= self.max_errors:
                print("Switching to fallback manager due to repeated errors")
                if self.fallback_manager is None:
                    self.fallback_manager = BucketManager(
                        self.fallback_config, 
                        self.primary_manager.device
                    )
                return self.fallback_manager.assign_gradients_bulk(gradients)
            else:
                # Try individual assignment as recovery
                return self._individual_assignment_fallback(gradients)
    
    def _individual_assignment_fallback(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Fallback to individual gradient assignment."""
        
        assignments = {}
        failed_gradients = []
        
        for param_name, gradient in gradients.items():
            try:
                bucket_id = self.primary_manager.assign_gradient(param_name, gradient)
                assignments[param_name] = bucket_id
            except Exception as e:
                print(f"Failed to assign {param_name}: {e}")
                failed_gradients.append(param_name)
        
        if failed_gradients:
            print(f"Failed to assign {len(failed_gradients)} gradients: {failed_gradients[:5]}...")
        
        return assignments
    
    def safe_synchronize(self, process_group=None) -> Dict[str, Any]:
        """Synchronization with error recovery."""
        
        active_manager = (self.fallback_manager if self.fallback_manager is not None 
                         else self.primary_manager)
        
        try:
            return active_manager.synchronize_buckets(process_group=process_group)
        
        except Exception as e:
            print(f"Synchronization error: {e}")
            
            # Try sequential synchronization as fallback
            return self._sequential_sync_fallback(active_manager, process_group)
    
    def _sequential_sync_fallback(self, manager: BucketManager, 
                                 process_group=None) -> Dict[str, Any]:
        """Fallback to sequential bucket synchronization."""
        
        successful_buckets = 0
        failed_buckets = 0
        total_time = 0
        
        for bucket in manager.buckets:
            if bucket.gradients:
                try:
                    start_time = time.time()
                    handle = bucket.start_communication(process_group=process_group)
                    if handle:
                        bucket.wait_communication()
                        successful_buckets += 1
                    total_time += time.time() - start_time
                except Exception as e:
                    print(f"Bucket {bucket.bucket_id} sync failed: {e}")
                    failed_buckets += 1
        
        return {
            'total_time': total_time,
            'successful_buckets': successful_buckets,
            'failed_buckets': failed_buckets,
            'fallback_used': True
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from active manager."""
        active_manager = (self.fallback_manager if self.fallback_manager is not None 
                         else self.primary_manager)
        
        stats = active_manager.get_statistics()
        stats['error_count'] = self.error_count
        stats['using_fallback'] = self.fallback_manager is not None
        
        return stats

# Usage with error recovery
def robust_training_step(model, robust_manager, batch, optimizer):
    """Training step with comprehensive error handling."""
    
    optimizer.zero_grad()
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    
    # Collect gradients with validation
    gradients = {}
    invalid_gradients = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Basic validation
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                invalid_gradients.append(name)
                # Zero out invalid gradients
                param.grad.zero_()
            else:
                gradients[name] = param.grad
    
    if invalid_gradients:
        print(f"Warning: Found invalid gradients in {len(invalid_gradients)} parameters")
    
    # Safe assignment and synchronization
    assignments = robust_manager.safe_assign_gradients(gradients)
    sync_stats = robust_manager.safe_synchronize()
    
    # Apply gradients
    active_manager = (robust_manager.fallback_manager 
                     if robust_manager.fallback_manager is not None 
                     else robust_manager.primary_manager)
    
    updated_gradients = active_manager.get_bucket_assignments()
    for name, param in model.named_parameters():
        if name in updated_gradients:
            param.grad = updated_gradients[name]
    
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'assignments': len(assignments),
        'sync_stats': sync_stats,
        'invalid_gradients': len(invalid_gradients),
        'manager_stats': robust_manager.get_statistics()
    }
```

This implementation guide provides practical, production-ready patterns for using RoseLLM's gradient bucketing system effectively. The examples demonstrate both basic usage and advanced optimization techniques that would be valuable in technical interviews and real-world deployments.