# Optimizer Factory & Configuration Design Patterns

## Executive Summary

The Optimizer Factory pattern in RoseLLM demonstrates enterprise-grade design principles for managing complex optimizer configurations in distributed training systems. This document explores the architectural decisions, design patterns, and implementation strategies that enable flexible, maintainable, and performant optimizer management at scale.

## Design Pattern Deep Dive

### 1. Factory Method Pattern Implementation

The `OptimizerFactory` class implements a sophisticated factory pattern with multiple creation strategies:

```python
class OptimizerFactory:
    """
    Implements Factory Method pattern with:
    1. Static factory methods for common use cases
    2. Registry pattern for extensibility
    3. Builder pattern for complex configurations
    4. Strategy pattern for optimizer selection
    """
    
    # Registry for custom optimizers
    _optimizer_registry: Dict[str, Type[Optimizer]] = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD,
        'LAMB': LAMB,  # Custom optimizer
        'FusedAdam': FusedAdam,  # APEX optimizer
    }
    
    # Preset configurations (Strategy pattern)
    _presets: Dict[str, DistributedOptimizerConfig] = {
        'baseline': DistributedOptimizerConfig(
            partition_parameters=False,
            partition_gradients=False,
            partition_optimizer_states=False
        ),
        'memory_efficient': DistributedOptimizerConfig(
            partition_parameters=True,
            partition_gradients=True,
            partition_optimizer_states=True,
            contiguous_gradients=True,
            cpu_offload=False
        ),
        'extreme_scale': DistributedOptimizerConfig(
            partition_parameters=True,
            partition_gradients=True,
            partition_optimizer_states=True,
            cpu_offload=True,
            mixed_precision=True,
            memory_efficient_fp16=True
        )
    }
```

**Interview Insight**: The factory pattern provides several benefits:
1. **Encapsulation**: Complex creation logic hidden from clients
2. **Flexibility**: Easy to add new optimizer types
3. **Testability**: Can mock optimizers for testing
4. **Configuration Management**: Centralized preset management

### 2. Builder Pattern for Configuration

The configuration system uses a fluent builder pattern:

```python
class OptimizerConfigBuilder:
    """
    Fluent interface for building optimizer configurations.
    
    Example usage:
    config = (OptimizerConfigBuilder()
        .with_mixed_precision()
        .with_gradient_clipping(1.0)
        .with_cpu_offload()
        .with_bucket_size(50)
        .build())
    """
    
    def __init__(self):
        self._config = DistributedOptimizerConfig()
    
    def with_mixed_precision(self, dtype: torch.dtype = torch.float16):
        self._config.mixed_precision = True
        self._config.dtype = dtype
        return self
    
    def with_gradient_clipping(self, max_norm: float):
        self._config.grad_clip_value = max_norm
        return self
    
    def with_cpu_offload(self):
        self._config.cpu_offload = True
        self._config.partition_optimizer_states = True  # Required dependency
        return self
    
    def validate(self):
        """Validates configuration consistency."""
        if self._config.cpu_offload and not self._config.partition_optimizer_states:
            raise ValueError("CPU offload requires optimizer state partitioning")
        return self
    
    def build(self) -> DistributedOptimizerConfig:
        self.validate()
        return self._config
```

**Megatron-LM Comparison**: Megatron uses a similar builder pattern but adds:
- Hierarchical configuration (global → model → layer specific)
- Configuration inheritance for parameter groups
- Automatic conflict resolution

### 3. Registry Pattern for Extensibility

The optimizer registry allows dynamic registration:

```python
class OptimizerRegistry:
    """
    Maintains registry of optimizer implementations with metadata.
    """
    
    def __init__(self):
        self._optimizers: Dict[str, OptimizerInfo] = {}
    
    def register(
        self,
        name: str,
        optimizer_class: Type[Optimizer],
        default_config: Optional[Dict] = None,
        compatible_with: Optional[List[str]] = None
    ):
        """
        Register a new optimizer with metadata.
        
        Args:
            name: Optimizer identifier
            optimizer_class: Optimizer implementation
            default_config: Default hyperparameters
            compatible_with: List of compatible features (e.g., 'mixed_precision')
        """
        self._optimizers[name] = OptimizerInfo(
            cls=optimizer_class,
            default_config=default_config or {},
            compatible_features=compatible_with or [],
            memory_multiplier=self._compute_memory_multiplier(optimizer_class)
        )
    
    def _compute_memory_multiplier(self, optimizer_class: Type[Optimizer]) -> float:
        """
        Computes memory requirements based on optimizer type.
        
        Returns multiplier relative to parameter size.
        """
        # Inspect optimizer to determine state variables
        if hasattr(optimizer_class, '__init__'):
            import inspect
            sig = inspect.signature(optimizer_class.__init__)
            
            # Heuristic based on common patterns
            if 'momentum' in sig.parameters and 'variance' in sig.parameters:
                return 2.0  # Adam-like (momentum + variance)
            elif 'momentum' in sig.parameters:
                return 1.0  # SGD with momentum
            else:
                return 0.0  # No state (vanilla SGD)
        
        return 2.0  # Conservative default
```

## Configuration Management System

### 1. Hierarchical Configuration

The configuration system supports inheritance and overrides:

```python
@dataclass
class HierarchicalConfig:
    """
    Implements hierarchical configuration with precedence:
    Global < Model < Layer < Parameter Group
    """
    
    global_config: DistributedOptimizerConfig
    model_overrides: Dict[str, Any] = field(default_factory=dict)
    layer_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    param_group_configs: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_effective_config(
        self, 
        param_name: str,
        layer_name: Optional[str] = None
    ) -> DistributedOptimizerConfig:
        """
        Resolves effective configuration for a parameter.
        
        Precedence order (highest to lowest):
        1. Parameter group specific
        2. Layer specific
        3. Model overrides
        4. Global defaults
        """
        config = dataclasses.replace(self.global_config)
        
        # Apply model-level overrides
        for key, value in self.model_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Apply layer-specific config if available
        if layer_name and layer_name in self.layer_configs:
            for key, value in self.layer_configs[layer_name].items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Check parameter group configs
        for pg_config in self.param_group_configs:
            if param_name in pg_config.get('params', []):
                for key, value in pg_config.items():
                    if key != 'params' and hasattr(config, key):
                        setattr(config, key, value)
        
        return config
```

### 2. Configuration Validation and Sanitization

```python
class ConfigValidator:
    """
    Validates and sanitizes optimizer configurations.
    """
    
    @staticmethod
    def validate_config(config: DistributedOptimizerConfig) -> List[str]:
        """
        Performs comprehensive validation.
        
        Returns list of warnings (empty if valid).
        """
        warnings = []
        
        # Check dependency constraints
        if config.overlap_grad_reduce and not config.contiguous_gradients:
            warnings.append("Overlapped reduction requires contiguous gradients")
        
        # Check memory constraints
        if config.cpu_offload and not config.partition_optimizer_states:
            warnings.append("CPU offload requires state partitioning")
        
        # Check performance implications
        if config.bucket_size_mb < 10:
            warnings.append(f"Small bucket size ({config.bucket_size_mb}MB) may hurt performance")
        
        if config.bucket_size_mb > 100:
            warnings.append(f"Large bucket size ({config.bucket_size_mb}MB) may cause memory spikes")
        
        # Check numerical stability
        if config.mixed_precision and config.grad_clip_value is None:
            warnings.append("Mixed precision without gradient clipping may be unstable")
        
        # Check hardware compatibility
        if config.use_hierarchical_allreduce and not self._check_nvlink_available():
            warnings.append("Hierarchical AllReduce requires NVLink for best performance")
        
        return warnings
    
    @staticmethod
    def sanitize_config(config: DistributedOptimizerConfig) -> DistributedOptimizerConfig:
        """
        Automatically fixes common configuration issues.
        """
        sanitized = dataclasses.replace(config)
        
        # Fix dependency violations
        if sanitized.partition_parameters:
            sanitized.partition_gradients = True
        
        if sanitized.cpu_offload:
            sanitized.partition_optimizer_states = True
        
        # Optimize settings based on hardware
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            
            # Adjust bucket size based on GPU memory
            if device_props.total_memory < 16 * 1024**3:  # <16GB
                sanitized.bucket_size_mb = min(sanitized.bucket_size_mb, 25)
            
            # Enable TF32 on Ampere+ GPUs
            if device_props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
        
        return sanitized
```

### 3. Dynamic Configuration Adaptation

The system can adapt configuration based on runtime conditions:

```python
class DynamicConfigAdapter:
    """
    Adapts configuration based on runtime metrics.
    """
    
    def __init__(self, base_config: DistributedOptimizerConfig):
        self.base_config = base_config
        self.metrics_history = []
        self.adaptation_count = 0
    
    def adapt_based_on_metrics(
        self,
        memory_usage: float,
        communication_time: float,
        computation_time: float
    ) -> DistributedOptimizerConfig:
        """
        Dynamically adjusts configuration based on performance metrics.
        """
        adapted = dataclasses.replace(self.base_config)
        
        # Memory pressure adaptation
        if memory_usage > 0.9:  # >90% memory used
            # Enable more aggressive memory saving
            adapted.cpu_offload = True
            adapted.gradient_accumulation_steps *= 2
            adapted.bucket_size_mb = max(10, adapted.bucket_size_mb // 2)
            
        elif memory_usage < 0.5:  # <50% memory used
            # Relax memory constraints for performance
            adapted.cpu_offload = False
            adapted.bucket_size_mb = min(100, adapted.bucket_size_mb * 2)
        
        # Communication bottleneck adaptation
        comm_ratio = communication_time / (communication_time + computation_time)
        if comm_ratio > 0.3:  # >30% time in communication
            # Increase bucket size to amortize communication
            adapted.bucket_size_mb = min(100, adapted.bucket_size_mb * 1.5)
            adapted.contiguous_gradients = True
            
        # Numerical stability adaptation
        if self._detect_gradient_explosion():
            adapted.grad_clip_value = (adapted.grad_clip_value or 1.0) * 0.5
        
        self.adaptation_count += 1
        return adapted
    
    def _detect_gradient_explosion(self) -> bool:
        """Detects gradient explosion from metrics history."""
        if len(self.metrics_history) < 10:
            return False
        
        recent_grads = [m['grad_norm'] for m in self.metrics_history[-10:]]
        return any(g > 100.0 for g in recent_grads)
```

## Factory Method Implementations

### 1. Create from Model

```python
@classmethod
def create_from_model(
    cls,
    model: nn.Module,
    optimizer_name: str = "AdamW",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    preset: str = "auto",
    process_group: Optional[dist.ProcessGroup] = None,
    **optimizer_kwargs
) -> DistributedOptimizer:
    """
    Creates optimizer from model with intelligent defaults.
    
    Interview Insight: The 'auto' preset demonstrates
    intelligent system design - adapting to model characteristics.
    """
    
    # Auto-detect best configuration
    if preset == "auto":
        preset = cls._detect_best_preset(model)
    
    # Get preset configuration
    config = cls._get_preset_config(preset)
    
    # Analyze model for parameter groups
    param_groups = cls._analyze_model_parameters(model)
    
    # Create optimizer
    optimizer_class = cls._optimizer_registry[optimizer_name]
    
    return DistributedOptimizer(
        params=param_groups,
        optimizer_class=optimizer_class,
        optimizer_kwargs={
            'lr': lr,
            'weight_decay': weight_decay,
            **optimizer_kwargs
        },
        config=config,
        process_group=process_group
    )

@staticmethod
def _detect_best_preset(model: nn.Module) -> str:
    """
    Intelligently selects preset based on model characteristics.
    
    Decision tree based on:
    1. Model size
    2. Available GPU memory
    3. Number of GPUs
    4. Model architecture
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Get available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        world_size = dist.get_world_size() if dist.is_initialized() else 1
    else:
        gpu_memory = 16 * 1024**3  # Assume 16GB for CPU
        world_size = 1
    
    # Estimate memory requirements
    param_memory = total_params * 2  # FP16
    optimizer_memory = total_params * 8  # Adam states
    total_memory = param_memory + optimizer_memory
    
    # Decision logic
    if total_memory > gpu_memory * 0.8:
        # Memory constrained - use aggressive optimization
        return 'extreme_scale'
    elif total_memory > gpu_memory * 0.5:
        # Moderate memory pressure
        return 'memory_efficient'
    elif world_size > 1:
        # Distributed but not memory constrained
        return 'speed_optimized'
    else:
        # Single GPU with plenty of memory
        return 'baseline'
```

### 2. Parameter Group Analysis

```python
@staticmethod
def _analyze_model_parameters(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Analyzes model parameters to create optimal parameter groups.
    
    Groups parameters by:
    1. Layer type (embeddings, attention, FFN, etc.)
    2. Parameter size
    3. Gradient characteristics
    """
    param_groups = []
    
    # Group 1: Embeddings (often need different LR)
    embedding_params = []
    embedding_names = []
    
    # Group 2: Layer normalization (often no weight decay)
    norm_params = []
    norm_names = []
    
    # Group 3: Attention weights
    attention_params = []
    attention_names = []
    
    # Group 4: Everything else
    other_params = []
    other_names = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'embedding' in name.lower():
            embedding_params.append(param)
            embedding_names.append(name)
        elif 'norm' in name.lower() or 'layernorm' in name.lower():
            norm_params.append(param)
            norm_names.append(name)
        elif 'attention' in name.lower() or 'attn' in name.lower():
            attention_params.append(param)
            attention_names.append(name)
        else:
            other_params.append(param)
            other_names.append(name)
    
    # Create parameter groups with specific settings
    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'names': embedding_names,
            'lr_multiplier': 0.5,  # Often use lower LR for embeddings
            'weight_decay': 0.0,   # No decay for embeddings
        })
    
    if norm_params:
        param_groups.append({
            'params': norm_params,
            'names': norm_names,
            'weight_decay': 0.0,  # No weight decay for normalization
        })
    
    if attention_params:
        param_groups.append({
            'params': attention_params,
            'names': attention_names,
            'lr_multiplier': 1.0,
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'names': other_names,
            'lr_multiplier': 1.0,
        })
    
    return param_groups if param_groups else [{'params': model.parameters()}]
```

## Interview Deep Dive Questions

### Q1: How would you extend the factory to support custom optimizers?

**Answer**:

Implement a plugin system with discovery and validation:

```python
class OptimizerPlugin:
    """Base class for optimizer plugins."""
    
    @abstractmethod
    def get_optimizer_class(self) -> Type[Optimizer]:
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def validate_compatibility(self, config: DistributedOptimizerConfig) -> bool:
        pass

class PluginManager:
    """Discovers and manages optimizer plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, OptimizerPlugin] = {}
    
    def discover_plugins(self, plugin_dir: str):
        """Dynamically loads plugins from directory."""
        import importlib.util
        import os
        
        for filename in os.listdir(plugin_dir):
            if filename.endswith('_optimizer.py'):
                spec = importlib.util.spec_from_file_location(
                    filename[:-3], 
                    os.path.join(plugin_dir, filename)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find OptimizerPlugin subclasses
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, OptimizerPlugin) and 
                        obj != OptimizerPlugin):
                        
                        plugin = obj()
                        self.register_plugin(name, plugin)
    
    def register_plugin(self, name: str, plugin: OptimizerPlugin):
        """Registers a plugin after validation."""
        # Validate plugin
        try:
            optimizer_cls = plugin.get_optimizer_class()
            default_config = plugin.get_default_config()
            
            # Test instantiation
            test_param = torch.zeros(1, requires_grad=True)
            test_optimizer = optimizer_cls([test_param], **default_config)
            
            self.plugins[name] = plugin
            
        except Exception as e:
            raise ValueError(f"Invalid plugin {name}: {e}")
```

### Q2: How do you handle configuration versioning and migration?

**Answer**:

Implement a version-aware configuration system:

```python
class ConfigVersion:
    """Manages configuration versioning and migration."""
    
    CURRENT_VERSION = "2.0.0"
    
    migrations = {
        "1.0.0": {
            "2.0.0": lambda c: cls._migrate_1_0_to_2_0(c)
        }
    }
    
    @classmethod
    def migrate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrates configuration to current version."""
        version = config.get('version', '1.0.0')
        
        if version == cls.CURRENT_VERSION:
            return config
        
        # Find migration path
        migration_path = cls._find_migration_path(version, cls.CURRENT_VERSION)
        
        migrated = config.copy()
        for from_version, to_version in migration_path:
            migration_fn = cls.migrations[from_version][to_version]
            migrated = migration_fn(migrated)
            migrated['version'] = to_version
        
        return migrated
    
    @staticmethod
    def _migrate_1_0_to_2_0(config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrates from version 1.0 to 2.0."""
        migrated = config.copy()
        
        # Rename deprecated fields
        if 'use_cpu_offload' in migrated:
            migrated['cpu_offload'] = migrated.pop('use_cpu_offload')
        
        # Add new required fields with defaults
        if 'memory_efficient_fp16' not in migrated:
            migrated['memory_efficient_fp16'] = False
        
        # Update changed semantics
        if migrated.get('bucket_size') is not None:
            # Changed from bytes to MB
            migrated['bucket_size_mb'] = migrated.pop('bucket_size') // (1024**2)
        
        return migrated
```

### Q3: Explain the memory profiler integration with the factory.

**Answer**:

The factory integrates with the memory profiler for intelligent configuration:

```python
class MemoryAwareFactory:
    """Factory that adapts to memory constraints."""
    
    @classmethod
    def create_with_memory_budget(
        cls,
        model: nn.Module,
        memory_budget_gb: float,
        optimizer_name: str = "AdamW",
        **kwargs
    ) -> DistributedOptimizer:
        """
        Creates optimizer that fits within memory budget.
        """
        profiler = MemoryProfiler()
        
        # Profile model memory
        model_memory = profiler.analyze_model_memory(model)
        
        # Calculate available memory for optimizer
        available_for_optimizer = memory_budget_gb - (model_memory['total_mb'] / 1024)
        
        # Determine optimal configuration
        config = cls._optimize_for_memory_budget(
            model,
            available_for_optimizer,
            optimizer_name
        )
        
        # Create optimizer with monitoring
        optimizer = cls.create(
            model.parameters(),
            optimizer_name,
            config=config,
            **kwargs
        )
        
        # Attach profiler for continuous monitoring
        optimizer.memory_profiler = profiler
        
        return optimizer
    
    @staticmethod
    def _optimize_for_memory_budget(
        model: nn.Module,
        budget_gb: float,
        optimizer_name: str
    ) -> DistributedOptimizerConfig:
        """
        Finds optimal configuration within memory budget.
        
        Uses binary search over configuration space.
        """
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate base memory requirements
        state_multiplier = {'SGD': 1, 'Adam': 2, 'AdamW': 2, 'LAMB': 2.25}[optimizer_name]
        base_memory_gb = (total_params * 4 * state_multiplier) / (1024**3)
        
        if base_memory_gb <= budget_gb:
            # Fits without optimization
            return DistributedOptimizerConfig(partition_parameters=False)
        
        # Need memory optimization
        config = DistributedOptimizerConfig()
        
        # Progressive optimization levels
        if base_memory_gb / 2 <= budget_gb:
            # Level 1: Partition optimizer states only
            config.partition_optimizer_states = True
            
        elif base_memory_gb / 4 <= budget_gb:
            # Level 2: Also partition gradients
            config.partition_optimizer_states = True
            config.partition_gradients = True
            
        else:
            # Level 3: Maximum memory saving
            config.partition_parameters = True
            config.partition_gradients = True
            config.partition_optimizer_states = True
            config.cpu_offload = True
            config.mixed_precision = True
        
        return config
```

### Q4: How does the factory handle distributed training nuances?

**Answer**:

The factory implements distributed-aware creation:

```python
class DistributedAwareFactory:
    """Factory with distributed training awareness."""
    
    @classmethod
    def create_for_distributed(
        cls,
        model: nn.Module,
        world_size: int,
        rank: int,
        backend: str = 'nccl',
        **kwargs
    ) -> DistributedOptimizer:
        """
        Creates optimizer optimized for distributed setup.
        """
        # Detect network topology
        topology = cls._detect_network_topology()
        
        # Optimize configuration for topology
        config = cls._optimize_for_topology(topology, world_size)
        
        # Handle backend-specific optimizations
        if backend == 'nccl':
            config.use_nccl_optimization = True
            config.bucket_size_mb = 25  # NCCL optimal
        elif backend == 'gloo':
            config.use_nccl_optimization = False
            config.bucket_size_mb = 10  # Smaller for Gloo
        
        # Create process group if needed
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank
            )
        
        process_group = dist.group.WORLD
        
        return cls.create(
            model.parameters(),
            config=config,
            process_group=process_group,
            **kwargs
        )
    
    @staticmethod
    def _detect_network_topology() -> Dict[str, Any]:
        """Detects network topology for optimization."""
        topology = {
            'has_nvlink': False,
            'has_infiniband': False,
            'numa_nodes': 1,
            'gpus_per_node': 1
        }
        
        if torch.cuda.is_available():
            # Check for NVLink
            import subprocess
            try:
                result = subprocess.run(
                    ['nvidia-smi', 'nvlink', '-s'],
                    capture_output=True,
                    text=True
                )
                topology['has_nvlink'] = 'Link 0' in result.stdout
            except:
                pass
            
            # Count GPUs per node
            topology['gpus_per_node'] = torch.cuda.device_count()
        
        return topology
    
    @staticmethod
    def _optimize_for_topology(
        topology: Dict[str, Any],
        world_size: int
    ) -> DistributedOptimizerConfig:
        """Optimizes configuration for network topology."""
        config = DistributedOptimizerConfig()
        
        if topology['has_nvlink'] and topology['gpus_per_node'] > 1:
            # NVLink available - use hierarchical communication
            config.use_hierarchical_allreduce = True
            config.bucket_size_mb = 50  # Larger buckets for NVLink
        
        if topology['has_infiniband']:
            # InfiniBand for inter-node
            config.allgather_bucket_size_mb = 100
        
        if world_size > 8:
            # Large scale - optimize for communication
            config.contiguous_gradients = True
            config.overlap_grad_reduce = True
        
        return config
```

## Conclusion

The Optimizer Factory and Configuration system in RoseLLM demonstrates several key software engineering principles:

1. **Design Patterns**: Effective use of Factory, Builder, Registry, and Strategy patterns
2. **Separation of Concerns**: Clear boundaries between creation, configuration, and optimization
3. **Extensibility**: Plugin architecture for custom optimizers
4. **Robustness**: Validation, sanitization, and migration support
5. **Performance**: Topology-aware and memory-aware optimizations
6. **Maintainability**: Clear interfaces and comprehensive documentation

These patterns enable scalable, maintainable, and performant distributed training systems that can adapt to diverse hardware configurations and training requirements.

## References

1. Gamma, E., et al. (1994). "Design Patterns: Elements of Reusable Object-Oriented Software"
2. Fowler, M. (2002). "Patterns of Enterprise Application Architecture"
3. PyTorch Distributed Training: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
4. NVIDIA Collective Communication Library: https://developer.nvidia.com/nccl
5. Meta's FairScale: https://github.com/facebookresearch/fairscale