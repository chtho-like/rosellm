# RoseLLM

A comprehensive RLHF (Reinforcement Learning from Human Feedback) framework with three main components:

## Components

### RoseTrainer (Distributed Training)
- `engine.py`: Main training engine with distributed initialization
- `parallelism/data_parallel.py`: Data parallelism with gradient averaging
- `parallelism/model_parallel.py`: Tensor parallelism with row/column parallel layers
- `parallelism/pipeline_parallel.py`: Pipeline parallelism with microbatching
- `parallelism/zero.py`: Zero Redundancy Optimizer for parameter partitioning
- `memory/activation_checkpoint.py`: Activation checkpointing for memory efficiency
- `memory/mixed_precision.py`: FP16 training with dynamic loss scaling
- `memory/cpu_offload.py`: CPU offloading for optimizer states and parameters

### RoseInfer (Distributed Inference)
- Optimized inference with tensor parallelism
- KV cache management
- Quantization support

### RoseRLHF (RLHF Implementation)
- Policy and value model training
- PPO implementation
- Reward model training

## Installation

```bash
# Clone the repository
git clone https://github.com/username/rosellm.git
cd rosellm

# Install the package
pip install -e .
```

## Basic Usage

```python
import torch
from rosellm.rosetrainer.engine import RoseTrainer

# Create a model (e.g., a HuggingFace model)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

# Create an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Configure training
config = {
    "max_grad_norm": 1.0,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
}

# Initialize the trainer
trainer = RoseTrainer(
    model=model,
    optimizer=optimizer,
    config=config,
    local_rank=0,  # Set from environment for multi-GPU
    world_size=1   # Set from environment for multi-GPU
)

# Prepare a batch (dict with tensors)
batch = {
    "input_ids": torch.tensor(...),
    "attention_mask": torch.tensor(...),
    ...
}

# Train for one step
result = trainer.train_step(batch)
print(f"Loss: {result['loss']}")

# Save a checkpoint
trainer.save_checkpoint("checkpoint.pt")

# Load a checkpoint
trainer.load_checkpoint("checkpoint.pt")
```

## Advanced Features

### Data Parallelism

```python
from rosellm.rosetrainer.parallelism.data_parallel import DataParallelTrainer

# Create data parallel trainer
dp_trainer = DataParallelTrainer(
    model=model,
    device=torch.device("cuda:0"),
    local_rank=0,
    world_size=8,  # Number of GPUs
    gradient_accumulation_steps=4
)

# Forward and backward pass with gradient accumulation
loss = dp_trainer.forward_backward(batch, optimizer)
```

### Tensor Parallelism

```python
from rosellm.rosetrainer.parallelism.model_parallel import TensorParallelism

# Initialize tensor parallelism
tp = TensorParallelism(
    local_rank=0,
    world_size=8,
    tp_size=2  # Split model across 2 GPUs
)

# Parallelize a linear layer
from rosellm.rosetrainer.parallelism.model_parallel import ColumnParallelLinear
parallel_layer = tp.parallelize_layer(model.linear, tp_type="column")
```

### Memory Optimizations

```python
# Apply activation checkpointing
from rosellm.rosetrainer.memory.activation_checkpoint import ActivationCheckpointing
model = ActivationCheckpointing.apply_to_transformer_layers(
    model, layer_attr="transformer.h"
)

# Use mixed precision training
from rosellm.rosetrainer.memory.mixed_precision import convert_model_to_fp16
convert_model_to_fp16(model)

# CPU offloading
from rosellm.rosetrainer.memory.cpu_offload import CPUOffloadOptimizer
optimizer = CPUOffloadOptimizer(optimizer, offload_params=True)
```

## Testing

The framework includes comprehensive tests for all components:

```bash
# Run all tests
python -m unittest discover -s tests

# Run specific tests
python -m unittest tests/rosetrainer/test_engine.py
python -m unittest tests/rosetrainer/memory/test_mixed_precision.py
```

## References

The implementation is based on the following papers and frameworks:

- Shoeybi, M. et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv:1909.08053 (2019)
- Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." arXiv:1910.02054 (2019)
- Huang, Y. et al. "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism." NeurIPS (2019)
- Micikevicius, P. et al. "Mixed Precision Training." arXiv:1710.03740 (2017)
- Chen, T. et al. "Training Deep Nets with Sublinear Memory Cost." arXiv:1604.06174 (2016)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

# Flash Attention CUDA Implementation

This repository contains a CUDA implementation of the Flash Attention algorithm based on the paper ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135).

## Overview

Flash Attention is a memory-efficient attention mechanism that avoids materializing the full attention matrix, enabling faster training and inference for transformer models with longer sequence lengths.

The implementation consists of:
- CUDA kernel for the forward pass of Flash Attention
- PyTorch C++ bindings
- Python wrapper for easy integration with PyTorch models

## Features

- Block-wise attention computation to reduce memory requirements
- Numerically stable softmax computation
- Multi-head attention support
- Compatible with PyTorch

## Installation

To build and install the CUDA extension:

```bash
python setup.py install
```

## Usage

```python
import torch
from flash_attention import FlashAttention, MultiHeadFlashAttention

# Single-head usage
model = FlashAttention()
output = model(q, k, v)

# Multi-head usage
model = MultiHeadFlashAttention(d_model=768, num_heads=12)
output = model(q, k, v)
```

## Requirements

- PyTorch >= 1.7.0
- CUDA Toolkit >= 10.2
- C++14 compatible compiler

## Implementation Details

The Flash Attention algorithm:
1. Divides Q, K, V into blocks to process sequentially
2. Uses shared memory to efficiently compute attention for each block
3. Maintains running statistics (m, l) to ensure numerical stability
4. Performs incremental updates to the output

This implementation is based on the algorithm description in the original paper and Python reference implementation.

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Tri Dao's Flash Attention implementation](https://github.com/HazyResearch/flash-attention)
