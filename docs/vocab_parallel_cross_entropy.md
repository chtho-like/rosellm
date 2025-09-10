# Vocabulary Parallel Cross-Entropy Loss

## Overview

The Vocabulary Parallel Cross-Entropy Loss is a critical optimization for training large language models with massive vocabularies. This feature splits the vocabulary dimension across tensor parallel ranks, enabling efficient memory usage and computation for models with 50K-256K+ token vocabularies.

## Key Features

- **Memory Efficiency**: Reduces per-GPU memory by factor of TP size for vocabulary embeddings
- **Compute Scalability**: Distributes cross-entropy computation across GPUs  
- **Label Smoothing**: Built-in support for label smoothing regularization
- **Numerical Stability**: Careful handling of logits max and exp operations
- **Megatron-LM Compatible**: Bit-to-bit accuracy with Megatron-LM reference

## Architecture

### Core Components

1. **VocabUtility**: Manages vocabulary partitioning across tensor parallel ranks
2. **VocabParallelCrossEntropy**: Core computation class with static methods
3. **_VocabParallelCrossEntropy**: Custom autograd function for forward/backward
4. **VocabParallelCrossEntropyLoss**: nn.Module wrapper for easy integration

### How It Works

1. **Vocabulary Partitioning**: Each TP rank handles vocab_size/tp_size tokens
2. **Forward Pass**:
   - Calculate local logits max, all-reduce across TP ranks
   - Compute exp(logits - max) for numerical stability
   - Extract predicted logits for target tokens in local partition
   - All-reduce predicted logits and sum_exp_logits
   - Calculate cross-entropy loss with optional label smoothing

3. **Backward Pass**:
   - Compute softmax gradients
   - Apply label smoothing adjustments if enabled
   - Scale by output gradients

## Usage

### Basic Example

```python
from rosellm.rosetrainer.tensor_parallel import vocab_parallel_cross_entropy

# Assume tensor parallel size = 4, vocab size = 50256
logits = model(input_ids)  # Shape: [seq_len, batch, 12564] (50256/4)
loss = vocab_parallel_cross_entropy(logits, targets, label_smoothing=0.1)
```

### Module Interface

```python
from rosellm.rosetrainer.tensor_parallel import VocabParallelCrossEntropyLoss

loss_fn = VocabParallelCrossEntropyLoss(
    label_smoothing=0.1,
    reduction='mean'
)
loss = loss_fn(logits, targets)
```

### Integration with Model

See `examples/vocab_parallel_training_example.py` for a complete training example with:
- Vocabulary parallel embeddings
- Vocabulary parallel output projection
- Full transformer model integration

## Memory Savings

For a model with:
- Vocab size: 50,256
- Hidden size: 768
- Sequence length: 512
- Batch size: 8

Standard memory for logits: 512 * 8 * 50256 * 4 bytes = ~786 MB
With TP=4: 512 * 8 * 12564 * 4 bytes = ~196 MB (4x reduction)

## Performance Considerations

1. **Communication Overhead**: Requires 2 all-reduce operations in forward, none in backward
2. **Load Balancing**: Ensure vocab_size is divisible by TP size for even partitioning
3. **Numerical Precision**: Uses float32 internally for stability, outputs match input dtype

## Testing

Run the test suite:
```bash
# Unit tests
pytest tests/rosetrainer/tensor_parallel/test_vocab_parallel_cross_entropy.py

# Megatron compatibility 
python tests/rosetrainer/tensor_parallel/test_megatron_compatibility.py

# End-to-end example
python examples/vocab_parallel_training_example.py
```

## Implementation Details

### Differences from Standard PyTorch

1. **Vocabulary Splitting**: Each rank only stores vocab_size/tp_size weights
2. **Distributed Reduction**: All-reduce operations for combining partial results
3. **Memory Layout**: Maintains [seq_len, batch, vocab_partition] format for Megatron compatibility

### Label Smoothing

Implements the formula:
```
smoothing = label_smoothing * vocab_size / (vocab_size - 1)
smoothed_loss = (1 - smoothing) * ce_loss - smoothing * mean_log_probs
```

This follows Megatron-LM's approach for consistency.

## Limitations

1. Vocabulary size must be divisible by tensor parallel size
2. Requires proper tensor parallel initialization
3. Currently supports only cross-entropy (not other loss functions)

## Future Enhancements

- [ ] Support for sparse targets
- [ ] Integration with mixture-of-experts models
- [ ] Adaptive vocabulary partitioning
- [ ] FP8 support for H100 GPUs

## References

- [Megatron-LM Paper](https://arxiv.org/abs/2104.04473)
- [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/1909.08053)
- [Label Smoothing](https://arxiv.org/abs/1512.00567)