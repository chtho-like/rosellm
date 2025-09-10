# Memory-Mapped Indexed Dataset Implementation Plan for RoseLLM

## Executive Summary

After analyzing both the RoseLLM and Megatron-LM codebases, I've identified **Memory-Mapped Indexed Dataset Support** as the next optimal mini-feature for implementation. This feature is crucial for efficient training on massive datasets, allowing RoseLLM to handle datasets larger than available RAM through memory-mapped I/O and intelligent indexing.

## Feature Description and Rationale

### What is Memory-Mapped Indexed Dataset?

The Memory-Mapped Indexed Dataset is a high-performance data loading mechanism that:
- Stores tokenized data in binary format with an accompanying index file
- Uses memory-mapped I/O for efficient, lazy loading of data
- Supports document boundaries and sequence-level indexing
- Enables training on datasets larger than available memory
- Provides zero-copy data access with minimal overhead

### Why This Feature?

1. **High Impact**: Enables training on massive datasets (100GB+) without loading into RAM
2. **Performance**: 10-100x faster data loading compared to dynamic tokenization
3. **Compatibility**: Direct compatibility with Megatron-LM preprocessed datasets
4. **Incremental**: Can be implemented alongside existing data loaders
5. **Well-Established**: Proven pattern in Megatron-LM with clear implementation

### Current Gap Analysis

RoseLLM currently lacks:
- Binary dataset format with indexing
- Memory-mapped I/O for large datasets
- Efficient document-aware data loading
- Preprocessing tools for dataset conversion
- Integration with distributed data loading

## Technical Architecture

### Core Components

```python
rosellm/rosetrainer/datasets/
├── __init__.py
├── indexed_dataset.py       # Core MMapIndexedDataset class
├── index_builder.py         # Dataset building utilities
├── data_types.py           # Data type definitions
├── distributed_dataset.py  # Distributed sampling
└── preprocessing/
    ├── __init__.py
    ├── tokenizer_wrapper.py
    └── text_processor.py
```

### Key Classes and Interfaces

#### 1. Data Type Management
```python
class DType(Enum):
    """NumPy data type enumeration for binary format."""
    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float32 = 7
    uint16 = 8
    
    @classmethod
    def optimal_dtype(cls, vocab_size: int) -> Type[np.number]:
        """Select optimal dtype based on vocabulary size."""
        if vocab_size < 256:
            return np.uint8
        elif vocab_size < 65536:
            return np.uint16
        else:
            return np.int32
```

#### 2. Index Reader/Writer
```python
class IndexWriter:
    """Write index files for efficient data access."""
    
    def __init__(self, path: str, dtype: Type[np.number]):
        self.path = path
        self.dtype = dtype
        
    def write(self, 
              sequence_lengths: List[int],
              sequence_offsets: List[int],
              document_indices: List[int]) -> None:
        """Write index with sequence and document boundaries."""

class IndexReader:
    """Read index files using memory mapping."""
    
    def __init__(self, path: str):
        self.mmap = np.memmap(path, mode='r', order='C')
        self._parse_header()
        self._load_indices()
```

#### 3. Core Dataset Class
```python
class MMapIndexedDataset(torch.utils.data.Dataset):
    """Memory-mapped dataset with lazy loading."""
    
    def __init__(self, 
                 path_prefix: str,
                 mmap: bool = True,
                 lazy: bool = True):
        """
        Args:
            path_prefix: Path without .idx/.bin extension
            mmap: Use memory mapping (vs file I/O)
            lazy: Load data on demand (vs preload)
        """
        self.index = IndexReader(f"{path_prefix}.idx")
        self.data_mmap = np.memmap(f"{path_prefix}.bin", 
                                   dtype=self.index.dtype,
                                   mode='r', order='C')
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """Get sequence at index with zero-copy efficiency."""
        ptr, length = self.index.get_sequence_bounds(idx)
        return self.data_mmap[ptr:ptr+length]
```

#### 4. Dataset Builder
```python
class IndexedDatasetBuilder:
    """Build indexed datasets from raw text."""
    
    def __init__(self, 
                 output_prefix: str,
                 vocab_size: int,
                 tokenizer: Any):
        self.output_prefix = output_prefix
        self.dtype = DType.optimal_dtype(vocab_size)
        self.tokenizer = tokenizer
        
    def add_document(self, text: str) -> None:
        """Process and add a document to the dataset."""
        tokens = self.tokenizer.encode(text)
        self._write_tokens(tokens)
        self._update_index()
    
    def finalize(self) -> None:
        """Write index file and close data file."""
```

### Integration Points

#### 1. Distributed Data Loading
```python
class DistributedIndexedDataset(MMapIndexedDataset):
    """Distributed wrapper with rank-aware sampling."""
    
    def __init__(self, 
                 path_prefix: str,
                 rank: int,
                 world_size: int,
                 shuffle: bool = True):
        super().__init__(path_prefix)
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        
    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch
        
    def __iter__(self):
        """Yield samples for this rank."""
        indices = self._get_rank_indices()
        if self.shuffle:
            indices = self._shuffle_with_seed(indices, self.epoch)
        for idx in indices:
            yield self[idx]
```

#### 2. Integration with RoseTrainer
```python
# In rosellm/rosetrainer/engine.py
class RoseTrainer:
    def create_dataloader(self, dataset_path: str, **kwargs):
        """Create dataloader with indexed dataset support."""
        if dataset_path.endswith('.idx'):
            # Use indexed dataset
            dataset = DistributedIndexedDataset(
                path_prefix=dataset_path[:-4],
                rank=self.rank,
                world_size=self.world_size,
                **kwargs
            )
        else:
            # Fall back to existing loader
            dataset = self._create_standard_dataset(dataset_path)
        
        return DataLoader(dataset, ...)
```

## Memory and Performance Analysis

### Memory Efficiency
- **Traditional Loading**: O(dataset_size) memory
- **MMap Loading**: O(1) memory + OS page cache
- **Effective for**: Datasets > 10GB

### Performance Characteristics
- **Initial Load**: < 100ms (index only)
- **Random Access**: O(1) with mmap
- **Sequential Access**: Optimized by OS prefetching
- **Cache Locality**: Automatic via OS page cache

### Benchmarks (Expected)
```
Dataset Size | Traditional RAM | MMap RAM | Load Time
10 GB        | 10 GB          | ~100 MB  | 0.1s
100 GB       | OOM            | ~100 MB  | 0.1s  
1 TB         | OOM            | ~100 MB  | 0.2s
```

## Implementation Milestones

### Phase 1: Core Infrastructure (200 lines)
- [ ] Implement DType enumeration and utilities
- [ ] Create IndexWriter for building index files
- [ ] Create IndexReader with memory mapping
- [ ] Implement basic MMapIndexedDataset

### Phase 2: Builder and Tools (150 lines)
- [ ] Implement IndexedDatasetBuilder
- [ ] Add text preprocessing utilities
- [ ] Create conversion script for existing datasets
- [ ] Add validation utilities

### Phase 3: Distributed Integration (100 lines)
- [ ] Implement DistributedIndexedDataset
- [ ] Add epoch-based shuffling
- [ ] Integrate with RoseTrainer
- [ ] Add configuration options

### Phase 4: Testing and Validation (150 lines)
- [ ] Unit tests for all components
- [ ] Bit-to-bit validation against Megatron-LM
- [ ] Performance benchmarks
- [ ] End-to-end training test

## Testing Strategy

### Unit Tests
```python
def test_index_writer_reader():
    """Test index file round-trip."""
    builder = IndexedDatasetBuilder("test", vocab_size=50000)
    builder.add_document("Hello world")
    builder.finalize()
    
    dataset = MMapIndexedDataset("test")
    assert len(dataset) == 1
    assert dataset[0].tolist() == [/* token ids */]

def test_distributed_sampling():
    """Test distributed dataset sharding."""
    dataset = DistributedIndexedDataset("data", rank=0, world_size=2)
    rank0_samples = list(dataset)
    
    dataset = DistributedIndexedDataset("data", rank=1, world_size=2)
    rank1_samples = list(dataset)
    
    # No overlap between ranks
    assert set(rank0_samples).isdisjoint(set(rank1_samples))
```

### Integration Tests
```python
def test_trainer_integration():
    """Test integration with RoseTrainer."""
    # Prepare indexed dataset
    prepare_indexed_dataset("train_data")
    
    # Create trainer
    trainer = RoseTrainer(model, config)
    dataloader = trainer.create_dataloader("train_data")
    
    # Train one step
    for batch in dataloader:
        loss = trainer.train_step(batch)
        assert loss.item() > 0
        break
```

### Validation Against Megatron-LM
```python
def test_megatron_compatibility():
    """Validate against Megatron-LM implementation."""
    # Load same dataset with both implementations
    megatron_ds = MegatronIndexedDataset("data")
    rose_ds = MMapIndexedDataset("data")
    
    # Compare samples
    for i in range(len(megatron_ds)):
        assert np.array_equal(megatron_ds[i], rose_ds[i])
```

## End-to-End Usage Example

```python
# Step 1: Preprocess and build indexed dataset
from rosellm.rosetrainer.datasets import IndexedDatasetBuilder
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
builder = IndexedDatasetBuilder(
    output_prefix="data/wiki_train",
    vocab_size=tokenizer.vocab_size,
    tokenizer=tokenizer
)

# Add documents
with open("wikipedia.txt") as f:
    for line in f:
        if line.strip():
            builder.add_document(line)
            
builder.finalize()
print(f"Built dataset with {builder.num_sequences} sequences")

# Step 2: Use in training
from rosellm.rosetrainer import RoseTrainer
from rosellm.rosetrainer.datasets import DistributedIndexedDataset

# Initialize trainer
trainer = RoseTrainer(
    model=model,
    optimizer=optimizer,
    config=config,
    local_rank=local_rank,
    world_size=world_size
)

# Create distributed dataset
train_dataset = DistributedIndexedDataset(
    path_prefix="data/wiki_train",
    rank=local_rank,
    world_size=world_size,
    shuffle=True
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    num_workers=4,
    pin_memory=True
)

# Training loop
for epoch in range(config.num_epochs):
    train_dataset.set_epoch(epoch)  # For reproducible shuffling
    
    for batch_idx, tokens in enumerate(train_loader):
        loss = trainer.train_step(tokens)
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
```

## Success Metrics

1. **Functional Correctness**
   - All unit tests pass
   - Bit-to-bit match with Megatron-LM for same data
   - No memory leaks or segfaults

2. **Performance**
   - Dataset loading < 1 second for any size
   - Memory usage < 500MB for 1TB dataset
   - Data loading not a bottleneck (< 5% of step time)

3. **Usability**
   - Simple API matching Megatron-LM patterns
   - Clear documentation and examples
   - Seamless integration with existing code

4. **Compatibility**
   - Works with existing RoseLLM features
   - Supports all parallelism dimensions
   - Compatible with checkpointing

## Potential Pitfalls and Solutions

### Pitfall 1: Platform Compatibility
**Issue**: mmap behavior differs across OS
**Solution**: Add platform-specific handling and fallback to file I/O

### Pitfall 2: Memory Pressure
**Issue**: OS may struggle with very large mmaps
**Solution**: Add chunked reading option for extreme cases

### Pitfall 3: Distributed Synchronization
**Issue**: Ranks may have different data views
**Solution**: Use deterministic shuffling with shared seed

### Pitfall 4: Data Corruption
**Issue**: Corrupted index/data files
**Solution**: Add checksums and validation on load

## Conclusion

The Memory-Mapped Indexed Dataset feature represents an ideal next step for RoseLLM:
- **High impact** on training efficiency and scalability
- **Manageable scope** (~600 lines of core code)
- **Clear patterns** from Megatron-LM reference
- **Incremental** implementation possible
- **Testable** with clear validation criteria

This feature will significantly enhance RoseLLM's ability to handle large-scale training workloads while maintaining compatibility with the Megatron-LM ecosystem.