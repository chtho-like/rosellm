"""End-to-end example of using memory-mapped indexed datasets with RoseLLM.

This example demonstrates:
1. Building an indexed dataset from raw text
2. Using the dataset for distributed training
3. Integration with RoseTrainer
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from rosellm.rosetrainer import RoseTrainer
from rosellm.rosetrainer.config import TrainingConfig
from rosellm.rosetrainer.datasets import (
    DistributedIndexedDataset,
    IndexedDatasetBuilder,
    MMapIndexedDataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dataset_from_text(
    input_file: str,
    output_prefix: str,
    vocab_size: int = 50000,
    max_seq_length: int = 512,
) -> dict:
    """Build indexed dataset from text file.

    Args:
        input_file: Path to input text file.
        output_prefix: Output path prefix for dataset.
        vocab_size: Vocabulary size for dtype selection.
        max_seq_length: Maximum sequence length.

    Returns:
        Dataset statistics.
    """
    logger.info(f"Building dataset from {input_file}")

    # Create builder
    builder = IndexedDatasetBuilder(output_prefix=output_prefix, vocab_size=vocab_size)

    # Simple tokenization (in practice, use a proper tokenizer)
    def simple_tokenize(text: str) -> list:
        """Simple character-level tokenization for demo."""
        return [ord(c) % vocab_size for c in text[:max_seq_length]]

    # Process input file
    num_docs = 0
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = simple_tokenize(line)
                builder.add_sequence(tokens)

                # Treat each line as a document
                builder.end_document()
                num_docs += 1

                if num_docs % 1000 == 0:
                    logger.info(f"Processed {num_docs} documents")

    # Finalize dataset
    stats = builder.finalize()
    logger.info(f"Built dataset: {stats}")

    return stats


def benchmark_dataset_loading(dataset_path: str, num_samples: int = 100):
    """Benchmark dataset loading performance.

    Args:
        dataset_path: Path to dataset (without extension).
        num_samples: Number of samples to load.
    """
    logger.info("Benchmarking dataset loading...")

    # Load with memory mapping
    logger.info("Testing memory-mapped loading...")
    start_time = time.time()
    mmap_dataset = MMapIndexedDataset(dataset_path)
    load_time = time.time() - start_time
    logger.info(f"  Load time: {load_time:.3f}s")

    # Measure random access
    start_time = time.time()
    for i in range(min(num_samples, len(mmap_dataset))):
        _ = mmap_dataset[np.random.randint(0, len(mmap_dataset))]
    access_time = time.time() - start_time
    logger.info(f"  Random access ({num_samples} samples): {access_time:.3f}s")

    # Get statistics
    stats = mmap_dataset.get_stats()
    logger.info(f"  Dataset stats: {stats}")

    # Compare with file-based loading
    logger.info("Testing file-based loading...")
    from rosellm.rosetrainer.datasets.indexed_dataset import IndexedDataset

    start_time = time.time()
    file_dataset = IndexedDataset(dataset_path, mmap=False)
    load_time = time.time() - start_time
    logger.info(f"  Load time: {load_time:.3f}s")

    start_time = time.time()
    for i in range(min(num_samples, len(file_dataset))):
        _ = file_dataset[np.random.randint(0, len(file_dataset))]
    access_time = time.time() - start_time
    logger.info(f"  Random access ({num_samples} samples): {access_time:.3f}s")


def distributed_training_example(
    dataset_path: str, model: nn.Module, num_epochs: int = 3, batch_size: int = 32
):
    """Example of distributed training with indexed dataset.

    Args:
        dataset_path: Path to indexed dataset.
        model: Model to train.
        num_epochs: Number of training epochs.
        batch_size: Batch size per GPU.
    """
    # Initialize distributed training if available
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    logger.info(f"Training on rank {local_rank}/{world_size}, device: {device}")

    # Create distributed dataset
    train_dataset = DistributedIndexedDataset(
        path_prefix=dataset_path,
        rank=local_rank,
        world_size=world_size,
        shuffle=True,
        seed=42,
    )

    logger.info(f"Dataset samples for this rank: {len(train_dataset)}")

    # Create dataloader with custom collate function
    def collate_fn(samples):
        """Collate variable-length sequences."""
        # Find max length
        max_len = max(len(s) for s in samples)

        # Pad sequences
        padded = torch.zeros(len(samples), max_len, dtype=torch.long)
        for i, seq in enumerate(samples):
            padded[i, : len(seq)] = torch.from_numpy(seq)

        return padded

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # Set epoch for shuffling
        train_dataset.set_epoch(epoch)

        model.train()
        total_loss = 0
        num_batches = 0

        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)

            # Simple next-token prediction task
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            # Forward pass
            if inputs.size(1) > 0:  # Skip if sequence too short
                outputs = model(inputs)
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if batch_idx % 10 == 0 and local_rank == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_time = time.time() - start_time

        if local_rank == 0:
            avg_loss = total_loss / max(num_batches, 1)
            logger.info(
                f"Epoch {epoch} complete: "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

    if world_size > 1:
        dist.destroy_process_group()


def integration_with_rosetrainer(dataset_path: str, model: nn.Module):
    """Example of integrating indexed dataset with RoseTrainer.

    Args:
        dataset_path: Path to indexed dataset.
        model: Model to train.
    """
    logger.info("Integrating with RoseTrainer...")

    # Configure trainer
    config = TrainingConfig(
        batch_size=32,
        num_epochs=3,
        log_interval=10,
        max_steps=1000,
        warmup_steps=100,
        seed=42,
        checkpoint_interval=100,
        eval_interval=50,
    )

    # Initialize trainer
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    trainer = RoseTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        config=config,
        local_rank=local_rank,
        world_size=world_size,
    )

    # Create distributed dataset
    train_dataset = DistributedIndexedDataset(
        path_prefix=dataset_path, rank=local_rank, world_size=world_size, shuffle=True
    )

    # Custom collate function
    def collate_fn(samples):
        max_len = max(len(s) for s in samples)
        padded = torch.zeros(len(samples), max_len, dtype=torch.long)
        for i, seq in enumerate(samples):
            padded[i, : len(seq)] = torch.from_numpy(seq)
        return padded

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Training loop with RoseTrainer
    num_epochs = config.num_epochs if config.num_epochs is not None else 3
    for epoch in range(num_epochs):
        train_dataset.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            # RoseTrainer handles device placement and distributed sync
            loss = trainer.train_step(batch)

            if batch_idx % config.log_interval == 0:
                logger.info(f"Step {batch_idx}, Loss: {loss:.4f}")

    logger.info("Training complete!")


class SimpleModel(nn.Module):
    """Simple model for demonstration."""

    def __init__(self, vocab_size: int = 50000, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.output(x)
        return x


def main():
    """Main function demonstrating full workflow."""
    parser = argparse.ArgumentParser(
        description="Memory-mapped indexed dataset example"
    )
    parser.add_argument(
        "--mode",
        choices=["build", "benchmark", "train", "integrate"],
        default="build",
        help="Mode to run",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input text file for building dataset",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/example_dataset",
        help="Path to dataset (without extension)",
    )
    parser.add_argument("--vocab-size", type=int, default=50000, help="Vocabulary size")

    args = parser.parse_args()

    if args.mode == "build":
        # Create sample data if no input file provided
        if args.input_file is None:
            logger.info("Creating sample text file...")
            sample_file = "sample_text.txt"
            with open(sample_file, "w") as f:
                for i in range(1000):
                    f.write(f"This is sample sentence number {i}. ")
                    f.write("It contains some text for testing. ")
                    f.write("The quick brown fox jumps over the lazy dog.\n")
            args.input_file = sample_file

        # Build dataset
        os.makedirs(os.path.dirname(args.dataset_path) or ".", exist_ok=True)
        build_dataset_from_text(args.input_file, args.dataset_path, args.vocab_size)

    elif args.mode == "benchmark":
        # Benchmark loading performance
        if not MMapIndexedDataset.exists(args.dataset_path):
            logger.error(f"Dataset not found at {args.dataset_path}")
            logger.info("Run with --mode=build first to create dataset")
            return

        benchmark_dataset_loading(args.dataset_path)

    elif args.mode == "train":
        # Run distributed training example
        if not MMapIndexedDataset.exists(args.dataset_path):
            logger.error(f"Dataset not found at {args.dataset_path}")
            logger.info("Run with --mode=build first to create dataset")
            return

        model = SimpleModel(vocab_size=args.vocab_size)
        distributed_training_example(args.dataset_path, model)

    elif args.mode == "integrate":
        # Demonstrate RoseTrainer integration
        if not MMapIndexedDataset.exists(args.dataset_path):
            logger.error(f"Dataset not found at {args.dataset_path}")
            logger.info("Run with --mode=build first to create dataset")
            return

        model = SimpleModel(vocab_size=args.vocab_size)
        integration_with_rosetrainer(args.dataset_path, model)


if __name__ == "__main__":
    main()
