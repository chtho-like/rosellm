"""
Advanced Parallel Training Example with Multi-Dimensional Parallelism

This example demonstrates how to use the advanced parallel state management system
to set up multi-dimensional parallelism for training large language models.
"""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from rosellm.rosetrainer import RoseTrainer
from rosellm.rosetrainer.parallelism import (
    NCCLConfig,
    get_context_parallel_size,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    initialize_model_parallel,
    set_nccl_config,
)


def setup_multi_dimensional_parallelism():
    """
    Set up advanced multi-dimensional parallelism with optimized NCCL configuration.
    """
    # Get world size from environment
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Configure parallelism dimensions based on world size
    if world_size == 1:
        # Single GPU - no parallelism
        tp_size = 1
        pp_size = 1
        dp_size = 1
        cp_size = 1
    elif world_size == 8:
        # 8 GPUs - balanced parallelism
        tp_size = 2  # Tensor parallel across 2 GPUs
        pp_size = 2  # Pipeline parallel across 2 stages
        dp_size = 2  # Data parallel across 2 replicas
        cp_size = 1  # No context parallelism
    elif world_size == 16:
        # 16 GPUs - with context parallelism
        tp_size = 2  # Tensor parallel
        pp_size = 2  # Pipeline parallel
        dp_size = 2  # Data parallel
        cp_size = 2  # Context parallel for long sequences
    elif world_size == 32:
        # 32 GPUs - large scale training
        tp_size = 4  # Larger tensor parallel
        pp_size = 2  # Pipeline parallel
        dp_size = 4  # More data parallel
        cp_size = 1  # No context parallel
    else:
        # Default configuration
        tp_size = min(8, world_size)
        pp_size = 1
        dp_size = world_size // tp_size
        cp_size = 1

    # Configure NCCL optimizations
    nccl_config = NCCLConfig(
        enable_sharp=True,  # Enable SHARP for better collective performance
        cta_size=8,  # CTA size for better GPU utilization
        min_nchannels=4,  # Minimum channels for communication
        tree_threshold=1000,  # Threshold for tree algorithm
    )
    set_nccl_config(nccl_config)

    # Initialize multi-dimensional parallelism
    initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        data_parallel_size=dp_size,
        context_parallel_size=cp_size,
        order="tp-cp-dp-pp",  # Optimized dimension ordering
    )

    # Print parallelism configuration
    if get_data_parallel_rank() == 0 and get_tensor_model_parallel_rank() == 0:
        print(f"Initialized multi-dimensional parallelism:")
        print(f"  World Size: {world_size}")
        print(f"  Tensor Parallel Size: {tp_size}")
        print(f"  Pipeline Parallel Size: {pp_size}")
        print(f"  Data Parallel Size: {dp_size}")
        print(f"  Context Parallel Size: {cp_size}")
        print(f"  Total: {tp_size * pp_size * dp_size * cp_size}")


def create_model_with_parallelism(model_name: str = "EleutherAI/pythia-70m"):
    """
    Create a model with appropriate parallelism configuration.

    Args:
        model_name: Name of the model to load

    Returns:
        Model configured for multi-dimensional parallelism
    """
    # Get parallelism configuration
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()
    dp_size = get_data_parallel_size()
    cp_size = get_context_parallel_size()

    tp_rank = get_tensor_model_parallel_rank()
    pp_rank = get_pipeline_model_parallel_rank()
    dp_rank = get_data_parallel_rank()

    # Load model on appropriate device
    device = torch.cuda.current_device()

    # For tensor parallelism, we would need to shard the model
    # For pipeline parallelism, we would need to split layers
    # For simplicity, this example uses standard DDP
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )

    # Apply data parallelism
    if dp_size > 1:
        model = DDP(
            model,
            device_ids=[device],
            process_group=get_data_parallel_group(),
        )

    return model


def main():
    """
    Main training function demonstrating advanced parallel training.
    """
    # Initialize distributed training
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        # Set CUDA device
        torch.cuda.set_device(rank % torch.cuda.device_count())

    # Set up multi-dimensional parallelism
    setup_multi_dimensional_parallelism()

    # Create model with parallelism
    model = create_model_with_parallelism()

    # Create dummy data for demonstration
    batch_size = 4
    seq_length = 512
    vocab_size = 50257

    # Generate random input
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_length), device=torch.cuda.current_device()
    )
    labels = input_ids.clone()

    # Training configuration
    training_config = {
        "learning_rate": 1e-4,
        "num_epochs": 1,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "checkpoint_dir": "./checkpoints",
    }

    # Initialize optimizer (required for RoseTrainer)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
    )

    # Initialize RoseTrainer with advanced parallelism
    trainer = RoseTrainer(
        model=model,
        optimizer=optimizer,
        config=training_config,
    )

    # Print current process information
    print(f"Process Info:")
    print(f"  Global Rank: {dist.get_rank()}")
    print(
        f"  TP Rank: {get_tensor_model_parallel_rank()}/{get_tensor_model_parallel_size()}"
    )
    print(
        f"  PP Rank: {get_pipeline_model_parallel_rank()}/{get_pipeline_model_parallel_size()}"
    )
    print(f"  DP Rank: {get_data_parallel_rank()}/{get_data_parallel_size()}")

    # Training step (simplified for demonstration)
    for epoch in range(training_config["num_epochs"]):
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Optimizer step (would be handled by trainer)
        # trainer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("Training completed successfully with advanced parallelism!")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
