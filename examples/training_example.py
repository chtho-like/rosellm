"""Example of using RoseTrainer for distributed training of a model."""

import os

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

# Import RoseTrainer components
from rosellm.rosetrainer.engine import RoseTrainer
from rosellm.rosetrainer.memory.activation_checkpoint import \
    ActivationCheckpointing


def main():
    """Run distributed training example using RoseTrainer."""
    # Configure distributed training
    config = {
        # Training hyperparameters
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "warmup_steps": 100,
        "max_steps": 10000,
        # Batch sizes
        "micro_batch_size": 4,
        "global_batch_size": 32,
    }

    # Get distributed training parameters
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Set up device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load a model (e.g., a pretrained LLM)
    print("Loading model...")
    model_name = "EleutherAI/pythia-70m"  # A small model for this example
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Move model to device
    model.to(device)
    print(f"Model loaded and moved to {device}")

    # Apply activation checkpointing if needed
    print("Applying activation checkpointing...")
    if model_name.lower().startswith("eleutherai/pythia"):
        # Find the correct path for transformer layers
        # For pythia models, the layers are under gpt_neox.layers
        checkpoint_manager = ActivationCheckpointing()
        model = checkpoint_manager.apply_to_transformer_layers(
            model,
            layer_attr="gpt_neox.layers",
            use_reentrant=False,  # Using False to avoid warnings
        )
    print("Activation checkpointing applied")

    # Set up the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Initialize the RoseTrainer
    trainer = RoseTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        local_rank=local_rank,
        world_size=world_size,
    )

    print("Starting training loop...")
    # Example training loop with a single batch
    for step in range(5):  # Just a few steps for the example
        # Create a dummy batch
        # For causal language models, labels are typically the same as input_ids
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (config["micro_batch_size"], 32), device=device
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(config["micro_batch_size"], 32, device=device),
            "labels": input_ids.clone(),  # Add labels for loss computation
        }

        # Perform a training step
        result = trainer.train_step(batch)

        # Print progress
        if local_rank == 0:  # Only print on main process
            print(f"Step {step + 1}, Loss: {result['loss']}")

    # Save the final checkpoint
    if local_rank == 0:  # Only save on main process
        trainer.save_checkpoint("trained_model.pt")
        print("Training complete.")


if __name__ == "__main__":
    main()
