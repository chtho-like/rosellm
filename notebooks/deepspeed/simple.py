import argparse
import os

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
        # Move the model to CUDA
        self.linear.to('cuda')

    def forward(self, x):
        # Ensure input is on the same device as model
        x = x.to(self.linear.weight.device)
        return self.linear(x)


# Initialize DeepSpeed
def init_deepspeed(model, dataset):
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)

    # Create a basic DeepSpeed config
    ds_config = {
        "train_batch_size": 8,
        "steps_per_print": 10,
        "optimizer": {"type": "Adam", "params": {"lr": 0.001}},
        "fp16": {"enabled": False},
    }

    # Save config to a file
    import json

    with open("ds_config.json", "w") as f:
        json.dump(ds_config, f)

    # Initialize DeepSpeed
    args = parser.parse_args(
        args=["--deepspeed", "--deepspeed_config", "ds_config.json"]
    )
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,  # Pass the dataset instead of dataloader
    )

    return model_engine


# Main function
def main():
    # Initialize distributed environment
    deepspeed.init_distributed()

    # Create a simple dataset
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=8)

    # Create model
    model = SimpleModel()

    # Initialize DeepSpeed - pass the dataset, not the dataloader
    model_engine = init_deepspeed(model, dataset)

    # Training loop
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to the same device as the model
            data = data.to(model_engine.device)
            target = target.to(model_engine.device)
            
            # Forward pass
            outputs = model_engine(data)
            loss = torch.nn.functional.mse_loss(outputs, target)

            # Backward pass
            model_engine.backward(loss)
            model_engine.step()

            # Print statistics
            if batch_idx % 10 == 0 and dist.get_rank() == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()

# To run this example:
# deepspeed --num_gpus=1 simple.py
# (or save this code to a .py file and run it with DeepSpeed)
