import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
        # Move the model to CUDA if available
        if torch.cuda.is_available():
            self.linear.to("cuda")

    def forward(self, x):
        # Ensure input is on the same device as model
        if torch.cuda.is_available():
            x = x.to(self.linear.weight.device)
        return self.linear(x)


# Main function
def main():
    # Create a simple dataset
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=8)

    # Create model
    model = SimpleModel()

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            print(batch_idx)
            # Move data to the same device as the model
            data = data.to(device)
            target = target.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)
            loss = torch.nn.functional.mse_loss(outputs, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Print statistics
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
