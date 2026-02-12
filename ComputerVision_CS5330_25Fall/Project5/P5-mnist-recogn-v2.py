# MNIST Digit Recognition Network
# Author: Guorun
# Builds, trains, and evaluates a convolutional neural network
# for MNIST digit recognition using PyTorch

# import statements
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# class definitions
class MyNetwork(nn.Module):
    """
    Convolutional Neural Network for MNIST digit recognition.
    Architecture:
    - Conv layer (10 filters, 5x5)
    - MaxPool (2x2) + ReLU
    - Conv layer (20 filters, 5x5)
    - Dropout (0.5)
    - MaxPool (2x2) + ReLU
    - Flatten + FC (50 nodes) + ReLU
    - FC (10 nodes) + LogSoftmax
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        # First convolutional layer: 1 input channel, 10 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Second convolutional layer: 10 input channels, 20 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout layer with 50% dropout rate
        self.dropout = nn.Dropout2d(p=0.5)
        # First fully connected layer: flattened input to 50 nodes
        self.fc1 = nn.Linear(320, 50)
        # Second fully connected layer: 50 nodes to 10 output classes
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        """
        Computes a forward pass for the network.
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            Output tensor of shape (batch_size, 10) with log probabilities
        """
        # First conv layer + max pooling + ReLU
        # Input: (batch, 1, 28, 28) -> Conv: (batch, 10, 24, 24) -> MaxPool: (batch, 10, 12, 12)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # Second conv layer + dropout + max pooling + ReLU
        # Conv: (batch, 20, 8, 8) -> Dropout -> MaxPool: (batch, 20, 4, 4)
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.relu(F.max_pool2d(x, 2))
        
        # Flatten: (batch, 20, 4, 4) -> (batch, 320)
        x = x.view(-1, 320)
        
        # First FC layer + ReLU
        x = F.relu(self.fc1(x))
        
        # Second FC layer + log_softmax
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_data(batch_size=64):
    """
    Load MNIST training and test datasets.
    Args:
        batch_size: Number of samples per batch
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def visualize_first_six(test_loader):
    """
    Display the first six digits from the test set.
    Args:
        test_loader: DataLoader for test data
    """
    # Get the first batch
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Plot first six images
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for i, ax in enumerate(axes.flat):
        if i < 6:
            # Convert tensor to numpy and remove channel dimension
            img = images[i].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Label: {labels[i].item()}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('first_six_digits.png')
    plt.show()
    print("First six digits saved to 'first_six_digits.png'")


def train_epoch(model, device, train_loader, optimizer, epoch, train_losses, examples_seen):
    """
    Train the model for one epoch.
    Args:
        model: Neural network model
        device: Device to train on (CPU or CUDA)
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        epoch: Current epoch number
        train_losses: List to store training losses
        examples_seen: List to store number of examples seen
    """
    model.train()
    running_loss = 0.0
    log_interval = 1000  # Log every 1000 examples
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Compute loss
        loss = F.nll_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        running_loss += loss.item()
        
        # Log training loss every log_interval examples
        if (batch_idx + 1) * len(data) % log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            examples = epoch * len(train_loader.dataset) + (batch_idx + 1) * len(data)
            train_losses.append(avg_loss)
            examples_seen.append(examples)
            print(f'Epoch {epoch}, Examples: {examples}/{len(train_loader.dataset)}, Loss: {avg_loss:.4f}')


def evaluate(model, device, data_loader, dataset_name):
    """
    Evaluate the model on a dataset.
    Args:
        model: Neural network model
        device: Device to evaluate on
        data_loader: DataLoader for evaluation data
        dataset_name: Name of dataset (for printing)
    Returns:
        avg_loss: Average loss on the dataset
        accuracy: Accuracy on the dataset
    """
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Sum up batch loss
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    
    print(f'{dataset_name} - Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)')
    
    return avg_loss, accuracy


def train_network(model, device, train_loader, test_loader, epochs=5, lr=0.01):
    """
    Train the network for multiple epochs and track performance.
    Args:
        model: Neural network model
        device: Device to train on
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of epochs to train
        lr: Learning rate
    Returns:
        Dictionary containing training history
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Lists to store metrics
    train_losses = []
    examples_seen = []
    test_losses = []
    test_examples = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(1, epochs + 1):
        print(f'\n--- Epoch {epoch}/{epochs} ---')
        
        # Train for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch, train_losses, examples_seen)
        
        # Evaluate on training set
        train_loss, train_acc = evaluate(model, device, train_loader, 'Training Set')
        train_accuracies.append(train_acc)
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, device, test_loader, 'Test Set')
        test_losses.append(test_loss)
        test_examples.append(epoch * len(train_loader.dataset))
        test_accuracies.append(test_acc)
    
    return {
        'train_losses': train_losses,
        'examples_seen': examples_seen,
        'test_losses': test_losses,
        'test_examples': test_examples,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }


def plot_results(history):
    """
    Plot training and testing losses and accuracies.
    Args:
        history: Dictionary containing training history
    """
    # Plot losses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['examples_seen'], history['train_losses'], 'b-', label='Training Loss', linewidth=1.5)
    ax1.plot(history['test_examples'], history['test_losses'], 'ro', label='Test Loss', markersize=8)
    ax1.set_xlabel('Number of Training Examples Seen')
    ax1.set_ylabel('Negative Log Likelihood Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    epochs = range(1, len(history['train_accuracies']) + 1)
    ax2.plot(epochs, history['train_accuracies'], 'b-o', label='Training Accuracy', linewidth=2, markersize=8)
    ax2.plot(epochs, history['test_accuracies'], 'r-s', label='Test Accuracy', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    print("Training results saved to 'training_results.png'")


def main(argv):
    """
    Main function to run the MNIST digit recognition training pipeline.
    Args:
        argv: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 64
    epochs = 5
    learning_rate = 0.01
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_data(batch_size)
    
    # Visualize first six test examples
    print("\nVisualizing first six test examples...")
    visualize_first_six(test_loader)
    
    # Create model
    print("\nCreating network...")
    model = MyNetwork().to(device)
    print(model)
    
    # Train network
    print("\nStarting training...")
    history = train_network(model, device, train_loader, test_loader, epochs, learning_rate)
    
    # Plot results
    print("\nPlotting results...")
    plot_results(history)
    
    # Save model
    print("\nSaving model...")
    model_path = 'mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")
    print(f"Model size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    print("\nTraining complete!")
    

if __name__ == "__main__":
    main(sys.argv)