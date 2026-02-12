# Examine MNIST Network
# Author: Guorun
# Loads a trained MNIST network and analyzes its structure,
# visualizing the first convolutional layer filters

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# class definitions
class MyNetwork(nn.Module):
    """
    Convolutional Neural Network for MNIST digit recognition.
    Must match the architecture used during training.
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        """
        Computes a forward pass for the network.
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            Output tensor of shape (batch_size, 10) with log probabilities
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_trained_model(model_path, device):
    """
    Load a trained model from file and print its structure.
    Args:
        model_path: Path to the saved model file
        device: Device to load model on
    Returns:
        model: Loaded model in evaluation mode
    """
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}\n")
    
    print("="*80)
    print("MODEL STRUCTURE")
    print("="*80)
    print(model)
    print("="*80 + "\n")
    
    return model


def analyze_first_layer(model):
    """
    Analyze and print information about the first convolutional layer.
    Args:
        model: Trained neural network model
    Returns:
        weights: Weight tensor of the first layer
    """
    print("="*80)
    print("FIRST LAYER ANALYSIS (conv1)")
    print("="*80)
    
    # Access the weights of the first convolutional layer
    weights = model.conv1.weight.data
    bias = model.conv1.bias.data
    
    # Print shape information
    print(f"\nWeight tensor shape: {weights.shape}")
    print(f"  - Number of filters: {weights.shape[0]}")
    print(f"  - Input channels: {weights.shape[1]}")
    print(f"  - Filter height: {weights.shape[2]}")
    print(f"  - Filter width: {weights.shape[3]}")
    
    print(f"\nBias tensor shape: {bias.shape}")
    print(f"  - Number of biases: {bias.shape[0]}")
    
    # Print statistics for each filter
    print("\nFilter Statistics:")
    print("-" * 80)
    for i in range(weights.shape[0]):
        filter_weights = weights[i, 0]
        print(f"Filter {i}:")
        print(f"  Min: {filter_weights.min().item():7.4f}, "
              f"Max: {filter_weights.max().item():7.4f}, "
              f"Mean: {filter_weights.mean().item():7.4f}, "
              f"Std: {filter_weights.std().item():7.4f}")
    
    # Print first filter as example
    print("\n" + "-" * 80)
    print("Example: Filter 0 weights (5x5):")
    print("-" * 80)
    filter_0 = weights[0, 0].cpu().numpy()
    for row in filter_0:
        print("  " + " ".join([f"{val:7.4f}" for val in row]))
    
    print("\n" + "="*80 + "\n")
    
    return weights


def visualize_first_layer_filters(weights, save_path='first_layer_filters.png'):
    """
    Visualize all filters from the first convolutional layer in a grid.
    Args:
        weights: Weight tensor from first layer (shape: [10, 1, 5, 5])
        save_path: Path to save the visualization
    """
    # Convert to numpy and get the number of filters
    weights_np = weights.cpu().numpy()
    num_filters = weights_np.shape[0]
    
    # Create a 3x4 grid (for 10 filters, last 2 spots will be empty)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle('First Layer Convolutional Filters (conv1)', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Plot each filter
    for i in range(len(axes)):
        ax = axes[i]
        
        if i < num_filters:
            # Get the i-th filter (5x5)
            filter_weights = weights_np[i, 0]
            
            # Plot the filter
            im = ax.imshow(filter_weights, cmap='gray', interpolation='nearest')
            ax.set_title(f'Filter {i}', fontsize=12, fontweight='bold')
            
            # Remove ticks for cleaner visualization
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar for each subplot
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            # Hide unused subplots
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"First layer filters saved to '{save_path}'")


def get_layer_summary(model):
    """
    Print a detailed summary of all layers in the network.
    Args:
        model: Neural network model
    """
    print("="*80)
    print("DETAILED LAYER SUMMARY")
    print("="*80)
    
    total_params = 0
    
    print("\n{:<20} {:<25} {:<15} {:<15}".format(
        "Layer Name", "Layer Type", "Output Shape", "Parameters"))
    print("-" * 80)
    
    # Conv1
    conv1_params = (model.conv1.weight.numel() + 
                    model.conv1.bias.numel())
    print("{:<20} {:<25} {:<15} {:<15}".format(
        "conv1", "Conv2d(1, 10, 5x5)", "[10, 24, 24]", f"{conv1_params:,}"))
    total_params += conv1_params
    
    # MaxPool1
    print("{:<20} {:<25} {:<15} {:<15}".format(
        "maxpool1", "MaxPool2d(2x2)", "[10, 12, 12]", "0"))
    
    # Conv2
    conv2_params = (model.conv2.weight.numel() + 
                    model.conv2.bias.numel())
    print("{:<20} {:<25} {:<15} {:<15}".format(
        "conv2", "Conv2d(10, 20, 5x5)", "[20, 8, 8]", f"{conv2_params:,}"))
    total_params += conv2_params
    
    # Dropout
    print("{:<20} {:<25} {:<15} {:<15}".format(
        "dropout", "Dropout2d(p=0.5)", "[20, 8, 8]", "0"))
    
    # MaxPool2
    print("{:<20} {:<25} {:<15} {:<15}".format(
        "maxpool2", "MaxPool2d(2x2)", "[20, 4, 4]", "0"))
    
    # Flatten
    print("{:<20} {:<25} {:<15} {:<15}".format(
        "flatten", "Flatten", "[320]", "0"))
    
    # FC1
    fc1_params = (model.fc1.weight.numel() + 
                  model.fc1.bias.numel())
    print("{:<20} {:<25} {:<15} {:<15}".format(
        "fc1", "Linear(320, 50)", "[50]", f"{fc1_params:,}"))
    total_params += fc1_params
    
    # FC2
    fc2_params = (model.fc2.weight.numel() + 
                  model.fc2.bias.numel())
    print("{:<20} {:<25} {:<15} {:<15}".format(
        "fc2", "Linear(50, 10)", "[10]", f"{fc2_params:,}"))
    total_params += fc2_params
    
    print("-" * 80)
    print(f"{'Total Parameters:':<46} {total_params:,}")
    print("="*80 + "\n")


def visualize_filters_alternate(weights, save_path='first_layer_filters_alt.png'):
    """
    Alternative visualization showing filters with their actual values displayed.
    Args:
        weights: Weight tensor from first layer
        save_path: Path to save the visualization
    """
    weights_np = weights.cpu().numpy()
    num_filters = weights_np.shape[0]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('First Layer Filters with Weight Values', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i in range(num_filters):
        ax = axes[i]
        filter_weights = weights_np[i, 0]
        
        # Create heatmap
        im = ax.imshow(filter_weights, cmap='RdBu_r', interpolation='nearest',
                      vmin=-0.5, vmax=0.5)
        
        # Add text annotations with weight values
        for y in range(5):
            for x in range(5):
                text = ax.text(x, y, f'{filter_weights[y, x]:.2f}',
                             ha="center", va="center", color="black", 
                             fontsize=8, fontweight='bold')
        
        ax.set_title(f'Filter {i}', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Alternative filter visualization saved to '{save_path}'")


def analyze_filter_properties(weights):
    """
    Analyze and categorize filters based on their properties.
    Args:
        weights: Weight tensor from first layer
    """
    print("="*80)
    print("FILTER PROPERTIES ANALYSIS")
    print("="*80)
    
    weights_np = weights.cpu().numpy()
    num_filters = weights_np.shape[0]
    
    print("\nFilter Characteristics:")
    print("-" * 80)
    
    for i in range(num_filters):
        filter_weights = weights_np[i, 0]
        
        # Calculate properties
        center_val = filter_weights[2, 2]
        edge_vals = np.concatenate([
            filter_weights[0, :],
            filter_weights[-1, :],
            filter_weights[1:-1, 0],
            filter_weights[1:-1, -1]
        ])
        
        horizontal_gradient = np.abs(filter_weights[:, :-1] - filter_weights[:, 1:]).mean()
        vertical_gradient = np.abs(filter_weights[:-1, :] - filter_weights[1:, :]).mean()
        
        # Determine filter type
        if horizontal_gradient > vertical_gradient * 1.5:
            filter_type = "Vertical Edge Detector"
        elif vertical_gradient > horizontal_gradient * 1.5:
            filter_type = "Horizontal Edge Detector"
        elif center_val > edge_vals.mean():
            filter_type = "Center-focused"
        else:
            filter_type = "General Feature Detector"
        
        print(f"Filter {i}: {filter_type}")
        print(f"  Center value: {center_val:7.4f}")
        print(f"  H-gradient: {horizontal_gradient:7.4f}, V-gradient: {vertical_gradient:7.4f}")
    
    print("\n" + "="*80 + "\n")


def main(argv):
    """
    Main function to examine and analyze the trained network.
    Args:
        argv: Command line arguments
              argv[1]: model path (optional, default: 'mnist_model.pth')
    """
    # Parse command line arguments
    model_path = argv[1] if len(argv) > 1 else 'mnist_model.pth'
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load trained model and print structure
    print("Loading trained model...\n")
    model = load_trained_model(model_path, device)
    
    # Get detailed layer summary
    get_layer_summary(model)
    
    # Analyze first convolutional layer
    print("Analyzing first convolutional layer...\n")
    weights = analyze_first_layer(model)
    
    # Analyze filter properties
    analyze_filter_properties(weights)
    
    # Visualize filters
    print("Visualizing first layer filters...\n")
    visualize_first_layer_filters(weights)
    
    # Create alternative visualization with values
    print("\nCreating alternative visualization with weight values...\n")
    visualize_filters_alternate(weights)
    
    print("Analysis complete!")
    print("\nGenerated files:")
    print("  - first_layer_filters.png (clean filter visualization)")
    print("  - first_layer_filters_alt.png (filters with weight values)")


if __name__ == "__main__":
    main(sys.argv)