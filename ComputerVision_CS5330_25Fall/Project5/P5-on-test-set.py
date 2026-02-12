# Test MNIST Network on Test Set
# Author: Guorun
# Loads a trained MNIST network and tests it on the first 10 examples
# from the test set, displaying predictions and network outputs

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


def load_test_data():
    """
    Load MNIST test dataset without shuffling.
    Returns:
        test_loader: DataLoader for test data
        test_dataset: Full test dataset for accessing individual samples
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Don't shuffle to get the same first 10 examples every time
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return test_loader, test_dataset


def load_trained_model(model_path, device):
    """
    Load a trained model from file.
    Args:
        model_path: Path to the saved model file
        device: Device to load model on
    Returns:
        model: Loaded model in evaluation mode
    """
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {model_path}")
    return model


def test_first_ten(model, device, test_loader):
    """
    Test the model on the first 10 examples and print detailed results.
    Args:
        model: Trained neural network model
        device: Device to run inference on
        test_loader: DataLoader for test data
    Returns:
        images: List of first 10 images
        predictions: List of predicted labels
        true_labels: List of true labels
        outputs: List of network output values
    """
    model.eval()
    
    images = []
    predictions = []
    true_labels = []
    outputs = []
    
    print("\n" + "="*80)
    print("Testing first 10 examples from test set")
    print("="*80)
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if idx >= 10:
                break
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get probabilities from log probabilities
            probs = torch.exp(output).squeeze().cpu().numpy()
            pred = output.argmax(dim=1).item()
            true_label = target.item()
            
            # Store results
            images.append(data.squeeze().cpu().numpy())
            predictions.append(pred)
            true_labels.append(true_label)
            outputs.append(probs)
            
            # Print detailed results
            print(f"\nExample {idx + 1}:")
            print(f"Network outputs: [{', '.join([f'{val:.2f}' for val in probs])}]")
            print(f"Predicted digit: {pred} (max output at index {pred})")
            print(f"True label: {true_label}")
            print(f"Result: {'✓ CORRECT' if pred == true_label else '✗ INCORRECT'}")
    
    print("\n" + "="*80)
    correct = sum([1 for p, t in zip(predictions, true_labels) if p == t])
    print(f"Accuracy on first 10 examples: {correct}/10 ({correct*10}%)")
    print("="*80 + "\n")
    
    return images, predictions, true_labels, outputs


def plot_first_nine(images, predictions, true_labels):
    """
    Plot the first 9 test examples in a 3x3 grid with predictions.
    Args:
        images: List of image arrays
        predictions: List of predicted labels
        true_labels: List of true labels
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < 9:
            ax.imshow(images[i], cmap='gray')
            
            # Color code: green for correct, red for incorrect
            color = 'green' if predictions[i] == true_labels[i] else 'red'
            ax.set_title(f'Pred: {predictions[i]}, True: {true_labels[i]}', 
                        color=color, fontsize=12, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('first_nine_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("First 9 predictions saved to 'first_nine_predictions.png'")


def main(argv):
    """
    Main function to test the trained network on the first 10 test examples.
    Args:
        argv: Command line arguments (argv[1] should be model path, default: 'mnist_model.pth')
    """
    # Get model path from command line or use default
    model_path = argv[1] if len(argv) > 1 else 'mnist_model.pth'
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    print("\nLoading MNIST test dataset...")
    test_loader, test_dataset = load_test_data()
    
    # Load trained model
    print("\nLoading trained model...")
    model = load_trained_model(model_path, device)
    
    # Test on first 10 examples
    images, predictions, true_labels, outputs = test_first_ten(model, device, test_loader)
    
    # Plot first 9 examples
    print("\nPlotting first 9 examples...")
    plot_first_nine(images, predictions, true_labels)
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main(sys.argv)