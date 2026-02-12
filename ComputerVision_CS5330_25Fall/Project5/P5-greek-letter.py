# Greek Letter Recognition using Transfer Learning
# Author: Guorun
# Greek letters (alpha, beta, gamma)
# set the default eopch to be 20 times

# import statements
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# class definitions
class MyNetwork(nn.Module):
    """
    Convolutional Neural Network for MNIST digit recognition.
    Modified for Greek letter recognition via transfer learning.
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)  # Original: 10 classes for digits
    
    def forward(self, x):
        """
        Computes a forward pass for the network.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class GreekTransform:
    """
    Transform for Greek letter images to match MNIST format.
    - Converts RGB to grayscale
    - Scales and crops to 28x28
    - Inverts colors to match MNIST (white on black)
    """
    def __init__(self):
        pass
    
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def load_pretrained_model(model_path, device):
    """
    Load pre-trained MNIST model.
    Args:
        model_path: Path to the saved model file
        device: Device to load model on
    Returns:
        model: Loaded model
    """
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Pre-trained MNIST model loaded from {model_path}")
    return model


def freeze_network_weights(model):
    """
    Freeze all parameters in the network so they won't be updated during training.
    Args:
        model: Neural network model
    """
    for param in model.parameters():
        param.requires_grad = False
    
    print("\nAll network weights frozen (requires_grad=False)")


def replace_final_layer(model, num_classes=3):
    """
    Replace the final layer to adapt for new classification task.
    Args:
        model: Neural network model
        num_classes: Number of classes in new task (3 for alpha, beta, gamma)
    Returns:
        model: Modified model
    """
    # Replace fc2 layer (50 inputs -> num_classes outputs)
    model.fc2 = nn.Linear(50, num_classes)
    
    # Move new layer to same device as model
    if next(model.parameters()).is_cuda:
        model.fc2 = model.fc2.cuda()
    
    print(f"\nReplaced final layer (fc2) with new Linear layer: 50 -> {num_classes}")
    print("New layer parameters are trainable (requires_grad=True)")
    
    return model


def load_greek_data(data_path, batch_size=5):
    """
    Load Greek letter dataset using ImageFolder.
    Args:
        data_path: Path to directory containing alpha, beta, gamma folders
        batch_size: Batch size for DataLoader
    Returns:
        greek_loader: DataLoader for Greek letters
        class_names: List of class names (alphabetically sorted)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset
    greek_dataset = datasets.ImageFolder(data_path, transform=transform)
    
    # Create DataLoader
    greek_loader = DataLoader(greek_dataset, batch_size=batch_size, shuffle=True)
    
    # Get class names (alphabetically sorted by ImageFolder)
    class_names = greek_dataset.classes
    
    print(f"\nGreek letter dataset loaded from: {data_path}")
    print(f"Number of images: {len(greek_dataset)}")
    print(f"Classes: {class_names}")
    print(f"Batch size: {batch_size}")
    
    return greek_loader, class_names, greek_dataset


def visualize_greek_samples(dataset, class_names, num_samples=9,
                            save_path='greek_samples.png'):
    """
    Visualize samples from the Greek letter dataset.
    Args:
        dataset: Greek letter dataset
        class_names: List of class names
        num_samples: Number of samples to show
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle('Greek Letter Training Samples', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # Denormalize for visualization
        image = image.squeeze().numpy()
        image = image * 0.3081 + 0.1307
        
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'{class_names[label]}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Greek letter samples saved to '{save_path}'")


def train_epoch(model, device, train_loader, optimizer, epoch):
    """
    Train the model for one epoch.
    Args:
        model: Neural network model
        device: Device to train on
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        epoch: Current epoch number
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    
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
        
        # Update weights (only fc2 layer will update)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    
    print(f'Epoch {epoch}: Average loss: {avg_loss:.4f}, '
          f'Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)')
    
    return avg_loss, accuracy


def evaluate(model, device, data_loader, class_names):
    """
    Evaluate the model on a dataset.
    Args:
        model: Neural network model
        device: Device to evaluate on
        data_loader: DataLoader for evaluation data
        class_names: List of class names
    Returns:
        accuracy: Accuracy on the dataset
        per_class_accuracy: Dictionary of per-class accuracies
    """
    model.eval()
    correct = 0
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Per-class accuracy
            for i in range(len(target)):
                label = target[i].item()
                class_name = class_names[label]
                class_total[class_name] += 1
                if pred[i].item() == label:
                    class_correct[class_name] += 1
    
    accuracy = 100. * correct / len(data_loader.dataset)
    
    per_class_accuracy = {}
    for name in class_names:
        if class_total[name] > 0:
            per_class_accuracy[name] = 100. * class_correct[name] / class_total[name]
        else:
            per_class_accuracy[name] = 0.0
    
    return accuracy, per_class_accuracy


def train_greek_model(model, device, train_loader, class_names, epochs=10, lr=0.01):
    """
    Train the model on Greek letters.
    Args:
        model: Neural network model (with frozen weights and new final layer)
        device: Device to train on
        train_loader: DataLoader for training data
        class_names: List of class names
        epochs: Number of epochs to train
        lr: Learning rate
    Returns:
        history: Dictionary containing training history
    """
    # Only optimize the final layer (fc2)
    optimizer = optim.SGD(model.fc2.parameters(), lr=lr, momentum=0.9)
    
    print("\n" + "="*80)
    print("TRAINING GREEK LETTER CLASSIFIER")
    print("="*80)
    print(f"Optimizer: SGD with lr={lr}, momentum=0.9")
    print(f"Only training final layer (fc2)")
    print("="*80 + "\n")
    
    losses = []
    accuracies = []
    
    for epoch in range(1, epochs + 1):
        avg_loss, accuracy = train_epoch(model, device, train_loader, optimizer, epoch)
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        # Evaluate per-class accuracy
        total_acc, per_class_acc = evaluate(model, device, train_loader, class_names)
        print(f"  Per-class accuracy: ", end="")
        for name, acc in per_class_acc.items():
            print(f"{name}: {acc:.1f}%  ", end="")
        print("\n")
        
        # Early stopping if perfect accuracy
        if accuracy == 100.0:
            print(f"Perfect accuracy achieved at epoch {epoch}!")
            break
    
    return {
        'losses': losses,
        'accuracies': accuracies,
        'epochs': list(range(1, len(losses) + 1))
    }


def plot_training_results(history, save_path='greek_training_results.png'):
    """
    Plot training loss and accuracy.
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['epochs'], history['losses'], 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Average Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['epochs'], history['accuracies'], 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training results saved to '{save_path}'")


def print_model_structure(model):
    """
    Print detailed model structure.
    Args:
        model: Neural network model
    """
    print("\n" + "="*80)
    print("MODIFIED NETWORK STRUCTURE")
    print("="*80)
    print(model)
    print("\n" + "-"*80)
    print("TRAINABLE PARAMETERS")
    print("-"*80)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            print(f"{name:20s}: {num_params:>8,} (trainable)")
        else:
            print(f"{name:20s}: {num_params:>8,} (frozen)")
    
    print("-"*80)
    print(f"{'Total parameters:':<20s} {total_params:>8,}")
    print(f"{'Trainable parameters:':<20s} {trainable_params:>8,}")
    print(f"{'Frozen parameters:':<20s} {total_params - trainable_params:>8,}")
    print("="*80 + "\n")


def test_on_custom_images(model, device, image_dir, class_names,
                          save_path='custom_greek_predictions.png'):
    """
    Test the model on custom handwritten Greek letters.
    Args:
        model: Trained model
        device: Device to run on
        image_dir: Directory containing custom images
        class_names: List of class names
        save_path: Path to save visualization
    """
    import glob
    
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if len(image_files) == 0:
        print(f"\nNo images found in {image_dir}")
        return
    
    image_files.sort()
    
    print(f"\nTesting on {len(image_files)} custom images...")
    print("="*80)
    
    # Transform for custom images
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for img_path in image_files:
            filename = os.path.basename(img_path)
            
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            output = model(image_tensor)
            probs = torch.exp(output).squeeze().cpu().numpy()
            pred = output.argmax(dim=1).item()
            confidence = probs[pred]
            
            # Store results
            results.append({
                'filename': filename,
                'image': image,
                'prediction': class_names[pred],
                'confidence': confidence,
                'probs': probs
            })
            
            print(f"{filename}:")
            print(f"  Prediction: {class_names[pred]}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Probabilities: ", end="")
            for i, name in enumerate(class_names):
                print(f"{name}={probs[i]:.2%}  ", end="")
            print("\n")
    
    # Visualize results
    if len(results) > 0:
        n_images = len(results)
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, result in enumerate(results):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[0, col]
            
            ax.imshow(result['image'])
            ax.set_title(f"Pred: {result['prediction']}\nConf: {result['confidence']:.1%}",
                        fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_images, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[0, col]
            ax.axis('off')
        
        plt.suptitle('Custom Greek Letter Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Custom predictions saved to '{save_path}'")
    
    print("="*80)


def main(argv):
    """
    Main function for Greek letter transfer learning.
    Args:
        argv: Command line arguments
              argv[1]: path to pre-trained MNIST model (default: 'mnist_model.pth')
              argv[2]: path to Greek letter dataset (default: './greek_train')
              argv[3]: number of epochs (default: 20)
    """
    # Parse arguments
    mnist_model_path = argv[1] if len(argv) > 1 else 'mnist_model.pth'
    greek_data_path = argv[2] if len(argv) > 2 else './greek_train'
    epochs = int(argv[3]) if len(argv) > 3 else 20
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Step 1: Load pre-trained MNIST model
    print("Step 1: Loading pre-trained MNIST model...")
    model = load_pretrained_model(mnist_model_path, device)
    
    # Step 2: Freeze network weights
    print("\nStep 2: Freezing network weights...")
    freeze_network_weights(model)
    
    # Step 3: Replace final layer
    print("\nStep 3: Replacing final layer...")
    model = replace_final_layer(model, num_classes=3)
    
    # Print modified network structure
    print_model_structure(model)
    
    # Step 4: Load Greek letter dataset
    print("Step 4: Loading Greek letter dataset...")
    greek_loader, class_names, greek_dataset = load_greek_data(greek_data_path, batch_size=5)
    
    # Visualize some samples
    print("\nVisualizing Greek letter samples...")
    visualize_greek_samples(greek_dataset, class_names)
    
    # Step 5: Train the model
    print(f"\nStep 5: Training for up to {epochs} epochs...")
    history = train_greek_model(model, device, greek_loader, class_names, 
                                epochs=epochs, lr=0.01)
    
    # Plot results
    print("\nPlotting training results...")
    plot_training_results(history)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print("="*80)
    final_acc, per_class_acc = evaluate(model, device, greek_loader, class_names)
    print(f"Overall Accuracy: {final_acc:.2f}%")
    print(f"Per-class Accuracy:")
    for name, acc in per_class_acc.items():
        print(f"  {name}: {acc:.2f}%")
    print("="*80)
    
    # Save trained model
    save_path = 'greek_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nTrained model saved to '{save_path}'")
    
    # Test on custom images if directory exists
    custom_image_dir = './custom_greek'
    if os.path.exists(custom_image_dir):
        print(f"\nTesting on custom images from '{custom_image_dir}'...")
        test_on_custom_images(model, device, custom_image_dir, class_names)
    else:
        print(f"\nTo test on custom images, create directory '{custom_image_dir}' "
              f"and add your handwritten Greek letters.")
    
    print("\nTransfer learning complete!")
    print(f"\nSummary:")
    print(f"  - Epochs needed: {len(history['epochs'])}")
    print(f"  - Final accuracy: {history['accuracies'][-1]:.2f}%")
    print(f"  - Final loss: {history['losses'][-1]:.4f}")


if __name__ == "__main__":
    main(sys.argv)