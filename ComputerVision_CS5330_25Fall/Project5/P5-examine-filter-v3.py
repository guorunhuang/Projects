# Show the Effect of Filters on MNIST Images
# Author: Guorun
# Applies the 10 learned filters to the first training image
# and visualizes them in a 5 rows x 4 columns format

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Try to import cv2, if not available use numpy
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("OpenCV not found, using numpy for filtering")


# class definitions
class MyNetwork(nn.Module):
    """
    Convolutional Neural Network for MNIST digit recognition.
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def apply_filter_numpy(image, filter_weights):
    """
    Apply a filter using numpy (manual convolution/correlation).
    This is equivalent to OpenCV's filter2D.
    
    Args:
        image: Input image (H x W)
        filter_weights: Filter kernel (5 x 5)
    Returns:
        filtered: Output image
    """
    h, w = image.shape
    fh, fw = filter_weights.shape
    
    # Pad image to handle borders
    pad_h = fh // 2
    pad_w = fw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Initialize output
    filtered = np.zeros((h, w), dtype=np.float32)
    
    # Apply filter (correlation operation)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+fh, j:j+fw]
            filtered[i, j] = np.sum(region * filter_weights)
    
    return filtered


def apply_filter(image, filter_weights):
    """
    Apply a filter to an image.
    Uses OpenCV if available, otherwise uses numpy implementation.
    
    Args:
        image: Input image
        filter_weights: Filter kernel (5x5)
    Returns:
        filtered: Filtered image
    """
    if HAS_CV2:
        # Use OpenCV filter2D
        filtered = cv2.filter2D(image, ddepth=-1, kernel=filter_weights)
    else:
        # Use numpy implementation
        filtered = apply_filter_numpy(image.astype(np.float32), filter_weights)
    
    return filtered


def load_trained_model(model_path, device):
    """
    Load a trained model from file.
    """
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def load_training_data():
    """
    Load MNIST training dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    return train_loader, train_dataset


def visualize_filters_and_effects(filters, filtered_images, original_image, label,
                                  save_path='filter_effects_5x4.png'):
    """
    Visualize filters and their effects in 5 rows x 4 columns format.
    
    Layout:
    Row 0: [Filter 0] [Filtered 0] [Filter 5] [Filtered 5]
    Row 1: [Filter 1] [Filtered 1] [Filter 6] [Filtered 6]
    Row 2: [Filter 2] [Filtered 2] [Filter 7] [Filtered 7]
    Row 3: [Filter 3] [Filtered 3] [Filter 8] [Filtered 8]
    Row 4: [Filter 4] [Filtered 4] [Filter 9] [Filtered 9]
    
    Args:
        filters: List of 10 filter weight arrays (5x5 each)
        filtered_images: List of 10 filtered result images
        original_image: Original input image
        label: Label of the digit
        save_path: Path to save visualization
    """
    # Create figure with 5 rows and 4 columns
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    
    fig.suptitle(f'Filter Effects on First Training Image (Digit {label})',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Fill the 5x4 grid
    for row in range(5):
        # Left half: Filters 0-4 and their effects
        filter_idx_left = row  # 0, 1, 2, 3, 4
        
        # Column 0: Filter weights (0-4)
        ax = axes[row, 0]
        im = ax.imshow(filters[filter_idx_left], cmap='gray', interpolation='nearest')
        ax.set_title(f'Filter {filter_idx_left}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Column 1: Filtered results (0-4)
        ax = axes[row, 1]
        im = ax.imshow(filtered_images[filter_idx_left], cmap='gray')
        ax.set_title(f'Filtered {filter_idx_left}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Right half: Filters 5-9 and their effects
        filter_idx_right = row + 5  # 5, 6, 7, 8, 9
        
        # Column 2: Filter weights (5-9)
        ax = axes[row, 2]
        im = ax.imshow(filters[filter_idx_right], cmap='gray', interpolation='nearest')
        ax.set_title(f'Filter {filter_idx_right}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Column 3: Filtered results (5-9)
        ax = axes[row, 3]
        im = ax.imshow(filtered_images[filter_idx_right], cmap='gray')
        ax.set_title(f'Filtered {filter_idx_right}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nFilter effects visualization saved to '{save_path}'")


def main(argv):
    """
    Main function to apply filters and visualize in 5x4 format.
    """
    model_path = argv[1] if len(argv) > 1 else 'mnist_model.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading trained model...")
    model = load_trained_model(model_path, device)
    
    # Load training data
    print("\nLoading training data...")
    train_loader, train_dataset = load_training_data()
    
    # Get first training example
    print("\nGetting first training example...")
    first_image, first_label = train_dataset[0]
    
    # Denormalize for visualization and filtering
    image_denorm = first_image.squeeze().numpy() * 0.3081 + 0.1307
    image_uint8 = (image_denorm * 255).astype(np.uint8)
    
    print(f"First training image: Digit {first_label}")
    print(f"Image shape: {image_uint8.shape}")
    
    # Extract filters from conv1 layer using torch.no_grad()
    print("\nExtracting filters from conv1 layer...")
    with torch.no_grad():
        # Access conv1 weights
        weights = model.conv1.weight.data  # Shape: [10, 1, 5, 5]
        print(f"Conv1 weight tensor shape: {weights.shape}")
        print(f"  - Number of filters: {weights.shape[0]}")
        print(f"  - Input channels: {weights.shape[1]}")
        print(f"  - Filter height: {weights.shape[2]}")
        print(f"  - Filter width: {weights.shape[3]}")
        
        # Convert to numpy and extract individual 5x5 filters
        weights_np = weights.cpu().numpy()
        filters = []
        for i in range(10):
            filter_2d = weights_np[i, 0]  # Extract i-th filter (5x5)
            filters.append(filter_2d)
        
        print(f"\nExtracted {len(filters)} filters")
        print(f"Each filter shape: {filters[0].shape}")
    
    # Apply each filter to the image
    print("\n" + "="*80)
    print("APPLYING 10 FILTERS TO FIRST TRAINING IMAGE")
    if HAS_CV2:
        print("Using OpenCV filter2D")
    else:
        print("Using NumPy implementation")
    print("="*80)
    
    filtered_images = []
    for i, filter_weights in enumerate(filters):
        # Apply filter
        filtered = apply_filter(image_uint8, filter_weights)
        filtered_images.append(filtered)
        
        print(f"Filter {i}: min={filtered.min():.2f}, max={filtered.max():.2f}, "
              f"mean={filtered.mean():.2f}")
    
    # Visualize in required format (5 rows x 4 columns)
    print("\nCreating visualization in 5x4 format...")
    print("Layout: 5 rows x 4 columns")
    print("  Column 1: Filter weights 0-4")
    print("  Column 2: Filtered results 0-4")
    print("  Column 3: Filter weights 5-9")
    print("  Column 4: Filtered results 5-9")
    
    visualize_filters_and_effects(filters, filtered_images, image_uint8, first_label)
    
    print("\n" + "="*80)
    print("FILTER EFFECTS ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated file:")
    print("  - filter_effects_5x4.png (5 rows x 4 columns format)")
    print("\nThis shows the 10 filters and their effects on the first training image.")
    print("Left half: Filters 0-4 and effects")
    print("Right half: Filters 5-9 and effects")


if __name__ == "__main__":
    main(sys.argv)