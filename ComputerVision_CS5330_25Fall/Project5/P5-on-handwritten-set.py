# Test MNIST Network on Handwritten Digits
# Author: Guorun
# Loads a trained MNIST network and tests it on handwritten digit images
# Images should be preprocessed to match MNIST format (white digits on black background)

# import statements
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob


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


def preprocess_image(image_path, show_steps=False):
    """
    Preprocess a handwritten digit image to match MNIST format.
    MNIST digits are white on black background with normalization.
    Args:
        image_path: Path to the image file
        show_steps: Whether to display preprocessing steps
    Returns:
        tensor: Preprocessed image tensor ready for the network
        processed_image: Processed image as numpy array for visualization
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Check if digits are black on white (need to invert)
    # MNIST has white digits on black background
    # If mean intensity > 127, image is mostly white, so we need to invert
    if img_array.mean() > 127:
        img_array = 255 - img_array
        print(f"  Inverted image (was black on white)")
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Store for visualization before MNIST normalization
    processed_image = img_array.copy()
    
    # Apply MNIST normalization: mean=0.1307, std=0.3081
    img_array = (img_array - 0.1307) / 0.3081
    
    # Convert to tensor and add batch and channel dimensions
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    if show_steps:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(image_path).convert('L'), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(processed_image, cmap='gray')
        plt.title('Processed (28x28)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return tensor, processed_image


def predict_digit(model, device, image_tensor):
    """
    Predict the digit in an image.
    Args:
        model: Trained neural network model
        device: Device to run inference on
        image_tensor: Preprocessed image tensor
    Returns:
        prediction: Predicted digit (0-9)
        confidence: Confidence score for the prediction
        all_probs: All class probabilities
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probs = torch.exp(output).squeeze().cpu().numpy()
        prediction = output.argmax(dim=1).item()
        confidence = probs[prediction]
    
    return prediction, confidence, probs


def test_handwritten_digits(model, device, image_dir='./handwritten_digits'):
    """
    Test the model on handwritten digit images.
    Args:
        model: Trained neural network model
        device: Device to run inference on
        image_dir: Directory containing handwritten digit images
    Returns:
        results: List of tuples (image, prediction, confidence, filename)
    """
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    # Sort files for consistent ordering
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        print("Please add your handwritten digit images (0-9) to this directory.")
        print("Supported formats: PNG, JPG, JPEG, BMP, GIF")
        return []
    
    print(f"\nFound {len(image_files)} images in {image_dir}")
    print("="*80)
    
    results = []
    
    for idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        print(f"\nProcessing: {filename}")
        
        try:
            # Preprocess image
            img_tensor, processed_img = preprocess_image(img_path)
            
            # Predict
            prediction, confidence, all_probs = predict_digit(model, device, img_tensor)
            
            # Print results
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  All probabilities: [{', '.join([f'{p:.2f}' for p in all_probs])}]")
            
            results.append((processed_img, prediction, confidence, filename, all_probs))
            
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
    
    print("\n" + "="*80)
    
    return results


def plot_handwritten_results(results):
    """
    Plot all handwritten digits with their predictions.
    Args:
        results: List of tuples (image, prediction, confidence, filename)
    """
    if len(results) == 0:
        print("No results to plot")
        return
    
    # Calculate grid size
    n_images = len(results)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img, pred, conf, filename, probs) in enumerate(results):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[0, col]
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Predicted: {pred}\nConfidence: {conf:.1%}', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[0, col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('handwritten_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nHandwritten digit predictions saved to 'handwritten_predictions.png'")


def create_sample_structure(image_dir):
    """
    Create sample directory structure and instructions.
    Args:
        image_dir: Directory to create for handwritten digits
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"\nCreated directory: {image_dir}")
        print("\nInstructions for preparing handwritten digits:")
        print("1. Write digits 0-9 on white paper using THICK black marker/sharpie")
        print("2. Take a photo of the paper")
        print("3. Crop each digit to a separate square image")
        print("4. Save images with descriptive names (e.g., digit_0.png, digit_1.png)")
        print(f"5. Place all images in the '{image_dir}' folder")
        print("6. Run this script again")
        print("\nNote: MNIST digits are WHITE on BLACK background.")
        print("      This script will automatically invert if needed.")


def main(argv):
    """
    Main function to test the trained network on handwritten digits.
    Args:
        argv: Command line arguments
              argv[1]: model path (optional, default: 'mnist_model.pth')
              argv[2]: image directory (optional, default: './handwritten_digits')
    """
    # Parse command line arguments
    model_path = argv[1] if len(argv) > 1 else 'mnist_model.pth'
    image_dir = argv[2] if len(argv) > 2 else './handwritten_digits'
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directory if it doesn't exist
    create_sample_structure(image_dir)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\nError: Model file '{model_path}' not found!")
        print("Please train the model first using the training script.")
        return
    
    # Load trained model
    print("\nLoading trained model...")
    model = load_trained_model(model_path, device)
    
    # Test on handwritten digits
    print("\nTesting on handwritten digits...")
    results = test_handwritten_digits(model, device, image_dir)
    
    # Plot results
    if len(results) > 0:
        print("\nPlotting results...")
        plot_handwritten_results(results)
        
        # Summary
        print("\nSummary:")
        print(f"Total images processed: {len(results)}")
        print(f"Average confidence: {np.mean([r[2] for r in results]):.1%}")
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main(sys.argv)