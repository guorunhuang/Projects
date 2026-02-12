# Prepare Custom Greek Letter Images
# Author: Guorun
# Prepare and test custom handwritten Greek letters

# import statements
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import cv2


# class definitions
class MyNetwork(nn.Module):
    """Network structure matching the transfer learning model."""
    def __init__(self, num_classes=3):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class GreekTransform:
    """Transform for Greek letter images."""
    def __init__(self):
        pass
    
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def preprocess_custom_image(image_path, target_size=128):
    """
    Preprocess a custom image to match the Greek letter format.
    Args:
        image_path: Path to image file
        target_size: Target size (default 128x128 to match training data)
    Returns:
        processed_image: PIL Image
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Get current size
    width, height = img.size
    
    # Make it square by padding
    max_dim = max(width, height)
    new_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    # Resize to target size
    new_img = new_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return new_img


def visualize_preprocessing_steps(image_path, save_path='preprocessing_steps.png'):
    """
    Visualize the preprocessing pipeline step by step.
    Args:
        image_path: Path to input image
        save_path: Path to save visualization
    """
    # Load original
    original = Image.open(image_path).convert('RGB')
    
    # Preprocess to 128x128
    preprocessed = preprocess_custom_image(image_path, target_size=128)
    
    # Apply Greek transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Get intermediate steps
    tensor = transforms.ToTensor()(preprocessed)
    after_greek = GreekTransform()(tensor)
    after_norm = transforms.Normalize((0.1307,), (0.3081,))(after_greek)
    
    # Denormalize for visualization
    after_greek_viz = after_greek.squeeze().numpy()
    after_norm_viz = after_norm.squeeze().numpy() * 0.3081 + 0.1307
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('1. Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(preprocessed)
    axes[0, 1].set_title('2. Resized to 128×128', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(tensor.permute(1, 2, 0))
    axes[0, 2].set_title('3. Converted to Tensor', fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(after_greek_viz, cmap='gray')
    axes[1, 0].set_title('4. After GreekTransform\n(grayscale, scaled, cropped, inverted)',
                        fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(after_norm_viz, cmap='gray')
    axes[1, 1].set_title('5. After Normalization\n(ready for network)', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Show final 28x28 with grid
    axes[1, 2].imshow(after_norm_viz, cmap='gray')
    axes[1, 2].set_title('6. Final 28×28 Image', fontweight='bold')
    axes[1, 2].set_xticks(np.arange(0, 28, 4))
    axes[1, 2].set_yticks(np.arange(0, 28, 4))
    axes[1, 2].grid(True, color='red', linewidth=0.5, alpha=0.3)
    
    plt.suptitle('Greek Letter Preprocessing Pipeline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Preprocessing visualization saved to '{save_path}'")


def batch_preprocess_images(input_dir, output_dir, target_size=128):
    """
    Batch preprocess all images in a directory.
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for preprocessed images
        target_size: Target size for images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        
        # Preprocess
        processed = preprocess_custom_image(img_path, target_size)
        
        # Save
        output_path = os.path.join(output_dir, f"{name}_processed.png")
        processed.save(output_path)
        print(f"  Processed: {filename} -> {os.path.basename(output_path)}")
    
    print(f"\nAll images saved to {output_dir}")


def create_directory_structure():
    """
    Create directory structure for custom Greek letters.
    """
    directories = [
        './custom_greek',
        './custom_greek_raw',
        './greek_train',
        './greek_train/alpha',
        './greek_train/beta',
        './greek_train/gamma'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("\nDirectory structure created!")
    print("\nInstructions:")
    print("1. Place your raw handwritten Greek letters in './custom_greek_raw/'")
    print("2. Run: python prepare_greek_images.py preprocess")
    print("3. Processed images will be in './custom_greek/'")
    print("4. Run: python greek_transfer_learning.py to test them")


def provide_instructions():
    """
    Provide detailed instructions for creating Greek letters.
    """
    print("\n" + "="*80)
    print("HOW TO CREATE GREEK LETTER IMAGES")
    print("="*80)
    print("\n1. WRITE THE LETTERS:")
    print("   - Use white paper and thick black marker/Sharpie")
    print("   - Write large (fill most of the paper)")
    print("   - Use THICK lines (like the MNIST digits)")
    print("   - Greek letters needed:")
    print("     α (alpha) ")
    print("     β (beta) ")
    print("     γ (gamma)  looks like an upside-down 'r' or 'y'")
    
    print("\n2. CROP AND PREPARE:")
    print("   - Crop each letter to a square image")
    print("   - Rough crop is fine (script will resize)")
    print("   - Save as PNG or JPG")
    print("   - Name descriptively: alpha1.png, beta1.jpg, gamma1.png, etc.")
    
    print("\n3. PLACE IN DIRECTORY:")
    print("   - Put raw images in: ./custom_greek_raw/")
    print("   - Or put preprocessed 128×128 images in: ./custom_greek/")
    
    print("\n4. PREPROCESS (if needed):")
    print("   - Run: python prepare_greek_images.py preprocess")
    print("   - This will resize to 128×128 and save to ./custom_greek/")
    
    print("\n5. TEST:")
    print("   - Run: python greek_transfer_learning.py")
    print("   - Script will automatically test images in ./custom_greek/")


def test_single_image(model_path, image_path, class_names=['alpha', 'beta', 'gamma']):
    """
    Test a single image through the model.
    Args:
        model_path: Path to trained model
        image_path: Path to image to test
        class_names: List of class names
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MyNetwork(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.exp(output).squeeze().cpu().numpy()
        pred = output.argmax(dim=1).item()
    
    # Display results
    print(f"\nResults for: {os.path.basename(image_path)}")
    print("="*50)
    print(f"Prediction: {class_names[pred]}")
    print(f"Confidence: {probs[pred]:.2%}")
    print("\nAll probabilities:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {probs[i]:.2%}")
    print("="*50)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # After transform
    transformed = transform(image).squeeze().numpy() * 0.3081 + 0.1307
    axes[1].imshow(transformed, cmap='gray')
    axes[1].set_title('After Transform (28×28)', fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    axes[2].bar(class_names, probs)
    axes[2].set_ylabel('Probability', fontweight='bold')
    axes[2].set_title('Prediction Probabilities', fontweight='bold')
    axes[2].set_ylim([0, 1])
    
    plt.suptitle(f'Predicted: {class_names[pred]} ({probs[pred]:.1%})',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main(argv):
    """
    Main function for preparing Greek letter images.
    Args:
        argv: Command line arguments
    """
    if len(argv) < 2:
        print("Greek Letter Image Preparation Tool")
        print("\nUsage:")
        print("  python prepare_greek_images.py preprocess - Batch preprocess images")
        print("  python prepare_greek_images.py visualize <image_path> - Show preprocessing steps")
        print("  python prepare_greek_images.py test <model_path> <image_path> - Test single image")
        return
    
    command = argv[1].lower()
    
    if command == 'setup':
        create_directory_structure()
        provide_instructions()
    
    elif command == 'help':
        provide_instructions()
    
    elif command == 'preprocess':
        input_dir = argv[2] if len(argv) > 2 else './custom_greek_raw'
        output_dir = argv[3] if len(argv) > 3 else './custom_greek'
        batch_preprocess_images(input_dir, output_dir, target_size=128)
    
    elif command == 'visualize':
        if len(argv) < 3:
            print("Usage: python prepare_greek_images.py visualize <image_path>")
            return
        image_path = argv[2]
        visualize_preprocessing_steps(image_path)
    
    elif command == 'test':
        if len(argv) < 4:
            print("Usage: python prepare_greek_images.py test <model_path> <image_path>")
            return
        model_path = argv[2]
        image_path = argv[3]
        test_single_image(model_path, image_path)
    
    else:
        print(f"Unknown command: {command}")
        print("Run without arguments to see usage.")


if __name__ == "__main__":
    main(sys.argv)