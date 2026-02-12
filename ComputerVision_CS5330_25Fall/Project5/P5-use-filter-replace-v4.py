# set the model saving path, and epoch=4
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Define the original network architecture
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Create custom filter banks
def create_sobel_x():
    """Sobel X filter for horizontal edge detection"""
    sobel = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float32)
    # Pad to 5x5
    filter_5x5 = np.zeros((5, 5), dtype=np.float32)
    filter_5x5[1:4, 1:4] = sobel
    return filter_5x5


def create_sobel_y():
    """Sobel Y filter for vertical edge detection"""
    sobel = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]], dtype=np.float32)
    # Pad to 5x5
    filter_5x5 = np.zeros((5, 5), dtype=np.float32)
    filter_5x5[1:4, 1:4] = sobel
    return filter_5x5


def create_laplacian():
    """Laplacian filter for edge detection"""
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float32)
    # Pad to 5x5
    filter_5x5 = np.zeros((5, 5), dtype=np.float32)
    filter_5x5[1:4, 1:4] = laplacian
    return filter_5x5


def create_gaussian():
    """Gaussian filter for smoothing"""
    # 5x5 Gaussian kernel
    gaussian = np.array([[1, 4, 6, 4, 1],
                         [4, 16, 24, 16, 4],
                         [6, 24, 36, 24, 6],
                         [4, 16, 24, 16, 4],
                         [1, 4, 6, 4, 1]], dtype=np.float32)
    gaussian = gaussian / 256.0  # Normalize
    return gaussian


def create_gabor_0():
    """Gabor filter at 0 degrees"""
    size = 5
    sigma = 1.0
    lambd = 3.0
    gamma = 0.5
    psi = 0
    
    gabor = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            x = i - size // 2
            y = j - size // 2
            x_theta = x * np.cos(0) + y * np.sin(0)
            y_theta = -x * np.sin(0) + y * np.cos(0)
            gabor[i, j] = np.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2) * \
                          np.cos(2 * np.pi * x_theta / lambd + psi)
    return gabor


def create_gabor_45():
    """Gabor filter at 45 degrees"""
    size = 5
    sigma = 1.0
    lambd = 3.0
    gamma = 0.5
    psi = 0
    theta = np.pi / 4  # 45 degrees
    
    gabor = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            x = i - size // 2
            y = j - size // 2
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)
            gabor[i, j] = np.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2) * \
                          np.cos(2 * np.pi * x_theta / lambd + psi)
    return gabor


def create_gabor_90():
    """Gabor filter at 90 degrees"""
    size = 5
    sigma = 1.0
    lambd = 3.0
    gamma = 0.5
    psi = 0
    theta = np.pi / 2  # 90 degrees
    
    gabor = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            x = i - size // 2
            y = j - size // 2
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)
            gabor[i, j] = np.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2) * \
                          np.cos(2 * np.pi * x_theta / lambd + psi)
    return gabor


def create_filter_bank(filter_type):
    """Create a bank of 10 filters based on the filter type"""
    if filter_type == 'sobel_x':
        base_filter = create_sobel_x()
    elif filter_type == 'sobel_y':
        base_filter = create_sobel_y()
    elif filter_type == 'laplacian':
        base_filter = create_laplacian()
    elif filter_type == 'gaussian':
        base_filter = create_gaussian()
    elif filter_type == 'gabor_0':
        base_filter = create_gabor_0()
    elif filter_type == 'gabor_45':
        base_filter = create_gabor_45()
    elif filter_type == 'gabor_90':
        base_filter = create_gabor_90()
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Create 10 variations of the filter
    filters = []
    for i in range(10):
        # Add small variations by rotating or scaling
        if i == 0:
            filters.append(base_filter.copy())
        else:
            # Create variations by adding small random perturbations
            variation = base_filter + np.random.randn(5, 5) * 0.1
            filters.append(variation)
    
    return np.array(filters)


def replace_first_layer(model, filter_bank):
    """Replace the first convolutional layer with custom filters"""
    # Convert filter bank to torch tensor
    # Shape: (10, 1, 5, 5) for 10 filters, 1 input channel, 5x5 kernel
    new_weights = torch.from_numpy(filter_bank).unsqueeze(1).float()
    
    # Replace the weights
    model.conv1.weight.data = new_weights
    
    # Freeze the first layer
    for param in model.conv1.parameters():
        param.requires_grad = False
    
    return model


def train(model, device, train_loader, optimizer, epoch):
    """Training function"""
    model.train()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    return avg_loss, accuracy


def test(model, device, test_loader):
    """Testing function"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy


def run_experiment(filter_type, epochs=4):
    """Run a complete experiment with a specific filter type"""
    print(f"\n{'='*60}")
    print(f"Running experiment with {filter_type} filter")
    print(f"{'='*60}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Load the original trained model
    model = MNISTNet().to(device)
    try:
        model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
        print("Loaded pretrained model successfully")
    except:
        print("Warning: Could not load pretrained model, using random initialization")
    
    # Create and apply filter bank
    filter_bank = create_filter_bank(filter_type)
    model = replace_first_layer(model, filter_bank)
    
    # Set up optimizer (only for layers that are not frozen)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # Train the model
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
    # Save the model
    torch.save(model.state_dict(), f'mnist_model_{filter_type}.pth')
    print(f"Model saved as mnist_model_{filter_type}.pth")
    
    return {
        'filter_type': filter_type,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_test_acc': test_accs[-1]
    }


def visualize_filters(filter_types):
    """Visualize all filter types"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, filter_type in enumerate(filter_types):
        filter_bank = create_filter_bank(filter_type)
        # Show the first filter of each bank
        axes[idx].imshow(filter_bank[0], cmap='gray')
        axes[idx].set_title(f'{filter_type}')
        axes[idx].axis('off')
    
    # Hide the last subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('P5-filter_banks.png')
    print("Filter visualizations saved as filter_banks.png")
    plt.show()


def main():
    """Main function to run all 7 experiments"""
    filter_types = ['sobel_x', 'sobel_y', 'laplacian', 'gaussian', 
                    'gabor_0', 'gabor_45', 'gabor_90']
    
    # Visualize all filters
    print("Creating filter visualizations...")
    visualize_filters(filter_types)
    
    # Run experiments
    results = []
    for filter_type in filter_types:
        result = run_experiment(filter_type, epochs=4)
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for result in results:
        print(f"{result['filter_type']:15s} - Final Test Accuracy: {result['final_test_acc']:.2f}%")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for result in results:
        ax1.plot(result['train_accs'], label=result['filter_type'])
        ax2.plot(result['test_accs'], label=result['filter_type'])
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy (%)')
    ax1.set_title('Training Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('P5-experiments_comparison.png')
    print("\nComparison plot saved as experiments_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()