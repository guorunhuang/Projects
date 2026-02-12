# Network Architecture Experiments
# Author: Guorun
# explores different network architectures 
# optimize performance on Fashion MNIST dataset

# 10 epochs take much time, so switch to 4

# import statements
import sys
import os
import time
import json
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime


# class definitions
class FlexibleNetwork(nn.Module):
    """
    Flexible CNN architecture that can be configured with different parameters.
    """
    def __init__(self, config):
        super(FlexibleNetwork, self).__init__()
        self.config = config
        
        # Store layers in lists
        self.conv_layers = nn.ModuleList()
        self.pool_layers = []
        self.dropout_layers = nn.ModuleList()
        
        # Build convolutional layers
        in_channels = 1
        current_size = 28
        
        for i in range(config['num_conv_layers']):
            # Add convolutional layer
            out_channels = config['conv_filters'][i] if i < len(config['conv_filters']) else config['conv_filters'][-1]
            kernel_size = config['conv_kernel_size'][i] if i < len(config['conv_kernel_size']) else config['conv_kernel_size'][-1]
            
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size))
            current_size = current_size - kernel_size + 1
            
            # Add pooling if specified
            if i < len(config['pool_positions']) and config['pool_positions'][i]:
                pool_size = config['pool_size']
                self.pool_layers.append((i, pool_size))
                current_size = current_size // pool_size
            
            # Add dropout after conv if specified
            if config['dropout_after_conv'] and i < config['num_conv_layers'] - 1:
                self.dropout_layers.append(nn.Dropout2d(p=config['conv_dropout_rate']))
            
            in_channels = out_channels
        
        # Calculate flattened size
        self.flat_size = out_channels * current_size * current_size
        
        # Build fully connected layers
        self.fc1 = nn.Linear(self.flat_size, config['fc_hidden_nodes'])
        
        # Dropout after fc1 if specified
        if config['dropout_after_fc1']:
            self.fc1_dropout = nn.Dropout(p=config['fc_dropout_rate'])
        else:
            self.fc1_dropout = None
        
        self.fc2 = nn.Linear(config['fc_hidden_nodes'], 10)
        
        # Activation function
        self.activation = self._get_activation(config['activation'])
    
    def _get_activation(self, name):
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu
        }
        return activations.get(name, F.relu)
    
    def forward(self, x):
        """Forward pass through the network."""
        dropout_idx = 0
        
        # Convolutional layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.activation(x)
            
            # Apply pooling if specified for this layer
            for pool_idx, pool_size in self.pool_layers:
                if pool_idx == i:
                    x = F.max_pool2d(x, pool_size)
            
            # Apply dropout if specified
            if self.config['dropout_after_conv'] and i < len(self.conv_layers) - 1:
                if dropout_idx < len(self.dropout_layers):
                    x = self.dropout_layers[dropout_idx](x)
                    dropout_idx += 1
        
        # Flatten
        x = x.view(-1, self.flat_size)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.activation(x)
        
        if self.fc1_dropout is not None:
            x = self.fc1_dropout(x)
        
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ExperimentRunner:
    """
    Manages running experiments with different network configurations.
    """
    def __init__(self, device, data_dir='./data'):
        self.device = device
        self.data_dir = data_dir
        self.results = []
        
        # Load data once
        print("Loading Fashion MNIST dataset...")
        self.train_loader, self.test_loader = self.load_data()
    
    def load_data(self, batch_size=64):
        """Load Fashion MNIST dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # Fashion MNIST stats
        ])
        
        train_dataset = datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.FashionMNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return train_loader, test_loader
    
    def train_epoch(self, model, optimizer, train_loader):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / len(train_loader.dataset)
        
        return avg_loss, accuracy
    
    def evaluate(self, model, test_loader):
        """Evaluate model on test set."""
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        return test_loss, accuracy
    
    def run_experiment(self, config, epochs=4, lr=0.01, verbose=False): # epoch changed
        """
        Run a single experiment with given configuration.
        Returns metrics including accuracy, loss, training time, and model size.
        """
        start_time = time.time()
        
        try:
            # Create model
            model = FlexibleNetwork(config).to(self.device)
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Optimizer
            # optimizer = optim.Adam(model.parameters(), lr=lr)
            # learning rate
            lr = config.get('learning_rate', 0.01)

            # choose optimizer
            opt_name = config.get('optimizer', 'adam')
            if opt_name == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif opt_name == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            elif opt_name == 'rmsprop':
                optimizer = optim.RMSprop(model.parameters(), lr=lr)
            else:
                raise ValueError(f"Unknown optimizer {opt_name}")
            
            # Training history
            train_losses = []
            train_accs = []
            test_losses = []
            test_accs = []
            
            # Train
            for epoch in range(1, epochs + 1):
                train_loss, train_acc = self.train_epoch(model, optimizer, self.train_loader)
                test_loss, test_acc = self.evaluate(model, self.test_loader)
                
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                
                if verbose and epoch % 2 == 0: # epoch changed
                    print(f"  Epoch {epoch}/{epochs}: "
                          f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
            
            training_time = time.time() - start_time
            
            # Final metrics
            final_train_acc = train_accs[-1]
            final_test_acc = test_accs[-1]
            best_test_acc = max(test_accs)
            final_test_loss = test_losses[-1]
            
            result = {
                'config': config.copy(),
                'num_params': num_params,
                'trainable_params': trainable_params,
                'training_time': training_time,
                'time_per_epoch': training_time / epochs,
                'final_train_acc': final_train_acc,
                'final_test_acc': final_test_acc,
                'best_test_acc': best_test_acc,
                'final_test_loss': final_test_loss,
                'train_history': {
                    'losses': train_losses,
                    'accuracies': train_accs
                },
                'test_history': {
                    'losses': test_losses,
                    'accuracies': test_accs
                },
                'success': True
            }
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            result = {
                'config': config.copy(),
                'error': str(e),
                'success': False
            }
        
        return result
    
    def save_results(self, filename='experiment_results.json'):
        """Save all results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")


def get_baseline_config():
    """Get baseline configuration similar to original MNIST network."""
    return {
        'num_conv_layers': 2,
        'conv_filters': [10, 20],
        'conv_kernel_size': [5, 5],
        'pool_positions': [True, True],  # Pool after each conv layer
        'pool_size': 2,
        'fc_hidden_nodes': 50,
        'dropout_after_conv': True,
        'conv_dropout_rate': 0.5,
        'dropout_after_fc1': False,
        'fc_dropout_rate': 0.5,
        'activation': 'relu',
        'optimizer': 'adam',
        'learning_rate': 0.01
    }


def generate_experiment_plan(strategy='linear_search', num_experiments=50):
    """
    Deterministic linear search:
      - 每个超参数选项都会被遍历一次（在它自己的维度上）
      - conv_filters / conv_kernel_size 和 num_conv_layers 一致
      - 不会产生非法配置
    """

    baseline = get_baseline_config()
    configs = [baseline]  # 先放一个 baseline

    # 所有 conv_filters / conv_kernel_size 选项
    conv_filters_options = [
        [8], [12], [16], [20], [32],                 # 1 层
        [6,12], [8,16], [10,20], [12,24], [16,32],   # 2 层
        [24,48], [32,64],
        [8,16,32], [10,20,40], [20,40,80]            # 3 层
    ]

    conv_kernel_options = [
        [3], [5], [7],                               # 1 层
        [1,1], [3,3], [3,5], [3,7], [5,3], [5,5],    # 2 层
        [5,7], [7,3], [7,5], [7,7],
        [3,3,3], [5,5,5], [7,7,7],                   # 3 层
        [7,5,3], [5,5,3], [7,7,5]
    ]

    # 按“长度”把这些选项分组，方便后面根据 num_conv_layers 匹配
    conv_filters_by_len = {1: [], 2: [], 3: []}
    for f in conv_filters_options:
        conv_filters_by_len[len(f)].append(f)

    conv_kernels_by_len = {1: [], 2: [], 3: []}
    for k in conv_kernel_options:
        conv_kernels_by_len[len(k)].append(k)

    # 其他维度（一次只改一个）
    dimensions = {
        'num_conv_layers': [1, 2, 3],
        'conv_filters': conv_filters_options,
        'conv_kernel_size': conv_kernel_options,
        'fc_hidden_nodes': [16, 25, 50, 64, 100, 128],
        'conv_dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'fc_dropout_rate':   [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'dropout_after_fc1': [False, True],
        'activation': ['relu', 'tanh', 'leaky_relu'],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'learning_rate': [0.2, 0.1, 0.05, 0.01, 0.001]
    }

    # 线性搜索：对每个维度，依次尝试该维度所有取值
    if strategy == 'linear_search':

        for dim_name, dim_values in dimensions.items():
            for value in dim_values:

                config = baseline.copy()

                # 情况 1：修改 num_conv_layers
                if dim_name == 'num_conv_layers':
                    n = value
                    config['num_conv_layers'] = n

                    # 选一组与层数匹配的 filters / kernels（各取该长度下的第一个）
                    config['conv_filters'] = conv_filters_by_len[n][0]
                    config['conv_kernel_size'] = conv_kernels_by_len[n][0]

                # 情况 2：修改 conv_filters
                elif dim_name == 'conv_filters':
                    filters = value
                    n = len(filters)
                    config['num_conv_layers'] = n
                    config['conv_filters'] = filters
                    # 给一个与层数匹配的 kernel（该长度下第一个）
                    config['conv_kernel_size'] = conv_kernels_by_len[n][0]

                # 情况 3：修改 conv_kernel_size
                elif dim_name == 'conv_kernel_size':
                    kernels = value
                    n = len(kernels)
                    config['num_conv_layers'] = n
                    config['conv_kernel_size'] = kernels
                    # 给一个与层数匹配的 filters（该长度下第一个）
                    config['conv_filters'] = conv_filters_by_len[n][0]

                # 情况 4：其他超参数，直接赋值
                else:
                    config[dim_name] = value
                    # 其它依赖仍保持 baseline 的 num_conv_layers / conv_xxx

                # 根据 num_conv_layers 设置 pool_positions 
                n_layers = config['num_conv_layers']
                if n_layers == 1:
                    config['pool_positions'] = [True]
                elif n_layers == 2:
                    config['pool_positions'] = [True, True]
                elif n_layers == 3:
                    config['pool_positions'] = [False, True, True]

                configs.append(config)


    unique_configs = []
    seen = set()
    for cfg in configs:
        cfg_s = json.dumps(cfg, sort_keys=True)
        if cfg_s not in seen:
            seen.add(cfg_s)
            unique_configs.append(cfg)

    return unique_configs[:num_experiments]

def run_all_experiments(configs, device, epochs=4, verbose=True):
    """
    Run all experiments and collect results.
    
    Args:
        configs: List of configurations
        device: Device to run on
        epochs: Number of epochs per experiment
        verbose: Print progress
    
    Returns:
        List of results
    """
    runner = ExperimentRunner(device)
    results = []
    
    print("\n" + "="*80)
    print(f"RUNNING {len(configs)} EXPERIMENTS")
    print("="*80)
    
    for i, config in enumerate(configs):
        if verbose:
            print(f"\nExperiment {i+1}/{len(configs)}")
            print(f"Config: {json.dumps(config, indent=2)}")
        
        result = runner.run_experiment(config, epochs=epochs, verbose=verbose)
        results.append(result)
        
        if result['success']:
            print(f"  √ Test Acc: {result['final_test_acc']:.2f}%, "
                  f"Params: {result['num_params']:,}, "
                  f"Time: {result['training_time']:.1f}s")
        else:
            print(f"  * Failed: {result.get('error', 'Unknown error')}")
    
    runner.results = results
    runner.save_results()
    
    return results


def analyze_results(results):
    """
    Analyze and visualize experiment results.
    
    Args:
        results: List of experiment results
    """
    # Filter successful results
    successful = [r for r in results if r.get('success', False)]
    
    if len(successful) == 0:
        print("No successful experiments to analyze!")
        return
    
    print("\n" + "="*80)
    print("EXPERIMENT ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    
    # Best performing models
    sorted_by_acc = sorted(successful, key=lambda x: x['best_test_acc'], reverse=True)
    
    print("\n" + "-"*80)
    print("TOP 5 MODELS BY ACCURACY")
    print("-"*80)
    for i, result in enumerate(sorted_by_acc[:5]):
        print(f"\n{i+1}. Test Accuracy: {result['best_test_acc']:.2f}%")
        print(f"   Parameters: {result['num_params']:,}")
        print(f"   Training time: {result['training_time']:.1f}s")
        print(f"   Config: {json.dumps(result['config'], indent=6)}")
    
    # Fastest models
    sorted_by_time = sorted(successful, key=lambda x: x['training_time'])
    
    print("\n" + "-"*80)
    print("TOP 5 FASTEST MODELS")
    print("-"*80)
    for i, result in enumerate(sorted_by_time[:5]):
        print(f"\n{i+1}. Training time: {result['training_time']:.1f}s")
        print(f"   Test Accuracy: {result['best_test_acc']:.2f}%")
        print(f"   Parameters: {result['num_params']:,}")
    
    # Most efficient (accuracy per parameter)
    for result in successful:
        result['efficiency'] = result['best_test_acc'] / (result['num_params'] / 1000)
    
    sorted_by_eff = sorted(successful, key=lambda x: x['efficiency'], reverse=True)
    
    print("\n" + "-"*80)
    print("TOP 5 MOST EFFICIENT MODELS (Accuracy/1K Parameters)")
    print("-"*80)
    for i, result in enumerate(sorted_by_eff[:5]):
        print(f"\n{i+1}. Efficiency: {result['efficiency']:.2f}")
        print(f"   Test Accuracy: {result['best_test_acc']:.2f}%")
        print(f"   Parameters: {result['num_params']:,}")



def main(argv):
    """
    Main function for network architecture experiments.
    
    Args:
        argv: Command line arguments
              argv[1]: strategy ('linear', or 'test')
              argv[2]: number of experiments (default: 50)
              argv[3]: epochs per experiment (default: 4)
    """
    # Parse arguments
    strategy = argv[1] if len(argv) > 1 else 'linear'
    num_experiments = int(argv[2]) if len(argv) > 2 else 50
    epochs = int(argv[3]) if len(argv) > 3 else 4 # epoch changed
    
    # Map strategy names
    strategy_map = {
        'linear': 'linear_search',
        'test': 'linear_search'  # Quick test with few experiments
    }
    
    strategy_full = strategy_map.get(strategy, 'linear_search')
    
    if strategy == 'test':
        num_experiments = 5
        epochs = 3
        print("\n*** TEST MODE: Running 5 experiments with 3 epochs each ***\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate experiment plan
    print(f"\nGenerating experiment plan: {strategy_full}")
    print(f"Target number of experiments: {num_experiments}")
    print(f"Epochs per experiment: {epochs}")
    
    configs = generate_experiment_plan(strategy=strategy_full, 
                                      num_experiments=num_experiments)
    
    print(f"Generated {len(configs)} unique configurations")
    
    # Run experiments
    results = run_all_experiments(configs, device, epochs=epochs, verbose=True)
    
    # Analyze results
    analyze_results(results)
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: experiment_results.json")
    print(f"Plots saved to: experiment_plots/")


if __name__ == "__main__":
    main(sys.argv)