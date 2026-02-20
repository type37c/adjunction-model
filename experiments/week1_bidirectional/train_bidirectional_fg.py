"""
Week 1: Train Bidirectional F/G (η + ε)

This script trains the bidirectional F/G model with both η and ε losses.

Expected outcomes:
- η (shape reconstruction error) decreases
- ε (action reconstruction error) decreases
- Both η and ε converge to low values (< 0.1)

This is the first step of the full implementation blitz plan.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.bidirectional_fg import BidirectionalFG
from src.data.synthetic_dataset import SyntheticAffordanceDataset
from torch.utils.data import DataLoader
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def train_bidirectional_fg(
    num_epochs: int = 50,
    batch_size: int = 4,
    num_train_samples: int = 100,
    num_val_samples: int = 20,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device('cpu')
):
    """
    Train bidirectional F/G model.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        learning_rate: Learning rate
        device: Device to run on
    """
    
    print("=" * 60)
    print("WEEK 1: BIDIRECTIONAL F/G TRAINING (η + ε)")
    print("=" * 60)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Train samples: {num_train_samples}")
    print(f"Val samples: {num_val_samples}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print("=" * 60)
    print()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SyntheticAffordanceDataset(
        num_samples=num_train_samples,
        num_points=512,
        shape_types=[0, 1, 2]  # cube, cylinder, sphere
    )
    
    val_dataset = SyntheticAffordanceDataset(
        num_samples=num_val_samples,
        num_points=512,
        shape_types=[0, 1, 2]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("Creating bidirectional F/G model...")
    model = BidirectionalFG(
        input_dim=3,
        hidden_dim=128,
        affordance_dim=16,
        action_dim=3,  # 3 discrete actions: Push, Pull, Rotate
        num_points=512
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_eta': [],
        'train_epsilon': [],
        'train_total': [],
        'val_eta': [],
        'val_epsilon': [],
        'val_total': []
    }
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_eta_sum = 0.0
        train_epsilon_sum = 0.0
        train_total_sum = 0.0
        train_batches = 0
        
        for batch_data in train_loader:
            pos = batch_data['pos'].to(device)
            batch = batch_data['batch'].to(device)
            
            # Generate random actions for ε training
            # In Week 1, we use random actions since we don't have Agent C yet
            batch_size_actual = batch.max().item() + 1
            actions = torch.randn(batch_size_actual, 3).to(device)
            
            # Forward pass
            results = model(pos, batch, actions)
            
            # Compute losses
            eta = results['eta'].mean()
            epsilon = results['epsilon'].mean()
            total_loss = eta + epsilon
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Record
            train_eta_sum += eta.item()
            train_epsilon_sum += epsilon.item()
            train_total_sum += total_loss.item()
            train_batches += 1
        
        # Average training losses
        train_eta_avg = train_eta_sum / train_batches
        train_epsilon_avg = train_epsilon_sum / train_batches
        train_total_avg = train_total_sum / train_batches
        
        # Validation
        model.eval()
        val_eta_sum = 0.0
        val_epsilon_sum = 0.0
        val_total_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                pos = batch_data['pos'].to(device)
                batch = batch_data['batch'].to(device)
                
                # Generate random actions
                batch_size_actual = batch.max().item() + 1
                actions = torch.randn(batch_size_actual, 3).to(device)
                
                # Forward pass
                results = model(pos, batch, actions)
                
                # Compute losses
                eta = results['eta'].mean()
                epsilon = results['epsilon'].mean()
                total_loss = eta + epsilon
                
                # Record
                val_eta_sum += eta.item()
                val_epsilon_sum += epsilon.item()
                val_total_sum += total_loss.item()
                val_batches += 1
        
        # Average validation losses
        val_eta_avg = val_eta_sum / val_batches
        val_epsilon_avg = val_epsilon_sum / val_batches
        val_total_avg = val_total_sum / val_batches
        
        # Update history
        history['train_eta'].append(train_eta_avg)
        history['train_epsilon'].append(train_epsilon_avg)
        history['train_total'].append(train_total_avg)
        history['val_eta'].append(val_eta_avg)
        history['val_epsilon'].append(val_epsilon_avg)
        history['val_total'].append(val_total_avg)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - η: {train_eta_avg:.6f}, ε: {train_epsilon_avg:.6f}, Total: {train_total_avg:.6f}")
        print(f"  Val   - η: {val_eta_avg:.6f}, ε: {val_epsilon_avg:.6f}, Total: {val_total_avg:.6f}")
        print()
    
    # Save model
    checkpoint_dir = Path('/home/ubuntu/adjunction-model/experiments/week1_bidirectional/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / 'bidirectional_fg_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, checkpoint_path)
    
    print(f"Model saved to {checkpoint_path}")
    
    # Save history
    results_dir = Path('/home/ubuntu/adjunction-model/experiments/week1_bidirectional/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, results_dir)
    
    # Analyze results
    analyze_results(model, val_loader, device, results_dir)
    
    print("Training completed!")
    return model, history


def plot_training_curves(history, results_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # η
    axes[0].plot(history['train_eta'], label='Train')
    axes[0].plot(history['val_eta'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('η (Shape Reconstruction Error)')
    axes[0].set_title('Unit η: Shape → F → G → Shape')
    axes[0].legend()
    axes[0].grid(True)
    
    # ε
    axes[1].plot(history['train_epsilon'], label='Train')
    axes[1].plot(history['val_epsilon'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ε (Action Reconstruction Error)')
    axes[1].set_title('Counit ε: Action → F_inv → G_inv → Action')
    axes[1].legend()
    axes[1].grid(True)
    
    # Total
    axes[2].plot(history['train_total'], label='Train')
    axes[2].plot(history['val_total'], label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Total Loss (η + ε)')
    axes[2].set_title('Total Coherence Signal')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=150)
    plt.close()
    
    print(f"Training curves saved to {results_dir / 'training_curves.png'}")


def analyze_results(model, val_loader, device, results_dir):
    """Analyze results and check if η and ε are both low for some actions."""
    model.eval()
    
    all_etas = []
    all_epsilons = []
    all_actions = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            pos = batch_data['pos'].to(device)
            batch = batch_data['batch'].to(device)
            
            # Generate multiple random actions
            batch_size_actual = batch.max().item() + 1
            for _ in range(10):  # 10 random actions per shape
                actions = torch.randn(batch_size_actual, 3).to(device)
                
                # Forward pass
                results = model(pos, batch, actions)
                
                # Record
                all_etas.extend(results['eta'].cpu().numpy())
                all_epsilons.extend(results['epsilon'].cpu().numpy())
                all_actions.extend(actions.cpu().numpy())
    
    all_etas = np.array(all_etas)
    all_epsilons = np.array(all_epsilons)
    all_actions = np.array(all_actions)
    
    # Find actions with both low η and low ε
    threshold = 0.1
    low_eta_mask = all_etas < threshold
    low_epsilon_mask = all_epsilons < threshold
    both_low_mask = low_eta_mask & low_epsilon_mask
    
    print("=" * 60)
    print("ANALYSIS: Actions with both low η and low ε")
    print("=" * 60)
    print(f"Total actions: {len(all_etas)}")
    print(f"Actions with η < {threshold}: {low_eta_mask.sum()} ({100 * low_eta_mask.sum() / len(all_etas):.1f}%)")
    print(f"Actions with ε < {threshold}: {low_epsilon_mask.sum()} ({100 * low_epsilon_mask.sum() / len(all_epsilons):.1f}%)")
    print(f"Actions with both < {threshold}: {both_low_mask.sum()} ({100 * both_low_mask.sum() / len(all_etas):.1f}%)")
    print()
    
    if both_low_mask.sum() > 0:
        print("Sample actions with both low η and low ε:")
        for i in range(min(5, both_low_mask.sum())):
            idx = np.where(both_low_mask)[0][i]
            print(f"  Action: {all_actions[idx]}, η: {all_etas[idx]:.6f}, ε: {all_epsilons[idx]:.6f}")
    
    # Plot scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(all_etas, all_epsilons, c=both_low_mask, cmap='RdYlGn', alpha=0.5)
    ax.set_xlabel('η (Shape Reconstruction Error)')
    ax.set_ylabel('ε (Action Reconstruction Error)')
    ax.set_title('η vs ε: Actions with both low values are "meaningful and coherent"')
    ax.axhline(threshold, color='r', linestyle='--', label=f'ε threshold ({threshold})')
    ax.axvline(threshold, color='r', linestyle='--', label=f'η threshold ({threshold})')
    ax.legend()
    ax.grid(True)
    plt.colorbar(scatter, label='Both low')
    plt.tight_layout()
    plt.savefig(results_dir / 'eta_vs_epsilon.png', dpi=150)
    plt.close()
    
    print(f"η vs ε scatter plot saved to {results_dir / 'eta_vs_epsilon.png'}")
    print("=" * 60)


if __name__ == '__main__':
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train
    model, history = train_bidirectional_fg(
        num_epochs=50,
        batch_size=4,
        num_train_samples=100,
        num_val_samples=20,
        learning_rate=1e-3,
        device=device
    )
