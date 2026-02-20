"""
Train Bidirectional F/G (η + ε)

This script trains the complete adjunction structure with both unit (η) and counit (ε).

Expected outcomes:
- η converges to < 0.1 (shape reconstruction)
- ε converges to < 0.1 (action reconstruction)
- Actions with both low η and low ε exist (coherent shape-action pairs)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from core.models.bidirectional_fg import BidirectionalFG
from src.data.synthetic_dataset import SyntheticAffordanceDataset
from torch.utils.data import DataLoader
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def train_bidirectional_fg(
    num_epochs: int = 100,
    batch_size: int = 8,
    num_train_samples: int = 200,
    num_val_samples: int = 40,
    learning_rate: float = 1e-3,
    affordance_dim: int = 16,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train bidirectional F/G model.
    """
    
    print("=" * 80)
    print("TRAINING BIDIRECTIONAL F/G (η + ε)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Affordance dim: {affordance_dim}")
    print("=" * 80)
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
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Create model
    print("Creating model...")
    model = BidirectionalFG(
        point_dim=3,
        affordance_dim=affordance_dim,
        action_dim=3,  # 3 discrete actions: Push, Pull, Rotate
        hidden_dim=128,
        num_layers=3
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training history
    history = {
        'train_eta': [],
        'train_epsilon': [],
        'train_total': [],
        'val_eta': [],
        'val_epsilon': [],
        'val_total': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_eta_sum = 0.0
        train_epsilon_sum = 0.0
        train_total_sum = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_data in pbar:
            # batch_dataは辞書で、各値は(B, N, ...)の形状
            points = batch_data['points'].to(device)  # (B, N, 3)
            B, N, _ = points.shape
            
            # Flatten for processing
            pos = points.view(-1, 3)  # (B*N, 3)
            
            # バッチインデックスを作成
            batch = torch.arange(B).repeat_interleave(N).to(device)  # (B*N,)
            
            # Generate random actions for ε training
            # In Phase 0, we use random actions since we don't have trained Agent C yet
            batch_size_actual = batch.max().item() + 1
            actions = torch.randn(batch_size_actual, 3).to(device)
            
            # Forward pass
            results = model(pos, actions, batch)
            
            # Compute losses
            eta = results['eta'].mean()
            epsilon = results['epsilon'].mean()
            total_loss = eta + epsilon
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Record
            train_eta_sum += eta.item()
            train_epsilon_sum += epsilon.item()
            train_total_sum += total_loss.item()
            train_batches += 1
            
            pbar.set_postfix({
                'η': f'{eta.item():.4f}',
                'ε': f'{epsilon.item():.4f}',
                'total': f'{total_loss.item():.4f}'
            })
        
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
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch_data in pbar:
                # batch_dataは辞書で、各値は(B, N, ...)の形状
                points = batch_data['points'].to(device)  # (B, N, 3)
                B, N, _ = points.shape
                
                # Flatten for processing
                pos = points.view(-1, 3)  # (B*N, 3)
                
                # バッチインデックスを作成
                batch = torch.arange(B).repeat_interleave(N).to(device)  # (B*N,)
                
                # Generate random actions
                batch_size_actual = B
                actions = torch.randn(batch_size_actual, 3).to(device)
                
                # Forward pass
                results = model(pos, actions, batch)
                
                # Compute losses
                eta = results['eta'].mean()
                epsilon = results['epsilon'].mean()
                total_loss = eta + epsilon
                
                # Record
                val_eta_sum += eta.item()
                val_epsilon_sum += epsilon.item()
                val_total_sum += total_loss.item()
                val_batches += 1
                
                pbar.set_postfix({
                    'η': f'{eta.item():.4f}',
                    'ε': f'{epsilon.item():.4f}',
                    'total': f'{total_loss.item():.4f}'
                })
        
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
        
        # Learning rate scheduling
        scheduler.step(val_total_avg)
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train - η: {train_eta_avg:.6f}, ε: {train_epsilon_avg:.6f}, Total: {train_total_avg:.6f}")
        print(f"  Val   - η: {val_eta_avg:.6f}, ε: {val_epsilon_avg:.6f}, Total: {val_total_avg:.6f}")
        
        # Save best model
        if val_total_avg < best_val_loss:
            best_val_loss = val_total_avg
            checkpoint_dir = Path('/home/ubuntu/adjunction-model/results/phase0')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_avg,
                'history': history
            }, checkpoint_dir / 'best_bidirectional_fg.pt')
            
            print(f"  ✓ Best model saved (val_loss: {val_total_avg:.6f})")
        
        print()
    
    # Save final model
    checkpoint_dir = Path('/home/ubuntu/adjunction-model/results/phase0')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, checkpoint_dir / 'final_bidirectional_fg.pt')
    
    print(f"Final model saved to {checkpoint_dir / 'final_bidirectional_fg.pt'}")
    
    # Save history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, checkpoint_dir)
    
    # Analyze results
    analyze_results(model, val_loader, device, checkpoint_dir)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    
    return model, history


def plot_training_curves(history, save_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_eta']) + 1)
    
    # η
    axes[0].plot(epochs, history['train_eta'], label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_eta'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('η (Shape Reconstruction Error)')
    axes[0].set_title('Unit η: Shape → F → G → Shape')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ε
    axes[1].plot(epochs, history['train_epsilon'], label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_epsilon'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ε (Action Reconstruction Error)')
    axes[1].set_title('Counit ε: Action → F_inv → G_inv → Action')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Total
    axes[2].plot(epochs, history['train_total'], label='Train', linewidth=2)
    axes[2].plot(epochs, history['val_total'], label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Total Loss (η + ε)')
    axes[2].set_title('Total Coherence Signal')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_dir / 'training_curves.png'}")


def analyze_results(model, val_loader, device, save_dir):
    """Analyze results and find coherent shape-action pairs."""
    model.eval()
    
    all_etas = []
    all_epsilons = []
    all_actions = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            # batch_dataは辞書で、各値は(B, N, ...)の形状
            points = batch_data['points'].to(device)  # (B, N, 3)
            B, N, _ = points.shape
            
            # Flatten for processing
            pos = points.view(-1, 3)  # (B*N, 3)
            
            # バッチインデックスを作成
            batch = torch.arange(B).repeat_interleave(N).to(device)  # (B*N,)
            
            # Generate multiple random actions per shape
            batch_size_actual = batch.max().item() + 1
            for _ in range(20):  # 20 random actions per shape
                actions = torch.randn(batch_size_actual, 3).to(device)
                
                # Forward pass
                results = model(pos, actions, batch)
                
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
    
    print("\n" + "=" * 80)
    print("ANALYSIS: Coherent Shape-Action Pairs")
    print("=" * 80)
    print(f"Total samples: {len(all_etas)}")
    print(f"Actions with η < {threshold}: {low_eta_mask.sum()} ({100 * low_eta_mask.sum() / len(all_etas):.1f}%)")
    print(f"Actions with ε < {threshold}: {low_epsilon_mask.sum()} ({100 * low_epsilon_mask.sum() / len(all_epsilons):.1f}%)")
    print(f"Actions with both < {threshold}: {both_low_mask.sum()} ({100 * both_low_mask.sum() / len(all_etas):.1f}%)")
    
    if both_low_mask.sum() > 0:
        print(f"\nSample coherent actions (η < {threshold}, ε < {threshold}):")
        for i in range(min(5, both_low_mask.sum())):
            idx = np.where(both_low_mask)[0][i]
            print(f"  Action: {all_actions[idx]}, η: {all_etas[idx]:.6f}, ε: {all_epsilons[idx]:.6f}")
    
    # Plot scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(all_etas, all_epsilons, c=both_low_mask, 
                        cmap='RdYlGn', alpha=0.6, s=20)
    ax.set_xlabel('η (Shape Reconstruction Error)', fontsize=12)
    ax.set_ylabel('ε (Action Reconstruction Error)', fontsize=12)
    ax.set_title('Coherence Analysis: η vs ε\n(Green = Both low = Coherent)', fontsize=14)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, 
              label=f'ε threshold ({threshold})')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
              label=f'η threshold ({threshold})')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Both low (coherent)')
    plt.tight_layout()
    plt.savefig(save_dir / 'eta_vs_epsilon.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"η vs ε scatter plot saved to {save_dir / 'eta_vs_epsilon.png'}")
    print("=" * 80)


if __name__ == '__main__':
    # Train
    model, history = train_bidirectional_fg(
        num_epochs=100,
        batch_size=8,
        num_train_samples=200,
        num_val_samples=40,
        learning_rate=1e-3,
        affordance_dim=16
    )
