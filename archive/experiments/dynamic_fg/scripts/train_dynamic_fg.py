"""
Train Dynamic F/G model

Training objective:
- Minimize prediction error: ||ee_pos_pred - ee_pos_true||²
- F learns to extract reachability affordance from temporal point clouds
- G learns to predict next end-effector position from affordance + action
"""

import sys
sys.path.append('/home/ubuntu/adjunction-model/experiments/dynamic_fg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.dynamic_fg import DynamicFGModel


class TemporalPointCloudDataset(Dataset):
    """
    Dataset for temporal point clouds
    """
    
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.samples = pickle.load(f)
        
        print(f"Loaded {len(self.samples)} samples from {dataset_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'pc_t': torch.FloatTensor(sample['pc_t']),
            'pc_t1': torch.FloatTensor(sample['pc_t1']),
            'action': torch.FloatTensor(sample['action']),
            'ee_pos_t': torch.FloatTensor(sample['ee_pos_t']),
            'ee_pos_t1': torch.FloatTensor(sample['ee_pos_t1']),
            'obj_pos': torch.FloatTensor(sample['obj_pos']),
        }


def train_dynamic_fg(
    dataset_path='data/temporal_pointcloud_dataset.pkl',
    affordance_dim=32,
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir='models/checkpoints',
    results_dir='results'
):
    """
    Train dynamic F/G model
    """
    print("="*60)
    print("Training Dynamic F/G")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Affordance dim: {affordance_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print()
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
    dataset = TemporalPointCloudDataset(dataset_path)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Create model
    model = DynamicFGModel(
        affordance_dim=affordance_dim,
        action_dim=3,
        output_dim=3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            pc_t = batch['pc_t'].to(device)
            pc_t1 = batch['pc_t1'].to(device)
            action = batch['action'].to(device)
            ee_pos_t1 = batch['ee_pos_t1'].to(device)
            
            # Forward
            affordance, ee_pos_pred = model(pc_t, pc_t1, action)
            loss = criterion(ee_pos_pred, ee_pos_t1)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                pc_t = batch['pc_t'].to(device)
                pc_t1 = batch['pc_t1'].to(device)
                action = batch['action'].to(device)
                ee_pos_t1 = batch['ee_pos_t1'].to(device)
                
                affordance, ee_pos_pred = model(pc_t, pc_t1, action)
                loss = criterion(ee_pos_pred, ee_pos_t1)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'dynamic_fg_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'affordance_dim': affordance_dim,
            }, checkpoint_path)
            print(f"  → Saved best model (val_loss={val_loss:.6f})")
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'dynamic_fg_final.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'affordance_dim': affordance_dim,
    }, final_checkpoint_path)
    print(f"\nSaved final model to {final_checkpoint_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Val Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Dynamic F/G Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved training curves to {plot_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.6f}")
    print("="*60)
    
    return model, train_losses, val_losses


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                        default='data/temporal_pointcloud_dataset.pkl',
                        help='Path to dataset')
    parser.add_argument('--affordance-dim', type=int, default=32,
                        help='Affordance dimension')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    args = parser.parse_args()
    
    train_dynamic_fg(
        dataset_path=args.dataset,
        affordance_dim=args.affordance_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )
