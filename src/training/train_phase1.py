"""
Phase 1 Training: Pre-training F ⊣ G without Agent Layer C

This implements the first stage of the 3-phase learning framework:
- Learn the adjunction structure F ⊣ G from shape-affordance data
- No agent state C is involved at this stage
- Self-supervised learning from video/demonstration data

Loss function:
    L = L_recon + λ_aff * L_aff + λ_coh * L_coherence
    
Where:
    L_recon: Reconstruction loss for G(F(shape))
    L_aff: Affordance prediction loss for F(shape)
    L_coherence: Coherence signal regularization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import os
from tqdm import tqdm


class Phase1Trainer:
    """
    Trainer for Phase 1: Pre-training the adjunction F ⊣ G.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        lambda_aff: float = 1.0,
        lambda_coh: float = 0.1,
        device: str = 'cpu',
        log_dir: str = './logs'
    ):
        """
        Args:
            model: AdjunctionModel instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Learning rate
            lambda_aff: Weight for affordance loss
            lambda_coh: Weight for coherence loss
            device: Device to train on
            log_dir: Directory for logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_dir = log_dir
        
        # Loss weights
        self.lambda_aff = lambda_aff
        self.lambda_coh = lambda_coh
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.affordance_criterion = nn.BCEWithLogitsLoss()
        
        # Logging
        os.makedirs(log_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute the total loss for a batch.
        
        Args:
            batch: Dictionary with 'points' and 'affordances'
            return_components: If True, return loss components as dict
        
        Returns:
            loss: Total loss (or dict of loss components)
        """
        points = batch['points']  # (B, N, 3)
        target_affordances = batch['affordances']  # (B, N, num_affordances)
        
        # Flatten batch for model input
        B, N, _ = points.shape
        pos = points.view(B * N, 3).to(self.device)
        batch_idx = torch.repeat_interleave(torch.arange(B, device=self.device), N)
        target_aff_flat = target_affordances.view(B * N, -1).to(self.device)
        
        # Forward pass
        results = self.model(pos, batch_idx)
        
        # 1. Affordance prediction loss
        pred_affordances = results['affordances']  # (B*N, num_affordances)
        loss_aff = self.affordance_criterion(pred_affordances, target_aff_flat)
        
        # 2. Reconstruction loss (Chamfer distance is already computed as coherence signal)
        # We want to minimize the coherence signal (distance between original and reconstructed)
        coherence_signal = results['coherence_signal']  # (B,)
        loss_recon = coherence_signal.mean()
        
        # 3. Coherence regularization
        # We also want coherence signal to be informative (not always zero)
        # Add a small penalty to prevent collapse
        loss_coh = -torch.log(coherence_signal.mean() + 1e-6)
        
        # Total loss
        loss = loss_recon + self.lambda_aff * loss_aff + self.lambda_coh * loss_coh
        
        if return_components:
            return {
                'total': loss,
                'recon': loss_recon,
                'affordance': loss_aff,
                'coherence': loss_coh,
                'coherence_signal_mean': coherence_signal.mean()
            }
        
        return loss
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        total_loss = 0.0
        total_recon = 0.0
        total_aff = 0.0
        total_coh = 0.0
        total_coherence_signal = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Compute loss
            loss_dict = self.compute_loss(batch, return_components=True)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon += loss_dict['recon'].item()
            total_aff += loss_dict['affordance'].item()
            total_coh += loss_dict['coherence'].item()
            total_coherence_signal += loss_dict['coherence_signal_mean'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'coh_sig': f"{loss_dict['coherence_signal_mean'].item():.4f}"
            })
        
        # Average losses
        n_batches = len(self.train_loader)
        avg_losses = {
            'total': total_loss / n_batches,
            'recon': total_recon / n_batches,
            'affordance': total_aff / n_batches,
            'coherence': total_coh / n_batches,
            'coherence_signal': total_coherence_signal / n_batches
        }
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary of average validation losses
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_recon = 0.0
        total_aff = 0.0
        total_coherence_signal = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                loss_dict = self.compute_loss(batch, return_components=True)
                
                total_loss += loss_dict['total'].item()
                total_recon += loss_dict['recon'].item()
                total_aff += loss_dict['affordance'].item()
                total_coherence_signal += loss_dict['coherence_signal_mean'].item()
        
        n_batches = len(self.val_loader)
        avg_losses = {
            'total': total_loss / n_batches,
            'recon': total_recon / n_batches,
            'affordance': total_aff / n_batches,
            'coherence_signal': total_coherence_signal / n_batches
        }
        
        return avg_losses
    
    def train(self, num_epochs: int, save_every: int = 10):
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting Phase 1 training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Loss weights: λ_aff={self.lambda_aff}, λ_coh={self.lambda_coh}")
        print("="*60)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses = self.validate()
            if val_losses:
                self.val_losses.append(val_losses)
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_losses['total']:.4f}, "
                  f"Recon: {train_losses['recon']:.4f}, "
                  f"Aff: {train_losses['affordance']:.4f}, "
                  f"Coh Signal: {train_losses['coherence_signal']:.4f}")
            
            if val_losses:
                print(f"  Val   - Loss: {val_losses['total']:.4f}, "
                      f"Recon: {val_losses['recon']:.4f}, "
                      f"Aff: {val_losses['affordance']:.4f}, "
                      f"Coh Signal: {val_losses['coherence_signal']:.4f}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)
        
        print("\nPhase 1 training completed!")
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.log_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    # Test the trainer
    print("Testing Phase 1 Trainer...")
    
    # Import dependencies
    import sys
    sys.path.append('/home/ubuntu/adjunction-model')
    from src.models import create_adjunction_model
    from src.data.synthetic_dataset import get_dataloader
    
    # Create model
    model = create_adjunction_model(
        num_affordances=5,
        num_points=512,
        f_hidden_dim=32,
        g_hidden_dim=64
    )
    
    # Create data loaders
    train_loader = get_dataloader(
        batch_size=4,
        num_samples=100,
        num_points=512,
        split='train'
    )
    
    val_loader = get_dataloader(
        batch_size=4,
        num_samples=20,
        num_points=512,
        split='val'
    )
    
    # Create trainer
    trainer = Phase1Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        lambda_aff=1.0,
        lambda_coh=0.1,
        device='cpu',
        log_dir='./logs/test'
    )
    
    # Train for 2 epochs
    trainer.train(num_epochs=2, save_every=1)
    
    print("\nPhase 1 Trainer test passed!")
