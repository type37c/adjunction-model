"""
Phase 2 Training: Fine-tuning with Agent Layer C

This training phase fine-tunes the pre-trained F⊣G adjunction with Agent Layer C.

Key differences from Phase 1:
- Phase 1: Pre-train F⊣G without C (self-supervised on shape-action pairs)
- Phase 2: Fine-tune F_C⊣G_C with C (agent learns to adapt based on coherence signal)

Loss components:
1. L_recon: Reconstruction loss (coherence signal)
2. L_aff: Affordance prediction loss
3. L_kl: KL divergence between posterior and prior (RSSM regularization)
4. L_coherence: Coherence regularization (prevent collapse)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.conditional_adjunction import ConditionalAdjunctionModel
from src.data.synthetic_dataset import SyntheticAffordanceDataset


class Phase2Trainer:
    """
    Trainer for Phase 2: Fine-tuning with Agent Layer C.
    """
    
    def __init__(
        self,
        model: ConditionalAdjunctionModel,
        device: torch.device = torch.device('cpu'),
        lr: float = 1e-4,
        lambda_aff: float = 1.0,
        lambda_kl: float = 0.1,
        lambda_coherence: float = 0.1
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Loss weights
        self.lambda_aff = lambda_aff
        self.lambda_kl = lambda_kl
        self.lambda_coherence = lambda_coherence
        
        # Loss functions
        self.aff_criterion = nn.BCEWithLogitsLoss()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_recon = 0.0
        total_aff = 0.0
        total_kl = 0.0
        total_coherence = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Extract data from batch
            # Note: DataLoader doesn't automatically create 'batch' for us
            # We need to create it manually
            points_list = batch_data['points']  # List of (num_points, 3)
            affordances_list = batch_data['affordances']  # List of (num_points, num_affordances)
            
            # Stack into batched tensors
            pos = torch.cat([p for p in points_list], dim=0).to(self.device)  # (B*N, 3)
            affordances_gt = torch.cat([a for a in affordances_list], dim=0).to(self.device)  # (B*N, num_affordances)
            
            # Create batch assignment
            batch_size = len(points_list)
            num_points = points_list[0].size(0)
            batch = torch.repeat_interleave(
                torch.arange(batch_size, device=self.device),
                num_points
            )
            
            batch_size = batch.max().item() + 1
            
            # Initialize agent state
            agent_state = self.model.initial_state(batch_size, self.device)
            coherence_signal_prev = torch.zeros(batch_size, 1, device=self.device)
            
            # Forward pass
            results = self.model(pos, batch, agent_state, coherence_signal_prev)
            
            # Extract results
            affordances_pred = results['affordances']
            reconstructed = results['reconstructed']
            coherence_signal = results['coherence_signal']
            rssm_info = results['rssm_info']
            
            # Compute losses
            # 1. Reconstruction loss (coherence signal)
            L_recon = coherence_signal.mean()
            
            # 2. Affordance loss
            L_aff = self.aff_criterion(affordances_pred, affordances_gt)
            
            # 3. KL divergence (RSSM regularization)
            L_kl = self.model.agent_c.rssm.kl_divergence(
                rssm_info['posterior_mean'],
                rssm_info['posterior_std'],
                rssm_info['prior_mean'],
                rssm_info['prior_std']
            ).mean()
            
            # 4. Coherence regularization (prevent collapse to zero)
            L_coherence = -torch.log(coherence_signal + 1e-8).mean()
            
            # Total loss
            loss = L_recon + self.lambda_aff * L_aff + self.lambda_kl * L_kl + self.lambda_coherence * L_coherence
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon += L_recon.item()
            total_aff += L_aff.item()
            total_kl += L_kl.item()
            total_coherence += L_coherence.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"Recon={L_recon.item():.4f}, "
                      f"Aff={L_aff.item():.4f}, "
                      f"KL={L_kl.item():.4f}, "
                      f"Coh={L_coherence.item():.4f}")
        
        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'recon': total_recon / num_batches,
            'aff': total_aff / num_batches,
            'kl': total_kl / num_batches,
            'coherence': total_coherence / num_batches
        }
        
        return metrics
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {path} (epoch {epoch})")
        return epoch


if __name__ == '__main__':
    # Test Phase 2 Trainer
    print("Testing Phase 2 Trainer...")
    
    # Create model
    model = ConditionalAdjunctionModel(
        num_affordances=5,
        num_points=256,
        f_hidden_dim=32,
        g_hidden_dim=64,
        agent_hidden_dim=128,
        agent_latent_dim=32,
        context_dim=64
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dataset
    dataset = SyntheticAffordanceDataset(
        num_samples=20,
        num_points=256
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Create trainer
    trainer = Phase2Trainer(
        model=model,
        device=torch.device('cpu'),
        lr=1e-3,
        lambda_aff=1.0,
        lambda_kl=0.1,
        lambda_coherence=0.1
    )
    
    print("Trainer created")
    
    # Train for 2 epochs
    print("\nTraining for 2 epochs...")
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}/2")
        metrics = trainer.train_epoch(dataloader, epoch)
        
        print(f"Epoch {epoch + 1} metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    print("\nPhase 2 Trainer test passed!")
