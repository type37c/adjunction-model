"""
Phase 2 Training v2: Fine-tuning with Agent Layer C v2

This training phase fine-tunes the pre-trained F⊣G adjunction with Agent Layer C v2.

Key differences from Phase 2 v1:
- v1: Agent C receives scalar coherence signal only
- v2: Agent C receives spatial coherence signal + priority-based attention

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

from src.models.conditional_adjunction_v2 import ConditionalAdjunctionModelV2
from src.data.synthetic_dataset import SyntheticAffordanceDataset


class Phase2TrainerV2:
    """
    Trainer for Phase 2 with Agent C v2.
    """
    
    def __init__(
        self,
        model: ConditionalAdjunctionModelV2,
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
            points_list = batch_data['points']
            affordances_list = batch_data['affordances']
            
            # Stack into batched tensors
            pos = torch.cat([p for p in points_list], dim=0).to(self.device)  # (B*N, 3)
            affordances_gt = torch.cat([a for a in affordances_list], dim=0).to(self.device)
            
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
            coherence_spatial_prev = torch.zeros(pos.size(0), device=self.device)
            
            # Forward pass (v2: includes spatial coherence)
            results = self.model(
                pos, batch, agent_state,
                coherence_signal_prev, coherence_spatial_prev
            )
            
            # Extract results
            affordances_pred = results['affordances']
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
