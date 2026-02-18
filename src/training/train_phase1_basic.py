"""
Phase 1 Training: Basic Adjunction Pre-training (F ⊣ G)

This implements the first stage of the 3-phase learning framework:
- Train Functor F and Functor G to learn basic reconstruction
- Agent C is frozen (not trained)
- Goal: F/G acquire "seeing" ability (basic shape-affordance mapping)

Loss components:
1. L_recon: Reconstruction loss (Chamfer distance) - MINIMIZED
2. L_aff: Affordance prediction loss - MINIMIZED
3. L_coherence: Coherence regularization (prevent collapse to zero)

Expected outcomes:
- Reconstruction error (η) decreases significantly
- F/G learn to map shapes to affordances and back
- F/G become "seeing eyes" for Agent C in later phases
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.adjunction_model import AdjunctionModel
from src.data.synthetic_dataset import SyntheticAffordanceDataset


class Phase1BasicTrainer:
    """
    Trainer for Phase 1: Basic Adjunction Pre-training.
    
    Trains F and G only, with Agent C frozen.
    """
    
    def __init__(
        self,
        model: AdjunctionModel,
        device: torch.device = torch.device('cpu'),
        lr_fg: float = 1e-3,
        lambda_aff: float = 1.0,
        lambda_recon: float = 1.0,
        lambda_coherence: float = 0.1
    ):
        """
        Args:
            model: AdjunctionModel instance
            device: Device to train on
            lr_fg: Learning rate for F/G
            lambda_aff: Weight for affordance loss
            lambda_recon: Weight for reconstruction loss
            lambda_coherence: Weight for coherence regularization
        """
        self.model = model.to(device)
        self.device = device
        
        # Freeze Agent C
        for param in self.model.agent_c.parameters():
            param.requires_grad = False
        
        # Only optimize F and G
        fg_params = (
            list(self.model.F.parameters()) + 
            list(self.model.G.parameters())
        )
        self.optimizer = optim.Adam(fg_params, lr=lr_fg)
        
        # Loss weights
        self.lambda_aff = lambda_aff
        self.lambda_recon = lambda_recon
        self.lambda_coherence = lambda_coherence
        
        # Loss functions
        self.aff_criterion = nn.MSELoss()
        
        print("Phase1BasicTrainer initialized:")
        print(f"  F/G parameters: {sum(p.numel() for p in fg_params):,}")
        print(f"  Agent C frozen: True")
        print(f"  Loss weights: λ_aff={lambda_aff}, λ_recon={lambda_recon}, λ_coh={lambda_coherence}")
    
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
        total_coherence = 0.0
        total_eta = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Unpack batch
            points = batch_data['points'].to(self.device)  # (B, N, 3)
            affordances_gt = batch_data['affordances'].to(self.device)  # (B, N, num_affordances)
            
            # Get batch dimensions
            B, N, _ = points.shape
            
            # Flatten to graph format
            pos = points.reshape(B * N, 3)
            batch = torch.repeat_interleave(torch.arange(B, device=self.device), N)
            affordances_gt_flat = affordances_gt.reshape(B * N, -1)
            
            # Initialize agent state (frozen, but needed for forward pass)
            agent_state = self.model.agent_c.initial_state(B, self.device)
            
            # Initialize coherence signal
            coherence_signal_prev = torch.zeros(B, 1, device=self.device)
            
            # Initialize coherence spatial (for first step)
            coherence_spatial_prev = torch.zeros(B * N, device=self.device)
            
            # Forward pass
            results = self.model(pos, batch, agent_state, coherence_signal_prev, coherence_spatial_prev)
            
            # Extract results
            affordances_pred = results['affordances']  # (B*N, num_affordances)
            coherence_signal = results['coherence_signal']  # (B, 1) - this is η (reconstruction error)
            
            # Compute losses
            # 1. Reconstruction loss (minimize η)
            L_recon = coherence_signal.mean()
            
            # 2. Affordance prediction loss
            L_aff = self.aff_criterion(affordances_pred, affordances_gt_flat)
            
            # 3. Coherence regularization (prevent collapse to zero)
            # This ensures η doesn't become exactly zero (numerical stability)
            L_coherence = -torch.log(coherence_signal + 1e-8).mean()
            
            # Total loss
            loss = (
                self.lambda_recon * L_recon +
                self.lambda_aff * L_aff +
                self.lambda_coherence * L_coherence
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.F.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.G.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon += L_recon.item()
            total_aff += L_aff.item()
            total_coherence += L_coherence.item()
            total_eta += coherence_signal.mean().item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"Recon={L_recon.item():.4f}, "
                      f"Aff={L_aff.item():.4f}, "
                      f"η={coherence_signal.mean().item():.4f}")
        
        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'recon': total_recon / num_batches,
            'aff': total_aff / num_batches,
            'coherence': total_coherence / num_batches,
            'eta': total_eta / num_batches,
        }
        
        return metrics
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        
        total_recon = 0.0
        total_aff = 0.0
        total_eta = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Unpack batch
                points = batch_data['points'].to(self.device)
                affordances_gt = batch_data['affordances'].to(self.device)
                
                # Get batch dimensions
                B, N, _ = points.shape
                
                # Flatten to graph format
                pos = points.reshape(B * N, 3)
                batch = torch.repeat_interleave(torch.arange(B, device=self.device), N)
                affordances_gt_flat = affordances_gt.reshape(B * N, -1)
                
                # Initialize agent state
                agent_state = self.model.agent_c.initial_state(B, self.device)
                
                # Initialize coherence signal
                coherence_signal_prev = torch.zeros(B, 1, device=self.device)
                
                # Initialize coherence spatial
                coherence_spatial_prev = torch.zeros(B * N, device=self.device)
                
                # Forward pass
                results = self.model(pos, batch, agent_state, coherence_signal_prev, coherence_spatial_prev)
                
                # Extract results
                affordances_pred = results['affordances']
                coherence_signal = results['coherence_signal']
                
                # Compute metrics
                L_recon = coherence_signal.mean()
                L_aff = self.aff_criterion(affordances_pred, affordances_gt_flat)
                
                total_recon += L_recon.item()
                total_aff += L_aff.item()
                total_eta += coherence_signal.mean().item()
                num_batches += 1
        
        metrics = {
            'recon': total_recon / num_batches,
            'aff': total_aff / num_batches,
            'eta': total_eta / num_batches,
        }
        
        return metrics
