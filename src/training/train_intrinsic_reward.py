"""
Intrinsic Reward-Driven Training

This trainer implements pure intrinsic motivation without external objectives (Affordance Loss).
It reproduces the 2/13 experiment where:
- F/G are frozen (fixed weights)
- Agent C is driven only by intrinsic rewards (Competence + Novelty)
- No Affordance Loss is used
- Value function is updated based on intrinsic rewards

Key differences from Phase2SlackTrainer:
- Phase2SlackTrainer: Affordance Loss drives training
- IntrinsicRewardTrainer: Intrinsic rewards drive training

Theoretical motivation:
- This tests the core hypothesis: Can structure emerge from intrinsic motivation alone?
- External objectives (like Affordance Loss) are extensions of existing AI paradigms
- The unique claim is that intrinsic rewards alone can drive structure formation

Loss components:
1. L_kl: KL divergence (RSSM regularization only)
2. L_value: Value function prediction error (driven by intrinsic rewards)

REMOVED: L_aff (Affordance Loss), L_recon (Reconstruction Loss)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.adjunction_model import AdjunctionModel


class IntrinsicRewardTrainer:
    """
    Trainer for intrinsic reward-driven experiments.
    
    This trainer:
    1. Freezes F/G (no learning)
    2. Trains only Agent C
    3. Uses intrinsic rewards (Competence + Novelty) as the driving force
    4. Does NOT use Affordance Loss
    """
    
    def __init__(
        self,
        model: AdjunctionModel,
        device: torch.device = torch.device('cpu'),
        lr: float = 1e-4,
        lambda_kl: float = 0.1,
        lambda_value: float = 1.0,
        freeze_fg: bool = True
    ):
        """
        Args:
            model: AdjunctionModel instance
            device: Device to train on
            lr: Learning rate
            lambda_kl: Weight for KL divergence loss
            lambda_value: Weight for value function loss
            freeze_fg: Whether to freeze F and G (default: True)
        """
        self.model = model.to(device)
        self.device = device
        
        # Freeze F and G if requested
        if freeze_fg:
            self._freeze_fg()
        
        # Optimizer only for Agent C parameters
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
        
        # Loss weights
        self.lambda_kl = lambda_kl
        self.lambda_value = lambda_value
        
        # Loss functions
        self.value_criterion = nn.MSELoss()
    
    def _freeze_fg(self):
        """Freeze F and G parameters."""
        # Freeze F (functor_f)
        for param in self.model.functor_f.parameters():
            param.requires_grad = False
        
        # Freeze G (functor_g)
        for param in self.model.functor_g.parameters():
            param.requires_grad = False
        
        print("F and G frozen. Only Agent C will be trained.")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch using intrinsic rewards only.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_kl = 0.0
        total_value = 0.0
        total_intrinsic_reward = 0.0
        total_competence = 0.0
        total_novelty = 0.0
        total_valence = 0.0
        total_coherence = 0.0
        total_unit = 0.0
        total_counit = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Unpack batch
            pos = batch_data['points'].to(self.device)
            batch = batch_data['batch'].to(self.device) if 'batch' in batch_data else None
            
            # Forward pass
            output = self.model(pos, batch=batch)
            
            # Extract outputs
            affordances_pred = output['affordances']
            rssm_info = output.get('rssm_info', {})
            
            # KL divergence (RSSM regularization)
            kl_div = rssm_info.get('kl_divergence', torch.tensor(0.0, device=self.device))
            if kl_div.dim() == 0:
                kl_div = kl_div.unsqueeze(0)
            L_kl = kl_div.mean()
            
            # Intrinsic rewards (computed by Agent C)
            intrinsic_rewards = rssm_info.get('intrinsic_rewards', {})
            R_intrinsic = intrinsic_rewards.get('R_intrinsic', torch.tensor(0.0, device=self.device))
            R_competence = intrinsic_rewards.get('R_competence', torch.tensor(0.0, device=self.device))
            R_novelty = intrinsic_rewards.get('R_novelty', torch.tensor(0.0, device=self.device))
            
            # Value function (predicted future intrinsic rewards)
            value_pred = rssm_info.get('value', torch.tensor(0.0, device=self.device))
            
            # Value function target: current intrinsic reward
            # In a full RL setup, this would be a discounted sum of future rewards
            # For now, we use the current reward as a simple target
            value_target = R_intrinsic.detach()
            
            # Value function loss
            if value_pred.dim() == 0:
                value_pred = value_pred.unsqueeze(0)
            if value_target.dim() == 0:
                value_target = value_target.unsqueeze(0)
            
            L_value = self.value_criterion(value_pred, value_target)
            
            # Total loss (no Affordance Loss!)
            loss = self.lambda_kl * L_kl + self.lambda_value * L_value
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_kl += L_kl.item()
            total_value += L_value.item()
            total_intrinsic_reward += R_intrinsic.mean().item()
            total_competence += R_competence.mean().item()
            total_novelty += R_novelty.mean().item()
            
            # Track slack signals
            valence_mean = rssm_info.get('valence_mean', torch.tensor(0.0, device=self.device))
            coherence_signal = output.get('coherence_signal', torch.tensor(0.0, device=self.device))
            unit_signal = output.get('unit_signal', torch.tensor(0.0, device=self.device))
            counit_signal = output.get('counit_signal', torch.tensor(0.0, device=self.device))
            
            total_valence += valence_mean.mean().item() if valence_mean.dim() > 0 else valence_mean.item()
            total_coherence += coherence_signal.mean().item() if coherence_signal.dim() > 0 else coherence_signal.item()
            total_unit += unit_signal.mean().item() if unit_signal.dim() > 0 else unit_signal.item()
            total_counit += counit_signal.mean().item() if counit_signal.dim() > 0 else counit_signal.item()
            
            num_batches += 1
        
        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'kl_loss': total_kl / num_batches,
            'value_loss': total_value / num_batches,
            'intrinsic_reward': total_intrinsic_reward / num_batches,
            'competence_reward': total_competence / num_batches,
            'novelty_reward': total_novelty / num_batches,
            'valence': total_valence / num_batches,
            'coherence': total_coherence / num_batches,
            'eta': total_unit / num_batches,
            'epsilon': total_counit / num_batches
        }
        
        return metrics
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Evaluation data loader
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_intrinsic_reward = 0.0
        total_competence = 0.0
        total_novelty = 0.0
        total_valence = 0.0
        total_coherence = 0.0
        total_unit = 0.0
        total_counit = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                pos = batch_data['points'].to(self.device)
                batch = batch_data['batch'].to(self.device) if 'batch' in batch_data else None
                
                output = self.model(pos, batch=batch)
                rssm_info = output.get('rssm_info', {})
                
                intrinsic_rewards = rssm_info.get('intrinsic_rewards', {})
                R_intrinsic = intrinsic_rewards.get('R_intrinsic', torch.tensor(0.0, device=self.device))
                R_competence = intrinsic_rewards.get('R_competence', torch.tensor(0.0, device=self.device))
                R_novelty = intrinsic_rewards.get('R_novelty', torch.tensor(0.0, device=self.device))
                
                valence_mean = rssm_info.get('valence_mean', torch.tensor(0.0, device=self.device))
                coherence_signal = output.get('coherence_signal', torch.tensor(0.0, device=self.device))
                unit_signal = output.get('unit_signal', torch.tensor(0.0, device=self.device))
                counit_signal = output.get('counit_signal', torch.tensor(0.0, device=self.device))
                
                total_intrinsic_reward += R_intrinsic.mean().item()
                total_competence += R_competence.mean().item()
                total_novelty += R_novelty.mean().item()
                total_valence += valence_mean.mean().item() if valence_mean.dim() > 0 else valence_mean.item()
                total_coherence += coherence_signal.mean().item() if coherence_signal.dim() > 0 else coherence_signal.item()
                total_unit += unit_signal.mean().item() if unit_signal.dim() > 0 else unit_signal.item()
                total_counit += counit_signal.mean().item() if counit_signal.dim() > 0 else counit_signal.item()
                
                num_batches += 1
        
        metrics = {
            'intrinsic_reward': total_intrinsic_reward / num_batches,
            'competence_reward': total_competence / num_batches,
            'novelty_reward': total_novelty / num_batches,
            'valence': total_valence / num_batches,
            'coherence': total_coherence / num_batches,
            'eta': total_unit / num_batches,
            'epsilon': total_counit / num_batches
        }
        
        return metrics


if __name__ == '__main__':
    print("Testing Intrinsic Reward Trainer...")
    
    # This is a placeholder test
    # Actual testing requires a full AdjunctionModel instance
    print("Trainer class defined successfully.")
    print("Key features:")
    print("  - F/G frozen (no learning)")
    print("  - Agent C trained with intrinsic rewards only")
    print("  - No Affordance Loss")
    print("  - Value function driven by intrinsic rewards")
