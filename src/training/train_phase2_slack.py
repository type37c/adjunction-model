"""
Phase 2 Training (Slack Preservation): Fine-tuning with Agent Layer C

This is a modified version of Phase 2 training that preserves η/ε slack.

Key differences from standard Phase 2:
- Standard: Minimizes reconstruction loss (L_recon) → η → 0
- Slack: Removes reconstruction loss → η preserved as "slack"

Theoretical motivation:
- Unit η: Shape → G(F(Shape)) represents "what cannot be captured by affordances"
- Counit ε: Affordance → F(G(Affordance)) represents "action ambiguity"
- η and ε are not "errors" to minimize, but "slack" to preserve
- Preserving slack allows suspension structure to emerge

Loss components:
1. L_aff: Affordance prediction loss (only this is minimized)
2. L_kl: KL divergence between posterior and prior (RSSM regularization)
3. L_coherence: Coherence regularization (prevent collapse)

REMOVED: L_recon (reconstruction loss)

Expected outcomes:
- η remains non-zero (slack is preserved)
- ε can be computed and observed
- Agent C learns to navigate the slack
- Suspension structure may emerge
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.conditional_adjunction_v4 import ConditionalAdjunctionModelV4
from src.data.synthetic_dataset import SyntheticAffordanceDataset


class Phase2SlackTrainer:
    """
    Trainer for Phase 2 (Slack Preservation): Fine-tuning with Agent Layer C.
    """
    
    def __init__(
        self,
        model: ConditionalAdjunctionModelV4,
        device: torch.device = torch.device('cpu'),
        lr: float = 1e-4,
        lambda_aff: float = 1.0,
        lambda_kl: float = 0.1,
        lambda_coherence: float = 0.1,
        lambda_counit: float = 0.0  # Optional: regularize counit
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Loss weights
        self.lambda_aff = lambda_aff
        self.lambda_kl = lambda_kl
        self.lambda_coherence = lambda_coherence
        self.lambda_counit = lambda_counit
        
        # Loss functions
        self.aff_criterion = nn.MSELoss()  # Changed from BCEWithLogitsLoss for continuous affordances
    
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
        total_aff = 0.0
        total_kl = 0.0
        total_coherence = 0.0
        total_counit = 0.0
        total_unit = 0.0  # Track η (unit signal)
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Unpack batch
            # SyntheticAffordanceDataset returns: {'points', 'affordances', 'shape_type'}
            pos = batch_data['points'].to(self.device)
            batch = batch_data['batch'].to(self.device) if 'batch' in batch_data else None
            affordances_gt = batch_data['affordances'].to(self.device)
            
            # Get batch size
            batch_size = pos.shape[0] if batch is None else batch.max().item() + 1
            
            # Initialize agent state
            agent_state = self.model.agent_c.initial_state(batch_size, self.device)
            
            # Initialize coherence signal (for first step)
            coherence_signal_prev = torch.zeros(batch_size, 1, device=self.device)
            
            # Forward pass
            results = self.model(pos, batch, agent_state, coherence_signal_prev)
            
            # Extract results
            affordances_pred = results['affordances']
            reconstructed = results['reconstructed']
            coherence_signal = results['coherence_signal']  # Unit η
            counit_signal = results['counit_signal']        # Counit ε
            rssm_info = results['rssm_info']
            
            # Compute losses
            # 1. Affordance loss (ONLY THIS IS MINIMIZED)
            # affordances_pred is (N, num_affordances) in graph format
            # affordances_gt is (B, num_points, num_affordances) in batch format
            # Convert affordances_gt to graph format
            B_gt, N_gt, A_gt = affordances_gt.shape
            affordances_gt_flat = affordances_gt.reshape(B_gt * N_gt, A_gt)  # (B*N, num_affordances)
            L_aff = self.aff_criterion(affordances_pred, affordances_gt_flat)
            
            # 2. KL divergence (RSSM regularization)
            L_kl = self.model.agent_c.rssm.kl_divergence(
                rssm_info['posterior_mean'],
                rssm_info['posterior_std'],
                rssm_info['prior_mean'],
                rssm_info['prior_std']
            ).mean()
            
            # 3. Coherence regularization (prevent collapse to zero)
            # NOTE: This prevents η → 0, but does not minimize η
            L_coherence = -torch.log(coherence_signal + 1e-8).mean()
            
            # 4. Optional: Counit regularization
            if self.lambda_counit > 0:
                L_counit = -torch.log(counit_signal + 1e-8).mean()
            else:
                L_counit = torch.tensor(0.0, device=self.device)
            
            # Total loss (NO RECONSTRUCTION LOSS)
            loss = (
                self.lambda_aff * L_aff + 
                self.lambda_kl * L_kl + 
                self.lambda_coherence * L_coherence +
                self.lambda_counit * L_counit
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_aff += L_aff.item()
            total_kl += L_kl.item()
            total_coherence += L_coherence.item()
            total_counit += L_counit.item()
            total_unit += coherence_signal.mean().item()  # Track η
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"Aff={L_aff.item():.4f}, "
                      f"KL={L_kl.item():.4f}, "
                      f"Coh={L_coherence.item():.4f}, "
                      f"η={coherence_signal.mean().item():.4f}, "
                      f"ε={counit_signal.mean().item():.4f}")
        
        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'aff': total_aff / num_batches,
            'kl': total_kl / num_batches,
            'coherence': total_coherence / num_batches,
            'counit': total_counit / num_batches,
            'unit_eta': total_unit / num_batches,  # Average η
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
        
        total_aff = 0.0
        total_unit = 0.0
        total_counit = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Unpack batch
                # SyntheticAffordanceDataset returns: {'points', 'affordances', 'shape_type'}
                pos = batch_data['points'].to(self.device)
                batch = batch_data['batch'].to(self.device) if 'batch' in batch_data else None
                affordances_gt = batch_data['affordances'].to(self.device)
                
                # Get batch size
                batch_size = pos.shape[0] if batch is None else batch.max().item() + 1
                
                # Initialize agent state
                agent_state = self.model.agent_c.initial_state(batch_size, self.device)
                
                # Initialize coherence signal
                coherence_signal_prev = torch.zeros(batch_size, 1, device=self.device)
                
                # Forward pass
                results = self.model(pos, batch, agent_state, coherence_signal_prev)
                
                # Extract results
                affordances_pred = results['affordances']
                coherence_signal = results['coherence_signal']
                counit_signal = results['counit_signal']
                
                # Compute metrics
                # affordances_pred is (N, num_affordances) in graph format
                # affordances_gt is (B, num_points, num_affordances) in batch format
                # Convert affordances_gt to graph format
                B_gt, N_gt, A_gt = affordances_gt.shape
                affordances_gt_flat = affordances_gt.reshape(B_gt * N_gt, A_gt)  # (B*N, num_affordances)
                L_aff = self.aff_criterion(affordances_pred, affordances_gt_flat)
                
                total_aff += L_aff.item()
                total_unit += coherence_signal.mean().item()
                total_counit += counit_signal.mean().item()
                num_batches += 1
        
        metrics = {
            'aff': total_aff / num_batches,
            'unit_eta': total_unit / num_batches,
            'counit_eps': total_counit / num_batches,
        }
        
        return metrics
