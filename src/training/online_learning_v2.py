"""
Online Learning v2 for Conditional Adjunction Model V2

This module implements online learning for the v2 model with spatial coherence signals.

Key differences from v1:
- Receives and passes spatial coherence signals to the model
- Tracks priority and uncertainty metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.conditional_adjunction_v2 import ConditionalAdjunctionModelV2


class OnlineLearnerV2:
    """
    Online learning wrapper for Conditional Adjunction Model V2.
    """
    
    def __init__(
        self,
        model: ConditionalAdjunctionModelV2,
        device: torch.device = torch.device('cpu'),
        base_lr: float = 1e-4,
        coherence_threshold: float = 0.3,
        adaptation_strength: float = 1.0,
        lambda_regularization: float = 0.01
    ):
        self.model = model.to(device)
        self.device = device
        
        self.base_lr = base_lr
        self.coherence_threshold = coherence_threshold
        self.adaptation_strength = adaptation_strength
        self.lambda_regularization = lambda_regularization
        
        self.optimizer = optim.Adam(model.parameters(), lr=base_lr)
        
        # Store initial parameters for regularization
        self.initial_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        
        # Metrics tracking
        self.update_count = 0
        self.total_exposures = 0
        self.coherence_history = []
    
    def adapt(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        agent_state: Optional[Dict[str, torch.Tensor]] = None,
        coherence_signal_prev: Optional[torch.Tensor] = None,
        coherence_spatial_prev: Optional[torch.Tensor] = None,
        affordances_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one step of online adaptation.
        
        Args:
            pos: Point cloud (N, 3) or (B*N, 3)
            batch: Batch assignment (N,), optional
            agent_state: Previous agent state, optional
            coherence_signal_prev: Previous scalar coherence signal (B, 1), optional
            coherence_spatial_prev: Previous spatial coherence signal (N,), optional
            affordances_gt: Ground truth affordances, optional
        
        Returns:
            results: Dictionary containing all model outputs + adaptation info
        """
        self.total_exposures += 1
        
        # Forward pass (v2: includes spatial coherence)
        results = self.model(
            pos, batch, agent_state,
            coherence_signal_prev, coherence_spatial_prev
        )
        
        coherence_signal = results['coherence_signal']
        self.coherence_history.append(coherence_signal.mean().item())
        
        # Decide whether to update based on coherence signal
        should_update = coherence_signal.mean().item() > self.coherence_threshold
        
        if should_update:
            # Adaptive learning rate
            adaptive_lr = self.base_lr * (1.0 + self.adaptation_strength * coherence_signal.mean().item())
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = adaptive_lr
            
            # Compute loss
            loss_coherence = coherence_signal.mean()
            
            # Regularization
            loss_reg = 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    loss_reg += torch.norm(param - self.initial_params[name]) ** 2
            
            loss_reg = self.lambda_regularization * loss_reg
            
            # Optional supervised loss
            loss_aff = 0.0
            if affordances_gt is not None:
                affordances_pred = results['affordances']
                loss_aff = nn.functional.binary_cross_entropy_with_logits(
                    affordances_pred, affordances_gt
                )
            
            loss = loss_coherence + loss_reg + loss_aff
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            self.update_count += 1
            results['updated'] = True
            results['lr'] = adaptive_lr
        else:
            results['updated'] = False
            results['lr'] = 0.0
        
        # Detach agent state and coherence signals
        if 'agent_state' in results:
            results['agent_state'] = {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in results['agent_state'].items()
            }
        
        if 'coherence_signal' in results:
            results['coherence_signal'] = results['coherence_signal'].detach()
        
        if 'coherence_spatial' in results:
            results['coherence_spatial'] = results['coherence_spatial'].detach()
        
        return results
    
    def get_metrics(self) -> Dict[str, float]:
        """Get online learning metrics."""
        return {
            'total_exposures': self.total_exposures,
            'update_count': self.update_count,
            'update_rate': self.update_count / max(self.total_exposures, 1),
            'avg_coherence': sum(self.coherence_history) / max(len(self.coherence_history), 1),
            'recent_coherence': self.coherence_history[-1] if self.coherence_history else 0.0
        }
