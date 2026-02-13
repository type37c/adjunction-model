"""
Online Learning for Conditional Adjunction Model

This module implements online learning (inference-time weight updates) for the
conditional adjunction model. Unlike standard training where the model is trained
offline and then deployed, online learning allows the model to adapt to novel
stimuli in real-time by updating its weights based on the coherence signal.

Key differences from Phase 2 training:
- Phase 2: Batch training with fixed dataset, eval mode for inference
- Online Learning: Single-sample updates during inference, continuous adaptation

The online learning mechanism is crucial for validating the "suspension structure"
hypothesis: the agent should be able to adapt to novel situations without catastrophic
forgetting, maintaining a balance between stability and plasticity.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.conditional_adjunction import ConditionalAdjunctionModel


class OnlineLearner:
    """
    Online learning wrapper for Conditional Adjunction Model.
    
    This class manages the online adaptation process, including:
    - Adaptive learning rate based on coherence signal
    - Selective parameter updates (only update when coherence is high)
    - Regularization to prevent catastrophic forgetting
    """
    
    def __init__(
        self,
        model: ConditionalAdjunctionModel,
        device: torch.device = torch.device('cpu'),
        base_lr: float = 1e-4,
        coherence_threshold: float = 0.3,
        adaptation_strength: float = 1.0,
        lambda_regularization: float = 0.01
    ):
        """
        Args:
            model: The conditional adjunction model
            device: Device
            base_lr: Base learning rate
            coherence_threshold: Only update when coherence > threshold
            adaptation_strength: Multiplier for learning rate based on coherence
            lambda_regularization: Weight for L2 regularization (prevent forgetting)
        """
        self.model = model.to(device)
        self.device = device
        
        self.base_lr = base_lr
        self.coherence_threshold = coherence_threshold
        self.adaptation_strength = adaptation_strength
        self.lambda_regularization = lambda_regularization
        
        # Optimizer for online updates
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
        affordances_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one step of online adaptation.
        
        Args:
            pos: Point cloud (N, 3) or (B*N, 3)
            batch: Batch assignment (N,) or (B*N,), optional
            agent_state: Previous agent state, optional
            coherence_signal_prev: Previous coherence signal, optional
            affordances_gt: Ground truth affordances (for supervised adaptation), optional
        
        Returns:
            results: Dictionary containing:
                - affordances: Predicted affordances
                - reconstructed: Reconstructed point cloud
                - coherence_signal: Current coherence signal
                - agent_state: Updated agent state
                - context: Context vector
                - updated: Whether weights were updated
                - lr: Learning rate used (0 if not updated)
        """
        self.total_exposures += 1
        
        # Forward pass
        results = self.model(pos, batch, agent_state, coherence_signal_prev)
        
        coherence_signal = results['coherence_signal']
        self.coherence_history.append(coherence_signal.mean().item())
        
        # Decide whether to update based on coherence signal
        should_update = coherence_signal.mean().item() > self.coherence_threshold
        
        if should_update:
            # Adaptive learning rate: higher coherence → higher learning rate
            adaptive_lr = self.base_lr * (1.0 + self.adaptation_strength * coherence_signal.mean().item())
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = adaptive_lr
            
            # Compute loss
            # Primary loss: coherence signal (reconstruction error)
            loss_coherence = coherence_signal.mean()
            
            # Regularization: L2 distance from initial parameters (prevent catastrophic forgetting)
            loss_reg = 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    loss_reg += torch.norm(param - self.initial_params[name]) ** 2
            
            loss_reg = self.lambda_regularization * loss_reg
            
            # Optional: supervised loss if ground truth affordances are provided
            loss_aff = 0.0
            if affordances_gt is not None:
                affordances_pred = results['affordances']
                loss_aff = nn.functional.binary_cross_entropy_with_logits(
                    affordances_pred, affordances_gt
                )
            
            # Total loss
            loss = loss_coherence + loss_reg + loss_aff
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            self.update_count += 1
            
            results['updated'] = True
            results['lr'] = adaptive_lr
        else:
            results['updated'] = False
            results['lr'] = 0.0
        
        # Detach agent state and coherence signal to prevent backprop through time accumulation
        if 'agent_state' in results:
            results['agent_state'] = {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in results['agent_state'].items()
            }
        
        # Also detach coherence_signal for next iteration
        if 'coherence_signal' in results:
            # Keep the original for return, but create a detached copy for internal use
            coherence_for_return = results['coherence_signal']
            results['coherence_signal'] = coherence_for_return.detach()
        
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
    
    def reset_metrics(self):
        """Reset metrics tracking."""
        self.update_count = 0
        self.total_exposures = 0
        self.coherence_history = []


if __name__ == '__main__':
    # Test online learner
    print("Testing Online Learner...")
    
    from src.models.conditional_adjunction import ConditionalAdjunctionModel
    from src.data.synthetic_dataset import SyntheticAffordanceDataset
    
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
    
    # Create online learner
    learner = OnlineLearner(
        model=model,
        device=torch.device('cpu'),
        base_lr=1e-4,
        coherence_threshold=0.3,
        adaptation_strength=1.0,
        lambda_regularization=0.01
    )
    
    print("Online learner created")
    
    # Generate test data
    dataset = SyntheticAffordanceDataset(num_samples=5, num_points=256)
    
    print(f"\nSimulating online adaptation with {len(dataset)} samples...")
    
    # Initialize agent state
    batch_size = 1
    agent_state = model.initial_state(batch_size, torch.device('cpu'))
    coherence_signal_prev = torch.zeros(batch_size, 1)
    
    for i, sample in enumerate(dataset):
        pos = sample['points']
        affordances_gt = sample['affordances']
        
        # Adapt
        results = learner.adapt(
            pos=pos,
            batch=None,
            agent_state=agent_state,
            coherence_signal_prev=coherence_signal_prev,
            affordances_gt=affordances_gt
        )
        
        # Update state for next iteration (detach to prevent graph accumulation)
        agent_state = results['agent_state']
        coherence_signal_prev = results['coherence_signal'].detach()
        
        print(f"  Sample {i+1}/{len(dataset)}: "
              f"Coherence={results['coherence_signal'].item():.4f}, "
              f"Updated={results['updated']}, "
              f"LR={results['lr']:.6f}")
    
    # Print metrics
    metrics = learner.get_metrics()
    print("\nOnline Learning Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nOnline Learner test passed!")
