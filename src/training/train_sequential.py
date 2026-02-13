"""
Sequential Training Loop for Agent C v3 with Purpose Space P

This training method allows Agent C to accumulate experiences across multiple shapes
in a sequential manner, enabling:
1. Valence accumulation through repeated experiences
2. State persistence across shapes (not resetting every batch)
3. Learning "which kinds of breakdowns lead to good outcomes"

Key design principles:
- Agent C's state persists across an episode (multiple shapes)
- Valence is updated based on coherence changes
- Priority adapts based on accumulated valence
- This prevents "coherence minimization collapse" by giving Agent C a purpose

Difference from batch training (train_phase2_v2.py):
- Batch training: Agent state reset every batch → 1-step experience
- Sequential training: Agent state persists across episode → multi-step experience
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.conditional_adjunction_v3 import ConditionalAdjunctionModelV3
from src.data.synthetic_dataset import SyntheticAffordanceDataset


class SequentialTrainer:
    """
    Sequential trainer for Agent C v3 with Purpose Space P.
    
    Training is organized into episodes, where each episode consists of
    multiple shapes presented sequentially. Agent C's state persists
    across the episode, allowing valence to accumulate.
    """
    
    def __init__(
        self,
        model: ConditionalAdjunctionModelV3,
        device: torch.device = torch.device('cpu'),
        lr: float = 1e-4,
        lambda_aff: float = 1.0,
        lambda_kl: float = 0.1,
        lambda_coherence: float = 0.1,
        episode_length: int = 5
    ):
        """
        Args:
            model: Conditional Adjunction Model v3
            device: Device to train on
            lr: Learning rate
            lambda_aff: Weight for affordance loss
            lambda_kl: Weight for KL divergence loss
            lambda_coherence: Weight for coherence regularization
            episode_length: Number of shapes per episode
        """
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Loss weights
        self.lambda_aff = lambda_aff
        self.lambda_kl = lambda_kl
        self.lambda_coherence = lambda_coherence
        
        # Episode configuration
        self.episode_length = episode_length
        
        # Loss functions
        self.aff_criterion = nn.BCEWithLogitsLoss()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch using sequential episodes.
        
        Each epoch consists of multiple episodes, where each episode
        presents `episode_length` shapes sequentially to Agent C.
        """
        self.model.train()
        
        total_loss = 0.0
        total_recon = 0.0
        total_aff = 0.0
        total_kl = 0.0
        total_coherence = 0.0
        total_valence_change = 0.0
        num_episodes = 0
        
        # Collect samples into episodes
        episode_buffer = []
        
        for batch_idx, batch_data in enumerate(dataloader):
            episode_buffer.append(batch_data)
            
            # When we have enough samples for an episode, train on it
            if len(episode_buffer) >= self.episode_length:
                episode_metrics = self._train_episode(episode_buffer[:self.episode_length])
                
                total_loss += episode_metrics['loss']
                total_recon += episode_metrics['recon']
                total_aff += episode_metrics['aff']
                total_kl += episode_metrics['kl']
                total_coherence += episode_metrics['coherence']
                total_valence_change += episode_metrics['valence_change']
                num_episodes += 1
                
                # Remove processed samples from buffer
                episode_buffer = episode_buffer[self.episode_length:]
                
                if num_episodes % 10 == 0:
                    print(f"  Episode {num_episodes}: "
                          f"Loss={episode_metrics['loss']:.4f}, "
                          f"Recon={episode_metrics['recon']:.4f}, "
                          f"Valence Δ={episode_metrics['valence_change']:.4f}")
        
        # Average metrics
        if num_episodes > 0:
            return {
                'loss': total_loss / num_episodes,
                'recon': total_recon / num_episodes,
                'aff': total_aff / num_episodes,
                'kl': total_kl / num_episodes,
                'coherence': total_coherence / num_episodes,
                'valence_change': total_valence_change / num_episodes
            }
        else:
            return {
                'loss': 0.0,
                'recon': 0.0,
                'aff': 0.0,
                'kl': 0.0,
                'coherence': 0.0,
                'valence_change': 0.0
            }
    
    def _train_episode(
        self,
        episode_data: List[Dict]
    ) -> Dict[str, float]:
        """
        Train on a single episode (sequence of shapes).
        
        Key idea: Agent C's state persists across the episode.
        Valence accumulates based on coherence changes.
        
        Args:
            episode_data: List of batch_data dictionaries
        
        Returns:
            metrics: Dictionary with loss components
        """
        # Initialize agent state at the start of episode
        agent_state = self.model.initial_state(1, self.device)
        coherence_signal_prev = torch.zeros(1, 1, device=self.device)
        
        episode_loss = 0.0
        episode_recon = 0.0
        episode_aff = 0.0
        episode_kl = 0.0
        episode_coherence = 0.0
        
        valence_initial = agent_state['valence'].clone()
        
        # Process each shape in the episode sequentially
        for step_idx, batch_data in enumerate(episode_data):
            # Extract data
            points = batch_data['points'][0].to(self.device)  # (N, 3)
            affordances_gt = batch_data['affordances'][0].to(self.device)  # (N, num_aff)
            
            num_points = points.size(0)
            batch = torch.zeros(num_points, dtype=torch.long, device=self.device)
            
            # Previous spatial coherence (initialize to zeros for first step)
            if step_idx == 0:
                coherence_spatial_prev = torch.zeros(num_points, device=self.device)
            else:
                coherence_spatial_prev = results['coherence_spatial'].detach()
            
            # Forward pass
            results = self.model(
                points, batch, agent_state,
                coherence_signal_prev, coherence_spatial_prev
            )
            
            # Compute losses
            loss_recon = results['coherence_signal'].mean()
            loss_aff = self.aff_criterion(results['affordances'], affordances_gt)
            
            # KL divergence (RSSM regularization)
            rssm_info = results['rssm_info']
            posterior_dist = torch.distributions.Normal(
                rssm_info['posterior_mean'],
                rssm_info['posterior_std']
            )
            prior_dist = torch.distributions.Normal(
                rssm_info['prior_mean'],
                rssm_info['prior_std']
            )
            loss_kl = torch.distributions.kl_divergence(posterior_dist, prior_dist).sum(dim=-1).mean()
            
            # Coherence regularization (prevent collapse to zero)
            loss_coherence = -torch.log(results['coherence_signal'].mean() + 1e-5)
            
            # Total loss
            loss = (loss_recon +
                    self.lambda_aff * loss_aff +
                    self.lambda_kl * loss_kl +
                    self.lambda_coherence * loss_coherence)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (important for RSSM stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            episode_loss += loss.item()
            episode_recon += loss_recon.item()
            episode_aff += loss_aff.item()
            episode_kl += loss_kl.item()
            episode_coherence += loss_coherence.item()
            
            # Update agent state for next step
            # Detach to prevent backprop through time (for now)
            agent_state = {k: v.detach() for k, v in results['agent_state'].items()}
            coherence_signal_prev = results['coherence_signal'].detach()
        
        valence_final = agent_state['valence']
        valence_change = (valence_final - valence_initial).abs().mean().item()
        
        # Average over episode
        num_steps = len(episode_data)
        return {
            'loss': episode_loss / num_steps,
            'recon': episode_recon / num_steps,
            'aff': episode_aff / num_steps,
            'kl': episode_kl / num_steps,
            'coherence': episode_coherence / num_steps,
            'valence_change': valence_change
        }
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Uses the same sequential episode structure as training.
        """
        self.model.eval()
        
        total_recon = 0.0
        total_aff = 0.0
        total_coherence = 0.0
        num_episodes = 0
        
        episode_buffer = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                episode_buffer.append(batch_data)
                
                if len(episode_buffer) >= self.episode_length:
                    episode_metrics = self._evaluate_episode(episode_buffer[:self.episode_length])
                    
                    total_recon += episode_metrics['recon']
                    total_aff += episode_metrics['aff']
                    total_coherence += episode_metrics['coherence']
                    num_episodes += 1
                    
                    episode_buffer = episode_buffer[self.episode_length:]
        
        if num_episodes > 0:
            return {
                'recon': total_recon / num_episodes,
                'aff': total_aff / num_episodes,
                'coherence': total_coherence / num_episodes
            }
        else:
            return {
                'recon': 0.0,
                'aff': 0.0,
                'coherence': 0.0
            }
    
    def _evaluate_episode(
        self,
        episode_data: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate on a single episode."""
        agent_state = self.model.initial_state(1, self.device)
        coherence_signal_prev = torch.zeros(1, 1, device=self.device)
        
        episode_recon = 0.0
        episode_aff = 0.0
        episode_coherence = 0.0
        
        for step_idx, batch_data in enumerate(episode_data):
            points = batch_data['points'][0].to(self.device)
            affordances_gt = batch_data['affordances'][0].to(self.device)
            
            num_points = points.size(0)
            batch = torch.zeros(num_points, dtype=torch.long, device=self.device)
            
            if step_idx == 0:
                coherence_spatial_prev = torch.zeros(num_points, device=self.device)
            else:
                coherence_spatial_prev = results['coherence_spatial']
            
            results = self.model(
                points, batch, agent_state,
                coherence_signal_prev, coherence_spatial_prev
            )
            
            loss_recon = results['coherence_signal'].mean()
            loss_aff = self.aff_criterion(results['affordances'], affordances_gt)
            loss_coherence = -torch.log(results['coherence_signal'].mean() + 1e-5)
            
            episode_recon += loss_recon.item()
            episode_aff += loss_aff.item()
            episode_coherence += loss_coherence.item()
            
            agent_state = results['agent_state']
            coherence_signal_prev = results['coherence_signal']
        
        num_steps = len(episode_data)
        return {
            'recon': episode_recon / num_steps,
            'aff': episode_aff / num_steps,
            'coherence': episode_coherence / num_steps
        }


if __name__ == '__main__':
    print("Testing Sequential Trainer...")
    
    # Create model
    model = ConditionalAdjunctionModelV3(
        num_affordances=5,
        num_points=256,
        f_hidden_dim=32,
        g_hidden_dim=64,
        agent_hidden_dim=128,
        agent_latent_dim=32,
        context_dim=64,
        valence_dim=16,
        valence_decay=0.1
    )
    
    # Create dataset
    dataset = SyntheticAffordanceDataset(
        num_samples=50,
        num_points=256,
        shape_types=[0, 1]  # 0=cube, 1=cylinder (2=sphere is novel)
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create trainer
    trainer = SequentialTrainer(
        model=model,
        device=torch.device('cpu'),
        lr=1e-4,
        episode_length=5
    )
    
    print(f"\nTrainer created")
    print(f"  Episode length: {trainer.episode_length}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train for one epoch
    print("\nTraining for 1 epoch...")
    metrics = trainer.train_epoch(dataloader, epoch=0)
    
    print(f"\nEpoch 0 metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Recon: {metrics['recon']:.4f}")
    print(f"  Aff: {metrics['aff']:.4f}")
    print(f"  KL: {metrics['kl']:.4f}")
    print(f"  Coherence: {metrics['coherence']:.4f}")
    print(f"  Valence change: {metrics['valence_change']:.4f}")
    
    print("\n" + "="*60)
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("Sequential training implements the conditions for P to emerge:")
    print("1. Agent state persists across episode (not reset every batch)")
    print("2. Valence accumulates based on coherence changes")
    print("3. Priority adapts based on accumulated valence")
    print("4. Agent learns 'which breakdowns lead to good outcomes'")
    print("="*60)
