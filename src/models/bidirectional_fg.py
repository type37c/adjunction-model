"""
Bidirectional F/G Model: η + ε

This module implements the complete adjunction structure with both unit (η) and counit (ε).

η: Shape → F → G → Shape' (shape → action → shape)
ε: Action → F_inv → G_inv → Action' (action → shape → action)

This is the first step towards implementing the full suspension structure from the initial experiment note.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
from src.models.functor_f import FunctorF
from src.models.functor_g import FunctorG


class BidirectionalFG(nn.Module):
    """
    Bidirectional F/G Model with η and ε.
    
    Components:
    - F: Shape → Affordance (existing)
    - G: Affordance → Shape (existing)
    - F_inv: Action → Affordance (new)
    - G_inv: Affordance → Action (new)
    
    Coherence signals:
    - η: ||Shape - G(F(Shape))||² (reconstruction error for shapes)
    - ε: ||Action - G_inv(F_inv(Action))||² (reconstruction error for actions)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        affordance_dim: int = 16,
        action_dim: int = 3,
        num_points: int = 512
    ):
        """
        Args:
            input_dim: Input dimension for point cloud (3 for xyz)
            hidden_dim: Hidden dimension for F/G
            affordance_dim: Affordance dimension
            action_dim: Action dimension (3 for discrete actions: Push, Pull, Rotate)
            num_points: Number of points in point cloud
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.affordance_dim = affordance_dim
        self.action_dim = action_dim
        self.num_points = num_points
        
        # Existing: Shape → Affordance → Shape (η)
        self.F = FunctorF(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            affordance_dim=affordance_dim
        )
        
        self.G = FunctorG(
            affordance_dim=affordance_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim
        )
        
        # New: Action → Affordance → Action (ε)
        # F_inv: Action → Affordance
        self.F_inv = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, affordance_dim)
        )
        
        # G_inv: Affordance → Action
        self.G_inv = nn.Sequential(
            nn.Linear(affordance_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def compute_eta(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute η: Shape → F → G → Shape'
        
        η measures how well the shape can be reconstructed through the affordance space.
        Low η means the shape is "graspable" or "understandable" by the current F/G.
        
        Args:
            pos: Point positions (N, 3) or (B, N, 3)
            batch: Batch indices (N,) or None
        
        Returns:
            eta: Reconstruction error per batch (B,) or scalar
            affordance: Affordance representation (N, affordance_dim) or (B, affordance_dim)
            reconstructed_pos: Reconstructed positions (N, 3) or (B, N, 3)
        """
        # Handle both batched and unbatched inputs
        if len(pos.shape) == 2:  # (N, 3)
            # F: Shape → Affordance
            affordance = self.F(pos)  # (N, affordance_dim)
            
            # G: Affordance → Shape
            reconstructed_pos = self.G(affordance)  # (N, 3)
            
            # Compute reconstruction error
            if batch is not None:
                # Per-batch error
                point_errors = torch.sum((pos - reconstructed_pos) ** 2, dim=-1)  # (N,)
                batch_size = batch.max().item() + 1
                eta = torch.zeros(batch_size, device=pos.device)
                
                for b in range(batch_size):
                    mask = (batch == b)
                    if mask.sum() > 0:
                        eta[b] = point_errors[mask].mean()
            else:
                # Global error
                eta = torch.mean((pos - reconstructed_pos) ** 2)
        
        else:  # (B, N, 3)
            B, N, _ = pos.shape
            
            # Flatten for processing
            pos_flat = pos.view(-1, 3)  # (B*N, 3)
            
            # F: Shape → Affordance
            affordance_flat = self.F(pos_flat)  # (B*N, affordance_dim)
            
            # G: Affordance → Shape
            reconstructed_pos_flat = self.G(affordance_flat)  # (B*N, 3)
            
            # Reshape back
            affordance = affordance_flat.view(B, N, self.affordance_dim)
            reconstructed_pos = reconstructed_pos_flat.view(B, N, 3)
            
            # Compute per-batch error
            eta = torch.mean((pos - reconstructed_pos) ** 2, dim=(1, 2))  # (B,)
        
        return eta, affordance, reconstructed_pos
    
    def compute_epsilon(
        self,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ε: Action → F_inv → G_inv → Action'
        
        ε measures how well the action can be reconstructed through the affordance space.
        Low ε means the action is "meaningful" or "corresponds to some shape" in the current F/G.
        
        Args:
            action: Action vector (B, action_dim) or (action_dim,)
        
        Returns:
            epsilon: Reconstruction error (B,) or scalar
            affordance: Affordance representation (B, affordance_dim) or (affordance_dim,)
            reconstructed_action: Reconstructed action (B, action_dim) or (action_dim,)
        """
        # F_inv: Action → Affordance
        affordance = self.F_inv(action)
        
        # G_inv: Affordance → Action
        reconstructed_action = self.G_inv(affordance)
        
        # Compute reconstruction error
        if len(action.shape) == 1:  # (action_dim,)
            epsilon = torch.mean((action - reconstructed_action) ** 2)
        else:  # (B, action_dim)
            epsilon = torch.mean((action - reconstructed_action) ** 2, dim=1)  # (B,)
        
        return epsilon, affordance, reconstructed_action
    
    def compute_coherence(
        self,
        pos: torch.Tensor,
        action: torch.Tensor,
        batch: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total coherence: η + ε
        
        This is the overall coherence signal that measures how well the shape-action pair
        is integrated in the current F/G structure.
        
        Args:
            pos: Point positions (N, 3) or (B, N, 3)
            action: Action vector (B, action_dim) or (action_dim,)
            batch: Batch indices (N,) or None
        
        Returns:
            coherence: Total coherence signal (B,) or scalar
            eta: Shape reconstruction error (B,) or scalar
            epsilon: Action reconstruction error (B,) or scalar
        """
        eta, _, _ = self.compute_eta(pos, batch)
        epsilon, _, _ = self.compute_epsilon(action)
        
        coherence = eta + epsilon
        
        return coherence, eta, epsilon
    
    def forward(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor = None,
        action: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the bidirectional F/G model.
        
        Args:
            pos: Point positions (N, 3) or (B, N, 3)
            batch: Batch indices (N,) or None
            action: Action vector (B, action_dim) or (action_dim,) or None
        
        Returns:
            results: Dictionary containing:
                - eta: Shape reconstruction error
                - affordance_from_shape: Affordance from shape
                - reconstructed_pos: Reconstructed positions
                - epsilon: Action reconstruction error (if action is provided)
                - affordance_from_action: Affordance from action (if action is provided)
                - reconstructed_action: Reconstructed action (if action is provided)
                - coherence: Total coherence (if action is provided)
        """
        # Compute η
        eta, affordance_from_shape, reconstructed_pos = self.compute_eta(pos, batch)
        
        results = {
            'eta': eta,
            'affordance_from_shape': affordance_from_shape,
            'reconstructed_pos': reconstructed_pos
        }
        
        # Compute ε if action is provided
        if action is not None:
            epsilon, affordance_from_action, reconstructed_action = self.compute_epsilon(action)
            
            results['epsilon'] = epsilon
            results['affordance_from_action'] = affordance_from_action
            results['reconstructed_action'] = reconstructed_action
            results['coherence'] = eta + epsilon
        
        return results
    
    def get_affordance_from_shape(self, pos: torch.Tensor) -> torch.Tensor:
        """Get affordance representation from shape."""
        return self.F(pos)
    
    def get_affordance_from_action(self, action: torch.Tensor) -> torch.Tensor:
        """Get affordance representation from action."""
        return self.F_inv(action)
    
    def reconstruct_shape_from_affordance(self, affordance: torch.Tensor) -> torch.Tensor:
        """Reconstruct shape from affordance."""
        return self.G(affordance)
    
    def reconstruct_action_from_affordance(self, affordance: torch.Tensor) -> torch.Tensor:
        """Reconstruct action from affordance."""
        return self.G_inv(affordance)
