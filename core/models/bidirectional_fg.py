"""
Bidirectional F/G: Complete Adjunction Structure (η + ε)

This module implements the complete adjunction F ⊣ G with both unit (η) and counit (ε).

Theory:
- η: Shape → F → G → Shape' (unit of adjunction)
- ε: Action → F_inv → G_inv → Action' (counit of adjunction)

Low η means "this shape is graspable/understandable"
Low ε means "this action is meaningful/corresponds to some shape"
Low η + ε means "this shape-action pair is coherent"

This is the foundation for the suspension structure.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import sys
sys.path.append('/home/ubuntu/adjunction-model')
from src.models.functor_f import FunctorF
from src.models.functor_g import FunctorG


class BidirectionalFG(nn.Module):
    """
    Bidirectional F/G with complete adjunction structure.
    
    Components:
    - F: Shape → Affordance (existing)
    - G: Affordance → Shape (existing)
    - F_inv: Action → Affordance (new)
    - G_inv: Affordance → Action (new)
    """
    
    def __init__(
        self,
        point_dim: int = 3,
        affordance_dim: int = 16,
        action_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        """
        Args:
            point_dim: Point cloud dimension (3 for xyz)
            affordance_dim: Affordance space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers in MLPs
        """
        super().__init__()
        
        self.point_dim = point_dim
        self.affordance_dim = affordance_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Existing: Shape ↔ Affordance (η)
        self.F = FunctorF(
            input_dim=point_dim,
            hidden_dim=hidden_dim,
            affordance_dim=affordance_dim,
            num_layers=num_layers
        )
        
        self.G = FunctorG(
            affordance_dim=affordance_dim,
            hidden_dim=hidden_dim,
            output_dim=point_dim,
            num_layers=num_layers
        )
        
        # New: Action ↔ Affordance (ε)
        # F_inv: Action → Affordance
        f_inv_layers = []
        in_dim = action_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else affordance_dim
            f_inv_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if i < num_layers - 1 else nn.Identity(),
                nn.ReLU() if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        self.F_inv = nn.Sequential(*f_inv_layers)
        
        # G_inv: Affordance → Action
        g_inv_layers = []
        in_dim = affordance_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else action_dim
            g_inv_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if i < num_layers - 1 else nn.Identity(),
                nn.ReLU() if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        self.G_inv = nn.Sequential(*g_inv_layers)
    
    def compute_eta(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute η: Shape → F → G → Shape'
        
        η is the unit of adjunction, measuring shape reconstruction error.
        Low η means "this shape is graspable/understandable by F/G".
        
        Args:
            pos: Point positions (N, 3) or (B, N, 3)
            batch: Batch indices (N,) or None
            return_components: Whether to return intermediate results
        
        Returns:
            eta: Reconstruction error (B,) or scalar
            (optional) affordance: Affordance representation
            (optional) reconstructed_pos: Reconstructed positions
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
        
        if return_components:
            return eta, affordance, reconstructed_pos
        return eta
    
    def compute_epsilon(
        self,
        action: torch.Tensor,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute ε: Action → F_inv → G_inv → Action'
        
        ε is the counit of adjunction, measuring action reconstruction error.
        Low ε means "this action is meaningful/corresponds to some shape".
        
        Args:
            action: Action vector (B, action_dim) or (action_dim,)
            return_components: Whether to return intermediate results
        
        Returns:
            epsilon: Reconstruction error (B,) or scalar
            (optional) affordance: Affordance representation
            (optional) reconstructed_action: Reconstructed action
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
        
        if return_components:
            return epsilon, affordance, reconstructed_action
        return epsilon
    
    def compute_coherence(
        self,
        pos: torch.Tensor,
        action: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total coherence: η + ε
        
        This is the overall coherence signal measuring how well the shape-action
        pair is integrated in the current F/G structure.
        
        Args:
            pos: Point positions (N, 3) or (B, N, 3)
            action: Action vector (B, action_dim) or (action_dim,)
            batch: Batch indices (N,) or None
        
        Returns:
            coherence: Total coherence (η + ε)
            eta: Shape reconstruction error
            epsilon: Action reconstruction error
        """
        eta = self.compute_eta(pos, batch)
        epsilon = self.compute_epsilon(action)
        
        coherence = eta + epsilon
        
        return coherence, eta, epsilon
    
    def forward(
        self,
        pos: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the bidirectional F/G model.
        
        Args:
            pos: Point positions (N, 3) or (B, N, 3)
            action: Action vector (B, action_dim) or None
            batch: Batch indices (N,) or None
        
        Returns:
            Dictionary containing:
                - eta: Shape reconstruction error
                - affordance_from_shape: Affordance from shape
                - reconstructed_pos: Reconstructed positions
                - epsilon: Action reconstruction error (if action provided)
                - affordance_from_action: Affordance from action (if action provided)
                - reconstructed_action: Reconstructed action (if action provided)
                - coherence: Total coherence (if action provided)
        """
        # Compute η
        eta, affordance_from_shape, reconstructed_pos = self.compute_eta(
            pos, batch, return_components=True
        )
        
        results = {
            'eta': eta,
            'affordance_from_shape': affordance_from_shape,
            'reconstructed_pos': reconstructed_pos
        }
        
        # Compute ε if action is provided
        if action is not None:
            epsilon, affordance_from_action, reconstructed_action = self.compute_epsilon(
                action, return_components=True
            )
            
            results['epsilon'] = epsilon
            results['affordance_from_action'] = affordance_from_action
            results['reconstructed_action'] = reconstructed_action
            results['coherence'] = eta + epsilon
        
        return results
    
    def get_affordance_from_shape(self, pos: torch.Tensor) -> torch.Tensor:
        """Get affordance representation from shape (F)."""
        return self.F(pos)
    
    def get_affordance_from_action(self, action: torch.Tensor) -> torch.Tensor:
        """Get affordance representation from action (F_inv)."""
        return self.F_inv(action)
    
    def reconstruct_shape(self, affordance: torch.Tensor) -> torch.Tensor:
        """Reconstruct shape from affordance (G)."""
        return self.G(affordance)
    
    def reconstruct_action(self, affordance: torch.Tensor) -> torch.Tensor:
        """Reconstruct action from affordance (G_inv)."""
        return self.G_inv(affordance)
