"""
Functor F: Shape → Affordance

F maps 3D point clouds (shapes) to affordance distributions.
Uses PointNet++ architecture for point cloud processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FunctorF(nn.Module):
    """
    Functor F: Shape → Affordance
    
    Maps 3D point clouds to affordance distributions using a simplified
    PointNet-style architecture.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        affordance_dim: int = 16,
        num_layers: int = 3
    ):
        """
        Args:
            input_dim: Input dimension (3 for xyz coordinates)
            hidden_dim: Hidden layer dimension
            affordance_dim: Output affordance dimension
            num_layers: Number of MLP layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.affordance_dim = affordance_dim
        
        # Point-wise feature extraction
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else affordance_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if i < num_layers - 1 else nn.Identity(),
                nn.ReLU() if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        
        self.point_encoder = nn.Sequential(*layers)
    
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pos: Point cloud positions (N, 3)
        
        Returns:
            affordances: Affordance features per point (N, affordance_dim)
        """
        # Point-wise encoding
        affordances = self.point_encoder(pos)  # (N, affordance_dim)
        
        return affordances
