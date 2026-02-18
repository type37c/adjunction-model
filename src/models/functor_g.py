"""
Functor G: Affordance → Shape

G maps affordance distributions back to 3D point clouds (shapes).
Uses a decoder architecture similar to FoldingNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FunctorG(nn.Module):
    """
    Functor G: Affordance → Shape
    
    Maps affordance distributions back to 3D point clouds using a
    simple MLP decoder.
    """
    
    def __init__(
        self,
        affordance_dim: int = 16,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 3
    ):
        """
        Args:
            affordance_dim: Input affordance dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (3 for xyz coordinates)
            num_layers: Number of MLP layers
        """
        super().__init__()
        
        self.affordance_dim = affordance_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Decoder: affordance → shape
        layers = []
        in_dim = affordance_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if i < num_layers - 1 else nn.Identity(),
                nn.ReLU() if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, affordances: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            affordances: Affordance features (N, affordance_dim)
        
        Returns:
            reconstructed_pos: Reconstructed point positions (N, 3)
        """
        # Decode affordances to point positions
        reconstructed_pos = self.decoder(affordances)  # (N, 3)
        
        return reconstructed_pos
