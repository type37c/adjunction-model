"""
Functor G: Action → Shape

This module implements the inverse model that reconstructs the "functional core"
of a shape from affordance distributions.

Architecture: Conditional GNN decoder that generates point clouds from affordance inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FunctorG(nn.Module):
    """
    Functor G: Action → Shape
    
    Reconstructs point cloud from affordance distribution.
    This is the "inverse" of F in the adjunction F ⊣ G.
    
    The model takes a global affordance representation and generates
    a point cloud that exhibits those affordances.
    """
    
    def __init__(
        self,
        num_affordances: int = 5,
        num_points: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 4,
        latent_dim: int = 256
    ):
        """
        Args:
            num_affordances: Number of affordance types
            num_points: Number of points to generate
            hidden_dim: Hidden dimension size
            num_layers: Number of decoder layers
            latent_dim: Latent code dimension
        """
        super().__init__()
        
        self.num_affordances = num_affordances
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Affordance encoder: aggregate per-point affordances to global code
        self.affordance_encoder = nn.Sequential(
            nn.Linear(num_affordances, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Point generator: decode latent code to point cloud
        # We use a simple MLP-based generator (similar to PointFlow)
        self.point_generator = nn.ModuleList()
        
        # First layer: latent + point index -> hidden
        self.point_generator.append(
            nn.Sequential(
                nn.Linear(latent_dim + 3, hidden_dim),  # +3 for grid position
                nn.ReLU()
            )
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.point_generator.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
            )
        
        # Final layer: hidden -> 3D offset
        self.point_generator.append(
            nn.Linear(hidden_dim, 3)
        )
        
        # Learnable grid positions (template for generation)
        self.register_buffer(
            'grid_positions',
            self._create_grid_positions(num_points)
        )
    
    def _create_grid_positions(self, num_points: int) -> torch.Tensor:
        """
        Create a regular 3D grid as template positions.
        
        Args:
            num_points: Number of points
        
        Returns:
            grid: (num_points, 3) grid positions
        """
        # Create a cube grid
        side = int(num_points ** (1/3)) + 1
        x = torch.linspace(-1, 1, side)
        y = torch.linspace(-1, 1, side)
        z = torch.linspace(-1, 1, side)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        grid = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
        
        # Sample num_points from grid
        indices = torch.randperm(grid.size(0))[:num_points]
        grid = grid[indices]
        
        return grid
    
    def encode_affordances(
        self,
        affordances: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode per-point affordances to global latent code.
        
        Args:
            affordances: (N, num_affordances) per-point affordance labels
            batch: (N,) batch assignment, optional
        
        Returns:
            latent: (B, latent_dim) global affordance code per graph
        """
        if batch is None:
            # Single sample: global average pooling
            affordance_global = affordances.mean(dim=0, keepdim=True)  # (1, num_affordances)
        else:
            # Batched: average pooling per graph
            batch_size = batch.max().item() + 1
            affordance_global = torch.zeros(
                batch_size, self.num_affordances,
                device=affordances.device, dtype=affordances.dtype
            )
            
            for b in range(batch_size):
                mask = batch == b
                affordance_global[b] = affordances[mask].mean(dim=0)
        
        # Encode to latent
        latent = self.affordance_encoder(affordance_global)  # (B, latent_dim)
        
        return latent
    
    def generate_points(
        self,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate point cloud from latent code.
        
        Args:
            latent: (B, latent_dim) latent codes
        
        Returns:
            points: (B, num_points, 3) generated point clouds
        """
        B = latent.size(0)
        
        # Expand grid positions for batch
        grid = self.grid_positions.unsqueeze(0).expand(B, -1, -1)  # (B, num_points, 3)
        
        # Expand latent for each point
        latent_expanded = latent.unsqueeze(1).expand(-1, self.num_points, -1)  # (B, num_points, latent_dim)
        
        # Concatenate latent and grid position
        x = torch.cat([latent_expanded, grid], dim=-1)  # (B, num_points, latent_dim + 3)
        
        # Reshape for MLP processing
        x = x.view(B * self.num_points, -1)  # (B * num_points, latent_dim + 3)
        
        # Pass through generator
        for layer in self.point_generator:
            x = layer(x)
        
        # Reshape back
        offsets = x.view(B, self.num_points, 3)  # (B, num_points, 3)
        
        # Add offsets to grid (residual connection)
        points = grid + offsets
        
        return points
    
    def forward(
        self,
        affordances: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: affordances → point cloud.
        
        Args:
            affordances: (N, num_affordances) per-point affordance labels
            batch: (N,) batch assignment, optional
        
        Returns:
            points: (B, num_points, 3) reconstructed point clouds
        """
        # Encode affordances to latent
        latent = self.encode_affordances(affordances, batch)
        
        # Generate points from latent
        points = self.generate_points(latent)
        
        return points


def create_functor_g(num_affordances: int = 5, num_points: int = 1024, **kwargs) -> FunctorG:
    """
    Factory function to create Functor G model.
    
    Args:
        num_affordances: Number of affordance types
        num_points: Number of points to generate
        **kwargs: Additional arguments for FunctorG
    
    Returns:
        FunctorG instance
    """
    return FunctorG(num_affordances=num_affordances, num_points=num_points, **kwargs)


if __name__ == '__main__':
    # Test the model
    print("Testing Functor G...")
    
    # Create model
    model = create_functor_g(num_affordances=5, num_points=512, hidden_dim=64, latent_dim=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input (batch of 2 affordance distributions)
    batch_size = 2
    num_points_input = 512
    
    # Simulate affordance input (per-point)
    affordances = torch.rand(batch_size * num_points_input, 5)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_points_input)
    
    print(f"Input affordances shape: {affordances.shape}")
    print(f"Batch shape: {batch.shape}")
    
    # Forward pass
    with torch.no_grad():
        points = model(affordances, batch)
    
    print(f"Output points shape: {points.shape}")
    print(f"Points range: [{points.min():.3f}, {points.max():.3f}]")
    
    # Test single sample
    affordances_single = torch.rand(num_points_input, 5)
    with torch.no_grad():
        points_single = model(affordances_single, batch=None)
    
    print(f"Single sample output shape: {points_single.shape}")
    
    print("\nFunctor G test passed!")
