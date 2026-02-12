"""
Adjunction Structure: F ⊣ G with Coherence Signal

This module integrates Functor F and Functor G into an adjunction structure
and implements the coherence signal computation.

The coherence signal η is defined as:
    η = distance(shape, G(F(shape)))

This is the core of the Physical-Semantic Adjunction Model.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from .functor_f import FunctorF
from .functor_g import FunctorG


class AdjunctionModel(nn.Module):
    """
    Complete adjunction structure F ⊣ G with coherence signal.
    
    This is the "riverbed" in the geological metaphor - a variable structure
    that represents the agent's current stable understanding of the world.
    """
    
    def __init__(
        self,
        functor_f: FunctorF,
        functor_g: FunctorG,
        distance_metric: str = 'chamfer'
    ):
        """
        Args:
            functor_f: The F: Shape → Action functor
            functor_g: The G: Action → Shape functor
            distance_metric: Distance metric for coherence signal ('chamfer' or 'hausdorff')
        """
        super().__init__()
        
        self.F = functor_f
        self.G = functor_g
        self.distance_metric = distance_metric
        
        # Ensure F and G have compatible dimensions
        assert self.F.num_affordances == self.G.num_affordances, \
            "F and G must have the same number of affordances"
    
    def chamfer_distance(
        self,
        pc1: torch.Tensor,
        pc2: torch.Tensor,
        batch1: Optional[torch.Tensor] = None,
        batch2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Chamfer distance between two point clouds.
        
        Chamfer distance is defined as:
            CD(P, Q) = mean(min_q ||p - q||^2 for p in P) + mean(min_p ||q - p||^2 for q in Q)
        
        Args:
            pc1: (N1, 3) or (B, N1, 3) first point cloud
            pc2: (N2, 3) or (B, N2, 3) second point cloud (typically from G, already batched)
            batch1: (N1,) batch assignment for pc1 (if unbatched input)
            batch2: Ignored (pc2 is assumed to be batched)
        
        Returns:
            distance: (B,) Chamfer distance per sample
        """
        # pc2 is already batched from G output: (B, N2, 3)
        # pc1 needs to be batched if it's not
        
        if pc2.dim() == 3:
            # pc2 is already batched
            if pc1.dim() == 2:
                # pc1 needs batching
                if batch1 is None:
                    pc1 = pc1.unsqueeze(0)
                else:
                    # Group by batch
                    batch_size = batch1.max().item() + 1
                    pc1_batched = []
                    
                    for b in range(batch_size):
                        pc1_batched.append(pc1[batch1 == b])
                    
                    # Pad to same length
                    max_len1 = max(p.size(0) for p in pc1_batched)
                    
                    pc1 = torch.stack([
                        torch.cat([p, torch.zeros(max_len1 - p.size(0), 3, device=p.device)], dim=0)
                        for p in pc1_batched
                    ])
        else:
            # Both unbatched
            if pc1.dim() == 2 and pc2.dim() == 2:
                pc1 = pc1.unsqueeze(0)
                pc2 = pc2.unsqueeze(0)
        
        B = pc1.size(0)
        N1 = pc1.size(1)
        N2 = pc2.size(1)
        
        # Compute pairwise distances
        # pc1: (B, N1, 3), pc2: (B, N2, 3)
        # dist: (B, N1, N2)
        pc1_expanded = pc1.unsqueeze(2)  # (B, N1, 1, 3)
        pc2_expanded = pc2.unsqueeze(1)  # (B, 1, N2, 3)
        dist = torch.sum((pc1_expanded - pc2_expanded) ** 2, dim=-1)  # (B, N1, N2)
        
        # Chamfer distance
        dist_pc1_to_pc2 = torch.min(dist, dim=2)[0].mean(dim=1)  # (B,)
        dist_pc2_to_pc1 = torch.min(dist, dim=1)[0].mean(dim=1)  # (B,)
        
        chamfer_dist = dist_pc1_to_pc2 + dist_pc2_to_pc1
        
        return chamfer_dist
    
    def chamfer_distance_spatial(
        self,
        pc1: torch.Tensor,
        pc2: torch.Tensor,
        batch1: Optional[torch.Tensor] = None,
        batch2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute per-point Chamfer distance (spatial decomposition of coherence signal).
        
        For each point in pc1, compute the minimum distance to any point in pc2.
        This provides spatial information about "where" the breakdown occurs.
        
        Args:
            pc1: (N1, 3) or (B, N1, 3) first point cloud
            pc2: (N2, 3) or (B, N2, 3) second point cloud (typically from G, already batched)
            batch1: (N1,) batch assignment for pc1 (if unbatched input)
            batch2: Ignored (pc2 is assumed to be batched)
        
        Returns:
            distance: (N1,) per-point coherence signal
        """
        # pc2 is already batched from G output: (B, N2, 3)
        # pc1 needs to be batched if it's not
        
        if pc2.dim() == 3:
            # pc2 is already batched
            if pc1.dim() == 2:
                # pc1 needs batching
                if batch1 is None:
                    pc1 = pc1.unsqueeze(0)
                    batch1 = torch.zeros(pc1.size(1), dtype=torch.long, device=pc1.device)
                else:
                    # Group by batch
                    batch_size = batch1.max().item() + 1
                    pc1_batched = []
                    
                    for b in range(batch_size):
                        pc1_batched.append(pc1[batch1 == b])
                    
                    # Pad to same length
                    max_len1 = max(p.size(0) for p in pc1_batched)
                    
                    pc1 = torch.stack([
                        torch.cat([p, torch.zeros(max_len1 - p.size(0), 3, device=p.device)], dim=0)
                        for p in pc1_batched
                    ])
        else:
            # Both unbatched
            if pc1.dim() == 2 and pc2.dim() == 2:
                pc1 = pc1.unsqueeze(0)
                pc2 = pc2.unsqueeze(0)
                if batch1 is None:
                    batch1 = torch.zeros(pc1.size(1), dtype=torch.long, device=pc1.device)
        
        B = pc1.size(0)
        N1 = pc1.size(1)
        N2 = pc2.size(1)
        
        # Compute pairwise distances
        # pc1: (B, N1, 3), pc2: (B, N2, 3)
        # dist: (B, N1, N2)
        pc1_expanded = pc1.unsqueeze(2)  # (B, N1, 1, 3)
        pc2_expanded = pc2.unsqueeze(1)  # (B, 1, N2, 3)
        dist = torch.sum((pc1_expanded - pc2_expanded) ** 2, dim=-1)  # (B, N1, N2)
        
        # Per-point minimum distance to reconstructed shape
        dist_per_point = torch.min(dist, dim=2)[0]  # (B, N1)
        
        # Flatten to (N1,) using batch assignment
        dist_per_point_flat = dist_per_point.reshape(-1)  # (B * N1,)
        
        return dist_per_point_flat
    
    def compute_coherence_signal(
        self,
        original_shape: torch.Tensor,
        reconstructed_shape: torch.Tensor,
        batch_original: Optional[torch.Tensor] = None,
        batch_reconstructed: Optional[torch.Tensor] = None,
        return_spatial: bool = False
    ) -> torch.Tensor:
        """
        Compute coherence signal η = distance(shape, G(F(shape))).
        
        This is the unit of the adjunction, measuring how well the
        adjunction F ⊣ G preserves the shape structure.
        
        Args:
            original_shape: (N, 3) or (B, N, 3) original point cloud
            reconstructed_shape: (M, 3) or (B, M, 3) reconstructed point cloud
            batch_original: (N,) batch assignment for original
            batch_reconstructed: (M,) batch assignment for reconstructed
            return_spatial: If True, return per-point coherence instead of scalar
        
        Returns:
            coherence: (B,) coherence signal per sample if return_spatial=False,
                      (N,) per-point coherence if return_spatial=True
        """
        if self.distance_metric == 'chamfer':
            if return_spatial:
                distance = self.chamfer_distance_spatial(
                    original_shape, reconstructed_shape,
                    batch_original, batch_reconstructed
                )
            else:
                distance = self.chamfer_distance(
                    original_shape, reconstructed_shape,
                    batch_original, batch_reconstructed
                )
        else:
            raise NotImplementedError(f"Distance metric '{self.distance_metric}' not implemented")
        
        return distance
    
    def forward(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through the adjunction.
        
        Computes: shape → F → affordances → G → reconstructed_shape
        And the coherence signal: distance(shape, reconstructed_shape)
        
        Args:
            pos: (N, 3) point cloud positions
            batch: (N,) batch assignment
            return_intermediate: If True, return intermediate results
        
        Returns:
            Dict with keys:
                - affordances: (N, num_affordances) predicted affordances
                - reconstructed_shape: (B, M, 3) reconstructed point cloud
                - coherence_signal: (B,) coherence signal
                - (optional) intermediate values if return_intermediate=True
        """
        # F: Shape → Action
        affordances = self.F(pos, batch)  # (N, num_affordances)
        
        # G: Action → Shape
        reconstructed_shape = self.G(affordances, batch)  # (B, M, 3)
        
        # Compute coherence signal
        coherence_signal = self.compute_coherence_signal(
            pos, reconstructed_shape, batch, None
        )
        
        results = {
            'affordances': affordances,
            'reconstructed_shape': reconstructed_shape,
            'coherence_signal': coherence_signal
        }
        
        if return_intermediate:
            results['original_shape'] = pos
            results['batch'] = batch
        
        return results


def create_adjunction_model(
    num_affordances: int = 5,
    num_points: int = 1024,
    f_hidden_dim: int = 64,
    g_hidden_dim: int = 128,
    **kwargs
) -> AdjunctionModel:
    """
    Factory function to create a complete adjunction model.
    
    Args:
        num_affordances: Number of affordance types
        num_points: Number of points in point clouds
        f_hidden_dim: Hidden dimension for F
        g_hidden_dim: Hidden dimension for G
        **kwargs: Additional arguments
    
    Returns:
        AdjunctionModel instance
    """
    from .functor_f import create_functor_f
    from .functor_g import create_functor_g
    
    F = create_functor_f(num_affordances=num_affordances, hidden_dim=f_hidden_dim)
    G = create_functor_g(num_affordances=num_affordances, num_points=num_points, hidden_dim=g_hidden_dim)
    
    model = AdjunctionModel(F, G, **kwargs)
    
    return model


if __name__ == '__main__':
    # Test the adjunction model
    print("Testing Adjunction Model...")
    
    # Create model
    model = create_adjunction_model(
        num_affordances=5,
        num_points=512,
        f_hidden_dim=32,
        g_hidden_dim=64
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    f_params = sum(p.numel() for p in model.F.parameters())
    g_params = sum(p.numel() for p in model.G.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"  F parameters: {f_params:,}")
    print(f"  G parameters: {g_params:,}")
    
    # Create dummy input
    batch_size = 2
    num_points = 512
    
    pos = torch.randn(batch_size * num_points, 3)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_points)
    
    print(f"\nInput shape: {pos.shape}")
    
    # Forward pass
    with torch.no_grad():
        results = model(pos, batch, return_intermediate=True)
    
    print(f"\nResults:")
    print(f"  Affordances shape: {results['affordances'].shape}")
    print(f"  Reconstructed shape: {results['reconstructed_shape'].shape}")
    print(f"  Coherence signal: {results['coherence_signal']}")
    print(f"  Coherence signal mean: {results['coherence_signal'].mean():.4f}")
    
    # Test single sample
    pos_single = torch.randn(num_points, 3)
    with torch.no_grad():
        results_single = model(pos_single, batch=None)
    
    print(f"\nSingle sample:")
    print(f"  Coherence signal: {results_single['coherence_signal'].item():.4f}")
    
    print("\nAdjunction Model test passed!")
    print("\n" + "="*60)
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("The coherence signal η = distance(shape, G(F(shape))) is now computable.")
    print("This implements the unit of the adjunction F ⊣ G.")
    print("Low coherence signal → stable adjunction (riverbed is solid)")
    print("High coherence signal → breakdown (riverbed is eroding)")
    print("="*60)
