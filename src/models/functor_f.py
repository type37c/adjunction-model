"""
Functor F: Shape → Action

This module implements the affordance prediction model that maps 3D shapes
(point clouds) to action possibilities (affordance distributions).

Architecture: PointNet++ style GNN for point cloud processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Optional, Tuple


class PointNetSetAbstraction(MessagePassing):
    """
    Set abstraction layer for PointNet++.
    Groups points using k-NN and applies local feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 16,
        mlp_hidden: int = 64
    ):
        super().__init__(aggr='max')
        self.k = k
        
        # MLP for local feature extraction
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, mlp_hidden),  # +3 for relative position
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def _build_knn_graph(self, pos: torch.Tensor, k: int, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Build k-NN graph manually (fallback when torch-cluster not available).
        
        Args:
            pos: Node positions (N, 3)
            k: Number of neighbors
            batch: Batch assignment (N,)
        
        Returns:
            edge_index: (2, E) edge indices
        """
        N = pos.size(0)
        
        if batch is None:
            # Single graph case
            # Compute pairwise distances
            dist = torch.cdist(pos, pos)  # (N, N)
            # Get k nearest neighbors (including self)
            _, indices = torch.topk(dist, k + 1, largest=False, dim=1)
            indices = indices[:, 1:]  # Exclude self
            
            # Build edge index
            row = torch.arange(N, device=pos.device).unsqueeze(1).repeat(1, k).reshape(-1)
            col = indices.reshape(-1)
            edge_index = torch.stack([row, col], dim=0)
        else:
            # Batched case: build k-NN within each graph
            edge_indices = []
            for b in range(batch.max().item() + 1):
                mask = batch == b
                pos_b = pos[mask]
                N_b = pos_b.size(0)
                
                # Compute pairwise distances
                dist = torch.cdist(pos_b, pos_b)
                _, indices = torch.topk(dist, min(k + 1, N_b), largest=False, dim=1)
                indices = indices[:, 1:]  # Exclude self
                
                # Build edge index for this graph
                k_actual = indices.size(1)
                row = torch.arange(N_b, device=pos.device).unsqueeze(1).repeat(1, k_actual).reshape(-1)
                col = indices.reshape(-1)
                
                # Map to global indices
                global_indices = torch.where(mask)[0]
                row = global_indices[row]
                col = global_indices[col]
                
                edge_indices.append(torch.stack([row, col], dim=0))
            
            edge_index = torch.cat(edge_indices, dim=1)
        
        return edge_index
    
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features (N, in_channels)
            pos: Node positions (N, 3)
            batch: Batch assignment (N,)
        
        Returns:
            x_out: Output features (N, out_channels)
            pos: Same positions (N, 3)
        """
        # Build k-NN graph
        edge_index = self._build_knn_graph(pos, self.k, batch)
        
        # Message passing
        x_out = self.propagate(edge_index, x=x, pos=pos)
        
        return x_out, pos
    
    def message(self, x_j: torch.Tensor, pos_i: torch.Tensor, pos_j: torch.Tensor) -> torch.Tensor:
        """
        Compute messages from neighbors.
        
        Args:
            x_j: Features of neighbors (E, in_channels)
            pos_i: Positions of center nodes (E, 3)
            pos_j: Positions of neighbors (E, 3)
        
        Returns:
            messages: (E, out_channels)
        """
        # Relative position encoding
        rel_pos = pos_j - pos_i  # (E, 3)
        
        # Concatenate features and relative position
        msg_input = torch.cat([x_j, rel_pos], dim=-1)  # (E, in_channels + 3)
        
        # Apply MLP
        msg = self.mlp(msg_input)  # (E, out_channels)
        
        return msg


class FunctorF(nn.Module):
    """
    Functor F: Shape → Action
    
    Maps point cloud representations to affordance predictions.
    Uses a PointNet++ style architecture with hierarchical feature learning.
    """
    
    def __init__(
        self,
        num_affordances: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 3,
        k_neighbors: int = 16,
        dropout: float = 0.1
    ):
        """
        Args:
            num_affordances: Number of affordance types to predict
            hidden_dim: Hidden dimension size
            num_layers: Number of set abstraction layers
            k_neighbors: Number of neighbors for k-NN graph
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_affordances = num_affordances
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial feature embedding (from 3D coordinates)
        self.input_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Hierarchical set abstraction layers
        self.sa_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else hidden_dim * (2 ** i)
            out_ch = hidden_dim * (2 ** (i + 1))
            
            self.sa_layers.append(
                PointNetSetAbstraction(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    k=k_neighbors,
                    mlp_hidden=out_ch
                )
            )
        
        # Final feature dimension
        final_dim = hidden_dim * (2 ** num_layers)
        
        # Affordance prediction head (per-point)
        self.affordance_head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_affordances)
        )
    
    def forward(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pos: Point positions (N, 3) where N is total points across batch
            batch: Batch assignment (N,), optional for single sample
        
        Returns:
            affordances: (N, num_affordances) affordance logits per point
        """
        # Initial embedding from coordinates
        x = self.input_embed(pos)  # (N, hidden_dim)
        
        # Hierarchical feature extraction
        for sa_layer in self.sa_layers:
            x, pos = sa_layer(x, pos, batch)
        
        # Per-point affordance prediction
        affordances = self.affordance_head(x)  # (N, num_affordances)
        
        return affordances
    
    def predict_affordances(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Predict binary affordance labels.
        
        Args:
            pos: Point positions (N, 3)
            batch: Batch assignment (N,)
            threshold: Threshold for binary prediction
        
        Returns:
            affordances: (N, num_affordances) binary predictions
        """
        logits = self.forward(pos, batch)
        probs = torch.sigmoid(logits)
        predictions = (probs > threshold).float()
        return predictions


def create_functor_f(num_affordances: int = 5, **kwargs) -> FunctorF:
    """
    Factory function to create Functor F model.
    
    Args:
        num_affordances: Number of affordance types
        **kwargs: Additional arguments for FunctorF
    
    Returns:
        FunctorF instance
    """
    return FunctorF(num_affordances=num_affordances, **kwargs)


if __name__ == '__main__':
    # Test the model
    print("Testing Functor F...")
    
    # Create model
    model = create_functor_f(num_affordances=5, hidden_dim=32, num_layers=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input (batch of 2 point clouds, 512 points each)
    batch_size = 2
    num_points = 512
    
    pos = torch.randn(batch_size * num_points, 3)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_points)
    
    print(f"Input shape: {pos.shape}")
    print(f"Batch shape: {batch.shape}")
    
    # Forward pass
    with torch.no_grad():
        affordances = model(pos, batch)
    
    print(f"Output shape: {affordances.shape}")
    print(f"Output range: [{affordances.min():.3f}, {affordances.max():.3f}]")
    
    # Test prediction
    predictions = model.predict_affordances(pos, batch)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions (first point): {predictions[0]}")
    
    print("\nFunctor F test passed!")
