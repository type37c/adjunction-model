"""
Dynamic Functor F/G for Reaching Task

Key differences from static F/G:
1. F processes TEMPORAL point clouds (t and t+1) → affordance (reachability)
2. G predicts NEXT STATE (motion prediction) from affordance + action
3. η measures PREDICTION ERROR (reachability difficulty)

Theoretical alignment:
- Affordance = "How reachable is the object from current state?"
- η = "How predictable is the motion?" (low η = easy to reach)
- This aligns with Reaching task requirements (dynamic motion)
"""

import torch
import torch.nn as nn
import numpy as np


class TemporalPointNetEncoder(nn.Module):
    """
    Encode temporal point clouds (t and t+1) into a single feature vector
    
    Architecture:
    - Process each point cloud independently with PointNet
    - Concatenate features
    - Fuse with temporal MLP
    """
    
    def __init__(self, point_dim=3, feature_dim=64):
        super().__init__()
        
        self.point_dim = point_dim
        self.feature_dim = feature_dim
        
        # Per-point MLP for each timestep
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )
        
        # Temporal fusion MLP
        self.temporal_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )
    
    def forward(self, pc_t, pc_t1):
        """
        Args:
            pc_t: Point cloud at time t (B, N, 3)
            pc_t1: Point cloud at time t+1 (B, N, 3)
        
        Returns:
            temporal_features: (B, feature_dim)
        """
        # Process each point cloud
        # (B, N, 3) -> (B, N, feature_dim) -> (B, feature_dim) via max pooling
        feat_t = self.point_mlp(pc_t).max(dim=1)[0]
        feat_t1 = self.point_mlp(pc_t1).max(dim=1)[0]
        
        # Concatenate and fuse
        combined = torch.cat([feat_t, feat_t1], dim=-1)
        temporal_features = self.temporal_fusion(combined)
        
        return temporal_features


class DynamicFunctorF(nn.Module):
    """
    Dynamic Functor F: Temporal point clouds → Affordance (reachability)
    
    Affordance represents: "How reachable is the object from current state?"
    """
    
    def __init__(self, affordance_dim=32, feature_dim=64):
        super().__init__()
        
        self.affordance_dim = affordance_dim
        
        # Temporal point cloud encoder
        self.temporal_encoder = TemporalPointNetEncoder(
            point_dim=3,
            feature_dim=feature_dim
        )
        
        # Affordance head
        self.affordance_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, affordance_dim),
        )
    
    def forward(self, pc_t, pc_t1):
        """
        Args:
            pc_t: Point cloud at time t (B, N, 3)
            pc_t1: Point cloud at time t+1 (B, N, 3)
        
        Returns:
            affordance: (B, affordance_dim)
        """
        temporal_features = self.temporal_encoder(pc_t, pc_t1)
        affordance = self.affordance_head(temporal_features)
        return affordance


class DynamicFunctorG(nn.Module):
    """
    Dynamic Functor G: Affordance + Action → Next state prediction (motion)
    
    Predicts: end-effector position at t+1 given affordance and action
    """
    
    def __init__(self, affordance_dim=32, action_dim=3, output_dim=3):
        super().__init__()
        
        self.affordance_dim = affordance_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        
        # Motion prediction network
        self.motion_predictor = nn.Sequential(
            nn.Linear(affordance_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, affordance, action):
        """
        Args:
            affordance: (B, affordance_dim)
            action: (B, action_dim)
        
        Returns:
            ee_pos_pred: Predicted end-effector position (B, output_dim)
        """
        combined = torch.cat([affordance, action], dim=-1)
        ee_pos_pred = self.motion_predictor(combined)
        return ee_pos_pred


class DynamicFGModel(nn.Module):
    """
    Combined Dynamic F/G model
    
    Training objective:
    - F learns to extract reachability affordance from temporal point clouds
    - G learns to predict next end-effector position from affordance + action
    - η = ||ee_pos_pred - ee_pos_true||² (prediction error = reachability difficulty)
    
    Low η → easy to reach (predictable motion)
    High η → hard to reach (unpredictable motion)
    """
    
    def __init__(self, affordance_dim=32, action_dim=3, output_dim=3):
        super().__init__()
        
        self.affordance_dim = affordance_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        
        self.functor_f = DynamicFunctorF(affordance_dim=affordance_dim)
        self.functor_g = DynamicFunctorG(
            affordance_dim=affordance_dim,
            action_dim=action_dim,
            output_dim=output_dim
        )
    
    def forward(self, pc_t, pc_t1, action):
        """
        Args:
            pc_t: Point cloud at time t (B, N, 3)
            pc_t1: Point cloud at time t+1 (B, N, 3)
            action: Action taken (B, action_dim)
        
        Returns:
            affordance: (B, affordance_dim)
            ee_pos_pred: Predicted end-effector position (B, output_dim)
        """
        affordance = self.functor_f(pc_t, pc_t1)
        ee_pos_pred = self.functor_g(affordance, action)
        return affordance, ee_pos_pred
    
    def compute_eta(self, ee_pos_pred, ee_pos_true):
        """
        Compute η (reconstruction/prediction error)
        
        Args:
            ee_pos_pred: Predicted end-effector position (B, 3)
            ee_pos_true: True end-effector position (B, 3)
        
        Returns:
            eta: Prediction error (B,)
        """
        eta = torch.sum((ee_pos_pred - ee_pos_true) ** 2, dim=-1)
        return eta
    
    def get_affordance(self, pc_t, pc_t1):
        """
        Extract affordance only (for Agent C integration)
        """
        return self.functor_f(pc_t, pc_t1)


def test_dynamic_fg():
    """
    Test dynamic F/G model
    """
    print("Testing Dynamic F/G model...")
    
    # Create model
    model = DynamicFGModel(affordance_dim=32, action_dim=3, output_dim=3)
    
    # Test input
    batch_size = 4
    num_points = 128
    pc_t = torch.randn(batch_size, num_points, 3)
    pc_t1 = torch.randn(batch_size, num_points, 3)
    action = torch.randn(batch_size, 3)
    ee_pos_true = torch.randn(batch_size, 3)
    
    # Forward pass
    affordance, ee_pos_pred = model(pc_t, pc_t1, action)
    eta = model.compute_eta(ee_pos_pred, ee_pos_true)
    
    print(f"  Affordance shape: {affordance.shape}")
    print(f"  EE pos pred shape: {ee_pos_pred.shape}")
    print(f"  η shape: {eta.shape}")
    print(f"  η values: {eta.detach().numpy()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    print("Test passed!")


if __name__ == '__main__':
    test_dynamic_fg()
