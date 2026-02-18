"""
Synthetic Affordance Dataset

Generates simple 3D shapes (cube, cylinder, sphere) with synthetic affordance labels.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SyntheticAffordanceDataset(Dataset):
    """
    Synthetic dataset for affordance learning.
    
    Generates simple 3D shapes with synthetic affordance labels.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        num_points: int = 512,
        shape_types: list = [0, 1, 2],  # 0: cube, 1: cylinder, 2: sphere
        affordance_dim: int = 16
    ):
        """
        Args:
            num_samples: Number of samples in the dataset
            num_points: Number of points per shape
            shape_types: List of shape types to generate
            affordance_dim: Dimension of affordance vectors
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.shape_types = shape_types
        self.affordance_dim = affordance_dim
        
        # Pre-generate all samples
        self.samples = []
        for _ in range(num_samples):
            shape_type = np.random.choice(shape_types)
            points = self._generate_shape(shape_type, num_points)
            affordances = self._generate_affordances(shape_type, num_points, affordance_dim)
            
            self.samples.append({
                'points': torch.FloatTensor(points),
                'affordances': torch.FloatTensor(affordances),
                'shape_type': shape_type
            })
    
    def _generate_shape(self, shape_type: int, num_points: int) -> np.ndarray:
        """
        Generate a 3D shape.
        
        Args:
            shape_type: 0 (cube), 1 (cylinder), 2 (sphere)
            num_points: Number of points
        
        Returns:
            points: (num_points, 3) array
        """
        if shape_type == 0:  # Cube
            points = np.random.uniform(-1, 1, (num_points, 3))
        
        elif shape_type == 1:  # Cylinder
            theta = np.random.uniform(0, 2 * np.pi, num_points)
            r = np.random.uniform(0, 1, num_points)
            z = np.random.uniform(-1, 1, num_points)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.stack([x, y, z], axis=-1)
        
        elif shape_type == 2:  # Sphere
            theta = np.random.uniform(0, 2 * np.pi, num_points)
            phi = np.random.uniform(0, np.pi, num_points)
            r = np.random.uniform(0, 1, num_points) ** (1/3)  # Uniform volume sampling
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            points = np.stack([x, y, z], axis=-1)
        
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
        
        return points.astype(np.float32)
    
    def _generate_affordances(
        self,
        shape_type: int,
        num_points: int,
        affordance_dim: int
    ) -> np.ndarray:
        """
        Generate synthetic affordance labels.
        
        Args:
            shape_type: Shape type
            num_points: Number of points
            affordance_dim: Affordance dimension
        
        Returns:
            affordances: (num_points, affordance_dim) array
        """
        # Simple synthetic affordances based on shape type
        # Each shape type has a characteristic affordance pattern
        base_affordance = np.zeros(affordance_dim, dtype=np.float32)
        
        if shape_type == 0:  # Cube: graspable, stackable
            base_affordance[0] = 1.0  # graspable
            base_affordance[1] = 1.0  # stackable
        
        elif shape_type == 1:  # Cylinder: rollable, graspable
            base_affordance[0] = 0.8  # graspable
            base_affordance[2] = 1.0  # rollable
        
        elif shape_type == 2:  # Sphere: rollable
            base_affordance[2] = 1.0  # rollable
            base_affordance[3] = 0.5  # bouncy
        
        # Repeat for all points with small noise
        affordances = np.tile(base_affordance, (num_points, 1))
        affordances += np.random.normal(0, 0.1, affordances.shape).astype(np.float32)
        affordances = np.clip(affordances, 0, 1)
        
        return affordances
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
