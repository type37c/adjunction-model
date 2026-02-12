"""
Synthetic dataset generator for MVP development.

This module generates simple 3D shapes (cubes, cylinders, spheres) with 
affordance labels to enable rapid prototyping without downloading large datasets.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict, List


class SyntheticAffordanceDataset(Dataset):
    """
    Generates synthetic 3D point clouds with affordance annotations.
    
    Shape types:
    - 0: Cube (graspable, liftable, stackable)
    - 1: Cylinder (graspable, rollable, containable)
    - 2: Sphere (graspable, rollable)
    
    Affordance types (simplified from 3D AffordanceNet's 18 types):
    - 0: grasp
    - 1: lift
    - 2: support (stack on top)
    - 3: contain
    - 4: roll
    """
    
    AFFORDANCE_NAMES = ['grasp', 'lift', 'support', 'contain', 'roll']
    NUM_AFFORDANCES = len(AFFORDANCE_NAMES)
    
    # Shape-affordance mapping (which affordances are valid for each shape)
    SHAPE_AFFORDANCES = {
        0: [0, 1, 2],      # Cube: grasp, lift, support
        1: [0, 3, 4],      # Cylinder: grasp, contain, roll
        2: [0, 4],         # Sphere: grasp, roll
    }
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_points: int = 1024,
        shape_types: List[int] = [0, 1, 2],
        noise_std: float = 0.01,
        seed: int = 42
    ):
        """
        Args:
            num_samples: Number of samples to generate
            num_points: Number of points per point cloud
            shape_types: List of shape type indices to include
            noise_std: Standard deviation of Gaussian noise added to points
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.shape_types = shape_types
        self.noise_std = noise_std
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Pre-generate all samples for consistency
        self.samples = self._generate_all_samples()
    
    def _generate_cube(self, size: float = 1.0) -> np.ndarray:
        """Generate a cube point cloud."""
        # Sample points uniformly on cube surface
        points = []
        points_per_face = self.num_points // 6
        
        for axis in range(3):
            for sign in [-1, 1]:
                # Generate points on one face
                face_points = np.random.uniform(-size/2, size/2, (points_per_face, 3))
                face_points[:, axis] = sign * size / 2
                points.append(face_points)
        
        points = np.vstack(points)
        # Ensure exact num_points
        if points.shape[0] > self.num_points:
            points = points[:self.num_points]
        elif points.shape[0] < self.num_points:
            # Pad with duplicates if needed
            shortage = self.num_points - points.shape[0]
            indices = np.random.choice(points.shape[0], shortage, replace=True)
            points = np.vstack([points, points[indices]])
        return points
    
    def _generate_cylinder(self, radius: float = 0.5, height: float = 1.0) -> np.ndarray:
        """Generate a cylinder point cloud."""
        points = []
        
        # Curved surface
        num_curved = int(self.num_points * 0.7)
        theta = np.random.uniform(0, 2*np.pi, num_curved)
        z = np.random.uniform(-height/2, height/2, num_curved)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append(np.stack([x, y, z], axis=1))
        
        # Top and bottom caps
        num_cap = (self.num_points - num_curved) // 2
        for z_val in [-height/2, height/2]:
            r = np.random.uniform(0, radius, num_cap)
            theta = np.random.uniform(0, 2*np.pi, num_cap)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.full(num_cap, z_val)
            points.append(np.stack([x, y, z], axis=1))
        
        points = np.vstack(points)
        # Ensure exact num_points
        if points.shape[0] > self.num_points:
            points = points[:self.num_points]
        elif points.shape[0] < self.num_points:
            # Pad with duplicates if needed
            shortage = self.num_points - points.shape[0]
            indices = np.random.choice(points.shape[0], shortage, replace=True)
            points = np.vstack([points, points[indices]])
        return points
    
    def _generate_sphere(self, radius: float = 0.5) -> np.ndarray:
        """Generate a sphere point cloud."""
        # Sample points uniformly on sphere surface using Fibonacci sphere
        indices = np.arange(0, self.num_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / self.num_points)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)
        
        points = np.stack([x, y, z], axis=1)
        return points
    
    def _generate_affordance_labels(self, shape_type: int) -> np.ndarray:
        """
        Generate per-point affordance labels for a shape.
        
        Returns:
            affordance_labels: (num_points, NUM_AFFORDANCES) binary matrix
        """
        labels = np.zeros((self.num_points, self.NUM_AFFORDANCES), dtype=np.float32)
        
        # Get valid affordances for this shape type
        valid_affordances = self.SHAPE_AFFORDANCES[shape_type]
        
        # For simplicity, mark all points as having the shape's affordances
        # In a real dataset, this would be spatially varying
        labels[:, valid_affordances] = 1.0
        
        return labels
    
    def _generate_all_samples(self) -> List[Dict]:
        """Pre-generate all samples."""
        samples = []
        
        for i in range(self.num_samples):
            # Randomly select shape type
            shape_type = np.random.choice(self.shape_types)
            
            # Generate point cloud
            if shape_type == 0:
                points = self._generate_cube()
            elif shape_type == 1:
                points = self._generate_cylinder()
            elif shape_type == 2:
                points = self._generate_sphere()
            else:
                raise ValueError(f"Unknown shape type: {shape_type}")
            
            # Add noise
            points += np.random.normal(0, self.noise_std, points.shape)
            
            # Generate affordance labels
            affordance_labels = self._generate_affordance_labels(shape_type)
            
            samples.append({
                'points': torch.from_numpy(points).float(),
                'affordances': torch.from_numpy(affordance_labels).float(),
                'shape_type': shape_type
            })
        
        return samples
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys:
                - points: (num_points, 3) point cloud
                - affordances: (num_points, NUM_AFFORDANCES) affordance labels
                - shape_type: int, shape category
        """
        return self.samples[idx]


def get_dataloader(
    batch_size: int = 32,
    num_samples: int = 1000,
    num_points: int = 1024,
    split: str = 'train',
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for synthetic affordance data.
    
    Args:
        batch_size: Batch size
        num_samples: Number of samples in dataset
        num_points: Points per point cloud
        split: 'train' or 'val' (affects random seed)
        **kwargs: Additional arguments passed to SyntheticAffordanceDataset
    
    Returns:
        DataLoader instance
    """
    seed = 42 if split == 'train' else 123
    
    dataset = SyntheticAffordanceDataset(
        num_samples=num_samples,
        num_points=num_points,
        seed=seed,
        **kwargs
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=0,  # Single-threaded for simplicity
        pin_memory=False
    )
    
    return dataloader


if __name__ == '__main__':
    # Test the dataset
    dataset = SyntheticAffordanceDataset(num_samples=10, num_points=512)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Points shape: {sample['points'].shape}")
    print(f"Affordances shape: {sample['affordances'].shape}")
    print(f"Shape type: {sample['shape_type']}")
    print(f"Affordance labels (first point): {sample['affordances'][0]}")
