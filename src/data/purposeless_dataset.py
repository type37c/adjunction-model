"""
Purposeless Assembly Dataset: No Target Shape, Full Point Cloud from Step 0

This module generates synthetic data for the Purpose-Emergent Active Assembly
experiment.  Unlike TemporalShapeDataset, there is NO target shape.  The agent
receives all 256 points from Step 0 and must *choose its own purpose*.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional


class PurposelessAssemblyDataset(Dataset):
    """
    Generates episodes for the Purpose-Emergent Active Assembly task.
    """

    REFERENCE_SHAPE_NAMES = ['cube', 'cylinder', 'sphere']

    def __init__(
        self,
        num_samples: int = 100,
        num_points: int = 256,
        scatter_radius: float = 1.5,
        noise_std: float = 0.02,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_points = num_points
        self.scatter_radius = scatter_radius
        self.noise_std = noise_std

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.reference_shapes = self._generate_reference_shapes()
        self.samples = self._generate_all_samples()

    def _generate_cube(self, n: int, size: float = 1.0) -> np.ndarray:
        points = []
        ppf = max(1, n // 6)
        for axis in range(3):
            for sign in [-1, 1]:
                face = np.random.uniform(-size / 2, size / 2, (ppf, 3))
                face[:, axis] = sign * size / 2
                points.append(face)
        return self._adjust_count(np.vstack(points), n)

    def _generate_cylinder(self, n: int, radius: float = 0.5,
                           height: float = 1.0) -> np.ndarray:
        n_curved = int(n * 0.7)
        n_cap = (n - n_curved) // 2
        theta = np.random.uniform(0, 2 * np.pi, n_curved)
        z = np.random.uniform(-height / 2, height / 2, n_curved)
        parts = [np.stack([radius * np.cos(theta),
                           radius * np.sin(theta), z], axis=1)]
        for z_val in [-height / 2, height / 2]:
            r = np.random.uniform(0, radius, n_cap)
            th = np.random.uniform(0, 2 * np.pi, n_cap)
            parts.append(np.stack([r * np.cos(th), r * np.sin(th),
                                   np.full(n_cap, z_val)], axis=1))
        return self._adjust_count(np.vstack(parts), n)

    def _generate_sphere(self, n: int, radius: float = 0.5) -> np.ndarray:
        idx = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * idx / n)
        theta = np.pi * (1 + 5 ** 0.5) * idx
        return np.stack([radius * np.cos(theta) * np.sin(phi),
                         radius * np.sin(theta) * np.sin(phi),
                         radius * np.cos(phi)], axis=1)

    @staticmethod
    def _adjust_count(pts: np.ndarray, target: int) -> np.ndarray:
        if pts.shape[0] > target:
            return pts[:target]
        elif pts.shape[0] < target:
            extra = np.random.choice(pts.shape[0], target - pts.shape[0],
                                     replace=True)
            return np.vstack([pts, pts[extra]])
        return pts

    def _generate_reference_shapes(self) -> Dict[str, torch.Tensor]:
        generators = {
            'cube': self._generate_cube,
            'cylinder': self._generate_cylinder,
            'sphere': self._generate_sphere,
        }
        shapes = {}
        for name, gen in generators.items():
            pts = gen(self.num_points)
            pts = pts + np.random.normal(0, self.noise_std, pts.shape)
            shapes[name] = torch.from_numpy(pts).float()
        return shapes

    def _generate_scattered_cloud(self) -> torch.Tensor:
        n = self.num_points
        r = self.scatter_radius * np.cbrt(np.random.uniform(0, 1, n))
        theta = np.random.uniform(0, 2 * np.pi, n)
        phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
        pts = np.stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ], axis=1)
        return torch.from_numpy(pts).float()

    def _generate_all_samples(self) -> List[Dict]:
        samples = []
        for _ in range(self.num_samples):
            samples.append({
                'initial_points': self._generate_scattered_cloud(),
            })
        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_purposeless_batch(batch: List[Dict]) -> Dict:
    B = len(batch)
    pts_list = []
    batch_list = []
    for i, s in enumerate(batch):
        pts = s['initial_points']
        pts_list.append(pts)
        batch_list.append(
            torch.full((pts.size(0),), i, dtype=torch.long))
    return {
        'initial_points': torch.cat(pts_list, dim=0),
        'initial_batch': torch.cat(batch_list, dim=0),
    }
