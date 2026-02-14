"""
Temporal Suspension Dataset: Active Point-Cloud Assembly

This module generates synthetic data for the *active* temporal suspension
experiment.  Unlike the previous "passive classification" design, the agent
must now **move** points to construct a target shape.

Theoretical motivation (from docs/docs/docs/01_temporal_suspension.md):
    - Action and understanding become inseparable: moving points *is*
      understanding the shape.
    - Suspension emerges naturally: early on, the target is ambiguous, so
      the agent cannot commit to large displacements.  Once enough evidence
      accumulates, it can act decisively.
    - Slack models should explore cautiously then commit; tight models
      should commit early and fail to correct.

Design:
    - Each episode has T time steps.
    - At step 0 the agent receives a small set of randomly scattered points
      plus a *hint* (a few points already near the target surface).
    - At each subsequent step, additional points are revealed (same
      progressive-revelation schedule as before).
    - The agent's task: output a displacement vector for every visible point
      so that the final configuration matches the target shape.
    - Ground truth: the target shape (sphere, cube, cylinder) as a point
      cloud.  Reward is the negative Chamfer Distance between the agent's
      assembled cloud and the target.

Data format (per sample):
    - 'initial_points':    (N_0, 3)  randomly scattered starting positions
    - 'target_points':     (N_final, 3)  the goal shape
    - 'target_affordances': (N_final, NUM_AFFORDANCES)  affordance labels
    - 'shape_type':        int  (0=cube, 1=cylinder, 2=sphere)
    - 'hint_mask':         (N_0,) bool  which initial points are "hints"
    - 'revelation_counts': (T,) int  cumulative point count at each step
    - 'ambiguity_schedule': (T,) float  ambiguity at each step
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional


class TemporalShapeDataset(Dataset):
    """
    Generates episodes for the active point-cloud assembly task.

    Shape types (compatible with SyntheticAffordanceDataset):
        0: Cube   (graspable, liftable, stackable)
        1: Cylinder (graspable, rollable, containable)
        2: Sphere  (graspable, rollable)

    Affordance types:
        0: grasp   1: lift   2: support   3: contain   4: roll
    """

    AFFORDANCE_NAMES = ['grasp', 'lift', 'support', 'contain', 'roll']
    NUM_AFFORDANCES = len(AFFORDANCE_NAMES)

    SHAPE_AFFORDANCES = {
        0: [0, 1, 2],      # Cube: grasp, lift, support
        1: [0, 3, 4],      # Cylinder: grasp, contain, roll
        2: [0, 4],          # Sphere: grasp, roll
    }

    SHAPE_NAMES = {0: 'cube', 1: 'cylinder', 2: 'sphere'}

    def __init__(
        self,
        num_samples: int = 100,
        num_points_final: int = 256,
        num_time_steps: int = 8,
        shape_types: Optional[List[int]] = None,
        noise_std: float = 0.02,
        scatter_radius: float = 1.5,
        hint_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Args:
            num_samples:     Number of episodes to generate.
            num_points_final: Total points in the fully revealed cloud.
            num_time_steps:  Number of time steps T per episode.
            shape_types:     Which shape categories to include.
            noise_std:       Gaussian noise added to target shapes.
            scatter_radius:  Radius of the initial random scatter.
            hint_ratio:      Fraction of initial points placed near the
                             target surface (shape hint).
            seed:            Random seed for reproducibility.
        """
        if shape_types is None:
            shape_types = [0, 1, 2]

        self.num_samples = num_samples
        self.num_points_final = num_points_final
        self.num_time_steps = num_time_steps
        self.shape_types = shape_types
        self.noise_std = noise_std
        self.scatter_radius = scatter_radius
        self.hint_ratio = hint_ratio

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.revelation_counts = self._compute_revelation_schedule()
        self.ambiguity_ratios = self._compute_ambiguity_schedule()
        self.samples = self._generate_all_samples()

    # ------------------------------------------------------------------
    # Shape generators
    # ------------------------------------------------------------------

    def _generate_cube(self, n: int, size: float = 1.0) -> np.ndarray:
        """Generate *n* points on a cube surface."""
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
        """Generate *n* points on a cylinder surface."""
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
        """Generate *n* points on a sphere surface (Fibonacci)."""
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

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------

    def _compute_revelation_schedule(self) -> np.ndarray:
        """Cumulative point count at each step.  counts[-1] == N_final."""
        T = self.num_time_steps
        raw = np.power(np.linspace(0.15, 1.0, T), 1.5)
        raw = raw / raw[-1]
        counts = np.round(raw * self.num_points_final).astype(int)
        counts[-1] = self.num_points_final
        for i in range(T):
            counts[i] = max(counts[i], 8)
            if i > 0:
                counts[i] = max(counts[i], counts[i - 1])
        return counts

    def _compute_ambiguity_schedule(self) -> np.ndarray:
        """Fraction of "ambiguous" (randomly scattered) points per step."""
        return np.linspace(0.9, 0.0, self.num_time_steps).astype(np.float32)

    # ------------------------------------------------------------------
    # Episode generation
    # ------------------------------------------------------------------

    def _generate_episode(self, shape_type: int) -> Dict:
        """Generate a single episode."""
        gen = {0: self._generate_cube,
               1: self._generate_cylinder,
               2: self._generate_sphere}

        # 1. Target shape
        target = gen[shape_type](self.num_points_final)
        target += np.random.normal(0, self.noise_std, target.shape)

        # 2. Initial scattered points (step 0)
        n0 = int(self.revelation_counts[0])
        n_hint = max(1, int(n0 * self.hint_ratio))
        n_scatter = n0 - n_hint

        # Scattered points: uniformly random inside a ball
        r = self.scatter_radius * np.cbrt(np.random.uniform(0, 1, n_scatter))
        theta = np.random.uniform(0, 2 * np.pi, n_scatter)
        phi = np.arccos(2 * np.random.uniform(0, 1, n_scatter) - 1)
        scattered = np.stack([r * np.sin(phi) * np.cos(theta),
                              r * np.sin(phi) * np.sin(theta),
                              r * np.cos(phi)], axis=1)

        # Hint points: sampled from target surface + extra noise
        hint_idx = np.random.choice(self.num_points_final, n_hint,
                                    replace=False)
        hints = target[hint_idx] + np.random.normal(
            0, self.noise_std * 3, (n_hint, 3))

        initial = np.vstack([scattered, hints])
        hint_mask = np.zeros(n0, dtype=bool)
        hint_mask[n_scatter:] = True

        # 3. Affordance labels
        aff = np.zeros((self.num_points_final, self.NUM_AFFORDANCES),
                       dtype=np.float32)
        aff[:, self.SHAPE_AFFORDANCES[shape_type]] = 1.0

        return {
            'initial_points': torch.from_numpy(initial).float(),
            'target_points': torch.from_numpy(target).float(),
            'target_affordances': torch.from_numpy(aff).float(),
            'shape_type': shape_type,
            'hint_mask': torch.from_numpy(hint_mask),
            'revelation_counts': torch.from_numpy(
                self.revelation_counts.copy()).long(),
            'ambiguity_schedule': torch.from_numpy(
                self.ambiguity_ratios.copy()).float(),
        }

    def _generate_all_samples(self) -> List[Dict]:
        samples = []
        for _ in range(self.num_samples):
            st = int(np.random.choice(self.shape_types))
            samples.append(self._generate_episode(st))
        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


# ======================================================================
# Collate function
# ======================================================================

def collate_temporal_batch(batch: List[Dict]) -> Dict:
    """
    Collate a batch of episodes into tensors.

    Because each sample has the same N_final and the same revelation
    schedule, we can stack most tensors directly.  The initial points
    are collated into graph format (concatenated + batch index) because
    N_0 may vary slightly across samples due to rounding.

    Returns:
        Dict with:
            'initial_points': (N_total_0, 3) concatenated
            'initial_batch':  (N_total_0,) batch assignment
            'hint_mask':      (N_total_0,) bool
            'target_points':  (B, N_final, 3)
            'target_affordances': (B, N_final, NUM_AFFORDANCES)
            'shape_types':    list of int
            'revelation_counts': (T,) int  (shared across batch)
            'ambiguity_schedule': (B, T)
    """
    B = len(batch)

    # Initial points → graph format
    init_pts, init_batch, hint_masks = [], [], []
    for i, s in enumerate(batch):
        pts = s['initial_points']
        init_pts.append(pts)
        init_batch.append(torch.full((pts.size(0),), i, dtype=torch.long))
        hint_masks.append(s['hint_mask'])

    # Target points (all same N_final → stackable)
    target_pts = torch.stack([s['target_points'] for s in batch])
    target_aff = torch.stack([s['target_affordances'] for s in batch])

    return {
        'initial_points': torch.cat(init_pts, dim=0),
        'initial_batch': torch.cat(init_batch, dim=0),
        'hint_mask': torch.cat(hint_masks, dim=0),
        'target_points': target_pts,
        'target_affordances': target_aff,
        'shape_types': [s['shape_type'] for s in batch],
        'revelation_counts': batch[0]['revelation_counts'],  # shared
        'ambiguity_schedule': torch.stack(
            [s['ambiguity_schedule'] for s in batch]),
    }


# ======================================================================
# Self-test
# ======================================================================

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    print("Testing TemporalShapeDataset (active assembly)...")

    ds = TemporalShapeDataset(num_samples=10, num_points_final=256,
                              num_time_steps=8, seed=42)
    print(f"Dataset size: {len(ds)}")

    s = ds[0]
    print(f"\nSample keys: {list(s.keys())}")
    print(f"Shape type: {s['shape_type']} "
          f"({TemporalShapeDataset.SHAPE_NAMES[s['shape_type']]})")
    print(f"Initial points: {s['initial_points'].shape}")
    print(f"  hints: {s['hint_mask'].sum().item()} / "
          f"{s['initial_points'].size(0)}")
    print(f"Target points:  {s['target_points'].shape}")
    print(f"Revelation counts: {s['revelation_counts'].tolist()}")
    print(f"Ambiguity schedule: "
          f"{[f'{v:.2f}' for v in s['ambiguity_schedule'].tolist()]}")

    loader = DataLoader(ds, batch_size=4,
                        collate_fn=collate_temporal_batch)
    b = next(iter(loader))
    print(f"\nBatch keys: {list(b.keys())}")
    print(f"Initial points: {b['initial_points'].shape}, "
          f"batch: {b['initial_batch'].shape}")
    print(f"Target points:  {b['target_points'].shape}")
    print(f"Ambiguity schedule: {b['ambiguity_schedule'].shape}")

    print("\nTemporalShapeDataset (active assembly) test passed!")
