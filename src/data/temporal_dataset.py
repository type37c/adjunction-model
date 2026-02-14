"""
Temporal Suspension Dataset: Progressive Shape Revelation

This module generates synthetic sequences where a 3D shape is progressively
revealed over multiple time steps. At early steps, the point cloud is ambiguous
(could be multiple shapes); as more points are added, the shape becomes clear.

Theoretical motivation (from docs/docs/docs/01_temporal_suspension.md):
- Level 0 (static): Single shape → affordance (already validated)
- Level 1 (temporal): Shape *sequence* → affordance over time
- Key insight: "One character doesn't trigger action" — context is needed

Design:
- Each sequence has T time steps
- At step t, a subset of the final point cloud is revealed
- Early steps are deliberately ambiguous (shared geometry between shapes)
- Later steps add discriminative points that resolve ambiguity
- The agent must decide at each step: "act" (classify) or "wait"

Data format:
- Each sample is a dict with:
    - 'points_sequence': list of T tensors, each (num_points_at_t, 3)
    - 'affordances': (num_points_final, NUM_AFFORDANCES) ground truth
    - 'shape_type': int, true shape category
    - 'ambiguity_schedule': (T,) float, how ambiguous the shape is at each step
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional


class TemporalShapeDataset(Dataset):
    """
    Generates sequences of progressively revealed 3D point clouds.

    Shape types (same as SyntheticAffordanceDataset for compatibility):
    - 0: Cube (graspable, liftable, stackable)
    - 1: Cylinder (graspable, rollable, containable)
    - 2: Sphere (graspable, rollable)

    Affordance types:
    - 0: grasp
    - 1: lift
    - 2: support (stack on top)
    - 3: contain
    - 4: roll

    Revelation strategy:
    - Steps 1-2: Only points from the "ambiguous core" (shared geometry)
    - Steps 3-4: Gradually add shape-specific points
    - Steps 5+:  Full shape with all discriminative features
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
        num_points_final: int = 512,
        num_time_steps: int = 8,
        shape_types: List[int] = [0, 1, 2],
        noise_std: float = 0.01,
        ambiguity_base: float = 0.3,
        seed: int = 42
    ):
        """
        Args:
            num_samples: Number of sequences to generate
            num_points_final: Number of points in the fully revealed shape
            num_time_steps: Number of time steps T in each sequence
            shape_types: List of shape type indices to include
            noise_std: Standard deviation of Gaussian noise added to points
            ambiguity_base: Base radius for the ambiguous core region
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_points_final = num_points_final
        self.num_time_steps = num_time_steps
        self.shape_types = shape_types
        self.noise_std = noise_std
        self.ambiguity_base = ambiguity_base

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Pre-generate all samples
        self.samples = self._generate_all_samples()

    # ------------------------------------------------------------------
    # Shape generators (reused from SyntheticAffordanceDataset patterns)
    # ------------------------------------------------------------------

    def _generate_cube(self, n: int, size: float = 1.0) -> np.ndarray:
        """Generate n points on a cube surface."""
        points = []
        points_per_face = max(1, n // 6)
        for axis in range(3):
            for sign in [-1, 1]:
                face = np.random.uniform(-size / 2, size / 2, (points_per_face, 3))
                face[:, axis] = sign * size / 2
                points.append(face)
        points = np.vstack(points)
        return self._adjust_count(points, n)

    def _generate_cylinder(self, n: int, radius: float = 0.5,
                           height: float = 1.0) -> np.ndarray:
        """Generate n points on a cylinder surface."""
        num_curved = int(n * 0.7)
        num_cap = (n - num_curved) // 2

        # Curved surface
        theta = np.random.uniform(0, 2 * np.pi, num_curved)
        z = np.random.uniform(-height / 2, height / 2, num_curved)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        parts = [np.stack([x, y, z], axis=1)]

        # Caps
        for z_val in [-height / 2, height / 2]:
            r = np.random.uniform(0, radius, num_cap)
            th = np.random.uniform(0, 2 * np.pi, num_cap)
            parts.append(np.stack([r * np.cos(th), r * np.sin(th),
                                   np.full(num_cap, z_val)], axis=1))
        points = np.vstack(parts)
        return self._adjust_count(points, n)

    def _generate_sphere(self, n: int, radius: float = 0.5) -> np.ndarray:
        """Generate n points on a sphere surface (Fibonacci)."""
        indices = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5 ** 0.5) * indices
        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def _adjust_count(points: np.ndarray, target: int) -> np.ndarray:
        """Pad or trim a point array to exactly *target* rows."""
        if points.shape[0] > target:
            return points[:target]
        elif points.shape[0] < target:
            shortage = target - points.shape[0]
            idx = np.random.choice(points.shape[0], shortage, replace=True)
            return np.vstack([points, points[idx]])
        return points

    # ------------------------------------------------------------------
    # Ambiguous core generation
    # ------------------------------------------------------------------

    def _generate_ambiguous_core(self, n: int) -> np.ndarray:
        """
        Generate points that are geometrically ambiguous — they could belong
        to any of the three shape categories.

        Strategy: sample from a sphere of radius ``ambiguity_base``.  All three
        shapes (cube, cylinder, sphere) share a roughly spherical core when only
        a few points are visible and noise is present.
        """
        # Uniform random points inside a sphere
        r = self.ambiguity_base * np.cbrt(np.random.uniform(0, 1, n))
        theta = np.random.uniform(0, 2 * np.pi, n)
        phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.stack([x, y, z], axis=1)

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    def _compute_revelation_schedule(self) -> np.ndarray:
        """
        Compute how many points are revealed at each time step.

        Returns:
            counts: (T,) int array — cumulative point counts at each step.
                    counts[-1] == num_points_final.
        """
        T = self.num_time_steps
        # Exponential-ish ramp: few points early, many later
        raw = np.power(np.linspace(0.15, 1.0, T), 1.5)
        raw = raw / raw[-1]  # normalise so last step = 1.0
        counts = np.round(raw * self.num_points_final).astype(int)
        counts[-1] = self.num_points_final  # guarantee exact final count
        # Ensure monotonically increasing and at least 8 points per step
        for i in range(T):
            counts[i] = max(counts[i], 8)
            if i > 0:
                counts[i] = max(counts[i], counts[i - 1])
        return counts

    def _compute_ambiguity_ratio(self) -> np.ndarray:
        """
        Compute the fraction of ambiguous-core points at each time step.

        Returns:
            ratios: (T,) float in [0, 1].  High early, low later.
        """
        T = self.num_time_steps
        # Linear decay from ~0.9 to ~0.0
        ratios = np.linspace(0.9, 0.0, T)
        return ratios

    def _generate_sequence(self, shape_type: int):
        """
        Generate a single progressive-revelation sequence.

        Returns:
            points_sequence: list of T np.ndarray, each (n_t, 3)
            full_points: (num_points_final, 3) — the complete shape
            ambiguity_schedule: (T,) float — ambiguity at each step
        """
        T = self.num_time_steps
        counts = self._compute_revelation_schedule()       # (T,)
        amb_ratios = self._compute_ambiguity_ratio()        # (T,)

        # Generate the full shape
        gen = {0: self._generate_cube,
               1: self._generate_cylinder,
               2: self._generate_sphere}
        full_points = gen[shape_type](self.num_points_final)

        # Add noise to full shape
        full_points = full_points + np.random.normal(0, self.noise_std,
                                                     full_points.shape)

        # Build the sequence
        points_sequence: List[np.ndarray] = []
        ambiguity_schedule = np.zeros(T, dtype=np.float32)

        for t in range(T):
            n_t = int(counts[t])
            n_amb = int(n_t * amb_ratios[t])   # ambiguous points
            n_real = n_t - n_amb                # shape-specific points

            parts = []
            if n_amb > 0:
                parts.append(self._generate_ambiguous_core(n_amb))
            if n_real > 0:
                # Sample from the full shape (without replacement if possible)
                idx = np.random.choice(self.num_points_final,
                                       min(n_real, self.num_points_final),
                                       replace=False)
                parts.append(full_points[idx[:n_real]])

            step_points = np.vstack(parts) if len(parts) > 1 else parts[0]
            step_points = self._adjust_count(step_points, n_t)

            # Add per-step noise
            step_points = step_points + np.random.normal(
                0, self.noise_std * (1 + amb_ratios[t]), step_points.shape)

            points_sequence.append(step_points)
            ambiguity_schedule[t] = amb_ratios[t]

        return points_sequence, full_points, ambiguity_schedule

    # ------------------------------------------------------------------
    # Affordance labels (same logic as SyntheticAffordanceDataset)
    # ------------------------------------------------------------------

    def _generate_affordance_labels(self, shape_type: int,
                                    n: int) -> np.ndarray:
        """Generate per-point affordance labels for n points."""
        labels = np.zeros((n, self.NUM_AFFORDANCES), dtype=np.float32)
        valid = self.SHAPE_AFFORDANCES[shape_type]
        labels[:, valid] = 1.0
        return labels

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _generate_all_samples(self) -> List[Dict]:
        """Pre-generate all samples."""
        samples = []
        for _ in range(self.num_samples):
            shape_type = int(np.random.choice(self.shape_types))
            pts_seq, full_pts, amb_sched = self._generate_sequence(shape_type)

            affordances = self._generate_affordance_labels(
                shape_type, self.num_points_final)

            samples.append({
                'points_sequence': [
                    torch.from_numpy(p).float() for p in pts_seq
                ],
                'points_final': torch.from_numpy(full_pts).float(),
                'affordances': torch.from_numpy(affordances).float(),
                'shape_type': shape_type,
                'ambiguity_schedule': torch.from_numpy(amb_sched).float(),
            })
        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys:
                - points_sequence: list of T tensors (n_t, 3)
                - points_final: (num_points_final, 3)
                - affordances: (num_points_final, NUM_AFFORDANCES)
                - shape_type: int
                - ambiguity_schedule: (T,) float
        """
        return self.samples[idx]


# ======================================================================
# Collate function for DataLoader
# ======================================================================

def collate_temporal_batch(batch: List[Dict]) -> Dict:
    """
    Collate a batch of temporal sequences into graph-format tensors.

    Because each time step may have a different number of points, we collate
    each step independently into graph format (concatenated points + batch
    index), following the project's tensor specification.

    Returns:
        Dict with:
            - 'points_sequence': list of T dicts, each with
                  'points' (N_t, 3) and 'batch' (N_t,)
            - 'points_final': dict with 'points' (N_final, 3) and 'batch' (N_final,)
            - 'affordances': (B, num_points_final, NUM_AFFORDANCES)
            - 'shape_types': list of int
            - 'ambiguity_schedule': (B, T)
    """
    B = len(batch)
    T = len(batch[0]['points_sequence'])

    # Collate each time step into graph format
    points_sequence = []
    for t in range(T):
        pts_list = []
        batch_idx_list = []
        for i, sample in enumerate(batch):
            pts = sample['points_sequence'][t]          # (n_t_i, 3)
            pts_list.append(pts)
            batch_idx_list.append(
                torch.full((pts.shape[0],), i, dtype=torch.long))
        points_sequence.append({
            'points': torch.cat(pts_list, dim=0),
            'batch': torch.cat(batch_idx_list, dim=0),
        })

    # Collate final points
    final_pts_list = []
    final_batch_list = []
    for i, sample in enumerate(batch):
        pts = sample['points_final']
        final_pts_list.append(pts)
        final_batch_list.append(
            torch.full((pts.shape[0],), i, dtype=torch.long))

    points_final = {
        'points': torch.cat(final_pts_list, dim=0),
        'batch': torch.cat(final_batch_list, dim=0),
    }

    # Affordances: (B, num_points_final, NUM_AFFORDANCES)
    affordances = torch.stack([s['affordances'] for s in batch], dim=0)

    # Shape types
    shape_types = [s['shape_type'] for s in batch]

    # Ambiguity schedule: (B, T)
    ambiguity_schedule = torch.stack(
        [s['ambiguity_schedule'] for s in batch], dim=0)

    return {
        'points_sequence': points_sequence,
        'points_final': points_final,
        'affordances': affordances,
        'shape_types': shape_types,
        'ambiguity_schedule': ambiguity_schedule,
    }


# ======================================================================
# Self-test
# ======================================================================

if __name__ == '__main__':
    print("Testing TemporalShapeDataset...")

    dataset = TemporalShapeDataset(
        num_samples=10,
        num_points_final=512,
        num_time_steps=8,
        seed=42
    )
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"Shape type: {sample['shape_type']} "
          f"({TemporalShapeDataset.SHAPE_NAMES[sample['shape_type']]})")
    print(f"Time steps: {len(sample['points_sequence'])}")
    for t, pts in enumerate(sample['points_sequence']):
        print(f"  Step {t}: {pts.shape[0]} points, "
              f"ambiguity={sample['ambiguity_schedule'][t]:.2f}")
    print(f"Final points: {sample['points_final'].shape}")
    print(f"Affordances: {sample['affordances'].shape}")

    # Test collate function
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4,
                        collate_fn=collate_temporal_batch)
    batch = next(iter(loader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Time steps in batch: {len(batch['points_sequence'])}")
    for t, step in enumerate(batch['points_sequence']):
        print(f"  Step {t}: points {step['points'].shape}, "
              f"batch {step['batch'].shape}")
    print(f"Final points: {batch['points_final']['points'].shape}")
    print(f"Affordances: {batch['affordances'].shape}")
    print(f"Ambiguity schedule: {batch['ambiguity_schedule'].shape}")

    print("\nTemporalShapeDataset test passed!")
