"""
Debug script to visualize and analyze point clouds.
"""

import sys
import numpy as np

sys.path.append('/home/ubuntu/adjunction-model')
sys.path.append('/home/ubuntu/adjunction-model/experiments/phase2.1_trajectory_prediction/step1_eta_validation')

from env import SimplePyBulletEnv


def analyze_point_cloud(points, name="Point Cloud"):
    """Analyze point cloud statistics."""
    print(f"\n{name}:")
    print(f"  Shape: {points.shape}")
    print(f"  Min: [{points[:, 0].min():.3f}, {points[:, 1].min():.3f}, {points[:, 2].min():.3f}]")
    print(f"  Max: [{points[:, 0].max():.3f}, {points[:, 1].max():.3f}, {points[:, 2].max():.3f}]")
    print(f"  Mean: [{points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f}]")
    print(f"  Std: [{points[:, 0].std():.3f}, {points[:, 1].std():.3f}, {points[:, 2].std():.3f}]")
    
    # Check if points are all the same
    unique_points = np.unique(points, axis=0)
    print(f"  Unique points: {len(unique_points)} / {len(points)}")
    
    if len(unique_points) == 1:
        print(f"  ⚠️  WARNING: All points are identical!")
        print(f"     Point value: {unique_points[0]}")


def main():
    """Main debug function."""
    print("="*60)
    print("Debug: Point Cloud Analysis")
    print("="*60)
    
    # Test box
    env = SimplePyBulletEnv(object_type='box', gui=False)
    env.step(10)
    
    # Get point cloud WITHOUT filling
    points_surface = env.get_point_cloud(num_points=512, fill_interior=False)
    analyze_point_cloud(points_surface, "Box - Surface Only")
    
    # Get point cloud WITH filling
    points_filled = env.get_point_cloud(num_points=512, fill_interior=True)
    analyze_point_cloud(points_filled, "Box - Filled")
    
    env.close()
    
    # Compare with Phase 1 style
    print("\n" + "-"*60)
    print("Comparison with Phase 1 style:")
    print("-"*60)
    
    from src.data.synthetic_dataset import SyntheticAffordanceDataset
    dataset = SyntheticAffordanceDataset(num_samples=1, num_points=512, shape_types=[0])
    phase1_points = dataset[0]['points'].numpy()
    analyze_point_cloud(phase1_points, "Phase 1 - Cube")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
