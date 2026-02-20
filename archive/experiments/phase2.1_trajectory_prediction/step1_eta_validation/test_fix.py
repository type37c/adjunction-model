"""
Quick test to verify that point cloud filling fixes the eta invariance problem.
"""

import sys
import torch
import numpy as np

sys.path.append('/home/ubuntu/adjunction-model')
sys.path.append('/home/ubuntu/adjunction-model/experiments/phase2.1_trajectory_prediction/step1_eta_validation')

from env import SimplePyBulletEnv
from src.models.adjunction_model import AdjunctionModel


def load_fg_model(checkpoint_path):
    """Load pre-trained F/G model from Phase 1."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = AdjunctionModel(input_dim=3, hidden_dim=128, affordance_dim=16)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.F, model.G


def compute_eta(functor_f, functor_g, points):
    """Compute reconstruction error eta."""
    with torch.no_grad():
        points_tensor = torch.from_numpy(points).float()
        affordance = functor_f(points_tensor)
        reconstructed = functor_g(affordance)
        eta = torch.mean((reconstructed - points_tensor) ** 2).item()
    return eta


def main():
    """Main test function."""
    print("="*60)
    print("Quick Test: Point Cloud Filling Fix")
    print("="*60)
    
    # Load F/G model
    checkpoint_path = '/home/ubuntu/adjunction-model/experiments/phase2.1_trajectory_prediction/step1_eta_validation/checkpoints/phase1_final.pt'
    functor_f, functor_g = load_fg_model(checkpoint_path)
    
    # Test each object type
    object_types = ['box', 'cup', 'bowl']
    
    print("\nTesting with FILLED point clouds (fix applied):")
    print("-"*60)
    
    for object_type in object_types:
        env = SimplePyBulletEnv(object_type=object_type, gui=False)
        
        etas_static = []
        etas_moved = []
        
        # Static observation
        for _ in range(3):
            env.step(10)
            points = env.get_point_cloud(num_points=512, fill_interior=True)
            eta = compute_eta(functor_f, functor_g, points)
            etas_static.append(eta)
        
        # After pushing
        env.apply_force(force=[5.0, 0, 0])
        for _ in range(3):
            env.step(10)
            points = env.get_point_cloud(num_points=512, fill_interior=True)
            eta = compute_eta(functor_f, functor_g, points)
            etas_moved.append(eta)
        
        print(f"\n{object_type.capitalize()}:")
        print(f"  Static: mean={np.mean(etas_static):.6f}, std={np.std(etas_static):.6f}")
        print(f"  Moved:  mean={np.mean(etas_moved):.6f}, std={np.std(etas_moved):.6f}")
        print(f"  Difference: {abs(np.mean(etas_moved) - np.mean(etas_static)):.6f}")
        
        env.close()
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


if __name__ == '__main__':
    main()
