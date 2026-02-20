"""
Verification Experiment A: Test eta with Phase 1 style point clouds.

This script generates point clouds in the same format as Phase 1 training data
(filled volumes) and computes eta to verify that F/G works correctly on
in-distribution data.
"""

import sys
import torch
import numpy as np

sys.path.append('/home/ubuntu/adjunction-model')

from src.models.adjunction_model import AdjunctionModel
from src.data.synthetic_dataset import SyntheticAffordanceDataset


def load_fg_model(checkpoint_path):
    """Load pre-trained F/G model from Phase 1."""
    print(f"Loading F/G model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = AdjunctionModel(
        input_dim=3,
        hidden_dim=128,
        affordance_dim=16
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
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
    """Main verification function."""
    print("="*60)
    print("Verification Experiment A: Phase 1 Style Point Clouds")
    print("="*60)
    
    # Load F/G model
    checkpoint_path = '/home/ubuntu/adjunction-model/experiments/phase2.1_trajectory_prediction/step1_eta_validation/checkpoints/phase1_final.pt'
    functor_f, functor_g = load_fg_model(checkpoint_path)
    
    # Create Phase 1 style dataset
    print("\nGenerating Phase 1 style point clouds...")
    dataset = SyntheticAffordanceDataset(
        num_samples=30,  # 10 samples per shape type
        num_points=512,  # Same as Phase 1
        shape_types=[0, 1, 2]
    )
    
    # Compute eta for each shape type
    print("\nComputing eta for each shape type...")
    print("-"*60)
    
    results = {0: [], 1: [], 2: []}
    shape_names = {0: 'Cube', 1: 'Cylinder', 2: 'Sphere'}
    
    for i in range(len(dataset)):
        sample = dataset[i]
        points = sample['points'].numpy()
        shape_type = sample['shape_type']
        
        eta = compute_eta(functor_f, functor_g, points)
        results[shape_type].append(eta)
        
        if i % 10 == 0:
            print(f"Sample {i+1}/{len(dataset)}: {shape_names[shape_type]}, eta = {eta:.6f}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    for shape_type in [0, 1, 2]:
        etas = results[shape_type]
        print(f"\n{shape_names[shape_type]}:")
        print(f"  Mean: {np.mean(etas):.6f}")
        print(f"  Std:  {np.std(etas):.6f}")
        print(f"  Min:  {np.min(etas):.6f}")
        print(f"  Max:  {np.max(etas):.6f}")
    
    # Overall statistics
    all_etas = []
    for etas in results.values():
        all_etas.extend(etas)
    
    print(f"\nOverall:")
    print(f"  Mean: {np.mean(all_etas):.6f}")
    print(f"  Std:  {np.std(all_etas):.6f}")
    print(f"  Min:  {np.min(all_etas):.6f}")
    print(f"  Max:  {np.max(all_etas):.6f}")
    print(f"  Unique values: {len(set(all_etas))}")
    
    # Check if eta varies
    print("\n" + "="*60)
    print("Conclusion")
    print("="*60)
    
    if len(set(all_etas)) == 1:
        print("❌ FAILED: Eta is constant even for Phase 1 style data")
        print("   This suggests a problem with F/G itself, not just data distribution")
    elif np.std(all_etas) < 0.001:
        print("⚠️  WARNING: Eta has very low variance")
        print("   F/G may not be learning meaningful features")
    else:
        print("✓ SUCCESS: Eta varies across different samples")
        print("   This confirms that the problem is data distribution mismatch")
        print(f"   Coefficient of variation: {np.std(all_etas) / np.mean(all_etas):.4f}")
    
    print("="*60)


if __name__ == '__main__':
    main()
