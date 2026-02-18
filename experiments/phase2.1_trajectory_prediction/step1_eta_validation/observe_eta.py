"""
Step 1: Observation experiment to validate eta's physical meaning.

This script performs various actions on objects in PyBullet and records
how the reconstruction error (eta) changes.
"""

import sys
import os
import torch
import numpy as np
import csv
from datetime import datetime

# Add project root to path
sys.path.append('/home/ubuntu/adjunction-model')
sys.path.append('/home/ubuntu/adjunction-model/experiments/step1_eta_validation')

from env import SimplePyBulletEnv
from src.models.adjunction_model import FunctorF, FunctorG


def load_fg_model(checkpoint_path):
    """Load pre-trained F/G model from Phase 1."""
    print(f"Loading F/G model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize AdjunctionModel (which contains F and G)
    from src.models.adjunction_model import AdjunctionModel
    model = AdjunctionModel(
        input_dim=3,
        hidden_dim=128,
        affordance_dim=16
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
    
    # Return F and G separately for convenience
    return model.F, model.G


def compute_eta(functor_f, functor_g, points):
    """
    Compute reconstruction error eta.
    
    Args:
        functor_f: Functor F model
        functor_g: Functor G model
        points: Nx3 numpy array
    
    Returns:
        eta: Reconstruction error (scalar)
    """
    with torch.no_grad():
        # Convert to tensor (FunctorF expects Nx3, not 1xNx3)
        points_tensor = torch.from_numpy(points).float()  # Nx3
        
        # Forward pass
        affordance = functor_f(points_tensor)  # NxD
        reconstructed = functor_g(affordance)  # Nx3
        
        # Compute reconstruction error
        eta = torch.mean((reconstructed - points_tensor) ** 2).item()
    
    return eta


def action_observe_static(env, functor_f, functor_g, num_frames=10):
    """Action 1: Static observation."""
    print("  Action: Static observation")
    data = []
    
    for frame in range(num_frames):
        env.step(10)  # Step simulation
        points = env.get_point_cloud()
        eta = compute_eta(functor_f, functor_g, points)
        state = env.get_object_state()
        
        data.append({
            'frame': frame,
            'action': 'static',
            'eta': eta,
            'pos_x': state['position'][0],
            'pos_y': state['position'][1],
            'pos_z': state['position'][2],
        })
        
        if frame % 5 == 0:
            print(f"    Frame {frame}: eta = {eta:.6f}")
    
    return data


def action_change_viewpoint(env, functor_f, functor_g):
    """Action 2: Change camera viewpoint."""
    print("  Action: Change viewpoint")
    data = []
    
    # 8 viewpoints around the object
    yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    for i, yaw in enumerate(yaw_angles):
        env.set_camera_view(yaw=yaw)
        env.step(10)
        points = env.get_point_cloud()
        eta = compute_eta(functor_f, functor_g, points)
        state = env.get_object_state()
        
        data.append({
            'frame': i,
            'action': f'viewpoint_yaw{yaw}',
            'eta': eta,
            'pos_x': state['position'][0],
            'pos_y': state['position'][1],
            'pos_z': state['position'][2],
        })
        
        print(f"    Yaw {yaw}°: eta = {eta:.6f}")
    
    # Reset viewpoint
    env.set_camera_view(yaw=0)
    return data


def action_push(env, functor_f, functor_g, num_frames=30):
    """Action 3: Push object."""
    print("  Action: Push object")
    data = []
    
    # Apply horizontal force
    env.apply_force(force=[5.0, 0, 0])
    
    for frame in range(num_frames):
        env.step(10)
        points = env.get_point_cloud()
        eta = compute_eta(functor_f, functor_g, points)
        state = env.get_object_state()
        
        data.append({
            'frame': frame,
            'action': 'push',
            'eta': eta,
            'pos_x': state['position'][0],
            'pos_y': state['position'][1],
            'pos_z': state['position'][2],
        })
        
        if frame % 10 == 0:
            print(f"    Frame {frame}: eta = {eta:.6f}, pos = ({state['position'][0]:.3f}, {state['position'][1]:.3f}, {state['position'][2]:.3f})")
    
    return data


def action_topple(env, functor_f, functor_g, num_frames=50):
    """Action 4: Topple object."""
    print("  Action: Topple object")
    data = []
    
    # Apply force at top to topple
    env.apply_force(force=[3.0, 0, 0], position=[0, 0, 0.1])
    
    for frame in range(num_frames):
        env.step(10)
        points = env.get_point_cloud()
        eta = compute_eta(functor_f, functor_g, points)
        state = env.get_object_state()
        
        data.append({
            'frame': frame,
            'action': 'topple',
            'eta': eta,
            'pos_x': state['position'][0],
            'pos_y': state['position'][1],
            'pos_z': state['position'][2],
        })
        
        if frame % 10 == 0:
            print(f"    Frame {frame}: eta = {eta:.6f}, pos_z = {state['position'][2]:.3f}")
    
    return data


def action_rotate(env, functor_f, functor_g, num_frames=30):
    """Action 5: Rotate object."""
    print("  Action: Rotate object")
    data = []
    
    # Apply torque
    for frame in range(num_frames):
        env.apply_force(force=[0, 2.0, 0], position=[0.05, 0, 0])
        env.step(10)
        points = env.get_point_cloud()
        eta = compute_eta(functor_f, functor_g, points)
        state = env.get_object_state()
        
        data.append({
            'frame': frame,
            'action': 'rotate',
            'eta': eta,
            'pos_x': state['position'][0],
            'pos_y': state['position'][1],
            'pos_z': state['position'][2],
        })
        
        if frame % 10 == 0:
            print(f"    Frame {frame}: eta = {eta:.6f}")
    
    return data


def run_experiment(object_type, functor_f, functor_g, num_trials=5):
    """
    Run observation experiment for one object type.
    
    Args:
        object_type: 'cup', 'bowl', or 'box'
        functor_f: Functor F model
        functor_g: Functor G model
        num_trials: Number of trials per action
    """
    print(f"\n{'='*60}")
    print(f"Running experiment for object: {object_type}")
    print(f"{'='*60}")
    
    all_data = []
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Create environment
        env = SimplePyBulletEnv(object_type=object_type, gui=False)
        
        # Action 1: Static observation
        env.reset_object()
        data = action_observe_static(env, functor_f, functor_g)
        for row in data:
            row['trial'] = trial
            row['object'] = object_type
        all_data.extend(data)
        
        # Action 2: Change viewpoint
        env.reset_object()
        data = action_change_viewpoint(env, functor_f, functor_g)
        for row in data:
            row['trial'] = trial
            row['object'] = object_type
        all_data.extend(data)
        
        # Action 3: Push
        env.reset_object()
        data = action_push(env, functor_f, functor_g)
        for row in data:
            row['trial'] = trial
            row['object'] = object_type
        all_data.extend(data)
        
        # Action 4: Topple
        env.reset_object()
        data = action_topple(env, functor_f, functor_g)
        for row in data:
            row['trial'] = trial
            row['object'] = object_type
        all_data.extend(data)
        
        # Action 5: Rotate
        env.reset_object()
        data = action_rotate(env, functor_f, functor_g)
        for row in data:
            row['trial'] = trial
            row['object'] = object_type
        all_data.extend(data)
        
        env.close()
    
    return all_data


def save_data(data, object_type):
    """Save data to CSV file."""
    output_dir = f'/home/ubuntu/adjunction-model/experiments/step1_eta_validation/data/{object_type}'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'eta_observations_{timestamp}.csv')
    
    fieldnames = ['object', 'trial', 'action', 'frame', 'eta', 'pos_x', 'pos_y', 'pos_z']
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\nData saved to: {output_path}")
    return output_path


def main():
    """Main function."""
    print("="*60)
    print("Step 1: Eta Validation Experiment")
    print("="*60)
    
    # Load F/G model
    checkpoint_path = '/home/ubuntu/adjunction-model/experiments/step1_eta_validation/checkpoints/phase1_final.pt'
    functor_f, functor_g = load_fg_model(checkpoint_path)
    
    # Run experiments for each object type
    object_types = ['box', 'cup', 'bowl']
    
    for object_type in object_types:
        data = run_experiment(object_type, functor_f, functor_g, num_trials=5)
        save_data(data, object_type)
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == '__main__':
    main()
