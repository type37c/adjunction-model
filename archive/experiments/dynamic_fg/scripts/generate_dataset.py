"""
Generate temporal point cloud dataset for dynamic F/G training

Strategy:
- Collect trajectories from random policy in Reaching environment
- Each sample: (point_cloud_t, point_cloud_t+1, action, success)
- F learns: affordance from temporal point clouds (reachability)
- G learns: predict next state (motion prediction)
- η becomes: prediction error (reachability difficulty)
"""

import sys
import os
sys.path.append('/home/ubuntu/adjunction-model/experiments/step2_v2_redesign')

import numpy as np
import torch
from env.reaching_env import ReachingEnv
from tqdm import tqdm
import pickle


def collect_trajectory(env, max_steps=100):
    """
    Collect a single trajectory with temporal point clouds
    
    Returns:
        trajectory: list of (pc_t, pc_t+1, action, ee_pos_t, ee_pos_t+1, obj_pos, success)
    """
    obs = env.reset()
    trajectory = []
    
    for step in range(max_steps):
        # Random action
        action = np.random.uniform(-1, 1, size=3)
        
        # Get current point cloud and state
        pc_t = obs['point_cloud'].copy()
        state_t = obs['state_vector'].copy()
        ee_pos_t = state_t[0:3]
        obj_pos = state_t[3:6]
        
        # Take action
        obs_next, reward, done, info = env.step(action)
        
        # Get next point cloud and state
        pc_t1 = obs_next['point_cloud'].copy()
        state_t1 = obs_next['state_vector'].copy()
        ee_pos_t1 = state_t1[0:3]
        
        # Record transition
        trajectory.append({
            'pc_t': pc_t,
            'pc_t1': pc_t1,
            'action': action,
            'ee_pos_t': ee_pos_t,
            'ee_pos_t1': ee_pos_t1,
            'obj_pos': obj_pos,
            'distance_t': np.linalg.norm(ee_pos_t - obj_pos),
            'distance_t1': np.linalg.norm(ee_pos_t1 - obj_pos),
            'success': info['success']
        })
        
        obs = obs_next
        
        if done:
            break
    
    return trajectory


def generate_dataset(num_trajectories=1000, output_dir='data'):
    """
    Generate dataset of temporal point clouds
    
    Args:
        num_trajectories: Number of trajectories to collect
        output_dir: Output directory
    """
    print(f"Generating dataset with {num_trajectories} trajectories...")
    
    env = ReachingEnv(gui=False, use_point_cloud=True, max_steps=100)
    
    all_samples = []
    total_steps = 0
    successful_episodes = 0
    
    for i in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        trajectory = collect_trajectory(env)
        all_samples.extend(trajectory)
        total_steps += len(trajectory)
        
        # Check if any step was successful
        if any(t['success'] for t in trajectory):
            successful_episodes += 1
    
    env.close()
    
    print(f"\nDataset statistics:")
    print(f"  Total trajectories: {num_trajectories}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Avg steps per trajectory: {total_steps / num_trajectories:.1f}")
    print(f"  Successful episodes: {successful_episodes} ({100*successful_episodes/num_trajectories:.1f}%)")
    
    # Calculate distance statistics
    distances = [s['distance_t'] for s in all_samples]
    print(f"  Distance statistics:")
    print(f"    Mean: {np.mean(distances):.3f}")
    print(f"    Std: {np.std(distances):.3f}")
    print(f"    Min: {np.min(distances):.3f}")
    print(f"    Max: {np.max(distances):.3f}")
    
    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'temporal_pointcloud_dataset.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_samples, f)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    return all_samples


def visualize_sample(sample_idx=0, dataset_path='data/temporal_pointcloud_dataset.pkl'):
    """
    Visualize a sample from the dataset
    """
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    sample = dataset[sample_idx]
    
    print(f"\nSample {sample_idx}:")
    print(f"  Point cloud shape: {sample['pc_t'].shape}")
    print(f"  Action: {sample['action']}")
    print(f"  EE position t: {sample['ee_pos_t']}")
    print(f"  EE position t+1: {sample['ee_pos_t1']}")
    print(f"  Object position: {sample['obj_pos']}")
    print(f"  Distance t: {sample['distance_t']:.3f}")
    print(f"  Distance t+1: {sample['distance_t1']:.3f}")
    print(f"  Distance change: {sample['distance_t1'] - sample['distance_t']:.3f}")
    print(f"  Success: {sample['success']}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-trajectories', type=int, default=1000,
                        help='Number of trajectories to collect')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize a sample after generation')
    args = parser.parse_args()
    
    # Generate dataset
    dataset = generate_dataset(args.num_trajectories, args.output_dir)
    
    # Visualize
    if args.visualize:
        visualize_sample(dataset_path=os.path.join(args.output_dir, 'temporal_pointcloud_dataset.pkl'))
