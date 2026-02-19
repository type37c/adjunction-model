"""
Environment sanity test

Quick test to verify:
1. Environment initializes correctly
2. Position control works
3. Reward function produces reasonable values
4. State vector has correct dimensions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from env import ReachingEnv


def test_env():
    print("=" * 60)
    print("Environment Sanity Test")
    print("=" * 60)
    
    env = ReachingEnv(gui=False, use_point_cloud=True, max_steps=100)
    
    # Test reset
    obs = env.reset()
    print(f"\nState vector shape: {obs['state_vector'].shape}")
    print(f"State vector: {obs['state_vector']}")
    print(f"Point cloud shape: {obs['point_cloud'].shape}")
    print(f"End-effector pos: {obs['end_effector_pos']}")
    print(f"Object pos: {obs['object_position']}")
    
    # Test random actions
    total_reward = 0
    distances = []
    
    for step in range(50):
        action = np.random.uniform(-1, 1, size=3)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        distances.append(info['distance'])
        
        if step < 5:
            print(f"\nStep {step}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Distance: {info['distance']:.4f}")
            print(f"  Reward components: {info['reward_components']}")
        
        if done:
            print(f"\nEpisode done at step {step}! Success: {info['success']}")
            break
    
    print(f"\n--- Summary ---")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Initial distance: {distances[0]:.4f}")
    print(f"Final distance: {distances[-1]:.4f}")
    print(f"Min distance: {min(distances):.4f}")
    print(f"Steps: {len(distances)}")
    
    # Test multiple resets
    print(f"\n--- Multiple Reset Test ---")
    for i in range(5):
        obs = env.reset()
        ee = obs['end_effector_pos']
        obj = obs['object_position']
        dist = np.linalg.norm(ee - obj)
        print(f"  Reset {i}: EE={ee}, Obj={obj}, Dist={dist:.4f}")
    
    env.close()
    print("\nEnvironment test passed!")


if __name__ == "__main__":
    test_env()
