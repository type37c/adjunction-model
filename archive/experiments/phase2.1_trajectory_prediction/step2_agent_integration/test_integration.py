"""
Test script to verify environment and agent integration

Tests that all components work together before full training.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import numpy as np
from env import ReachingEnv
from agent import AgentCBaseline
from src.models.functor_f_v2 import FunctorF_v2
from src.models.functor_g import FunctorG


def test_environment():
    """Test PyBullet environment"""
    print("Testing environment...")
    
    env = ReachingEnv(gui=False, use_point_cloud=True)
    
    # Reset
    obs = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"  joint_angles: {obs['joint_angles'].shape}")
    print(f"  joint_velocities: {obs['joint_velocities'].shape}")
    print(f"  object_position: {obs['object_position'].shape}")
    print(f"  point_cloud: {obs['point_cloud'].shape}")
    
    # Step
    action = np.random.randn(env.get_action_space_dim())
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward:.4f}, Done: {done}, Success: {info['success']}")
    
    env.close()
    print("✓ Environment test passed\n")


def test_agent_baseline():
    """Test Agent C baseline (without F/G)"""
    print("Testing Agent C baseline...")
    
    robot_state_dim = 6  # 3 joint angles + 3 joint velocities
    object_pos_dim = 3
    action_dim = 3
    
    agent = AgentCBaseline(robot_state_dim, object_pos_dim, action_dim, hidden_dim=64)
    
    # Test forward pass
    robot_state = torch.randn(1, robot_state_dim)
    object_position = torch.randn(1, object_pos_dim)
    
    action, log_prob, value, hidden = agent.get_action(robot_state, object_position)
    print(f"Action: {action.shape}, Log prob: {log_prob.shape}, Value: {value.shape}")
    
    print("✓ Agent C baseline test passed\n")


def test_fg_integration():
    """Test F/G integration"""
    print("Testing F/G integration...")
    
    # Load Phase 1.5 checkpoint
    checkpoint_path = "../../phase1.5_fg_retraining/checkpoints/phase1.5_best.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print("Skipping F/G integration test\n")
        return
    
    # Initialize F/G
    functor_f = FunctorF_v2(
        input_dim=3,
        affordance_dim=32,
        goal_dim=4,
        k=16,
        hidden_dim=128
    )
    
    functor_g = FunctorG(
        affordance_dim=32,
        output_dim=3
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    functor_f.load_state_dict(checkpoint['functor_f_state_dict'])
    functor_g.load_state_dict(checkpoint['functor_g_state_dict'])
    
    functor_f.eval()
    functor_g.eval()
    
    # Test with point cloud
    point_cloud = torch.randn(1, 512, 3)
    goal = torch.zeros(1, 4)  # Reaching task (no specific action)
    
    with torch.no_grad():
        affordances = functor_f(point_cloud, goal=goal)  # (1, 512, 32)
        global_affordance = affordances.mean(dim=1)  # (1, 32)
    
    print(f"Point cloud: {point_cloud.shape}")
    print(f"Affordances: {affordances.shape}")
    print(f"Global affordance: {global_affordance.shape}")
    
    print("✓ F/G integration test passed\n")


def test_full_episode():
    """Test full episode with baseline agent"""
    print("Testing full episode...")
    
    env = ReachingEnv(gui=False, use_point_cloud=False)
    
    robot_state_dim = 6
    object_pos_dim = 3
    action_dim = 3
    
    agent = AgentCBaseline(robot_state_dim, object_pos_dim, action_dim, hidden_dim=64)
    
    obs = env.reset()
    hidden = None
    total_reward = 0
    
    for step in range(50):
        # Prepare observation
        robot_state = torch.FloatTensor(np.concatenate([
            obs['joint_angles'],
            obs['joint_velocities']
        ])).unsqueeze(0)
        
        object_position = torch.FloatTensor(obs['object_position']).unsqueeze(0)
        
        # Get action
        action, log_prob, value, hidden = agent.get_action(robot_state, object_position, hidden)
        action = action.detach().numpy().flatten()
        
        # Step
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode finished at step {step+1}")
            print(f"Success: {info['success']}, Distance: {info['distance']:.4f}")
            break
    
    print(f"Total reward: {total_reward:.4f}")
    
    env.close()
    print("✓ Full episode test passed\n")


def main():
    print("=" * 50)
    print("Step 2 Integration Test")
    print("=" * 50 + "\n")
    
    test_environment()
    test_agent_baseline()
    test_fg_integration()
    test_full_episode()
    
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
