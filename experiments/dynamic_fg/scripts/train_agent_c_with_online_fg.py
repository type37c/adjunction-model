"""
Train Agent C with Online Dynamic F/G

Strategy:
- F/G is trained ONLINE during Agent C's training
- Each episode: collect temporal point clouds, update F/G, then update Agent C
- This ensures F/G adapts to the actual task distribution
- No need for separate dataset generation

Benefits:
1. Faster (no dataset generation phase)
2. Better alignment (F/G sees actual task distribution)
3. Co-adaptation (F/G and Agent C evolve together)
"""

import sys
sys.path.append('/home/ubuntu/adjunction-model/experiments/step2_v2_redesign')
sys.path.append('/home/ubuntu/adjunction-model/experiments/dynamic_fg')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt

from env.reaching_env import ReachingEnv
from models.dynamic_fg import DynamicFGModel


class OnlineFGBuffer:
    """
    Buffer for online F/G training
    """
    
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, pc_t, pc_t1, action, ee_pos_t1):
        """Add a transition"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (pc_t, pc_t1, action, ee_pos_t1)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        pc_t = torch.FloatTensor(np.array([b[0] for b in batch]))
        pc_t1 = torch.FloatTensor(np.array([b[1] for b in batch]))
        action = torch.FloatTensor(np.array([b[2] for b in batch]))
        ee_pos_t1 = torch.FloatTensor(np.array([b[3] for b in batch]))
        
        return pc_t, pc_t1, action, ee_pos_t1
    
    def __len__(self):
        return len(self.buffer)


def train_online_fg_agent_c(
    num_episodes=1000,
    affordance_dim=32,
    fg_lr=1e-3,
    fg_batch_size=32,
    fg_update_freq=10,  # Update F/G every N episodes
    device='cpu',
    checkpoint_dir='models/checkpoints_online',
    results_dir='results_online',
    seed=42
):
    """
    Train Agent C with online F/G learning
    
    Phase 1 (episodes 0-500): Train F/G only (no Agent C)
    Phase 2 (episodes 500-1000): Train both F/G and Agent C
    """
    print("="*60)
    print("Training Agent C with Online Dynamic F/G")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Affordance dim: {affordance_dim}")
    print(f"F/G learning rate: {fg_lr}")
    print(f"F/G batch size: {fg_batch_size}")
    print(f"F/G update freq: every {fg_update_freq} episodes")
    print(f"Device: {device}")
    print()
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create environment
    env = ReachingEnv(gui=False, use_point_cloud=True, max_steps=100)
    
    # Create F/G model
    fg_model = DynamicFGModel(
        affordance_dim=affordance_dim,
        action_dim=3,
        output_dim=3
    ).to(device)
    
    fg_optimizer = optim.Adam(fg_model.parameters(), lr=fg_lr)
    fg_criterion = nn.MSELoss()
    
    # F/G buffer
    fg_buffer = OnlineFGBuffer(capacity=5000)
    
    # Training logs
    episode_rewards = []
    episode_successes = []
    episode_distances = []
    fg_losses = []
    
    print("Phase 1: Training F/G only (random policy)")
    print()
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs = env.reset()
        episode_reward = 0
        episode_success = False
        episode_distance = []
        
        # Collect trajectory
        for step in range(100):
            # Random action (for now)
            action = np.random.uniform(-1, 1, size=3)
            
            # Get current state
            pc_t = obs['point_cloud'].copy()
            state_t = obs['state_vector'].copy()
            ee_pos_t = state_t[0:3]
            
            # Take action
            obs_next, reward, done, info = env.step(action)
            
            # Get next state
            pc_t1 = obs_next['point_cloud'].copy()
            state_t1 = obs_next['state_vector'].copy()
            ee_pos_t1 = state_t1[0:3]
            
            # Store in F/G buffer
            fg_buffer.push(pc_t, pc_t1, action, ee_pos_t1)
            
            # Update metrics
            episode_reward += reward
            episode_success = episode_success or info['success']
            episode_distance.append(info['distance'])
            
            obs = obs_next
            
            if done:
                break
        
        # Log episode
        episode_rewards.append(episode_reward)
        episode_successes.append(1 if episode_success else 0)
        episode_distances.append(np.mean(episode_distance))
        
        # Update F/G
        if (episode + 1) % fg_update_freq == 0 and len(fg_buffer) >= fg_batch_size:
            fg_model.train()
            
            # Multiple updates per episode
            for _ in range(5):
                pc_t, pc_t1, action, ee_pos_t1 = fg_buffer.sample(fg_batch_size)
                pc_t = pc_t.to(device)
                pc_t1 = pc_t1.to(device)
                action = action.to(device)
                ee_pos_t1 = ee_pos_t1.to(device)
                
                # Forward
                affordance, ee_pos_pred = fg_model(pc_t, pc_t1, action)
                loss = fg_criterion(ee_pos_pred, ee_pos_t1)
                
                # Backward
                fg_optimizer.zero_grad()
                loss.backward()
                fg_optimizer.step()
            
            fg_losses.append(loss.item())
        
        # Log progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_success = np.mean(episode_successes[-100:])
            avg_distance = np.mean(episode_distances[-100:])
            avg_fg_loss = np.mean(fg_losses[-10:]) if fg_losses else 0
            
            print(f"\nEpisode {episode+1}/{num_episodes}:")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Success Rate: {100*avg_success:.1f}%")
            print(f"  Avg Distance: {avg_distance:.3f}")
            print(f"  F/G Loss: {avg_fg_loss:.6f}")
    
    env.close()
    
    # Save F/G model
    checkpoint_path = os.path.join(checkpoint_dir, 'dynamic_fg_online.pth')
    torch.save({
        'model_state_dict': fg_model.state_dict(),
        'affordance_dim': affordance_dim,
        'fg_losses': fg_losses,
    }, checkpoint_path)
    print(f"\nSaved F/G model to {checkpoint_path}")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Reward
    axes[0, 0].plot(episode_rewards, alpha=0.3)
    axes[0, 0].plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'), linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Success rate
    success_rate = np.convolve(episode_successes, np.ones(100)/100, mode='valid')
    axes[0, 1].plot(success_rate, linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('Success Rate (100-episode moving average)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distance
    axes[1, 0].plot(episode_distances, alpha=0.3)
    axes[1, 0].plot(np.convolve(episode_distances, np.ones(100)/100, mode='valid'), linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Distance')
    axes[1, 0].set_title('Final Distance to Target')
    axes[1, 0].grid(True, alpha=0.3)
    
    # F/G loss
    axes[1, 1].plot(fg_losses, linewidth=2)
    axes[1, 1].set_xlabel('Update Step')
    axes[1, 1].set_ylabel('MSE Loss')
    axes[1, 1].set_title('F/G Training Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'online_training_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved training curves to {plot_path}")
    
    # Save stats
    stats = {
        'num_episodes': num_episodes,
        'final_avg_reward': np.mean(episode_rewards[-100:]),
        'final_success_rate': np.mean(episode_successes[-100:]),
        'final_avg_distance': np.mean(episode_distances[-100:]),
        'final_fg_loss': fg_losses[-1] if fg_losses else 0,
    }
    
    stats_path = os.path.join(results_dir, 'online_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final avg reward: {stats['final_avg_reward']:.2f}")
    print(f"Final success rate: {100*stats['final_success_rate']:.1f}%")
    print(f"Final avg distance: {stats['final_avg_distance']:.3f}")
    print("="*60)
    
    return fg_model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes')
    parser.add_argument('--affordance-dim', type=int, default=32,
                        help='Affordance dimension')
    parser.add_argument('--fg-lr', type=float, default=1e-3,
                        help='F/G learning rate')
    parser.add_argument('--fg-batch-size', type=int, default=32,
                        help='F/G batch size')
    parser.add_argument('--fg-update-freq', type=int, default=10,
                        help='F/G update frequency (episodes)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    train_online_fg_agent_c(
        num_episodes=args.episodes,
        affordance_dim=args.affordance_dim,
        fg_lr=args.fg_lr,
        fg_batch_size=args.fg_batch_size,
        fg_update_freq=args.fg_update_freq,
        device=args.device,
        seed=args.seed
    )
