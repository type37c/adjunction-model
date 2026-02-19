"""
Training script for Variant A: With F/G affordance features

Agent C uses robot state + F/G affordance features.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from env import ReachingEnv
from agent import AgentCWithFG, PPOTrainer
from src.models.functor_f_v2 import FunctorF_v2
from src.models.functor_g import FunctorG


def train_with_fg(
    num_episodes=1000,
    update_interval=20,
    checkpoint_dir='checkpoints_with_fg',
    results_dir='results',
    fg_checkpoint='../../phase1.5_fg_retraining/checkpoints/phase1.5_best.pth',
    device='cpu'
):
    """
    Train agent with F/G affordance features
    
    Args:
        num_episodes: Number of training episodes
        update_interval: Update agent every N episodes
        checkpoint_dir: Directory to save checkpoints
        results_dir: Directory to save results
        fg_checkpoint: Path to Phase 1.5 F/G checkpoint
        device: Device to use
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize environment
    env = ReachingEnv(gui=False, use_point_cloud=True)
    
    # Load F/G models
    print("Loading F/G models...")
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
    
    checkpoint = torch.load(fg_checkpoint, map_location=device)
    functor_f.load_state_dict(checkpoint['functor_f_state_dict'])
    functor_g.load_state_dict(checkpoint['functor_g_state_dict'])
    
    functor_f.to(device)
    functor_g.to(device)
    functor_f.eval()
    functor_g.eval()
    
    print("F/G models loaded successfully")
    
    # Initialize agent
    robot_state_dim = 6  # 3 joint angles + 3 joint velocities
    affordance_dim = 32  # Global affordance feature dimension
    action_dim = 3
    
    agent = AgentCWithFG(robot_state_dim, affordance_dim, action_dim, hidden_dim=128)
    agent.to(device)
    
    # Initialize PPO trainer
    trainer = PPOTrainer(
        agent=agent.agent_c,  # Use the underlying AgentC
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        batch_size=64
    )
    
    # Training loop
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    
    best_avg_reward = -np.inf
    
    with tqdm(total=num_episodes, desc='Training with F/G') as pbar:
        for episode in range(num_episodes):
            obs = env.reset()
            hidden = None
            episode_reward = 0
            episode_length = 0
            success = False
            
            for step in range(env.max_steps):
                # Prepare robot state
                robot_state = torch.FloatTensor(np.concatenate([
                    obs['joint_angles'],
                    obs['joint_velocities']
                ])).unsqueeze(0).to(device)
                
                # Compute affordance features from point cloud
                with torch.no_grad():
                    point_cloud = torch.FloatTensor(obs['point_cloud']).unsqueeze(0).to(device)  # (1, 512, 3)
                    goal = torch.zeros(1, 4).to(device)  # Reaching task (no specific action)
                    
                    affordances = functor_f(point_cloud, goal=goal)  # (1, 512, 32)
                    global_affordance = affordances.mean(dim=1)  # (1, 32)
                
                # Combine observations for trainer
                obs_combined = torch.cat([robot_state, global_affordance], dim=-1)
                
                # Get action
                action, log_prob, value, hidden = agent.get_action(
                    robot_state, global_affordance, hidden
                )
                
                action_np = action.detach().cpu().numpy().flatten()
                log_prob_np = log_prob.detach().cpu().item()
                value_np = value.detach().cpu().item()
                
                # Step environment
                obs, reward, done, info = env.step(action_np)
                
                # Store transition
                trainer.store_transition(
                    obs_combined.detach().cpu().numpy().flatten(),
                    action_np,
                    reward,
                    done,
                    log_prob_np,
                    value_np
                )
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    success = info['success']
                    break
            
            episode_rewards.append(episode_reward)
            episode_successes.append(1.0 if success else 0.0)
            episode_lengths.append(episode_length)
            
            # Update agent
            if (episode + 1) % update_interval == 0:
                stats = trainer.update()
                
                # Compute moving averages
                window = 50
                if len(episode_rewards) >= window:
                    avg_reward = np.mean(episode_rewards[-window:])
                    avg_success = np.mean(episode_successes[-window:])
                    
                    pbar.set_postfix({
                        'reward': f'{avg_reward:.2f}',
                        'success': f'{avg_success:.2%}',
                        'policy_loss': f'{stats["policy_loss"]:.4f}'
                    })
                    
                    # Save best model
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        torch.save({
                            'agent_state_dict': agent.state_dict(),
                            'episode': episode,
                            'avg_reward': avg_reward,
                            'avg_success': avg_success
                        }, os.path.join(checkpoint_dir, 'best_model.pth'))
            
            pbar.update(1)
    
    env.close()
    
    # Save final model
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'episode': num_episodes,
        'episode_rewards': episode_rewards,
        'episode_successes': episode_successes,
        'episode_lengths': episode_lengths
    }, os.path.join(checkpoint_dir, 'final_model.pth'))
    
    # Save training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Episode reward')
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, label=f'Moving avg ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward (With F/G)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    if len(episode_successes) >= window:
        success_rate = np.convolve(episode_successes, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_successes)), success_rate)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate (With F/G)')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(episode_lengths, alpha=0.3)
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_lengths)), moving_avg)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Length (With F/G)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'with_fg_training_curves.png'), dpi=150)
    print(f"Saved training curves to {results_dir}/with_fg_training_curves.png")
    
    # Save statistics
    stats = {
        'num_episodes': num_episodes,
        'final_avg_reward': float(np.mean(episode_rewards[-50:])),
        'final_success_rate': float(np.mean(episode_successes[-50:])),
        'best_avg_reward': float(best_avg_reward),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_successes': [float(s) for s in episode_successes],
        'episode_lengths': [int(l) for l in episode_lengths]
    }
    
    with open(os.path.join(results_dir, 'with_fg_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Final avg reward (last 50): {stats['final_avg_reward']:.2f}")
    print(f"Final success rate (last 50): {stats['final_success_rate']:.2%}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train agent with F/G features')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--update-interval', type=int, default=20, help='Update interval')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_with_fg', help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    parser.add_argument('--fg-checkpoint', type=str, 
                        default='../../phase1.5_fg_retraining/checkpoints/phase1.5_best.pth',
                        help='Path to Phase 1.5 F/G checkpoint')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_with_fg(
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        fg_checkpoint=args.fg_checkpoint,
        device=device
    )


if __name__ == "__main__":
    main()
