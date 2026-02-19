"""
Train Agent C v3 with trained Dynamic F/G

Agent C v3 integrates the trained Dynamic F/G:
- State: state_vector(16) + affordance(32) + η(1) = 49 dimensions
- F/G is frozen (not updated during Agent C training)
- Compare with baseline (state_vector only, 16 dimensions)
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
from agent.agent_c_v2 import AgentC_v2
from agent.ppo_v2 import PPOTrainer_v2


def train_agent_c_v3(
    num_episodes=1500,
    use_fg=True,
    fg_checkpoint='models/checkpoints_online/dynamic_fg_online.pth',
    affordance_dim=32,
    state_dim=16,
    action_dim=3,
    hidden_dim=256,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    ppo_epochs=10,
    mini_batch_size=64,
    device='cpu',
    checkpoint_dir='models/checkpoints_agent_c_v3',
    results_dir='results_agent_c_v3',
    seed=42
):
    """
    Train Agent C v3 with dynamic F/G
    """
    print("="*60)
    print(f"Training Agent C v3 {'WITH' if use_fg else 'WITHOUT'} Dynamic F/G")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Use F/G: {use_fg}")
    if use_fg:
        print(f"F/G checkpoint: {fg_checkpoint}")
        print(f"Affordance dim: {affordance_dim}")
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print()
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create environment
    env = ReachingEnv(gui=False, use_point_cloud=use_fg, max_steps=100)
    
    # Load F/G if using
    fg_model = None
    if use_fg:
        print(f"Loading F/G from {fg_checkpoint}...")
        checkpoint = torch.load(fg_checkpoint, map_location=device)
        fg_model = DynamicFGModel(
            affordance_dim=affordance_dim,
            action_dim=action_dim,
            output_dim=3
        ).to(device)
        fg_model.load_state_dict(checkpoint['model_state_dict'])
        fg_model.eval()  # Freeze F/G
        for param in fg_model.parameters():
            param.requires_grad = False
        print("F/G loaded and frozen.")
        print()
    
    # Create Agent C
    total_state_dim = state_dim
    if use_fg:
        total_state_dim += affordance_dim + 1  # +1 for η
    
    agent = AgentC_v2(
        state_dim=total_state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        affordance_dim=0,  # Already included in state_dim
        eta_dim=0
    ).to(device)
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer_v2(
        agent=agent,
        lr=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        ppo_epochs=ppo_epochs,
        mini_batch_size=mini_batch_size
    )
    
    # Training logs
    episode_rewards = []
    episode_successes = []
    episode_distances = []
    episode_etas = [] if use_fg else None
    
    # Previous observation for temporal F/G
    prev_obs = None
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs = env.reset()
        prev_obs = obs  # Initialize previous observation
        
        episode_reward = 0
        episode_success = False
        episode_distance = []
        episode_eta = [] if use_fg else None
        
        # Collect trajectory
        for step in range(100):
            # Prepare state
            state = obs['state_vector'].copy()
            
            # Add F/G features if using
            if use_fg and fg_model is not None:
                pc_t = torch.FloatTensor(prev_obs['point_cloud']).unsqueeze(0).to(device)
                pc_t1 = torch.FloatTensor(obs['point_cloud']).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    affordance = fg_model.functor_f(pc_t, pc_t1)
                    affordance_np = affordance.cpu().numpy()[0]
                
                # Compute η (will be computed after action)
                eta_placeholder = np.array([0.0])  # Will be updated after action
                
                # Concatenate
                state = np.concatenate([state, affordance_np, eta_placeholder])
            
            # Get action
            action, log_prob, value = agent.get_action(
                torch.FloatTensor(state).to(device),
                affordance=None,  # Already in state
                eta=None
            )
            action_np = action.cpu().numpy()
            
            # Compute η if using F/G
            eta_value = 0.0
            if use_fg and fg_model is not None:
                with torch.no_grad():
                    pc_t = torch.FloatTensor(prev_obs['point_cloud']).unsqueeze(0).to(device)
                    pc_t1 = torch.FloatTensor(obs['point_cloud']).unsqueeze(0).to(device)
                    action_tensor = torch.FloatTensor(action_np).unsqueeze(0).to(device)
                    
                    affordance, ee_pos_pred = fg_model(pc_t, pc_t1, action_tensor)
                    ee_pos_true = torch.FloatTensor(obs['state_vector'][0:3]).unsqueeze(0).to(device)
                    eta = fg_model.compute_eta(ee_pos_pred, ee_pos_true)
                    eta_value = eta.item()
                
                # Update state with actual η
                state[-1] = eta_value
            
            # Take action
            obs_next, reward, done, info = env.step(action_np)
            
            # Store transition
            ppo_trainer.buffer.add(
                state=state,
                action=action_np,
                reward=reward,
                log_prob=log_prob.item(),
                value=value.item(),
                done=done,
                affordance=None,  # Already in state
                eta=None
            )
            
            # Update metrics
            episode_reward += reward
            episode_success = episode_success or info['success']
            episode_distance.append(info['distance'])
            if use_fg:
                episode_eta.append(eta_value)
            
            # Update previous observation
            prev_obs = obs
            obs = obs_next
            
            if done:
                break
        
        # Log episode
        episode_rewards.append(episode_reward)
        episode_successes.append(1 if episode_success else 0)
        episode_distances.append(np.mean(episode_distance))
        if use_fg:
            episode_etas.append(np.mean(episode_eta))
        
        # Update policy
        if len(ppo_trainer.buffer) >= mini_batch_size:
            ppo_trainer.update()
        
        # Log progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_success = np.mean(episode_successes[-100:])
            avg_distance = np.mean(episode_distances[-100:])
            
            log_str = f"\nEpisode {episode+1}/{num_episodes}:"
            log_str += f"\n  Avg Reward: {avg_reward:.2f}"
            log_str += f"\n  Success Rate: {100*avg_success:.1f}%"
            log_str += f"\n  Avg Distance: {avg_distance:.3f}"
            if use_fg:
                avg_eta = np.mean(episode_etas[-100:])
                log_str += f"\n  Avg η: {avg_eta:.4f}"
            print(log_str)
    
    env.close()
    
    # Save model
    checkpoint_path = os.path.join(checkpoint_dir, 'agent_c_v3_final.pth')
    torch.save({
        'model_state_dict': agent.state_dict(),
        'use_fg': use_fg,
        'affordance_dim': affordance_dim if use_fg else 0,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': hidden_dim,
    }, checkpoint_path)
    print(f"\nSaved Agent C v3 to {checkpoint_path}")
    
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
    
    # η (if using F/G)
    if use_fg:
        axes[1, 1].plot(episode_etas, alpha=0.3)
        axes[1, 1].plot(np.convolve(episode_etas, np.ones(100)/100, mode='valid'), linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('η')
        axes[1, 1].set_title('Average η per Episode')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No F/G (Baseline)', 
                        ha='center', va='center', fontsize=16)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'agent_c_v3_training_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved training curves to {plot_path}")
    
    # Save stats
    stats = {
        'num_episodes': num_episodes,
        'use_fg': use_fg,
        'final_avg_reward': np.mean(episode_rewards[-100:]),
        'final_success_rate': np.mean(episode_successes[-100:]),
        'final_avg_distance': np.mean(episode_distances[-100:]),
    }
    if use_fg:
        stats['final_avg_eta'] = np.mean(episode_etas[-100:])
    
    stats_path = os.path.join(results_dir, 'agent_c_v3_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final avg reward: {stats['final_avg_reward']:.2f}")
    print(f"Final success rate: {100*stats['final_success_rate']:.1f}%")
    print(f"Final avg distance: {stats['final_avg_distance']:.3f}")
    if use_fg:
        print(f"Final avg η: {stats['final_avg_eta']:.4f}")
    print("="*60)
    
    return agent, stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1500,
                        help='Number of episodes')
    parser.add_argument('--use-fg', action='store_true',
                        help='Use dynamic F/G features')
    parser.add_argument('--fg-checkpoint', type=str,
                        default='models/checkpoints_online/dynamic_fg_online.pth',
                        help='Path to F/G checkpoint')
    parser.add_argument('--affordance-dim', type=int, default=32,
                        help='Affordance dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    train_agent_c_v3(
        num_episodes=args.episodes,
        use_fg=args.use_fg,
        fg_checkpoint=args.fg_checkpoint,
        affordance_dim=args.affordance_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed
    )
