"""
Step 2 v2 Training Script

Unified training for:
  Variant A: Baseline (state vector only)
  Variant B: F/G-enhanced (state vector + affordance features + η)

実験設計:
- 同一環境、同一ハイパーパラメータで比較
- F/G-enhanced版はΔη（ηの変化率）を内発的報酬として使用
- 二重報酬: r(t) = r_ext(t) + α_int·(-Δη(t))

哲学的根拠:
- 価値関数分析v3: 「F/Gという川床の上をAgent Cという水が流れると、
  はじめて随伴が成立する」
- Baseline = 川床なしの水の流れ
- F/G-enhanced = 川床ありの水の流れ
- 差異が随伴の効果を示す
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time

from env import ReachingEnv
from agent import AgentC_v2, AgentCWithFG, PPOTrainer_v2
from src.models.functor_f_v2 import FunctorF_v2
from src.models.functor_g import FunctorG


def train(
    mode='baseline',
    num_episodes=2000,
    update_interval=10,
    checkpoint_dir='checkpoints_baseline',
    results_dir='results',
    fg_checkpoint_path=None,
    device='cpu',
    seed=42,
):
    """
    Train Agent C
    
    Args:
        mode: 'baseline' or 'fg_enhanced'
        num_episodes: Number of training episodes
        update_interval: Update agent every N episodes
        checkpoint_dir: Directory to save checkpoints
        results_dir: Directory to save results
        fg_checkpoint_path: Path to F/G checkpoint (required for fg_enhanced)
        device: Device
        seed: Random seed
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize environment
    use_point_cloud = (mode == 'fg_enhanced')
    env = ReachingEnv(gui=False, use_point_cloud=use_point_cloud, max_steps=100)
    
    # Initialize agent
    if mode == 'baseline':
        agent = AgentC_v2(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=256,
            affordance_dim=0,
            eta_dim=0,
        )
        agent.to(device)
        
        trainer = PPOTrainer_v2(
            agent=agent,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=10,
            mini_batch_size=64,
            target_kl=0.015,
            normalize_rewards=True,
            alpha_intrinsic=0.0,
        )
    
    elif mode == 'fg_enhanced':
        assert fg_checkpoint_path is not None, "F/G checkpoint required"
        
        # Load F/G models
        ckpt = torch.load(fg_checkpoint_path, map_location=device)
        
        functor_f = FunctorF_v2(
            input_dim=3, affordance_dim=32, goal_dim=4, k=16, hidden_dim=128
        )
        functor_f.load_state_dict(ckpt['functor_f_state_dict'])
        
        functor_g = FunctorG(affordance_dim=32, output_dim=3)
        functor_g.load_state_dict(ckpt['functor_g_state_dict'])
        
        agent = AgentCWithFG(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=256,
            affordance_dim=32,
            functor_f=functor_f,
            functor_g=functor_g,
            freeze_fg=True,
            use_eta=True,
        )
        agent.to(device)
        
        trainer = PPOTrainer_v2(
            agent=agent,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=10,
            mini_batch_size=64,
            target_kl=0.015,
            normalize_rewards=True,
            alpha_intrinsic=0.1,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters() if p.requires_grad):,}")
    
    # Training loop
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    episode_distances = []
    episode_etas = []
    
    best_avg_reward = -np.inf
    start_time = time.time()
    
    with tqdm(total=num_episodes, desc=f'Training {mode}') as pbar:
        for episode in range(num_episodes):
            obs = env.reset()
            
            if mode == 'fg_enhanced':
                agent.reset_eta_tracking()
            
            episode_reward = 0
            episode_length = 0
            success = False
            ep_etas = []
            
            for step in range(env.max_steps):
                # Prepare state
                state = torch.FloatTensor(obs['state_vector']).to(device)
                
                if mode == 'fg_enhanced':
                    point_cloud = torch.FloatTensor(obs['point_cloud']).to(device)
                    action, log_prob, value, eta, delta_eta, affordance = agent.get_action(
                        state, point_cloud
                    )
                    
                    eta_val = eta.item() if eta is not None else 0.0
                    delta_eta_val = delta_eta.item() if delta_eta is not None else 0.0
                    affordance_np = affordance.detach().cpu().numpy() if affordance is not None else None
                    ep_etas.append(eta_val)
                else:
                    action, log_prob, value = agent.get_action(state)
                    eta_val = 0.0
                    delta_eta_val = 0.0
                    affordance_np = None
                
                action_np = action.detach().cpu().numpy()
                log_prob_val = log_prob.detach().cpu().item()
                value_val = value.detach().cpu().item()
                
                # Step environment
                obs, reward, done, info = env.step(action_np)
                
                # Store transition
                state_np = obs['state_vector'] if not done else np.zeros(env.state_dim)
                
                # For buffer: use the state before the step
                trainer.buffer.add(
                    state=state.detach().cpu().numpy(),
                    action=action_np,
                    reward=reward,
                    done=done,
                    log_prob=log_prob_val,
                    value=value_val,
                    affordance=affordance_np,
                    eta=eta_val if mode == 'fg_enhanced' else None,
                    delta_eta=delta_eta_val if mode == 'fg_enhanced' else None,
                )
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    success = info['success']
                    break
            
            episode_rewards.append(episode_reward)
            episode_successes.append(1.0 if success else 0.0)
            episode_lengths.append(episode_length)
            episode_distances.append(info['distance'])
            if ep_etas:
                episode_etas.append(np.mean(ep_etas))
            
            # Update agent
            if (episode + 1) % update_interval == 0:
                stats = trainer.update()
                
                window = min(50, len(episode_rewards))
                if len(episode_rewards) >= window:
                    avg_reward = np.mean(episode_rewards[-window:])
                    avg_success = np.mean(episode_successes[-window:])
                    avg_distance = np.mean(episode_distances[-window:])
                    
                    postfix = {
                        'R': f'{avg_reward:.1f}',
                        'SR': f'{avg_success:.0%}',
                        'D': f'{avg_distance:.3f}',
                        'PL': f'{stats["policy_loss"]:.4f}',
                    }
                    if mode == 'fg_enhanced' and episode_etas:
                        postfix['η'] = f'{np.mean(episode_etas[-window:]):.4f}'
                    
                    pbar.set_postfix(postfix)
                    
                    # Save best model
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        torch.save({
                            'agent_state_dict': agent.state_dict(),
                            'episode': episode,
                            'avg_reward': avg_reward,
                            'avg_success': avg_success,
                        }, os.path.join(checkpoint_dir, 'best_model.pth'))
            
            pbar.update(1)
    
    env.close()
    
    elapsed = time.time() - start_time
    
    # Save final model
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'episode': num_episodes,
        'episode_rewards': episode_rewards,
        'episode_successes': episode_successes,
        'episode_lengths': episode_lengths,
        'episode_distances': episode_distances,
        'episode_etas': episode_etas,
    }, os.path.join(checkpoint_dir, 'final_model.pth'))
    
    # Compute final statistics
    window = min(100, len(episode_rewards))
    final_stats = {
        'mode': mode,
        'num_episodes': num_episodes,
        'elapsed_seconds': elapsed,
        'final_avg_reward': float(np.mean(episode_rewards[-window:])),
        'final_success_rate': float(np.mean(episode_successes[-window:])),
        'final_avg_distance': float(np.mean(episode_distances[-window:])),
        'best_avg_reward': float(best_avg_reward),
        'total_successes': int(sum(episode_successes)),
    }
    
    if episode_etas:
        final_stats['final_avg_eta'] = float(np.mean(episode_etas[-window:]))
    
    with open(os.path.join(results_dir, f'{mode}_stats.json'), 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Save training curves
    plot_training_curves(
        episode_rewards, episode_successes, episode_distances, episode_etas,
        mode, results_dir
    )
    
    print(f"\n{'='*60}")
    print(f"Training complete! ({mode})")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Final avg reward (last {window}): {final_stats['final_avg_reward']:.2f}")
    print(f"Final success rate (last {window}): {final_stats['final_success_rate']:.2%}")
    print(f"Final avg distance (last {window}): {final_stats['final_avg_distance']:.4f}")
    print(f"Total successes: {final_stats['total_successes']}/{num_episodes}")
    print(f"{'='*60}")
    
    return final_stats


def plot_training_curves(rewards, successes, distances, etas, mode, output_dir):
    """Plot and save training curves"""
    n_plots = 4 if etas else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    
    window = 50
    
    # Rewards
    axes[0].plot(rewards, alpha=0.2, color='blue')
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), ma, color='blue', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title(f'Reward ({mode})')
    axes[0].grid(True, alpha=0.3)
    
    # Success rate
    if len(successes) >= window:
        sr = np.convolve(successes, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(successes)), sr, color='green', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_title(f'Success Rate ({mode})')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    
    # Distance
    axes[2].plot(distances, alpha=0.2, color='red')
    if len(distances) >= window:
        ma = np.convolve(distances, np.ones(window)/window, mode='valid')
        axes[2].plot(range(window-1, len(distances)), ma, color='red', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Final Distance')
    axes[2].set_title(f'Final Distance ({mode})')
    axes[2].grid(True, alpha=0.3)
    
    # η (if available)
    if etas and n_plots > 3:
        axes[3].plot(etas, alpha=0.2, color='purple')
        if len(etas) >= window:
            ma = np.convolve(etas, np.ones(window)/window, mode='valid')
            axes[3].plot(range(window-1, len(etas)), ma, color='purple', linewidth=2)
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('η (Reconstruction Error)')
        axes[3].set_title(f'η Trajectory ({mode})')
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{mode}_training_curves.png'), dpi=150)
    plt.close()
    print(f"Saved training curves to {output_dir}/{mode}_training_curves.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Agent C v2')
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'fg_enhanced'],
                        help='Training mode')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of episodes')
    parser.add_argument('--update-interval', type=int, default=10,
                        help='PPO update interval (episodes)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results directory')
    parser.add_argument('--fg-checkpoint', type=str, default=None,
                        help='Path to F/G checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f'checkpoints_{args.mode}'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train(
        mode=args.mode,
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        fg_checkpoint_path=args.fg_checkpoint,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
