"""
Step 2 v2 実験実行スクリプト

Variant A (Baseline) と Variant B (F/G-enhanced) を順番に訓練し、
結果を比較する。

実験の目的:
- F/G特徴量がAgent Cの学習を促進するかを検証
- ηの変化率（Δη）が内発的報酬として有効かを検証
- 「随伴の成立」を実験的に確認する
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import json
import matplotlib.pyplot as plt
import numpy as np

from train import train


def run_comparison_experiment():
    """Run baseline vs F/G-enhanced comparison"""
    
    fg_checkpoint = os.path.join(
        os.path.dirname(__file__), '..', 
        'phase1.5_fg_retraining', 'checkpoints', 'phase1.5_best.pth'
    )
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    num_episodes = 1500
    
    # === Variant A: Baseline ===
    print("\n" + "=" * 70)
    print("VARIANT A: BASELINE (no F/G features)")
    print("=" * 70)
    
    baseline_stats = train(
        mode='baseline',
        num_episodes=num_episodes,
        update_interval=10,
        checkpoint_dir='checkpoints_baseline',
        results_dir=results_dir,
        fg_checkpoint_path=None,
        device='cpu',
        seed=42,
    )
    
    # === Variant B: F/G-enhanced ===
    print("\n" + "=" * 70)
    print("VARIANT B: F/G-ENHANCED (with affordance features + η)")
    print("=" * 70)
    
    fg_stats = train(
        mode='fg_enhanced',
        num_episodes=num_episodes,
        update_interval=10,
        checkpoint_dir='checkpoints_with_fg',
        results_dir=results_dir,
        fg_checkpoint_path=fg_checkpoint,
        device='cpu',
        seed=42,
    )
    
    # === Comparison ===
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    comparison = {
        'baseline': baseline_stats,
        'fg_enhanced': fg_stats,
    }
    
    print(f"\n{'Metric':<30} {'Baseline':>15} {'F/G-Enhanced':>15}")
    print("-" * 60)
    print(f"{'Final Avg Reward':<30} {baseline_stats['final_avg_reward']:>15.2f} {fg_stats['final_avg_reward']:>15.2f}")
    print(f"{'Final Success Rate':<30} {baseline_stats['final_success_rate']:>15.2%} {fg_stats['final_success_rate']:>15.2%}")
    print(f"{'Final Avg Distance':<30} {baseline_stats['final_avg_distance']:>15.4f} {fg_stats['final_avg_distance']:>15.4f}")
    print(f"{'Total Successes':<30} {baseline_stats['total_successes']:>15d} {fg_stats['total_successes']:>15d}")
    print(f"{'Training Time (s)':<30} {baseline_stats['elapsed_seconds']:>15.1f} {fg_stats['elapsed_seconds']:>15.1f}")
    
    # Save comparison
    with open(os.path.join(results_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Plot comparison
    plot_comparison(results_dir)
    
    print(f"\nResults saved to {results_dir}/")


def plot_comparison(results_dir):
    """Plot comparison between baseline and F/G-enhanced"""
    
    # Load data
    baseline_ckpt = None
    fg_ckpt = None
    
    import torch
    
    baseline_path = os.path.join('checkpoints_baseline', 'final_model.pth')
    fg_path = os.path.join('checkpoints_with_fg', 'final_model.pth')
    
    if os.path.exists(baseline_path):
        baseline_ckpt = torch.load(baseline_path, map_location='cpu', weights_only=False)
    if os.path.exists(fg_path):
        fg_ckpt = torch.load(fg_path, map_location='cpu', weights_only=False)
    
    if baseline_ckpt is None or fg_ckpt is None:
        print("Cannot create comparison plot: missing checkpoints")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    window = 50
    
    # Rewards
    ax = axes[0, 0]
    for data, label, color in [
        (baseline_ckpt['episode_rewards'], 'Baseline', 'blue'),
        (fg_ckpt['episode_rewards'], 'F/G-Enhanced', 'red'),
    ]:
        ax.plot(data, alpha=0.1, color=color)
        if len(data) >= window:
            ma = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(data)), ma, color=color, linewidth=2, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success Rate
    ax = axes[0, 1]
    for data, label, color in [
        (baseline_ckpt['episode_successes'], 'Baseline', 'blue'),
        (fg_ckpt['episode_successes'], 'F/G-Enhanced', 'red'),
    ]:
        if len(data) >= window:
            sr = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(data)), sr, color=color, linewidth=2, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distance
    ax = axes[1, 0]
    for data, label, color in [
        (baseline_ckpt['episode_distances'], 'Baseline', 'blue'),
        (fg_ckpt['episode_distances'], 'F/G-Enhanced', 'red'),
    ]:
        ax.plot(data, alpha=0.1, color=color)
        if len(data) >= window:
            ma = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(data)), ma, color=color, linewidth=2, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Final Distance')
    ax.set_title('Final Distance to Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # η trajectory (F/G only)
    ax = axes[1, 1]
    if 'episode_etas' in fg_ckpt and fg_ckpt['episode_etas']:
        etas = fg_ckpt['episode_etas']
        ax.plot(etas, alpha=0.2, color='purple')
        if len(etas) >= window:
            ma = np.convolve(etas, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(etas)), ma, color='purple', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('η (Reconstruction Error)')
    ax.set_title('η Trajectory (F/G-Enhanced only)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Step 2 v2: Baseline vs F/G-Enhanced Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison_plot.png'), dpi=150)
    plt.close()
    print(f"Saved comparison plot to {results_dir}/comparison_plot.png")


if __name__ == "__main__":
    run_comparison_experiment()
