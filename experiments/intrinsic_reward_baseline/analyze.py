"""
Analyze Intrinsic Reward Baseline Experiment Results

This script generates visualizations and analysis for the intrinsic reward baseline experiment.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def load_metrics(results_dir: str) -> dict:
    """Load metrics from JSON file."""
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def plot_metrics(metrics: dict, save_path: str):
    """Generate comprehensive plots of all metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Intrinsic Reward Baseline Experiment Results', fontsize=16)
    
    epochs = list(range(1, len(metrics['loss']) + 1))
    
    # Row 1: Losses
    axes[0, 0].plot(epochs, metrics['loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, metrics['kl_loss'], 'g-', linewidth=2)
    axes[0, 1].set_title('KL Divergence Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('KL Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(epochs, metrics['value_loss'], 'r-', linewidth=2)
    axes[0, 2].set_title('Value Function Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Value Loss')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Intrinsic Rewards
    axes[1, 0].plot(epochs, metrics['intrinsic_reward'], 'purple', linewidth=2, label='Total')
    axes[1, 0].set_title('Total Intrinsic Reward')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].plot(epochs, metrics['competence_reward'], 'orange', linewidth=2, label='Competence')
    axes[1, 1].plot(epochs, metrics['novelty_reward'], 'cyan', linewidth=2, label='Novelty')
    axes[1, 1].set_title('Intrinsic Reward Components')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Competence vs Novelty contribution
    competence_contrib = np.array(metrics['competence_reward'])
    novelty_contrib = np.array(metrics['novelty_reward'])
    total_contrib = competence_contrib + novelty_contrib
    competence_pct = (competence_contrib / (total_contrib + 1e-8)) * 100
    novelty_pct = (novelty_contrib / (total_contrib + 1e-8)) * 100
    
    axes[1, 2].plot(epochs, competence_pct, 'orange', linewidth=2, label='Competence %')
    axes[1, 2].plot(epochs, novelty_pct, 'cyan', linewidth=2, label='Novelty %')
    axes[1, 2].set_title('Reward Contribution (%)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Percentage')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    axes[1, 2].set_ylim([0, 100])
    
    # Row 3: Slack Signals
    axes[2, 0].plot(epochs, metrics['valence'], 'magenta', linewidth=2)
    axes[2, 0].set_title('Valence (Purpose Space P Axis 3)')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Valence')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(epochs, metrics['coherence'], 'brown', linewidth=2, label='Coherence')
    axes[2, 1].set_title('Coherence (Breakdown Signal)')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Coherence')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    axes[2, 2].plot(epochs, metrics['eta'], 'blue', linewidth=2, label='η (unit)')
    axes[2, 2].plot(epochs, metrics['epsilon'], 'red', linewidth=2, label='ε (counit)')
    axes[2, 2].set_title('Slack Signals (η and ε)')
    axes[2, 2].set_xlabel('Epoch')
    axes[2, 2].set_ylabel('Slack')
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def print_summary(metrics: dict):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    # Initial and final values
    print("\nKey Metrics (Initial → Final):")
    print(f"  Intrinsic Reward: {metrics['intrinsic_reward'][0]:.6f} → {metrics['intrinsic_reward'][-1]:.6f}")
    ir_change = (metrics['intrinsic_reward'][-1] / (metrics['intrinsic_reward'][0] + 1e-8) - 1) * 100
    print(f"    Change: {ir_change:+.1f}%")
    
    print(f"  Valence: {metrics['valence'][0]:.6f} → {metrics['valence'][-1]:.6f}")
    val_change = (metrics['valence'][-1] / (metrics['valence'][0] + 1e-8) - 1) * 100
    print(f"    Change: {val_change:+.1f}%")
    
    print(f"  Coherence: {metrics['coherence'][0]:.6f} → {metrics['coherence'][-1]:.6f}")
    coh_change = (metrics['coherence'][-1] / (metrics['coherence'][0] + 1e-8) - 1) * 100
    print(f"    Change: {coh_change:+.1f}%")
    
    # Reward composition (final epoch)
    comp_final = metrics['competence_reward'][-1]
    nov_final = metrics['novelty_reward'][-1]
    total_final = comp_final + nov_final
    
    print("\nReward Composition (Final Epoch):")
    print(f"  Competence: {comp_final:.6f} ({comp_final/(total_final+1e-8)*100:.1f}%)")
    print(f"  Novelty: {nov_final:.6f} ({nov_final/(total_final+1e-8)*100:.1f}%)")
    
    # Comparison with 2/13
    print("\n" + "="*60)
    print("COMPARISON WITH 2/13 RESULTS")
    print("="*60)
    print("\nExpected (2/13):")
    print("  Intrinsic Reward: +1584%")
    print("  Valence: +15% (0.58 → 0.667)")
    print("  Coherence: stable (~0.43)")
    print("  Competence contribution: 59.5%")
    
    print("\nActual (This Run):")
    print(f"  Intrinsic Reward: {ir_change:+.1f}%")
    print(f"  Valence: {val_change:+.1f}%")
    print(f"  Coherence: {coh_change:+.1f}% (mean: {np.mean(metrics['coherence']):.3f})")
    print(f"  Competence contribution: {comp_final/(total_final+1e-8)*100:.1f}%")
    
    # Valence growth rate analysis
    valence_arr = np.array(metrics['valence'])
    if len(valence_arr) >= 50:
        first_half_growth = valence_arr[25] - valence_arr[0]
        second_half_growth = valence_arr[-1] - valence_arr[25]
        print("\nValence Growth Analysis:")
        print(f"  First half growth: {first_half_growth:+.3f}")
        print(f"  Second half growth: {second_half_growth:+.3f}")
        if second_half_growth > first_half_growth:
            print("  ✓ Accelerating growth detected (matches 2/13)")
        else:
            print("  ✗ No acceleration detected (differs from 2/13)")


def main():
    # Determine results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    
    if not os.path.exists(os.path.join(results_dir, 'metrics.json')):
        print(f"Error: No metrics.json found in {results_dir}")
        print("Please run the experiment first: python run.py")
        return
    
    print("Loading metrics...")
    metrics = load_metrics(results_dir)
    
    print("Generating plots...")
    plot_path = os.path.join(results_dir, 'analysis.png')
    plot_metrics(metrics, plot_path)
    
    print_summary(metrics)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
