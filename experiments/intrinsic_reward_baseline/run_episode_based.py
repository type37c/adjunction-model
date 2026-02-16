"""
Intrinsic Reward Baseline Experiment (Episode-Based)

This experiment faithfully reproduces the 2/13 experiment structure:
- 100 episodes × 10 steps
- Episode-based training with state reset
- F/G frozen
- Value-based Agent C training

Design principle: "Temporal persistence with appropriate boundaries (sleep)"
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.data.synthetic_dataset import SyntheticAffordanceDataset
from src.models.adjunction_model import AdjunctionModel
from src.models.value_function import ValueFunction
from src.training.train_agent_value_based import ValueBasedAgentTrainer


class ExperimentLogger:
    """Logs and visualizes experiment results"""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics
        self.metrics = {
            'episode': [],
            'total_reward': [],
            'avg_coherence': [],
            'avg_uncertainty': [],
            'avg_valence': [],
            'avg_eta': [],
            'avg_epsilon': [],
            'value_start': [],
            'value_end': [],
            'td_loss': [],
            'r_curiosity': [],
            'r_competence': [],
            'r_novelty': [],
            'r_intrinsic': [],
        }
    
    def log_episode(self, episode: int, episode_data: dict):
        """Log metrics from one episode"""
        self.metrics['episode'].append(episode)
        
        for key in ['total_reward', 'avg_coherence', 'avg_uncertainty', 
                    'avg_valence', 'avg_eta', 'avg_epsilon',
                    'value_start', 'value_end', 'td_loss',
                    'r_curiosity', 'r_competence', 'r_novelty', 'r_intrinsic']:
            self.metrics[key].append(episode_data.get(key, 0.0))
    
    def save_metrics(self):
        """Save metrics to JSON"""
        filepath = self.save_dir / "metrics.json"
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    def plot_metrics(self):
        """Plot key metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        episodes = self.metrics['episode']
        
        # Intrinsic reward
        axes[0, 0].plot(episodes, self.metrics['r_intrinsic'], 'b-', linewidth=2)
        axes[0, 0].set_title('Intrinsic Reward', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('R_intrinsic')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Coherence (η)
        axes[0, 1].plot(episodes, self.metrics['avg_coherence'], 'r-', linewidth=2)
        axes[0, 1].set_title('Coherence (η)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('η')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Valence
        axes[0, 2].plot(episodes, self.metrics['avg_valence'], 'g-', linewidth=2)
        axes[0, 2].set_title('Valence', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Valence')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Value function
        axes[1, 0].plot(episodes, self.metrics['value_start'], 'b-', label='Start', linewidth=2)
        axes[1, 0].plot(episodes, self.metrics['value_end'], 'r-', label='End', linewidth=2)
        axes[1, 0].set_title('Value Function', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Eta (η)
        axes[1, 1].plot(episodes, self.metrics['avg_eta'], 'm-', linewidth=2)
        axes[1, 1].set_title('Eta (Shape Slack)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('η')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Epsilon (ε)
        axes[1, 2].plot(episodes, self.metrics['avg_epsilon'], 'c-', linewidth=2)
        axes[1, 2].set_title('Epsilon (Affordance Slack)', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('ε')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.save_dir / "metrics.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved to {filepath}")
        plt.close()
    
    def generate_report(self):
        """Generate summary report"""
        report = []
        report.append("=" * 60)
        report.append("INTRINSIC REWARD BASELINE EXPERIMENT REPORT")
        report.append("=" * 60)
        report.append("")
        report.append("Summary Statistics:")
        report.append("-" * 60)
        
        # Final values
        for metric_name, display_name in [
            ('r_intrinsic', 'Final Intrinsic Reward'),
            ('value_start', 'Final Value (Start)'),
            ('td_loss', 'Final TD Loss'),
            ('avg_coherence', 'Final Coherence'),
            ('avg_valence', 'Final Valence'),
            ('avg_eta', 'Final Eta'),
            ('avg_epsilon', 'Final Epsilon')
        ]:
            if len(self.metrics[metric_name]) > 0:
                final_value = self.metrics[metric_name][-1]
                report.append(f"  {display_name}: {final_value:.4f}")
        
        report.append("")
        report.append("Trends:")
        report.append("-" * 60)
        
        # Trends (first 10 vs last 10)
        for metric_name in ['r_intrinsic', 'value_start', 'avg_coherence', 'avg_valence', 'avg_eta', 'avg_epsilon']:
            if len(self.metrics[metric_name]) >= 20:
                early = np.mean(self.metrics[metric_name][:10])
                late = np.mean(self.metrics[metric_name][-10:])
                change = late - early
                report.append(f"  {metric_name}: {early:.4f} → {late:.4f} (Δ {change:+.4f})")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Print to console
        print(report_text)
        
        # Save to file
        filepath = self.save_dir / "report.txt"
        with open(filepath, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to {filepath}")


def run_experiment(
    num_episodes: int = 100,
    episode_length: int = 10,
    num_shapes: int = 50,
    device: torch.device = torch.device('cpu')
):
    """
    Run intrinsic reward baseline experiment (episode-based)
    
    Args:
        num_episodes: Number of training episodes (default: 100)
        episode_length: Number of shapes per episode (default: 10)
        num_shapes: Total number of unique shapes in dataset (default: 50)
        device: Device to run on
    """
    
    print("=" * 60)
    print("INTRINSIC REWARD BASELINE EXPERIMENT (EPISODE-BASED)")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Episode length: {episode_length}")
    print(f"Unique shapes: {num_shapes}")
    print(f"Device: {device}")
    print("=" * 60)
    print()
    
    # Create dataset
    print("Creating synthetic dataset...")
    dataset = SyntheticAffordanceDataset(
        num_samples=num_shapes,
        num_points=512,
        shape_types=[0, 1, 2]  # cube, cylinder, sphere
    )
    print(f"  Created {len(dataset)} shapes")
    print()
    
    # Create model
    print("Creating model...")
    model = AdjunctionModel(
        num_affordances=5,
        num_points=512,
        f_hidden_dim=64,
        g_hidden_dim=128,
        agent_hidden_dim=256,
        agent_latent_dim=64,
        context_dim=128,
        valence_dim=32,
        valence_decay=0.1,
        alpha_curiosity=0.3,
        beta_competence=0.5,
        gamma_novelty=0.2
    ).to(device)
    
    value_function = ValueFunction(
        hidden_dim=256,
        latent_dim=64,
        valence_dim=32,
        value_hidden_dim=128
    ).to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Value function parameters: {sum(p.numel() for p in value_function.parameters()):,}")
    print()
    
    # Create trainer
    print("Creating trainer...")
    trainer = ValueBasedAgentTrainer(
        model=model,
        value_function=value_function,
        device=device,
        agent_lr=1e-4,
        value_lr=1e-3,
        fg_lr=0.0,  # Freeze F/G
        gamma=0.99,
        episode_length=episode_length,
        value_update_freq=1,
        agent_update_freq=1,
        reward_scale=1.0  # No scaling
    )
    print("  Trainer created (F/G frozen)")
    print()
    
    # Create logger
    logger = ExperimentLogger(save_dir="results")
    
    # Training loop
    print("Starting training...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        # Sample shapes for this episode
        indices = np.random.choice(len(dataset), size=episode_length, replace=True)
        episode_shapes = [dataset[i]['points'] for i in indices]
        
        # Convert to batch
        batch = torch.stack(episode_shapes).to(device)
        
        # Run episode
        episode_data = trainer.train_episode(batch)
        
        # Log
        logger.log_episode(episode, episode_data)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  R_intrinsic: {episode_data['r_intrinsic']:.4f}")
            print(f"  Value (start): {episode_data['value_start']:.4f}")
            print(f"  TD loss: {episode_data['td_loss']:.4f}")
            print(f"  Coherence: {episode_data['avg_coherence']:.4f}")
            print(f"  Valence: {episode_data['avg_valence']:.4f}")
            print(f"  Eta: {episode_data['avg_eta']:.4f}")
            print(f"  Epsilon: {episode_data['avg_epsilon']:.4f}")
    
    print("-" * 60)
    print("Training complete!")
    print()
    
    # Save results
    print("Saving results...")
    logger.save_metrics()
    logger.plot_metrics()
    logger.generate_report()
    print()
    
    # Save model
    model_path = logger.save_dir / "model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'value_function_state_dict': value_function.state_dict(),
    }, model_path)
    print(f"Model saved to {model_path}")
    print()
    
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    device = torch.device('cpu')
    run_experiment(
        num_episodes=100,
        episode_length=10,
        num_shapes=50,
        device=device
    )
