"""
Full-Scale Training Experiment for Agent C v4

This experiment runs a comprehensive training session to analyze:
1. Agent C's emergent behavior over many episodes
2. Value function convergence
3. Intrinsic reward patterns
4. Valence evolution
5. Potential suspension structure emergence

Based on TODO.md High Priority Task #1
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
from src.models.conditional_adjunction_v4 import ConditionalAdjunctionModelV4
from src.models.value_function import ValueFunction
from src.training.train_agent_value_based import ValueBasedAgentTrainer


class ExperimentLogger:
    """Logs and visualizes experiment results"""
    
    def __init__(self, save_dir: str = "results/full_scale_training"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics
        self.metrics = {
            'episode': [],
            'total_reward': [],
            'avg_coherence': [],
            'avg_uncertainty': [],
            'avg_valence': [],
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
                    'avg_valence', 'value_start', 'value_end', 'td_loss',
                    'r_curiosity', 'r_competence', 'r_novelty', 'r_intrinsic']:
            self.metrics[key].append(episode_data.get(key, 0.0))
    
    def save_metrics(self):
        """Save metrics to JSON"""
        filepath = self.save_dir / "metrics.json"
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    def plot_intrinsic_rewards(self):
        """Plot intrinsic reward components over time"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        episodes = self.metrics['episode']
        
        # Total intrinsic reward
        axes[0, 0].plot(episodes, self.metrics['r_intrinsic'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Intrinsic Reward', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('R_intrinsic')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Curiosity
        axes[0, 1].plot(episodes, self.metrics['r_curiosity'], 'g-', linewidth=2)
        axes[0, 1].set_title('Curiosity Reward', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('R_curiosity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Competence
        axes[1, 0].plot(episodes, self.metrics['r_competence'], 'r-', linewidth=2)
        axes[1, 0].set_title('Competence Reward', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('R_competence')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Novelty
        axes[1, 1].plot(episodes, self.metrics['r_novelty'], 'm-', linewidth=2)
        axes[1, 1].set_title('Novelty Reward', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('R_novelty')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.save_dir / "intrinsic_rewards.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Intrinsic rewards plot saved to {filepath}")
        plt.close()
    
    def plot_value_function(self):
        """Plot value function evolution"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        episodes = self.metrics['episode']
        
        # Value at start and end of episodes
        axes[0].plot(episodes, self.metrics['value_start'], 'b-', label='Start', linewidth=2)
        axes[0].plot(episodes, self.metrics['value_end'], 'r-', label='End', linewidth=2)
        axes[0].set_title('Value Function (Start vs End)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # TD loss
        axes[1].plot(episodes, self.metrics['td_loss'], 'g-', linewidth=2)
        axes[1].set_title('TD Loss', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.save_dir / "value_function.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Value function plot saved to {filepath}")
        plt.close()
    
    def plot_agent_state(self):
        """Plot Agent C's internal state metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        episodes = self.metrics['episode']
        
        # Coherence
        axes[0].plot(episodes, self.metrics['avg_coherence'], 'b-', linewidth=2)
        axes[0].set_title('Average Coherence Signal', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Coherence')
        axes[0].grid(True, alpha=0.3)
        
        # Uncertainty
        axes[1].plot(episodes, self.metrics['avg_uncertainty'], 'g-', linewidth=2)
        axes[1].set_title('Average Uncertainty', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Uncertainty')
        axes[1].grid(True, alpha=0.3)
        
        # Valence
        axes[2].plot(episodes, self.metrics['avg_valence'], 'r-', linewidth=2)
        axes[2].set_title('Average Valence', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Valence')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.save_dir / "agent_state.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Agent state plot saved to {filepath}")
        plt.close()
    
    def generate_report(self):
        """Generate a summary report"""
        report = []
        report.append("=" * 60)
        report.append("FULL-SCALE TRAINING EXPERIMENT REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("Summary Statistics:")
        report.append("-" * 60)
        
        metrics_summary = {
            'Total Episodes': len(self.metrics['episode']),
            'Final Intrinsic Reward': self.metrics['r_intrinsic'][-1] if self.metrics['r_intrinsic'] else 0,
            'Final Value (Start)': self.metrics['value_start'][-1] if self.metrics['value_start'] else 0,
            'Final TD Loss': self.metrics['td_loss'][-1] if self.metrics['td_loss'] else 0,
            'Final Coherence': self.metrics['avg_coherence'][-1] if self.metrics['avg_coherence'] else 0,
            'Final Valence': self.metrics['avg_valence'][-1] if self.metrics['avg_valence'] else 0,
        }
        
        for key, value in metrics_summary.items():
            report.append(f"  {key}: {value:.4f}")
        
        report.append("")
        report.append("Trends:")
        report.append("-" * 60)
        
        # Compute trends (first 10 vs last 10 episodes)
        if len(self.metrics['episode']) >= 20:
            for metric_name in ['r_intrinsic', 'value_start', 'avg_coherence', 'avg_valence']:
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


def run_full_scale_experiment(
    num_episodes: int = 100,
    episode_length: int = 10,
    num_shapes: int = 50,
    device: torch.device = torch.device('cpu')
):
    """
    Run full-scale training experiment
    
    Args:
        num_episodes: Number of training episodes
        episode_length: Number of shapes per episode
        num_shapes: Total number of unique shapes in dataset
        device: Device to run on
    """
    
    print("=" * 60)
    print("FULL-SCALE TRAINING EXPERIMENT")
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
    model = ConditionalAdjunctionModelV4(
        num_affordances=5,
        num_points=512,
        f_hidden_dim=64,
        g_hidden_dim=128,
        agent_hidden_dim=256,
        agent_latent_dim=64,
        context_dim=128,
        valence_dim=32,
        valence_decay=0.1,
        alpha_curiosity=0.0,  # Disabled (sign issue)
        beta_competence=0.6,  # Increased
        gamma_novelty=0.4     # Increased
    ).to(device)
    
    value_function = ValueFunction(
        hidden_dim=256,  # agent_hidden_dim
        latent_dim=64,   # agent_latent_dim
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
        agent_update_freq=1
    )
    print("  Trainer created (F/G frozen)")
    print()
    
    # Create logger
    logger = ExperimentLogger()
    
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
    
    print("-" * 60)
    print("Training complete!")
    print()
    
    # Save results
    print("Saving results...")
    logger.save_metrics()
    logger.plot_intrinsic_rewards()
    logger.plot_value_function()
    logger.plot_agent_state()
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


if __name__ == "__main__":
    # Run experiment
    run_full_scale_experiment(
        num_episodes=100,
        episode_length=10,
        num_shapes=50,
        device=torch.device('cpu')
    )
