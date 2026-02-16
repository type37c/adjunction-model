"""
Intrinsic Reward Baseline Experiment: Run Script

This script reproduces the 2/13 experiment using value-based training.

Key features:
- F/G frozen (no external objective)
- Agent C trained to maximize expected future intrinsic rewards
- TD learning for value function
- Episode-based training (temporal continuity)

Intrinsic rewards (2/13 configuration):
- α (curiosity): 0.0 (disabled)
- β (competence): 0.6
- γ (novelty): 0.4

Competence reward definition (2/13):
- R_competence = coherence_curr × attention × 100
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.adjunction_model import AdjunctionModel
from src.models.value_function import ValueFunction
from src.training.train_agent_value_based import ValueBasedAgentTrainer
from src.data.synthetic_dataset import SyntheticAffordanceDataset


def load_config():
    """Load experiment configuration."""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, device):
    """Create AdjunctionModel with 2/13 configuration."""
    model = AdjunctionModel(
        num_affordances=config['model']['num_affordances'],
        num_points=config['model']['num_points'],
        f_hidden_dim=config['model']['f_hidden_dim'],
        g_hidden_dim=config['model']['g_hidden_dim'],
        agent_hidden_dim=config['model']['agent_hidden_dim'],
        agent_latent_dim=config['model']['agent_latent_dim'],
        context_dim=config['model']['context_dim'],
        valence_dim=config['model']['valence_dim'],
        valence_decay=config['model']['valence_decay'],
        alpha_curiosity=config['intrinsic_rewards']['alpha_curiosity'],
        beta_competence=config['intrinsic_rewards']['beta_competence'],
        gamma_novelty=config['intrinsic_rewards']['gamma_novelty'],
        uncertainty_type=config['model']['uncertainty_type'],
        attention_temperature=config['model']['attention_temperature']
    ).to(device)
    
    # Freeze F/G (critical for 2/13 reproduction)
    model.freeze_fg()
    
    return model


def create_value_function(config, device):
    """Create value function."""
    value_fn = ValueFunction(
        hidden_dim=config['model']['agent_hidden_dim'],
        latent_dim=config['model']['agent_latent_dim'],
        valence_dim=config['model']['valence_dim'],
        value_hidden_dim=config['value_function']['hidden_dim']
    ).to(device)
    
    return value_fn


def create_trainer(model, value_fn, config, device):
    """Create ValueBasedAgentTrainer."""
    trainer = ValueBasedAgentTrainer(
        model=model,
        value_function=value_fn,
        device=device,
        agent_lr=config['training']['agent_lr'],
        value_lr=config['training']['value_lr'],
        fg_lr=0.0,  # F/G frozen
        gamma=config['training']['gamma'],
        episode_length=config['training']['episode_length'],
        value_update_freq=1,
        agent_update_freq=1
    )
    
    return trainer


def train_epoch(trainer, dataloader, epoch, config):
    """Train for one epoch."""
    total_metrics = {
        'total_reward': 0.0,
        'avg_coherence': 0.0,
        'avg_uncertainty': 0.0,
        'avg_valence': 0.0,
        'value_start': 0.0,
        'value_end': 0.0,
        'td_loss': 0.0,
        'r_curiosity': 0.0,
        'r_competence': 0.0,
        'r_novelty': 0.0,
        'r_intrinsic': 0.0
    }
    
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # batch is a dict with 'points' key
        points = batch['points']  # (batch_size, num_points, 3)
        
        # Train episode
        metrics = trainer.train_episode(points)
        
        # Accumulate metrics
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        num_batches += 1
        
        # Print progress
        if (batch_idx + 1) % config['training']['log_interval'] == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"R_intrinsic={metrics['r_intrinsic']:.4f}, "
                  f"Valence={metrics['avg_valence']:.4f}, "
                  f"Value={metrics['value_end']:.4f}")
    
    # Average metrics
    for key in total_metrics:
        total_metrics[key] /= num_batches
    
    return total_metrics


def main():
    """Main training loop."""
    # Load config
    config = load_config()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, device)
    
    # Create value function
    print("Creating value function...")
    value_fn = create_value_function(config, device)
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(model, value_fn, config, device)
    
    # Create dataset
    print("\nCreating dataset...")
    
    # Convert shape_types from string names to indices
    shape_name_to_idx = {'cube': 0, 'cylinder': 1, 'sphere': 2}
    shape_types_config = config['data']['shape_types']
    if isinstance(shape_types_config[0], str):
        shape_types = [shape_name_to_idx[name] for name in shape_types_config]
    else:
        shape_types = shape_types_config
    
    dataset = SyntheticAffordanceDataset(
        num_samples=config['data']['num_samples'],
        num_points=config['model']['num_points'],
        shape_types=shape_types
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Episode length: {config['training']['episode_length']}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    history = {
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
        'r_intrinsic': []
    }
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train epoch
        metrics = train_epoch(trainer, dataloader, epoch, config)
        
        # Store history
        for key in history:
            history[key].append(metrics[key])
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  R_intrinsic: {metrics['r_intrinsic']:.4f}")
        print(f"  R_curiosity: {metrics['r_curiosity']:.4f}")
        print(f"  R_competence: {metrics['r_competence']:.4f}")
        print(f"  R_novelty: {metrics['r_novelty']:.4f}")
        print(f"  Coherence: {metrics['avg_coherence']:.4f}")
        print(f"  Uncertainty: {metrics['avg_uncertainty']:.4f}")
        print(f"  Valence: {metrics['avg_valence']:.4f}")
        print(f"  Value (start): {metrics['value_start']:.4f}")
        print(f"  Value (end): {metrics['value_end']:.4f}")
        print(f"  TD Loss: {metrics['td_loss']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_dir = Path(__file__).parent / 'checkpoints'
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'value_fn_state_dict': value_fn.state_dict(),
                'agent_optimizer_state_dict': trainer.agent_optimizer.state_dict(),
                'value_optimizer_state_dict': trainer.value_optimizer.state_dict(),
                'history': history,
                'config': config
            }, checkpoint_path)
            
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final results
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Save history
    history_path = results_dir / 'training_history.pt'
    torch.save(history, history_path)
    print(f"\nTraining history saved: {history_path}")
    
    # Save final model
    model_path = results_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'value_fn_state_dict': value_fn.state_dict(),
        'config': config
    }, model_path)
    print(f"Final model saved: {model_path}")
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(f"  R_intrinsic: {history['r_intrinsic'][-1]:.4f}")
    print(f"  Valence: {history['avg_valence'][-1]:.4f}")
    print(f"  Coherence: {history['avg_coherence'][-1]:.4f}")
    print(f"  Uncertainty: {history['avg_uncertainty'][-1]:.4f}")
    print(f"  Value (end): {history['value_end'][-1]:.4f}")
    
    # Compare with 2/13 results
    print("\n" + "="*60)
    print("Comparison with 2/13 Results:")
    print("="*60)
    print("2/13 Final Metrics:")
    print("  Valence: 0.667 (started at 0.58)")
    print("  Coherence: 0.43 (stable)")
    print("  Acceleration: Valence growth accelerated in second half")
    print("\nCurrent Experiment:")
    print(f"  Valence: {history['avg_valence'][-1]:.4f} (started at {history['avg_valence'][0]:.4f})")
    print(f"  Coherence: {history['avg_coherence'][-1]:.4f}")
    if len(history['avg_valence']) >= 10:
        first_half_growth = history['avg_valence'][len(history['avg_valence'])//2] - history['avg_valence'][0]
        second_half_growth = history['avg_valence'][-1] - history['avg_valence'][len(history['avg_valence'])//2]
        print(f"  First half Valence growth: {first_half_growth:.4f}")
        print(f"  Second half Valence growth: {second_half_growth:.4f}")
        if second_half_growth > first_half_growth:
            print("  ✓ Valence growth accelerated (matches 2/13)")
        else:
            print("  ✗ Valence growth did not accelerate (differs from 2/13)")


if __name__ == '__main__':
    main()
