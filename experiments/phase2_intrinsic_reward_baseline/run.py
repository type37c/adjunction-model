"""
Phase 2 Experiment: Intrinsic Reward Baseline

Train Agent C with intrinsic rewards only (F/G frozen).

Expected outcomes:
- Agent C learns to maximize intrinsic rewards
- Value function learns to predict future rewards
- Agent explores states with low η (high coherence)
"""

import torch
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.adjunction_model import AdjunctionModel
from src.models.value_function import ValueFunction
from src.data.synthetic_dataset import SyntheticAffordanceDataset
from src.training.train_phase2_intrinsic import Phase2IntrinsicTrainer
from torch.utils.data import DataLoader
import json
from pathlib import Path


def run_phase2_experiment(
    num_epochs: int = 30,
    batch_size: int = 4,
    num_train_samples: int = 100,
    num_episodes_per_epoch: int = 10,
    device: torch.device = torch.device('cpu'),
    phase1_checkpoint: str = None
):
    """
    Run Phase 2 experiment.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        num_train_samples: Number of training samples
        num_episodes_per_epoch: Episodes per epoch for trajectory collection
        device: Device to run on
        phase1_checkpoint: Path to Phase 1 checkpoint (if None, train from scratch)
    """
    
    print("=" * 60)
    print("PHASE 2: INTRINSIC REWARD BASELINE")
    print("=" * 60)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Train samples: {num_train_samples}")
    print(f"Episodes per epoch: {num_episodes_per_epoch}")
    print(f"Device: {device}")
    print(f"Phase 1 checkpoint: {phase1_checkpoint or 'None (training from scratch)'}")
    print("=" * 60)
    print()
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = SyntheticAffordanceDataset(
        num_samples=num_train_samples,
        num_points=512,
        shape_types=[0, 1, 2]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    print("Creating model...")
    model = AdjunctionModel(
        input_dim=3,
        hidden_dim=128,
        affordance_dim=16,
        agent_hidden_dim=64,
        agent_latent_dim=32,
        agent_action_dim=8
    )
    
    # Load Phase 1 checkpoint if provided
    if phase1_checkpoint and Path(phase1_checkpoint).exists():
        print(f"Loading Phase 1 checkpoint: {phase1_checkpoint}")
        checkpoint = torch.load(phase1_checkpoint, map_location=device)
        model.F.load_state_dict(checkpoint['F_state_dict'])
        model.G.load_state_dict(checkpoint['G_state_dict'])
        print("Phase 1 checkpoint loaded successfully")
    else:
        print("No Phase 1 checkpoint provided. F/G will be randomly initialized.")
    
    # Create value function
    value_function = ValueFunction(
        state_dim=64 + 32,  # agent_hidden_dim + agent_latent_dim
        hidden_dim=64
    )
    
    # Create trainer
    trainer = Phase2IntrinsicTrainer(
        model=model,
        value_function=value_function,
        device=device,
        lr_agent=1e-4,
        lr_value=1e-3,
        gamma=0.99,
        competence_weight=1.0,
        novelty_weight=0.1
    )
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    train_history = []
    
    for epoch in range(1, num_epochs + 1):
        metrics = trainer.train_epoch(
            train_loader,
            epoch,
            num_episodes=num_episodes_per_epoch
        )
        
        train_history.append(metrics)
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Reward: {metrics['reward_total']:.4f}")
        print(f"  η: {metrics['eta']:.4f}")
        print(f"  TD Error: {metrics['td_error']:.4f}")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print("=" * 60)
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_dir = Path(__file__).parent / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"phase2_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'agent_state_dict': model.agent_c.state_dict(),
                'value_state_dict': value_function.state_dict(),
                'train_history': train_history
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "metrics.json", 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    # Run experiment
    run_phase2_experiment(
        num_epochs=30,
        batch_size=4,
        num_train_samples=100,
        num_episodes_per_epoch=10,
        device=torch.device('cpu'),
        phase1_checkpoint=None  # Set to Phase 1 checkpoint path if available
    )
