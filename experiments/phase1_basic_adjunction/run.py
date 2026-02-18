"""
Phase 1 Experiment: Basic Adjunction Pre-training

Train F and G to learn basic shape-affordance mapping.
Agent C is frozen during this phase.

Expected outcomes:
- η (reconstruction error) decreases significantly
- F/G learn to "see" shapes and affordances
- Checkpoint saved for Phase 2

RESUME FEATURE:
- Automatically detects and loads the latest checkpoint
- Continues training from the saved epoch
"""

import torch
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.adjunction_model import AdjunctionModel
from src.data.synthetic_dataset import SyntheticAffordanceDataset
from src.training.train_phase1_basic import Phase1BasicTrainer
from torch.utils.data import DataLoader
import json
from pathlib import Path
import glob


def find_latest_checkpoint(checkpoint_dir: Path):
    """Find the latest checkpoint file."""
    checkpoint_files = glob.glob(str(checkpoint_dir / "phase1_epoch_*.pt"))
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the latest
    epoch_nums = []
    for f in checkpoint_files:
        try:
            epoch_num = int(Path(f).stem.split('_')[-1])
            epoch_nums.append((epoch_num, f))
        except:
            continue
    
    if not epoch_nums:
        return None
    
    epoch_nums.sort(reverse=True)
    return epoch_nums[0][1], epoch_nums[0][0]


def run_phase1_experiment(
    num_epochs: int = 50,
    batch_size: int = 4,
    num_train_samples: int = 100,
    num_val_samples: int = 20,
    device: torch.device = torch.device('cpu'),
    resume: bool = True
):
    """
    Run Phase 1 experiment.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        device: Device to run on
        resume: Whether to resume from checkpoint if available
    """
    
    print("=" * 60)
    print("PHASE 1: BASIC ADJUNCTION PRE-TRAINING")
    print("=" * 60)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Train samples: {num_train_samples}")
    print(f"Val samples: {num_val_samples}")
    print(f"Device: {device}")
    print(f"Resume: {resume}")
    print("=" * 60)
    print()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SyntheticAffordanceDataset(
        num_samples=num_train_samples,
        num_points=512,
        shape_types=[0, 1, 2]  # cube, cylinder, sphere
    )
    
    val_dataset = SyntheticAffordanceDataset(
        num_samples=num_val_samples,
        num_points=512,
        shape_types=[0, 1, 2]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Val dataset: {len(val_dataset)} samples")
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
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create trainer
    print("Creating trainer...")
    trainer = Phase1BasicTrainer(
        model=model,
        device=device,
        lr_fg=1e-3,
        lambda_aff=1.0,
        lambda_recon=1.0,
        lambda_coherence=0.1
    )
    print()
    
    # Check for checkpoint to resume from
    start_epoch = 1
    train_history = []
    val_history = []
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if resume:
        checkpoint_info = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_info:
            checkpoint_path, last_epoch = checkpoint_info
            print(f"Found checkpoint: {checkpoint_path}")
            print(f"Resuming from epoch {last_epoch}...")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = last_epoch + 1
            
            # Load history if available
            if 'train_history' in checkpoint:
                train_history = checkpoint.get('train_history', [])
                val_history = checkpoint.get('val_history', [])
            
            print(f"  Loaded model state from epoch {last_epoch}")
            print(f"  Continuing from epoch {start_epoch}")
            print()
    
    if start_epoch > num_epochs:
        print(f"Training already complete (epoch {start_epoch-1}/{num_epochs})")
        return
    
    # Training loop
    print("Starting training...")
    print("-" * 60)
    
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        train_history.append(train_metrics)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        val_history.append(val_metrics)
        
        # Print summary
        print(f"\n  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Recon: {train_metrics['recon']:.4f}, "
              f"Aff: {train_metrics['aff']:.4f}, "
              f"η: {train_metrics['eta']:.4f}")
        
        print(f"  Val   - Recon: {val_metrics['recon']:.4f}, "
              f"Aff: {val_metrics['aff']:.4f}, "
              f"η: {val_metrics['eta']:.4f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f"phase1_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'train_history': train_history,
                'val_history': val_history
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    print("\n" + "-" * 60)
    print("Training complete!")
    print()
    
    # Save final checkpoint
    print("Saving final checkpoint...")
    final_checkpoint_path = checkpoint_dir / "phase1_final.pt"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'train_history': train_history,
        'val_history': val_history
    }, final_checkpoint_path)
    print(f"  Saved: {final_checkpoint_path}")
    print()
    
    # Save metrics
    print("Saving metrics...")
    metrics_path = Path("results/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_history': train_history,
            'val_history': val_history
        }, f, indent=2)
    print(f"  Saved: {metrics_path}")
    print()
    
    # Print summary
    print("=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    print(f"Initial η (train): {train_history[0]['eta']:.4f}")
    print(f"Final η (train): {train_history[-1]['eta']:.4f}")
    print(f"η reduction: {(1 - train_history[-1]['eta'] / train_history[0]['eta']) * 100:.1f}%")
    print()
    print(f"Initial η (val): {val_history[0]['eta']:.4f}")
    print(f"Final η (val): {val_history[-1]['eta']:.4f}")
    print(f"η reduction: {(1 - val_history[-1]['eta'] / val_history[0]['eta']) * 100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    device = torch.device('cpu')
    run_phase1_experiment(
        num_epochs=50,
        batch_size=4,
        num_train_samples=100,
        num_val_samples=20,
        device=device,
        resume=True  # Enable automatic resume
    )
