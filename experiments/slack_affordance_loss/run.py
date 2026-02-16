"""
Phase 2 Slack Preservation Experiment

This experiment trains the model with η/ε slack preservation.

Key differences from standard training:
- Removes reconstruction loss (L_recon)
- Preserves unit η and counit ε as "slack"
- Observes η and ε during training
- Analyzes whether suspension structure emerges

Expected outcomes:
- η remains non-zero (not minimized)
- ε can be observed and analyzed
- Correlation between η and ε
- Dynamic changes in confidence
"""

import sys
sys.path.append('/home/ubuntu/adjunction-model')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.models.adjunction_model import AdjunctionModel
from src.data.synthetic_dataset import SyntheticAffordanceDataset
from src.training.train_phase2_slack import Phase2SlackTrainer


def collate_graph_batch(batch):
    """
    Collate function to convert batch of point clouds to graph format.
    
    Args:
        batch: List of dicts with keys {'points', 'affordances', 'shape_type'}
               Each 'points' is (num_points, 3), 'affordances' is (affordance_dim,)
    
    Returns:
        Dict with:
            - 'points': (total_points, 3) - all points concatenated
            - 'batch': (total_points,) - batch index for each point
            - 'affordances': (batch_size, affordance_dim) - stacked affordances
            - 'shape_type': list of shape types
    """
    points_list = []
    batch_indices = []
    affordances_list = []
    shape_types = []
    
    for i, sample in enumerate(batch):
        points = sample['points']  # (num_points, 3)
        num_points = points.shape[0]
        
        points_list.append(points)
        batch_indices.append(torch.full((num_points,), i, dtype=torch.long))
        affordances_list.append(sample['affordances'])
        shape_types.append(sample['shape_type'])
    
    # Concatenate all points and batch indices
    all_points = torch.cat(points_list, dim=0)  # (total_points, 3)
    all_batch = torch.cat(batch_indices, dim=0)  # (total_points,)
    all_affordances = torch.stack(affordances_list, dim=0)  # (batch_size, affordance_dim)
    
    return {
        'points': all_points,
        'batch': all_batch,
        'affordances': all_affordances,
        'shape_type': shape_types
    }


def run_phase2_slack_experiment(
    num_epochs: int = 50,
    num_shapes: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    lambda_aff: float = 1.0,
    lambda_kl: float = 0.1,
    lambda_coherence: float = 0.1,
    device: str = 'cpu',
    pretrained_model_path: str = None
):
    """
    Run Phase 2 slack preservation experiment.
    
    Args:
        num_epochs: Number of training epochs
        num_shapes: Number of shapes in dataset
        batch_size: Batch size
        lr: Learning rate
        lambda_aff: Weight for affordance loss
        lambda_kl: Weight for KL divergence
        lambda_coherence: Weight for coherence regularization
        device: Device to use ('cpu' or 'cuda')
    """
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path("/home/ubuntu/adjunction-model/results/phase2_slack")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    print(f"\nCreating dataset with {num_shapes} samples...")
    dataset = SyntheticAffordanceDataset(
        num_samples=num_shapes,
        num_points=512,
        shape_types=[0, 1, 2],
        noise_std=0.01
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graph_batch
    )
    
    # Create model
    print("\nInitializing model...")
    print("Training F⊣G from scratch (without reconstruction loss)")
    print("This preserves η and ε as 'slack' rather than minimizing them.\n")
    
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
        alpha_curiosity=0.0,  # Disabled due to sign issues
        beta_competence=0.6,  # Attention to breakdown
        gamma_novelty=0.4     # KL divergence
    )
    
    # Create trainer
    trainer = Phase2SlackTrainer(
        model=model,
        device=device,
        lr=lr,
        lambda_aff=lambda_aff,
        lambda_kl=lambda_kl,
        lambda_coherence=lambda_coherence
    )
    
    # Training history
    history = {
        'loss': [],
        'aff': [],
        'kl': [],
        'coherence': [],
        'unit_eta': [],
        'counit_eps': []
    }
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = trainer.train_epoch(dataloader, epoch)
        
        # Validate
        val_metrics = trainer.validate(dataloader)
        
        # Record history
        for key in history.keys():
            if key in train_metrics:
                history[key].append(train_metrics[key])
        
        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Affordance: {train_metrics['aff']:.4f}")
        print(f"  KL: {train_metrics['kl']:.4f}")
        print(f"  Coherence: {train_metrics['coherence']:.4f}")
        print(f"  Unit η: {train_metrics['unit_eta']:.4f}")
        print(f"  Validation:")
        print(f"    Affordance: {val_metrics['aff']:.4f}")
        print(f"    Unit η: {val_metrics['unit_eta']:.4f}")
        print(f"    Counit ε: {val_metrics['counit_eps']:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    
    # Save metrics
    print("\nSaving metrics...")
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), output_dir / "model_final.pt")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(history, output_dir)
    
    # Analyze η and ε
    print("\nAnalyzing η and ε...")
    analyze_slack(history, output_dir)
    
    # Generate report
    print("Generating report...")
    generate_report(history, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)


def create_visualizations(history, output_dir):
    """Create visualization plots."""
    
    # Plot 1: Training losses
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['aff'])
    axes[0, 1].set_title('Affordance Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['kl'])
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['coherence'])
    axes[1, 1].set_title('Coherence Regularization')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_losses.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: η and ε
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history['unit_eta'], label='Unit η (shape slack)', linewidth=2)
    # Note: counit_eps is only available from validation, so we don't have it per epoch
    
    ax.set_title('Unit η (Shape Slack) Over Training', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "slack_signals.png", dpi=150, bbox_inches='tight')
    plt.close()


def analyze_slack(history, output_dir):
    """Analyze η and ε slack signals."""
    
    eta_values = np.array(history['unit_eta'])
    
    analysis = {
        'eta': {
            'initial': float(eta_values[0]),
            'final': float(eta_values[-1]),
            'mean': float(eta_values.mean()),
            'std': float(eta_values.std()),
            'min': float(eta_values.min()),
            'max': float(eta_values.max()),
            'trend': 'increasing' if eta_values[-1] > eta_values[0] else 'decreasing'
        }
    }
    
    # Save analysis
    with open(output_dir / "slack_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    return analysis


def generate_report(history, output_dir):
    """Generate a text report."""
    
    eta_values = np.array(history['unit_eta'])
    
    report = f"""Phase 2 Slack Preservation Experiment Report
{'=' * 80}

## Training Configuration

- Number of epochs: {len(history['loss'])}
- Final loss: {history['loss'][-1]:.4f}
- Final affordance loss: {history['aff'][-1]:.4f}

## Slack Analysis

### Unit η (Shape Slack)

- Initial value: {eta_values[0]:.4f}
- Final value: {eta_values[-1]:.4f}
- Change: {eta_values[-1] - eta_values[0]:.4f} ({((eta_values[-1] - eta_values[0]) / eta_values[0] * 100):.1f}%)
- Mean: {eta_values.mean():.4f}
- Std: {eta_values.std():.4f}
- Min: {eta_values.min():.4f}
- Max: {eta_values.max():.4f}

### Interpretation

"""
    
    if eta_values[-1] < eta_values[0] * 0.5:
        report += "⚠️  WARNING: η decreased significantly (>50%). Slack may not be preserved.\n"
    elif eta_values[-1] > eta_values[0] * 0.9:
        report += "✅ SUCCESS: η is well preserved (>90% of initial value).\n"
    else:
        report += "⚡ PARTIAL: η decreased but remains non-zero.\n"
    
    report += f"""
## Next Steps

1. Examine visualizations in slack_signals.png
2. Compare with standard Phase 2 training (with L_recon)
3. Analyze correlation between η and ε
4. Observe Agent C's behavior with preserved slack

{'=' * 80}
"""
    
    # Save report
    with open(output_dir / "report.txt", 'w') as f:
        f.write(report)
    
    print(report)


if __name__ == "__main__":
    run_phase2_slack_experiment(
        num_epochs=50,
        num_shapes=50,
        batch_size=4,
        lr=1e-4,
        device='cpu'
    )
