"""
Phase 2.5 Valence Role Experiment Runner

This script runs three experimental conditions to test the role of valence:

Condition 1 (Baseline): Phase 2 Slack with no valence updates (alpha_curiosity=0.0)
Condition 2 (Emergent): AgentCV3 with valence fed directly to RSSM
Condition 3 (Designed): Agent C with Priority = coherence × uncertainty × valence

Each condition is run for 50 epochs with the same dataset and hyperparameters.
Results are saved to experiments/phase2_valence_experiment/condition_{1,2,3}/
"""

import torch
import sys
import os
import json
from pathlib import Path

sys.path.append('/home/ubuntu/adjunction-model')

from src.models.adjunction_model import AdjunctionModel
from src.data.synthetic_dataset import SyntheticAffordanceDataset
from src.training.train_phase2_slack import Phase2SlackTrainer
from torch.utils.data import DataLoader


def run_condition_1(output_dir: Path, num_epochs: int = 50):
    """
    Condition 1: Baseline (Phase 2 Slack, no valence updates)
    """
    print("\n" + "="*60)
    print("CONDITION 1: Baseline (No Valence Updates)")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Create model with alpha_curiosity=0.0 (no valence updates)
    model = AdjunctionModel(
        in_channels=3,
        hidden_channels=64,
        out_channels=5,
        agent_hidden_dim=256,
        agent_latent_dim=64,
        context_dim=128,
        valence_dim=32,
        alpha_curiosity=0.0,  # KEY: No valence updates
        beta_competence=0.5,
        gamma_novelty=0.2
    )
    
    # Create dataset
    dataset = SyntheticAffordanceDataset(
        num_samples=100,
        num_points=256,
        shape_types=['sphere', 'cube', 'cylinder']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    # Create trainer
    trainer = Phase2SlackTrainer(
        model=model,
        device=device,
        lr=1e-4,
        lambda_aff=1.0,
        lambda_kl=0.1,
        lambda_coherence=0.1
    )
    
    # Train
    results = []
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(dataloader, epoch)
        results.append(metrics)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: L_aff={metrics['aff_loss']:.4f}, "
                  f"η={metrics['unit_mean']:.4f}, ε={metrics['counit_mean']:.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), output_dir / 'model_final.pt')
    
    print(f"\nCondition 1 complete. Results saved to {output_dir}")
    return results


def run_condition_2(output_dir: Path, num_epochs: int = 50):
    """
    Condition 2: Emergent Valence (AgentCV3)
    """
    print("\n" + "="*60)
    print("CONDITION 2: Emergent Valence (AgentCV3)")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Import AdjunctionModelV3
    from src.models.adjunction_model_v3 import AdjunctionModelV3
    
    # Create model with AgentCV3
    model = AdjunctionModelV3(
        num_affordances=5,
        num_points=256,
        f_hidden_dim=64,
        g_hidden_dim=128,
        agent_hidden_dim=256,
        agent_latent_dim=64,
        context_dim=128,
        valence_dim=32,
        valence_decay=0.1,
        valence_learning_rate=0.1
    )
    
    # Create dataset
    dataset = SyntheticAffordanceDataset(
        num_samples=100,
        num_points=256,
        shape_types=['sphere', 'cube', 'cylinder']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    # Create trainer
    trainer = Phase2SlackTrainer(
        model=model,
        device=device,
        lr=1e-4,
        lambda_aff=1.0,
        lambda_kl=0.1,
        lambda_coherence=0.1
    )
    
    # Train
    results = []
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(dataloader, epoch)
        results.append(metrics)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: L_aff={metrics['aff_loss']:.4f}, "
                  f"η={metrics['unit_mean']:.4f}, ε={metrics['counit_mean']:.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), output_dir / 'model_final.pt')
    
    print(f"\nCondition 2 complete. Results saved to {output_dir}")
    return results


def run_condition_3(output_dir: Path, num_epochs: int = 50):
    """
    Condition 3: Designed Valence (Priority = coherence × uncertainty × valence)
    """
    print("\n" + "="*60)
    print("CONDITION 3: Designed Valence (Priority Computation)")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Create model with alpha_curiosity=1.0 (enable valence updates)
    model = AdjunctionModel(
        in_channels=3,
        hidden_channels=64,
        out_channels=5,
        agent_hidden_dim=256,
        agent_latent_dim=64,
        context_dim=128,
        valence_dim=32,
        alpha_curiosity=1.0,  # KEY: Enable valence updates
        beta_competence=0.5,
        gamma_novelty=0.2
    )
    
    # Create dataset
    dataset = SyntheticAffordanceDataset(
        num_samples=100,
        num_points=256,
        shape_types=['sphere', 'cube', 'cylinder']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    # Create trainer
    trainer = Phase2SlackTrainer(
        model=model,
        device=device,
        lr=1e-4,
        lambda_aff=1.0,
        lambda_kl=0.1,
        lambda_coherence=0.1
    )
    
    # Train
    results = []
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(dataloader, epoch)
        results.append(metrics)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: L_aff={metrics['aff_loss']:.4f}, "
                  f"η={metrics['unit_mean']:.4f}, ε={metrics['counit_mean']:.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), output_dir / 'model_final.pt')
    
    print(f"\nCondition 3 complete. Results saved to {output_dir}")
    return results


def main():
    """
    Run all three conditions of the valence experiment.
    """
    base_dir = Path('/home/ubuntu/adjunction-model/experiments/phase2_valence_experiment')
    
    print("\n" + "="*60)
    print("PHASE 2.5 VALENCE ROLE EXPERIMENT")
    print("="*60)
    print("\nThis experiment tests three conditions:")
    print("  1. Baseline: No valence updates (alpha_curiosity=0.0)")
    print("  2. Emergent: Valence fed directly to RSSM (AgentCV3)")
    print("  3. Designed: Priority = coherence × uncertainty × valence")
    print("\nEach condition runs for 50 epochs.")
    print("="*60)
    
    # Run conditions
    results_1 = run_condition_1(base_dir / 'condition_1', num_epochs=50)
    results_2 = run_condition_2(base_dir / 'condition_2', num_epochs=50)
    results_3 = run_condition_3(base_dir / 'condition_3', num_epochs=50)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {base_dir}")
    print("\nNext steps:")
    print("  1. Run analysis script to compare conditions")
    print("  2. Generate visualizations")
    print("  3. Update documentation with findings")


if __name__ == '__main__':
    main()
