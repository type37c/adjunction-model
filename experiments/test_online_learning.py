"""
Online Learning Validation Experiment

This experiment compares the adaptation performance with and without online learning.

Hypothesis:
With online learning (weight updates during inference), the agent should adapt more
effectively to novel shapes, showing a larger decrease in coherence signal compared
to the eval-only mode (Phase 2 experiment).

Experimental Setup:
1. Train the model on {cube, cylinder, sphere}
2. Test on a novel shape (torus) with two conditions:
   - Condition A: Eval only (no weight updates) - baseline from Phase 2
   - Condition B: Online learning (weight updates enabled)
3. Compare coherence signal evolution over multiple exposures
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.conditional_adjunction import ConditionalAdjunctionModel
from src.data.synthetic_dataset import SyntheticAffordanceDataset
from src.training.train_phase2 import Phase2Trainer
from src.training.online_learning import OnlineLearner
from torch.utils.data import DataLoader


def generate_torus(num_points: int = 512, R: float = 1.0, r: float = 0.3) -> np.ndarray:
    """Generate a torus point cloud."""
    u = np.random.uniform(0, 2*np.pi, num_points)
    v = np.random.uniform(0, 2*np.pi, num_points)
    
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    
    points = np.stack([x, y, z], axis=1)
    return points


def run_eval_only(
    model: ConditionalAdjunctionModel,
    device: torch.device,
    num_exposures: int = 20
) -> dict:
    """Run adaptation without weight updates (eval only)."""
    model.eval()
    
    torus_points = generate_torus(num_points=model.num_points)
    torus_tensor = torch.from_numpy(torus_points).float().to(device)
    
    batch_size = 1
    agent_state = model.initial_state(batch_size, device)
    coherence_signal_prev = torch.zeros(batch_size, 1, device=device)
    
    coherence_history = []
    
    print(f"  Running eval-only condition ({num_exposures} exposures)...")
    
    with torch.no_grad():
        for exposure in range(num_exposures):
            results = model(
                torus_tensor,
                batch=None,
                agent_state=agent_state,
                coherence_signal_prev=coherence_signal_prev
            )
            
            coherence_signal = results['coherence_signal']
            agent_state = results['agent_state']
            
            coherence_history.append(coherence_signal.item())
            coherence_signal_prev = coherence_signal
            
            if exposure % 5 == 0:
                print(f"    Exposure {exposure + 1}: Coherence={coherence_signal.item():.4f}")
    
    return {'coherence_history': coherence_history}


def run_online_learning(
    model: ConditionalAdjunctionModel,
    device: torch.device,
    num_exposures: int = 20
) -> dict:
    """Run adaptation with online learning (weight updates enabled)."""
    # Create online learner
    learner = OnlineLearner(
        model=model,
        device=device,
        base_lr=1e-4,
        coherence_threshold=0.2,  # Lower threshold to enable more updates
        adaptation_strength=2.0,   # Stronger adaptation
        lambda_regularization=0.001  # Weaker regularization for faster adaptation
    )
    
    torus_points = generate_torus(num_points=model.num_points)
    torus_tensor = torch.from_numpy(torus_points).float().to(device)
    
    batch_size = 1
    agent_state = model.initial_state(batch_size, device)
    coherence_signal_prev = torch.zeros(batch_size, 1, device=device)
    
    coherence_history = []
    update_history = []
    
    print(f"  Running online learning condition ({num_exposures} exposures)...")
    
    for exposure in range(num_exposures):
        results = learner.adapt(
            pos=torus_tensor,
            batch=None,
            agent_state=agent_state,
            coherence_signal_prev=coherence_signal_prev
        )
        
        coherence_signal = results['coherence_signal']
        agent_state = results['agent_state']
        
        coherence_history.append(coherence_signal.item())
        update_history.append(1 if results['updated'] else 0)
        coherence_signal_prev = coherence_signal
        
        if exposure % 5 == 0:
            print(f"    Exposure {exposure + 1}: Coherence={coherence_signal.item():.4f}, "
                  f"Updated={results['updated']}")
    
    metrics = learner.get_metrics()
    
    return {
        'coherence_history': coherence_history,
        'update_history': update_history,
        'metrics': metrics
    }


def visualize_comparison(
    eval_results: dict,
    online_results: dict,
    save_dir: Path
):
    """Visualize the comparison between eval-only and online learning."""
    exposures = np.arange(1, len(eval_results['coherence_history']) + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Coherence Signal Comparison
    axes[0].plot(exposures, eval_results['coherence_history'], 
                 'o-', label='Eval Only (No Weight Updates)', 
                 linewidth=2, markersize=6, color='blue')
    axes[0].plot(exposures, online_results['coherence_history'], 
                 's-', label='Online Learning (With Weight Updates)', 
                 linewidth=2, markersize=6, color='red')
    axes[0].set_xlabel('Exposure', fontsize=12)
    axes[0].set_ylabel('Coherence Signal', fontsize=12)
    axes[0].set_title('Coherence Signal: Eval Only vs. Online Learning', 
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Update Frequency
    if 'update_history' in online_results:
        update_cumsum = np.cumsum(online_results['update_history'])
        axes[1].plot(exposures, update_cumsum, 'o-', linewidth=2, markersize=6, color='green')
        axes[1].set_xlabel('Exposure', fontsize=12)
        axes[1].set_ylabel('Cumulative Weight Updates', fontsize=12)
        axes[1].set_title('Online Learning: Cumulative Weight Updates', 
                          fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'online_learning_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_dir / 'online_learning_comparison.png'}")
    plt.close()


def main():
    print("=" * 80)
    print("ONLINE LEARNING VALIDATION EXPERIMENT")
    print("=" * 80)
    
    device = torch.device('cpu')
    save_dir = Path('logs/online_learning_test')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and train model
    print("\n1. Training model on {cube, cylinder, sphere}...")
    model = ConditionalAdjunctionModel(
        num_affordances=5,
        num_points=256,
        f_hidden_dim=32,
        g_hidden_dim=64,
        agent_hidden_dim=128,
        agent_latent_dim=32,
        context_dim=64
    ).to(device)
    
    dataset = SyntheticAffordanceDataset(num_samples=50, num_points=256)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    trainer = Phase2Trainer(
        model=model,
        device=device,
        lr=1e-3,
        lambda_aff=1.0,
        lambda_kl=0.1,
        lambda_coherence=0.1
    )
    
    for epoch in range(5):
        metrics = trainer.train_epoch(dataloader, epoch)
        print(f"   Epoch {epoch + 1}/5: Loss={metrics['loss']:.4f}")
    
    # Condition A: Eval only
    print("\n2. Condition A: Eval Only (No Weight Updates)")
    model_eval = model  # Use the same model
    eval_results = run_eval_only(model_eval, device, num_exposures=20)
    
    # Condition B: Online learning (need a fresh copy of the model)
    print("\n3. Condition B: Online Learning (With Weight Updates)")
    # Create a new model with the same trained weights
    model_online = ConditionalAdjunctionModel(
        num_affordances=5,
        num_points=256,
        f_hidden_dim=32,
        g_hidden_dim=64,
        agent_hidden_dim=128,
        agent_latent_dim=32,
        context_dim=64
    ).to(device)
    model_online.load_state_dict(model.state_dict())
    
    online_results = run_online_learning(model_online, device, num_exposures=20)
    
    # Visualize comparison
    print("\n4. Visualizing results...")
    visualize_comparison(eval_results, online_results, save_dir)
    
    # Analysis
    print("\n5. Analysis:")
    eval_initial = eval_results['coherence_history'][0]
    eval_final = eval_results['coherence_history'][-1]
    eval_change = ((eval_final - eval_initial) / eval_initial) * 100
    
    online_initial = online_results['coherence_history'][0]
    online_final = online_results['coherence_history'][-1]
    online_change = ((online_final - online_initial) / online_initial) * 100
    
    print(f"\n   Eval Only:")
    print(f"     Initial coherence: {eval_initial:.4f}")
    print(f"     Final coherence: {eval_final:.4f}")
    print(f"     Change: {eval_change:+.1f}%")
    
    print(f"\n   Online Learning:")
    print(f"     Initial coherence: {online_initial:.4f}")
    print(f"     Final coherence: {online_final:.4f}")
    print(f"     Change: {online_change:+.1f}%")
    print(f"     Update rate: {online_results['metrics']['update_rate']:.1%}")
    
    improvement = online_change - eval_change
    print(f"\n   Improvement with online learning: {improvement:.1f} percentage points")
    
    if improvement < -10:
        print("\n   ✓ HYPOTHESIS CONFIRMED: Online learning significantly improves adaptation!")
    elif improvement < -5:
        print("\n   ~ PARTIAL CONFIRMATION: Online learning shows moderate improvement.")
    else:
        print("\n   ✗ UNEXPECTED: Online learning did not improve adaptation significantly.")
    
    # Save results
    np.save(save_dir / 'eval_results.npy', eval_results)
    np.save(save_dir / 'online_results.npy', online_results)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
