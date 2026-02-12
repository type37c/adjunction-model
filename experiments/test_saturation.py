"""
Saturation (Boredom) Experiment (Suspension Structure Validation 2)

This experiment tests whether the agent exhibits "saturation" or "boredom" when
repeatedly exposed to the same shape, even after coherence signal has stabilized.

Hypothesis:
According to the Claude session discussion, there are two types of creativity drivers:
1. External: Coherence Signal (breakdown-driven creativity)
2. Internal: Saturation (boredom-driven creativity)

When the agent is repeatedly exposed to the same shape and coherence signal stabilizes
at a low value, the internal state should show signs of "saturation":
- Stochastic state (z) entropy increases (exploration desire)
- Deterministic state (h) change rate decreases (reduced responsiveness)

Experimental Setup:
1. Train the model on {cube, cylinder, sphere}
2. Expose the agent to a single shape (cube) 100 times with online learning
3. Track:
   - Coherence signal
   - Stochastic state (z) entropy
   - Deterministic state (h) norm and change rate
4. Identify saturation: coherence low + entropy high + h-change low
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


def compute_entropy(z: torch.Tensor) -> float:
    """Compute entropy of stochastic state z."""
    # Normalize z to approximate a probability distribution
    z_normalized = torch.softmax(z, dim=-1)
    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(z_normalized * torch.log(z_normalized + 1e-8))
    return entropy.item()


def run_saturation_experiment(
    model: ConditionalAdjunctionModel,
    device: torch.device,
    num_exposures: int = 100
) -> dict:
    """Run the saturation experiment."""
    
    learner = OnlineLearner(
        model=model,
        device=device,
        base_lr=1e-4,
        coherence_threshold=0.15,  # Lower threshold to keep updating even when coherence is low
        adaptation_strength=1.0,
        lambda_regularization=0.001
    )
    
    # Generate a single cube
    dataset = SyntheticAffordanceDataset(num_samples=1, num_points=model.num_points)
    cube_sample = dataset[0]
    cube_tensor = cube_sample['points'].to(device)
    
    # Initialize agent state
    batch_size = 1
    agent_state = model.initial_state(batch_size, device)
    coherence_signal_prev = torch.zeros(batch_size, 1, device=device)
    
    # Track metrics
    coherence_history = []
    z_entropy_history = []
    h_norm_history = []
    h_change_history = []
    
    h_prev = None
    
    print(f"  Exposing agent to the same cube {num_exposures} times...")
    
    for exposure in range(num_exposures):
        results = learner.adapt(
            pos=cube_tensor,
            batch=None,
            agent_state=agent_state,
            coherence_signal_prev=coherence_signal_prev
        )
        
        # Extract metrics
        coherence_signal = results['coherence_signal']
        agent_state = results['agent_state']
        
        h = agent_state['h']
        z = agent_state['z']
        
        # Compute metrics
        coherence = coherence_signal.item()
        z_entropy = compute_entropy(z)
        h_norm = torch.norm(h).item()
        
        if h_prev is not None:
            h_change = torch.norm(h - h_prev).item()
        else:
            h_change = 0.0
        
        # Store metrics
        coherence_history.append(coherence)
        z_entropy_history.append(z_entropy)
        h_norm_history.append(h_norm)
        h_change_history.append(h_change)
        
        # Update for next iteration
        coherence_signal_prev = coherence_signal
        h_prev = h.clone()
        
        if exposure % 20 == 0 or exposure == num_exposures - 1:
            print(f"    Exposure {exposure + 1}: "
                  f"Coherence={coherence:.4f}, "
                  f"Z-Entropy={z_entropy:.4f}, "
                  f"H-Change={h_change:.4f}")
    
    return {
        'coherence_history': coherence_history,
        'z_entropy_history': z_entropy_history,
        'h_norm_history': h_norm_history,
        'h_change_history': h_change_history
    }


def visualize_saturation(results: dict, save_dir: Path):
    """Visualize the saturation experiment."""
    
    exposures = np.arange(1, len(results['coherence_history']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Coherence Signal
    axes[0, 0].plot(exposures, results['coherence_history'], 'o-', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Exposure', fontsize=11)
    axes[0, 0].set_ylabel('Coherence Signal', fontsize=11)
    axes[0, 0].set_title('Coherence Signal (External Driver)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Z Entropy
    axes[0, 1].plot(exposures, results['z_entropy_history'], 'o-', linewidth=2, markersize=4, color='orange')
    axes[0, 1].set_xlabel('Exposure', fontsize=11)
    axes[0, 1].set_ylabel('Stochastic State (z) Entropy', fontsize=11)
    axes[0, 1].set_title('Z Entropy (Exploration Desire)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: H Norm
    axes[1, 0].plot(exposures, results['h_norm_history'], 'o-', linewidth=2, markersize=4, color='green')
    axes[1, 0].set_xlabel('Exposure', fontsize=11)
    axes[1, 0].set_ylabel('Deterministic State (h) Norm', fontsize=11)
    axes[1, 0].set_title('H Norm (Internal State Magnitude)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: H Change Rate
    axes[1, 1].plot(exposures, results['h_change_history'], 'o-', linewidth=2, markersize=4, color='red')
    axes[1, 1].set_xlabel('Exposure', fontsize=11)
    axes[1, 1].set_ylabel('Deterministic State (h) Change', fontsize=11)
    axes[1, 1].set_title('H Change Rate (Responsiveness)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'saturation.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_dir / 'saturation.png'}")
    plt.close()


def main():
    print("=" * 80)
    print("SATURATION (BOREDOM) EXPERIMENT (Suspension Structure Validation 2)")
    print("=" * 80)
    
    device = torch.device('cpu')
    save_dir = Path('logs/saturation_test')
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
    
    # Run saturation experiment
    print("\n2. Running saturation experiment...")
    results = run_saturation_experiment(model, device, num_exposures=100)
    
    # Visualize
    print("\n3. Visualizing results...")
    visualize_saturation(results, save_dir)
    
    # Analysis
    print("\n4. Analysis:")
    
    # Divide into early and late phases
    early_coherence = np.mean(results['coherence_history'][:20])
    late_coherence = np.mean(results['coherence_history'][-20:])
    
    early_z_entropy = np.mean(results['z_entropy_history'][:20])
    late_z_entropy = np.mean(results['z_entropy_history'][-20:])
    
    early_h_change = np.mean(results['h_change_history'][1:21])  # Skip first (always 0)
    late_h_change = np.mean(results['h_change_history'][-20:])
    
    print(f"\n   Early phase (exposures 1-20):")
    print(f"     Avg coherence: {early_coherence:.4f}")
    print(f"     Avg Z entropy: {early_z_entropy:.4f}")
    print(f"     Avg H change: {early_h_change:.4f}")
    
    print(f"\n   Late phase (exposures 81-100):")
    print(f"     Avg coherence: {late_coherence:.4f}")
    print(f"     Avg Z entropy: {late_z_entropy:.4f}")
    print(f"     Avg H change: {late_h_change:.4f}")
    
    coherence_decrease = ((late_coherence - early_coherence) / early_coherence) * 100
    z_entropy_change = ((late_z_entropy - early_z_entropy) / early_z_entropy) * 100
    h_change_decrease = ((late_h_change - early_h_change) / early_h_change) * 100
    
    print(f"\n   Changes from early to late:")
    print(f"     Coherence: {coherence_decrease:+.1f}%")
    print(f"     Z entropy: {z_entropy_change:+.1f}%")
    print(f"     H change: {h_change_decrease:+.1f}%")
    
    # Saturation criteria:
    # 1. Coherence decreases (agent adapts)
    # 2. Z entropy increases (exploration desire rises)
    # 3. H change decreases (responsiveness drops)
    
    saturation_score = 0
    if coherence_decrease < -10:
        saturation_score += 1
        print("\n   ✓ Criterion 1: Coherence decreased significantly (adaptation)")
    
    if z_entropy_change > 5:
        saturation_score += 1
        print("   ✓ Criterion 2: Z entropy increased (exploration desire)")
    
    if h_change_decrease < -10:
        saturation_score += 1
        print("   ✓ Criterion 3: H change decreased (reduced responsiveness)")
    
    print(f"\n   Saturation score: {saturation_score}/3")
    
    if saturation_score >= 2:
        print("\n   ✓ SATURATION DETECTED: Agent exhibits signs of 'boredom' after repeated exposure!")
    else:
        print("\n   ~ PARTIAL SATURATION: Some signs present, but not conclusive.")
    
    # Save results
    np.save(save_dir / 'saturation_results.npy', results)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
