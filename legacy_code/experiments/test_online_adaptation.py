"""
Online Adaptation Experiment

This experiment tests the agent's ability to adapt to novel shapes in real-time.

Hypothesis:
When the agent encounters a novel shape (not seen during training), the coherence signal
should increase, triggering a mode shift in the Agent Layer C. Over multiple exposures,
the agent should adapt its internal state to accommodate the new shape, leading to a
gradual decrease in the coherence signal.

Experimental Setup:
1. Train the model on {cube, cylinder, sphere}
2. Test on a novel shape (torus)
3. Measure coherence signal and agent state evolution over multiple exposures
4. Visualize the adaptation trajectory
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
from torch.utils.data import DataLoader


def generate_torus(num_points: int = 512, R: float = 1.0, r: float = 0.3) -> np.ndarray:
    """
    Generate a torus point cloud.
    
    Args:
        num_points: Number of points
        R: Major radius
        r: Minor radius
    
    Returns:
        points: (num_points, 3) numpy array
    """
    u = np.random.uniform(0, 2*np.pi, num_points)
    v = np.random.uniform(0, 2*np.pi, num_points)
    
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    
    points = np.stack([x, y, z], axis=1)
    return points


def run_adaptation_experiment(
    model: ConditionalAdjunctionModel,
    device: torch.device,
    num_exposures: int = 10,
    save_dir: Path = Path('logs/adaptation_test')
):
    """
    Run the online adaptation experiment.
    
    Args:
        model: Trained conditional adjunction model
        device: Device
        num_exposures: Number of times to expose the agent to the novel shape
        save_dir: Directory to save results
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Generate a novel torus
    torus_points = generate_torus(num_points=model.num_points)
    torus_tensor = torch.from_numpy(torus_points).float().to(device)
    
    # Initialize agent state
    batch_size = 1
    agent_state = model.initial_state(batch_size, device)
    coherence_signal_prev = torch.zeros(batch_size, 1, device=device)
    
    # Track metrics over exposures
    coherence_history = []
    context_history = []
    h_norm_history = []
    z_norm_history = []
    
    print(f"Running online adaptation experiment with {num_exposures} exposures...")
    
    with torch.no_grad():
        for exposure in range(num_exposures):
            # Forward pass
            results = model(
                torus_tensor,
                batch=None,
                agent_state=agent_state,
                coherence_signal_prev=coherence_signal_prev
            )
            
            # Extract results
            coherence_signal = results['coherence_signal']
            agent_state = results['agent_state']
            context = results['context']
            
            # Record metrics
            coherence_history.append(coherence_signal.item())
            context_history.append(context.cpu().numpy())
            h_norm_history.append(torch.norm(agent_state['h']).item())
            z_norm_history.append(torch.norm(agent_state['z']).item())
            
            # Update for next iteration
            coherence_signal_prev = coherence_signal
            
            print(f"  Exposure {exposure + 1}/{num_exposures}: "
                  f"Coherence={coherence_signal.item():.4f}, "
                  f"||h||={h_norm_history[-1]:.4f}, "
                  f"||z||={z_norm_history[-1]:.4f}")
    
    # Save results
    results_dict = {
        'coherence_history': coherence_history,
        'context_history': np.array(context_history),
        'h_norm_history': h_norm_history,
        'z_norm_history': z_norm_history
    }
    
    np.save(save_dir / 'adaptation_results.npy', results_dict)
    
    # Visualize
    visualize_adaptation(results_dict, save_dir)
    
    return results_dict


def visualize_adaptation(results: dict, save_dir: Path):
    """
    Visualize the adaptation trajectory.
    
    Args:
        results: Dictionary with adaptation metrics
        save_dir: Directory to save figures
    """
    coherence_history = results['coherence_history']
    h_norm_history = results['h_norm_history']
    z_norm_history = results['z_norm_history']
    
    exposures = np.arange(1, len(coherence_history) + 1)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot 1: Coherence Signal
    axes[0].plot(exposures, coherence_history, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Exposure', fontsize=12)
    axes[0].set_ylabel('Coherence Signal', fontsize=12)
    axes[0].set_title('Coherence Signal Over Exposures to Novel Shape (Torus)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Deterministic State Norm ||h||
    axes[1].plot(exposures, h_norm_history, 'o-', color='orange', linewidth=2, markersize=8)
    axes[1].set_xlabel('Exposure', fontsize=12)
    axes[1].set_ylabel('||h|| (Deterministic State Norm)', fontsize=12)
    axes[1].set_title('Agent Internal State Evolution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Stochastic State Norm ||z||
    axes[2].plot(exposures, z_norm_history, 'o-', color='green', linewidth=2, markersize=8)
    axes[2].set_xlabel('Exposure', fontsize=12)
    axes[2].set_ylabel('||z|| (Stochastic State Norm)', fontsize=12)
    axes[2].set_title('Exploration/Uncertainty Evolution', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'adaptation_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_dir / 'adaptation_trajectory.png'}")
    
    plt.close()


def main():
    print("=" * 80)
    print("ONLINE ADAPTATION EXPERIMENT")
    print("=" * 80)
    
    device = torch.device('cpu')
    
    # Create model
    print("\n1. Creating model...")
    model = ConditionalAdjunctionModel(
        num_affordances=5,
        num_points=256,
        f_hidden_dim=32,
        g_hidden_dim=64,
        agent_hidden_dim=128,
        agent_latent_dim=32,
        context_dim=64
    ).to(device)
    
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model (quick training for demonstration)
    print("\n2. Training model on {cube, cylinder, sphere}...")
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
        print(f"   Epoch {epoch + 1}/5: Loss={metrics['loss']:.4f}, Coherence={metrics['coherence']:.4f}")
    
    # Run adaptation experiment
    print("\n3. Testing online adaptation to novel shape (torus)...")
    results = run_adaptation_experiment(
        model=model,
        device=device,
        num_exposures=10,
        save_dir=Path('logs/adaptation_test')
    )
    
    # Analyze results
    print("\n4. Analysis:")
    coherence_history = results['coherence_history']
    initial_coherence = coherence_history[0]
    final_coherence = coherence_history[-1]
    coherence_change = ((final_coherence - initial_coherence) / initial_coherence) * 100
    
    print(f"   Initial coherence signal: {initial_coherence:.4f}")
    print(f"   Final coherence signal: {final_coherence:.4f}")
    print(f"   Change: {coherence_change:+.1f}%")
    
    if coherence_change < -10:
        print("\n   ✓ HYPOTHESIS CONFIRMED: Agent adapted to the novel shape!")
        print("     The coherence signal decreased over exposures, indicating successful adaptation.")
    elif coherence_change > 10:
        print("\n   ✗ UNEXPECTED: Coherence signal increased over exposures.")
        print("     This may indicate instability or divergence in the agent's internal state.")
    else:
        print("\n   ~ INCONCLUSIVE: Coherence signal remained relatively stable.")
        print("     The agent may need more exposures or a stronger learning signal.")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
