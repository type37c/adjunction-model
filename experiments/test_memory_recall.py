"""
Memory Recall Experiment (Suspension Structure Validation 1)

This experiment tests whether the model can distinguish between "first-time novel"
and "previously encountered novel" shapes, which would indicate that memory is
emergent from GNN topology changes.

Hypothesis:
If memory emerges from GNN topology changes (as discussed in the Claude session),
then when the agent re-encounters a previously seen novel shape, the coherence
signal should be lower than the initial encounter, indicating "recognition".

Experimental Setup:
1. Train the model on {cube, cylinder, sphere}
2. Expose the agent to a novel shape A (torus) multiple times with online learning
3. Expose the agent to a different novel shape B (cone)
4. Re-expose the agent to shape A
5. Compare coherence signals:
   - Initial exposure to A vs. re-exposure to A (should decrease if memory exists)
   - Initial exposure to B vs. initial exposure to A (should be similar, both novel)
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


def generate_cone(num_points: int = 512, radius: float = 1.0, height: float = 2.0) -> np.ndarray:
    """Generate a cone point cloud."""
    # Sample from cone surface
    h = np.random.uniform(0, height, num_points)
    theta = np.random.uniform(0, 2*np.pi, num_points)
    r = radius * (1 - h / height)  # Radius decreases linearly with height
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = h
    
    points = np.stack([x, y, z], axis=1)
    return points


def run_memory_experiment(
    model: ConditionalAdjunctionModel,
    device: torch.device
) -> dict:
    """Run the memory recall experiment."""
    
    learner = OnlineLearner(
        model=model,
        device=device,
        base_lr=1e-4,
        coherence_threshold=0.2,
        adaptation_strength=2.0,
        lambda_regularization=0.001
    )
    
    # Generate shapes
    torus_points = generate_torus(num_points=model.num_points)
    cone_points = generate_cone(num_points=model.num_points)
    
    torus_tensor = torch.from_numpy(torus_points).float().to(device)
    cone_tensor = torch.from_numpy(cone_points).float().to(device)
    
    # Initialize agent state
    batch_size = 1
    agent_state = model.initial_state(batch_size, device)
    coherence_signal_prev = torch.zeros(batch_size, 1, device=device)
    
    # Track results
    phase_labels = []
    coherence_history = []
    shape_labels = []
    
    # Phase 1: Initial exposure to Torus (10 times)
    print("  Phase 1: Initial exposure to Torus (10 times)...")
    for i in range(10):
        results = learner.adapt(
            pos=torus_tensor,
            batch=None,
            agent_state=agent_state,
            coherence_signal_prev=coherence_signal_prev
        )
        
        agent_state = results['agent_state']
        coherence_signal_prev = results['coherence_signal']
        
        coherence_history.append(results['coherence_signal'].item())
        phase_labels.append('Phase 1: Torus (Initial)')
        shape_labels.append('Torus')
        
        if i == 0 or i == 9:
            print(f"    Exposure {i+1}: Coherence={results['coherence_signal'].item():.4f}")
    
    # Phase 2: Exposure to Cone (5 times)
    print("\n  Phase 2: Exposure to Cone (5 times)...")
    for i in range(5):
        results = learner.adapt(
            pos=cone_tensor,
            batch=None,
            agent_state=agent_state,
            coherence_signal_prev=coherence_signal_prev
        )
        
        agent_state = results['agent_state']
        coherence_signal_prev = results['coherence_signal']
        
        coherence_history.append(results['coherence_signal'].item())
        phase_labels.append('Phase 2: Cone')
        shape_labels.append('Cone')
        
        if i == 0 or i == 4:
            print(f"    Exposure {i+1}: Coherence={results['coherence_signal'].item():.4f}")
    
    # Phase 3: Re-exposure to Torus (5 times)
    print("\n  Phase 3: Re-exposure to Torus (5 times)...")
    for i in range(5):
        results = learner.adapt(
            pos=torus_tensor,
            batch=None,
            agent_state=agent_state,
            coherence_signal_prev=coherence_signal_prev
        )
        
        agent_state = results['agent_state']
        coherence_signal_prev = results['coherence_signal']
        
        coherence_history.append(results['coherence_signal'].item())
        phase_labels.append('Phase 3: Torus (Recall)')
        shape_labels.append('Torus')
        
        if i == 0 or i == 4:
            print(f"    Exposure {i+1}: Coherence={results['coherence_signal'].item():.4f}")
    
    return {
        'coherence_history': coherence_history,
        'phase_labels': phase_labels,
        'shape_labels': shape_labels
    }


def visualize_memory_experiment(results: dict, save_dir: Path):
    """Visualize the memory recall experiment."""
    
    coherence_history = results['coherence_history']
    phase_labels = results['phase_labels']
    
    exposures = np.arange(1, len(coherence_history) + 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Plot coherence signal with phase annotations
    ax.plot(exposures, coherence_history, 'o-', linewidth=2, markersize=6)
    
    # Add phase boundaries
    ax.axvline(x=10.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Phase Boundary')
    ax.axvline(x=15.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add phase labels
    ax.text(5, max(coherence_history) * 0.95, 'Phase 1:\nTorus (Initial)', 
            ha='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(13, max(coherence_history) * 0.95, 'Phase 2:\nCone', 
            ha='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(18, max(coherence_history) * 0.95, 'Phase 3:\nTorus (Recall)', 
            ha='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    ax.set_xlabel('Exposure', fontsize=12)
    ax.set_ylabel('Coherence Signal', fontsize=12)
    ax.set_title('Memory Recall Experiment: Coherence Signal Across Phases', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'memory_recall.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_dir / 'memory_recall.png'}")
    plt.close()


def main():
    print("=" * 80)
    print("MEMORY RECALL EXPERIMENT (Suspension Structure Validation 1)")
    print("=" * 80)
    
    device = torch.device('cpu')
    save_dir = Path('logs/memory_recall_test')
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
    
    # Run memory experiment
    print("\n2. Running memory recall experiment...")
    results = run_memory_experiment(model, device)
    
    # Visualize
    print("\n3. Visualizing results...")
    visualize_memory_experiment(results, save_dir)
    
    # Analysis
    print("\n4. Analysis:")
    coherence_history = results['coherence_history']
    
    torus_initial_first = coherence_history[0]
    torus_initial_last = coherence_history[9]
    cone_first = coherence_history[10]
    torus_recall_first = coherence_history[15]
    
    print(f"\n   Torus (Initial, 1st exposure): {torus_initial_first:.4f}")
    print(f"   Torus (Initial, 10th exposure): {torus_initial_last:.4f}")
    print(f"   Cone (1st exposure): {cone_first:.4f}")
    print(f"   Torus (Recall, 1st exposure): {torus_recall_first:.4f}")
    
    memory_evidence = ((torus_initial_first - torus_recall_first) / torus_initial_first) * 100
    
    print(f"\n   Memory evidence: {memory_evidence:.1f}%")
    print(f"   (Reduction in coherence signal when re-encountering Torus)")
    
    if memory_evidence > 20:
        print("\n   ✓ STRONG EVIDENCE: Agent exhibits clear memory of previously encountered shape!")
    elif memory_evidence > 10:
        print("\n   ~ MODERATE EVIDENCE: Agent shows some memory of previously encountered shape.")
    else:
        print("\n   ✗ WEAK EVIDENCE: Memory effect is not significant.")
    
    # Save results
    np.save(save_dir / 'memory_results.npy', results)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
