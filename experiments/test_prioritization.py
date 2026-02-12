"""
Prioritization Experiment (Suspension Structure Validation 3)

This experiment tests whether the agent exhibits "intentionality" by prioritizing
attention to shapes with higher coherence signals (more uncertain/novel).

Hypothesis:
If the suspension structure is "always active" and exhibits intentionality, the agent
should allocate more "attention" (measured by internal state change) to shapes with
higher coherence signals, even when multiple shapes are presented.

Experimental Setup:
1. Train the model on {cube, cylinder, sphere}
2. Present the agent with alternating sequences:
   - Sequence A: Known shape (cube) → Novel shape (torus)
   - Sequence B: Novel shape (torus) → Known shape (cube)
3. Measure attention allocation:
   - Internal state change (||h_after - h_before||) for each shape
4. Hypothesis: Attention to novel shape > Attention to known shape
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


def run_prioritization_experiment(
    model: ConditionalAdjunctionModel,
    device: torch.device,
    num_trials: int = 10
) -> dict:
    """Run the prioritization experiment."""
    
    learner = OnlineLearner(
        model=model,
        device=device,
        base_lr=1e-4,
        coherence_threshold=0.2,
        adaptation_strength=2.0,
        lambda_regularization=0.001
    )
    
    # Generate shapes
    dataset = SyntheticAffordanceDataset(num_samples=1, num_points=model.num_points)
    cube_sample = dataset[0]
    cube_tensor = cube_sample['points'].to(device)
    
    torus_points = generate_torus(num_points=model.num_points)
    torus_tensor = torch.from_numpy(torus_points).float().to(device)
    
    # Initialize agent state
    batch_size = 1
    agent_state = model.initial_state(batch_size, device)
    coherence_signal_prev = torch.zeros(batch_size, 1, device=device)
    
    # Track results
    cube_coherence = []
    torus_coherence = []
    cube_attention = []  # Measured as ||h_after - h_before||
    torus_attention = []
    
    print(f"  Running {num_trials} trials of alternating presentations...")
    
    for trial in range(num_trials):
        # Sequence: Cube → Torus
        
        # Present cube
        h_before = agent_state['h'].clone()
        
        results_cube = learner.adapt(
            pos=cube_tensor,
            batch=None,
            agent_state=agent_state,
            coherence_signal_prev=coherence_signal_prev
        )
        
        h_after = results_cube['agent_state']['h']
        attention_cube = torch.norm(h_after - h_before).item()
        
        cube_coherence.append(results_cube['coherence_signal'].item())
        cube_attention.append(attention_cube)
        
        agent_state = results_cube['agent_state']
        coherence_signal_prev = results_cube['coherence_signal']
        
        # Present torus
        h_before = agent_state['h'].clone()
        
        results_torus = learner.adapt(
            pos=torus_tensor,
            batch=None,
            agent_state=agent_state,
            coherence_signal_prev=coherence_signal_prev
        )
        
        h_after = results_torus['agent_state']['h']
        attention_torus = torch.norm(h_after - h_before).item()
        
        torus_coherence.append(results_torus['coherence_signal'].item())
        torus_attention.append(attention_torus)
        
        agent_state = results_torus['agent_state']
        coherence_signal_prev = results_torus['coherence_signal']
        
        if trial % 3 == 0 or trial == num_trials - 1:
            print(f"    Trial {trial + 1}: "
                  f"Cube(Coh={results_cube['coherence_signal'].item():.4f}, Att={attention_cube:.4f}), "
                  f"Torus(Coh={results_torus['coherence_signal'].item():.4f}, Att={attention_torus:.4f})")
    
    return {
        'cube_coherence': cube_coherence,
        'torus_coherence': torus_coherence,
        'cube_attention': cube_attention,
        'torus_attention': torus_attention
    }


def visualize_prioritization(results: dict, save_dir: Path):
    """Visualize the prioritization experiment."""
    
    trials = np.arange(1, len(results['cube_coherence']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Coherence Signal
    axes[0].plot(trials, results['cube_coherence'], 'o-', label='Cube (Known)', 
                 linewidth=2, markersize=6, color='blue')
    axes[0].plot(trials, results['torus_coherence'], 's-', label='Torus (Novel)', 
                 linewidth=2, markersize=6, color='red')
    axes[0].set_xlabel('Trial', fontsize=12)
    axes[0].set_ylabel('Coherence Signal', fontsize=12)
    axes[0].set_title('Coherence Signal: Known vs. Novel', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Attention (Internal State Change)
    axes[1].plot(trials, results['cube_attention'], 'o-', label='Cube (Known)', 
                 linewidth=2, markersize=6, color='blue')
    axes[1].plot(trials, results['torus_attention'], 's-', label='Torus (Novel)', 
                 linewidth=2, markersize=6, color='red')
    axes[1].set_xlabel('Trial', fontsize=12)
    axes[1].set_ylabel('Attention (||Δh||)', fontsize=12)
    axes[1].set_title('Attention Allocation: Known vs. Novel', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'prioritization.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_dir / 'prioritization.png'}")
    plt.close()


def main():
    print("=" * 80)
    print("PRIORITIZATION EXPERIMENT (Suspension Structure Validation 3)")
    print("=" * 80)
    
    device = torch.device('cpu')
    save_dir = Path('logs/prioritization_test')
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
    
    # Run prioritization experiment
    print("\n2. Running prioritization experiment...")
    results = run_prioritization_experiment(model, device, num_trials=10)
    
    # Visualize
    print("\n3. Visualizing results...")
    visualize_prioritization(results, save_dir)
    
    # Analysis
    print("\n4. Analysis:")
    
    avg_cube_coherence = np.mean(results['cube_coherence'])
    avg_torus_coherence = np.mean(results['torus_coherence'])
    
    avg_cube_attention = np.mean(results['cube_attention'])
    avg_torus_attention = np.mean(results['torus_attention'])
    
    print(f"\n   Average Coherence Signal:")
    print(f"     Cube (known): {avg_cube_coherence:.4f}")
    print(f"     Torus (novel): {avg_torus_coherence:.4f}")
    print(f"     Ratio (novel/known): {avg_torus_coherence / avg_cube_coherence:.2f}x")
    
    print(f"\n   Average Attention (||Δh||):")
    print(f"     Cube (known): {avg_cube_attention:.4f}")
    print(f"     Torus (novel): {avg_torus_attention:.4f}")
    print(f"     Ratio (novel/known): {avg_torus_attention / avg_cube_attention:.2f}x")
    
    # Hypothesis: Attention to novel > Attention to known
    attention_ratio = avg_torus_attention / avg_cube_attention
    
    if attention_ratio > 1.5:
        print("\n   ✓ STRONG EVIDENCE: Agent prioritizes attention to novel shapes!")
        print("     This indicates intentionality: the suspension structure directs")
        print("     cognitive resources toward higher coherence signals.")
    elif attention_ratio > 1.2:
        print("\n   ~ MODERATE EVIDENCE: Agent shows preference for novel shapes.")
    else:
        print("\n   ✗ WEAK EVIDENCE: No clear prioritization detected.")
    
    # Save results
    np.save(save_dir / 'prioritization_results.npy', results)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
