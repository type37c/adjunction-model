"""
Experiment: Test Coherence Signal on Known vs Unknown Shapes

This experiment verifies the core hypothesis:
- Coherence signal should be LOW for shapes similar to training data (stable adjunction)
- Coherence signal should be HIGH for novel/unknown shapes (adjunction breakdown)

This is a preliminary test for Setting A (Zero-shot Affordance) from TODO.md.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models import create_adjunction_model
from src.data import SyntheticAffordanceDataset, get_dataloader
from src.training import Phase1Trainer


def train_model(num_epochs=10):
    """Train a model on synthetic data."""
    print("="*60)
    print("STEP 1: Training model on known shapes (cube, cylinder, sphere)")
    print("="*60)
    
    # Create model
    model = create_adjunction_model(
        num_affordances=5,
        num_points=256,
        f_hidden_dim=32,
        g_hidden_dim=64
    )
    
    # Create data loaders
    train_loader = get_dataloader(
        batch_size=8,
        num_samples=200,
        num_points=256,
        split='train'
    )
    
    val_loader = get_dataloader(
        batch_size=8,
        num_samples=50,
        num_points=256,
        split='val'
    )
    
    # Train
    trainer = Phase1Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        lambda_aff=1.0,
        lambda_coh=0.1,
        device='cpu',
        log_dir='./logs/coherence_test'
    )
    
    trainer.train(num_epochs=num_epochs, save_every=num_epochs)
    
    return model


def test_coherence_on_known_shapes(model):
    """Test coherence signal on shapes from training distribution."""
    print("\n" + "="*60)
    print("STEP 2: Testing coherence signal on KNOWN shapes")
    print("="*60)
    
    model.eval()
    
    # Generate test samples from same distribution
    test_dataset = SyntheticAffordanceDataset(
        num_samples=30,
        num_points=256,
        shape_types=[0, 1, 2],  # Same as training
        seed=999
    )
    
    coherence_signals = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            pos = sample['points']  # (256, 3)
            
            results = model(pos, batch=None)
            coherence = results['coherence_signal'].item()
            coherence_signals.append(coherence)
    
    mean_coherence = np.mean(coherence_signals)
    std_coherence = np.std(coherence_signals)
    
    print(f"\nKnown shapes coherence signal:")
    print(f"  Mean: {mean_coherence:.4f}")
    print(f"  Std:  {std_coherence:.4f}")
    print(f"  Range: [{np.min(coherence_signals):.4f}, {np.max(coherence_signals):.4f}]")
    
    return coherence_signals


def create_novel_shape(num_points=256):
    """
    Create a novel shape not in training distribution.
    Here we create a torus, which is topologically different from cube/cylinder/sphere.
    """
    # Torus parameters
    R = 0.5  # Major radius
    r = 0.2  # Minor radius
    
    # Sample points on torus surface
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, 2*np.pi, num_points)
    
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    points = np.stack([x, y, z], axis=1)
    points = torch.from_numpy(points).float()
    
    return points


def test_coherence_on_novel_shapes(model, num_samples=30):
    """Test coherence signal on novel shapes (torus)."""
    print("\n" + "="*60)
    print("STEP 3: Testing coherence signal on NOVEL shapes (torus)")
    print("="*60)
    
    model.eval()
    
    coherence_signals = []
    
    with torch.no_grad():
        for i in range(num_samples):
            pos = create_novel_shape(num_points=256)
            
            results = model(pos, batch=None)
            coherence = results['coherence_signal'].item()
            coherence_signals.append(coherence)
    
    mean_coherence = np.mean(coherence_signals)
    std_coherence = np.std(coherence_signals)
    
    print(f"\nNovel shapes coherence signal:")
    print(f"  Mean: {mean_coherence:.4f}")
    print(f"  Std:  {std_coherence:.4f}")
    print(f"  Range: [{np.min(coherence_signals):.4f}, {np.max(coherence_signals):.4f}]")
    
    return coherence_signals


def visualize_results(known_coherence, novel_coherence):
    """Visualize coherence signal distribution."""
    print("\n" + "="*60)
    print("STEP 4: Visualizing results")
    print("="*60)
    
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(known_coherence, bins=15, alpha=0.7, label='Known shapes', color='blue')
    plt.hist(novel_coherence, bins=15, alpha=0.7, label='Novel shapes', color='red')
    plt.xlabel('Coherence Signal')
    plt.ylabel('Frequency')
    plt.title('Coherence Signal Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot([known_coherence, novel_coherence], labels=['Known', 'Novel'])
    plt.ylabel('Coherence Signal')
    plt.title('Coherence Signal Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./logs/coherence_test/coherence_comparison.png', dpi=150)
    print(f"\nSaved visualization: ./logs/coherence_test/coherence_comparison.png")
    
    return './logs/coherence_test/coherence_comparison.png'


def main():
    print("\n" + "="*70)
    print(" EXPERIMENT: Coherence Signal on Known vs Novel Shapes")
    print("="*70)
    print("\nHypothesis:")
    print("  - Known shapes (from training) → LOW coherence signal")
    print("  - Novel shapes (unseen topology) → HIGH coherence signal")
    print("\nThis tests whether the adjunction F ⊣ G captures the")
    print("'functional core' and can detect when it breaks down.")
    print("="*70 + "\n")
    
    # Train model
    model = train_model(num_epochs=15)
    
    # Test on known shapes
    known_coherence = test_coherence_on_known_shapes(model)
    
    # Test on novel shapes
    novel_coherence = test_coherence_on_novel_shapes(model)
    
    # Visualize
    viz_path = visualize_results(known_coherence, novel_coherence)
    
    # Statistical test
    print("\n" + "="*60)
    print("STEP 5: Statistical Analysis")
    print("="*60)
    
    known_mean = np.mean(known_coherence)
    novel_mean = np.mean(novel_coherence)
    difference = novel_mean - known_mean
    percent_increase = (difference / known_mean) * 100
    
    print(f"\nCoherence signal comparison:")
    print(f"  Known shapes mean:  {known_mean:.4f}")
    print(f"  Novel shapes mean:  {novel_mean:.4f}")
    print(f"  Difference:         {difference:.4f}")
    print(f"  Percent increase:   {percent_increase:.1f}%")
    
    # Conclusion
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if novel_mean > known_mean * 1.2:  # At least 20% higher
        print("✓ HYPOTHESIS CONFIRMED")
        print(f"  Novel shapes show {percent_increase:.1f}% higher coherence signal.")
        print("  The adjunction F ⊣ G successfully detects distributional shift.")
        print("  This validates the theoretical prediction that coherence signal")
        print("  measures the stability of the adjunction structure.")
    else:
        print("✗ HYPOTHESIS NOT CONFIRMED")
        print(f"  Novel shapes show only {percent_increase:.1f}% higher coherence signal.")
        print("  The model may need:")
        print("    - More training epochs")
        print("    - Different loss weights (λ_aff, λ_coh)")
        print("    - Larger model capacity")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    import os
    os.makedirs('./logs/coherence_test', exist_ok=True)
    main()
