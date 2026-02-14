"""
Integration test for the temporal suspension experiment.

Validates:
1. All imports resolve correctly
2. TemporalShapeDataset generates valid data
3. AdjunctionModel forward pass works with temporal data
4. ConfidenceGate and ShapeClassifier produce correct shapes
5. TemporalSuspensionTrainer can run one training step
"""

import sys
sys.path.append('/home/ubuntu/adjunction-model')

import torch
from torch.utils.data import DataLoader

from src.data.temporal_dataset import (
    TemporalShapeDataset,
    collate_temporal_batch,
)
from src.models.adjunction_model import AdjunctionModel
from experiments.temporal_suspension_experiment import (
    ConfidenceGate,
    ShapeClassifier,
    TemporalSuspensionTrainer,
)


def test_dataset():
    """Test dataset generation and collation."""
    print("1. Testing TemporalShapeDataset...")
    ds = TemporalShapeDataset(num_samples=8, num_points_final=256,
                              num_time_steps=6, seed=0)
    assert len(ds) == 8
    sample = ds[0]
    assert len(sample['points_sequence']) == 6
    assert sample['points_final'].shape == (256, 3)
    assert sample['affordances'].shape == (256, 5)
    assert sample['ambiguity_schedule'].shape == (6,)

    loader = DataLoader(ds, batch_size=4, collate_fn=collate_temporal_batch)
    batch = next(iter(loader))
    assert len(batch['points_sequence']) == 6
    assert batch['affordances'].shape == (4, 256, 5)
    assert batch['ambiguity_schedule'].shape == (4, 6)
    print("   PASSED")


def test_model_forward():
    """Test AdjunctionModel forward pass with temporal data."""
    print("2. Testing AdjunctionModel forward pass...")
    model = AdjunctionModel(
        num_affordances=5, num_points=256, f_hidden_dim=64,
        g_hidden_dim=128, agent_hidden_dim=256, agent_latent_dim=64,
        context_dim=128, valence_dim=32, valence_decay=0.1,
        alpha_curiosity=0.0, beta_competence=0.6, gamma_novelty=0.4,
    )
    model.eval()

    ds = TemporalShapeDataset(num_samples=4, num_points_final=256,
                              num_time_steps=4, seed=0)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_temporal_batch)
    batch = next(iter(loader))

    B = 2
    state = model.initial_state(B, torch.device('cpu'))
    coh = torch.zeros(B, 1)

    for t in range(4):
        pos = batch['points_sequence'][t]['points']
        bidx = batch['points_sequence'][t]['batch']
        N = pos.size(0)
        coh_spatial = torch.zeros(N)

        # Remove stale per-point tensors whose length matches the
        # *previous* step's N (same fix as TemporalSuspensionTrainer._step)
        state_clean = {k: v for k, v in state.items()
                       if k not in ('priority_normalized',)}

        with torch.no_grad():
            results = model(pos, bidx, state_clean, coh, coh_spatial)

        assert results['affordances'].shape[0] == N
        assert results['affordances'].shape[1] == 5
        assert results['coherence_signal'].shape == (B, 1)
        assert results['counit_signal'].shape == (B, 1)
        assert results['context'].shape == (B, 128)

        state = results['agent_state']
        coh = results['coherence_signal']
        coh_spatial = results['coherence_spatial']

    print("   PASSED")


def test_confidence_gate():
    """Test ConfidenceGate shape."""
    print("3. Testing ConfidenceGate...")
    gate = ConfidenceGate(context_dim=128)
    ctx = torch.randn(4, 128)
    out = gate(ctx)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()
    print("   PASSED")


def test_classifier():
    """Test ShapeClassifier shape."""
    print("4. Testing ShapeClassifier...")
    clf = ShapeClassifier(num_affordances=5, num_classes=3)
    aff = torch.randn(4, 5)
    logits = clf(aff)
    assert logits.shape == (4, 3)
    print("   PASSED")


def test_trainer_one_step():
    """Test that the trainer can run one training step without error."""
    print("5. Testing TemporalSuspensionTrainer (1 epoch)...")
    model = AdjunctionModel(
        num_affordances=5, num_points=256, f_hidden_dim=64,
        g_hidden_dim=128, agent_hidden_dim=256, agent_latent_dim=64,
        context_dim=128, valence_dim=32, valence_decay=0.1,
        alpha_curiosity=0.0, beta_competence=0.6, gamma_novelty=0.4,
    )
    gate = ConfidenceGate(context_dim=128)
    clf = ShapeClassifier(num_affordances=5, num_classes=3)

    trainer = TemporalSuspensionTrainer(
        model=model, confidence_gate=gate, classifier=clf,
        device=torch.device('cpu'), lr=1e-3,
        confidence_threshold=0.5, mode='slack',
    )

    ds = TemporalShapeDataset(num_samples=8, num_points_final=256,
                              num_time_steps=4, seed=0)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_temporal_batch)

    metrics = trainer.train_epoch(loader, epoch=0)
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'eta_by_step' in metrics
    assert isinstance(metrics['eta_by_step'], dict)
    print(f"   Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2f}")
    print("   PASSED")


def test_trainer_tight_mode():
    """Test tight mode (with reconstruction loss)."""
    print("6. Testing TemporalSuspensionTrainer tight mode...")
    model = AdjunctionModel(
        num_affordances=5, num_points=256, f_hidden_dim=64,
        g_hidden_dim=128, agent_hidden_dim=256, agent_latent_dim=64,
        context_dim=128, valence_dim=32, valence_decay=0.1,
        alpha_curiosity=0.0, beta_competence=0.6, gamma_novelty=0.4,
    )
    gate = ConfidenceGate(context_dim=128)
    clf = ShapeClassifier(num_affordances=5, num_classes=3)

    trainer = TemporalSuspensionTrainer(
        model=model, confidence_gate=gate, classifier=clf,
        device=torch.device('cpu'), lr=1e-3,
        confidence_threshold=0.5, mode='tight',
        lambda_recon=1.0,
    )

    ds = TemporalShapeDataset(num_samples=8, num_points_final=256,
                              num_time_steps=4, seed=0)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_temporal_batch)

    metrics = trainer.train_epoch(loader, epoch=0)
    assert metrics['recon'] > 0, "Tight mode should have non-zero recon loss"
    print(f"   Loss={metrics['loss']:.4f}, Recon={metrics['recon']:.4f}")
    print("   PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("Temporal Suspension Integration Tests")
    print("=" * 60)
    print()

    test_dataset()
    test_model_forward()
    test_confidence_gate()
    test_classifier()
    test_trainer_one_step()
    test_trainer_tight_mode()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
