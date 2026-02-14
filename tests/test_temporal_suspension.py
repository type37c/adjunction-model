"""
Integration tests for the Temporal Suspension Experiment (Active Assembly).

Tests:
    1. TemporalShapeDataset generates valid episodes
    2. collate_temporal_batch produces correct graph-format tensors
    3. DisplacementHead forward pass (context + affordances → displacement)
    4. chamfer_distance_graph computes valid distances
    5. Full model forward pass through one time step (_step)
    6. One training step (slack mode) runs without error
    7. One training step (tight mode) runs without error

These tests follow the project guideline:
    "Early integration tests (forward pass tests) catch 90% of shape bugs."
"""

import sys
sys.path.append('/home/ubuntu/adjunction-model')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.adjunction_model import AdjunctionModel
from src.data.temporal_dataset import (
    TemporalShapeDataset,
    collate_temporal_batch,
)
from experiments.temporal_suspension_experiment import (
    DisplacementHead,
    TemporalSuspensionTrainer,
    chamfer_distance_graph,
)


def test_dataset():
    """Test 1: Dataset generates valid episodes."""
    print("1. Testing TemporalShapeDataset ... ", end="")
    ds = TemporalShapeDataset(num_samples=5, num_points_final=64,
                              num_time_steps=4, seed=0)
    assert len(ds) == 5
    s = ds[0]
    assert s['initial_points'].dim() == 2
    assert s['initial_points'].size(1) == 3
    assert s['target_points'].shape == (64, 3)
    assert s['target_affordances'].shape == (64, 5)
    assert s['hint_mask'].shape[0] == s['initial_points'].size(0)
    assert s['revelation_counts'].size(0) == 4
    assert s['ambiguity_schedule'].size(0) == 4
    print("PASSED")


def test_collate():
    """Test 2: Collate function produces correct shapes."""
    print("2. Testing collate_temporal_batch ... ", end="")
    ds = TemporalShapeDataset(num_samples=8, num_points_final=64,
                              num_time_steps=4, seed=0)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_temporal_batch)
    batch = next(iter(loader))

    assert batch['initial_points'].dim() == 2
    assert batch['initial_points'].size(1) == 3
    assert batch['initial_batch'].dim() == 1
    assert batch['target_points'].shape == (4, 64, 3)
    assert batch['target_affordances'].shape == (4, 64, 5)
    assert batch['hint_mask'].dim() == 1
    assert batch['revelation_counts'].size(0) == 4
    assert batch['ambiguity_schedule'].shape == (4, 4)
    print("PASSED")


def test_displacement_head():
    """Test 3: DisplacementHead forward pass."""
    print("3. Testing DisplacementHead ... ", end="")
    head = DisplacementHead(context_dim=64, num_affordances=5)
    B, N = 2, 20
    context = torch.randn(B, 64)
    aff = torch.randn(N, 5)
    batch_idx = torch.cat([torch.zeros(10, dtype=torch.long),
                           torch.ones(10, dtype=torch.long)])
    disp = head(context, aff, batch_idx)
    assert disp.shape == (N, 3), f"Expected ({N}, 3), got {disp.shape}"
    print("PASSED")


def test_chamfer_distance():
    """Test 4: chamfer_distance_graph computes valid distances."""
    print("4. Testing chamfer_distance_graph ... ", end="")
    B, N, M = 2, 15, 20
    assembled = torch.randn(N, 3)
    target = torch.randn(B, M, 3)
    batch_idx = torch.cat([torch.zeros(8, dtype=torch.long),
                           torch.ones(7, dtype=torch.long)])
    cd_per, cd_mean = chamfer_distance_graph(assembled, target, batch_idx, B)
    assert cd_per.shape == (B,)
    assert cd_mean.dim() == 0
    assert cd_mean.item() >= 0
    # Identical clouds should have zero CD
    pts = torch.randn(10, 3)
    target_id = pts.unsqueeze(0)  # (1, 10, 3)
    batch_id = torch.zeros(10, dtype=torch.long)
    _, cd_zero = chamfer_distance_graph(pts, target_id, batch_id, 1)
    assert cd_zero.item() < 1e-5, f"Expected ~0, got {cd_zero.item()}"
    print("PASSED")


def _make_model_and_trainer(mode='slack'):
    """Helper: create small model + trainer for testing."""
    model = AdjunctionModel(
        num_affordances=5,
        num_points=64,
        f_hidden_dim=32,
        g_hidden_dim=64,
        agent_hidden_dim=128,
        agent_latent_dim=32,
        context_dim=64,
        valence_dim=16,
        valence_decay=0.1,
        alpha_curiosity=0.0,
        beta_competence=0.6,
        gamma_novelty=0.4,
    )
    head = DisplacementHead(context_dim=64, num_affordances=5)
    trainer = TemporalSuspensionTrainer(
        model=model,
        displacement_head=head,
        device=torch.device('cpu'),
        lr=1e-3,
        mode=mode,
    )
    return trainer


def test_single_step():
    """Test 5: Full model forward pass through one time step."""
    print("5. Testing single-step forward pass ... ", end="")
    trainer = _make_model_and_trainer('slack')
    B = 2
    N = 20
    pos = torch.randn(N, 3)
    batch_idx = torch.cat([torch.zeros(10, dtype=torch.long),
                           torch.ones(10, dtype=torch.long)])
    agent_state = trainer.model.initial_state(B, torch.device('cpu'))
    coherence_prev = torch.zeros(B, 1)

    out = trainer._step(pos, batch_idx, agent_state, coherence_prev, None)
    assert out['displacement'].shape == (N, 3)
    assert out['eta'].shape == (B, 1)
    assert out['eps'].shape == (B, 1)
    assert out['aff_batched'].shape == (B, 5)
    print("PASSED")


def test_train_step_slack():
    """Test 6: One training step (slack mode)."""
    print("6. Testing training step (slack) ... ", end="")
    trainer = _make_model_and_trainer('slack')
    ds = TemporalShapeDataset(num_samples=8, num_points_final=64,
                              num_time_steps=4, seed=0)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_temporal_batch)

    metrics = trainer.train_epoch(loader, epoch=0)
    assert 'loss' in metrics
    assert 'chamfer' in metrics
    assert 'disp_mag_by_step' in metrics
    assert 'cd_by_step' in metrics
    assert metrics['loss'] > 0
    print("PASSED")


def test_train_step_tight():
    """Test 7: One training step (tight mode)."""
    print("7. Testing training step (tight) ... ", end="")
    trainer = _make_model_and_trainer('tight')
    ds = TemporalShapeDataset(num_samples=8, num_points_final=64,
                              num_time_steps=4, seed=0)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_temporal_batch)

    metrics = trainer.train_epoch(loader, epoch=0)
    assert 'loss' in metrics
    assert 'recon' in metrics
    assert metrics['loss'] > 0
    print("PASSED")


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Temporal Suspension — Integration Tests (Active Assembly)")
    print("=" * 60)
    print()

    test_dataset()
    test_collate()
    test_displacement_head()
    test_chamfer_distance()
    test_single_step()
    test_train_step_slack()
    test_train_step_tight()

    print()
    print("=" * 60)
    print("All 7 tests PASSED!")
    print("=" * 60)
