"""
Temporal Suspension Experiment

This experiment validates whether an agent with slack (Phase 2 trained) can
appropriately *defer* action when information is ambiguous and *commit* only
when the shape becomes sufficiently clear.

Theoretical basis:
    docs/docs/docs/01_temporal_suspension.md
    docs/theory/suspension_and_confidence.md

Experiment design:
    1. A shape is progressively revealed over T time steps.
    2. At each step the model processes the current partial point cloud and
       produces η(t) (unit slack) and ε(t) (counit slack).
    3. Agent C decides "act" or "wait" based on a learned confidence gate.
    4. We compare two conditions:
       (a) Phase-2-Slack model  — trained with slack preservation
       (b) Phase-1-Only model   — trained with reconstruction loss (tight η)
    5. Metrics: η(t) trajectory, action timing, classification accuracy.

Key hypothesis:
    The slack model should *wait longer* (defer action) when the shape is
    ambiguous, and achieve higher accuracy when it finally acts.

Implementation notes:
    - Follows the same structure as phase2_slack_experiment.py
    - Uses collate_temporal_batch from temporal_dataset.py
    - Reuses AdjunctionModel and Phase2SlackTrainer from existing codebase
    - CPU-friendly: 50 epochs, small dataset
"""

import sys
sys.path.append('/home/ubuntu/adjunction-model')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models.adjunction_model import AdjunctionModel
from src.data.temporal_dataset import (
    TemporalShapeDataset,
    collate_temporal_batch,
)
from src.training.train_phase2_slack import Phase2SlackTrainer


# ======================================================================
# Confidence Gate — decides "act" vs "wait"
# ======================================================================

class ConfidenceGate(nn.Module):
    """
    A small network that maps the agent's internal state to a scalar
    confidence c(t) in [0, 1].  The agent "acts" when c(t) > threshold.

    Input:  agent context vector (B, context_dim)
    Output: confidence (B, 1) via sigmoid
    """

    def __init__(self, context_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Returns confidence in [0, 1], shape (B, 1)."""
        return torch.sigmoid(self.net(context))


# ======================================================================
# Affordance Classifier — maps per-point affordances to shape category
# ======================================================================

class ShapeClassifier(nn.Module):
    """
    Maps aggregated affordance predictions to a shape-category logit.

    Input:  affordance vector (B, num_affordances)
    Output: logits (B, num_classes)
    """

    def __init__(self, num_affordances: int = 5, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_affordances, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, affordances: torch.Tensor) -> torch.Tensor:
        return self.net(affordances)


# ======================================================================
# Temporal Suspension Trainer
# ======================================================================

class TemporalSuspensionTrainer:
    """
    Trainer for the temporal suspension experiment.

    For each sequence:
        1. Process time steps 0 … T-1 through the adjunction model.
        2. At each step, record η(t), ε(t), confidence c(t).
        3. The agent "acts" at the first step where c(t) > threshold.
        4. Loss = classification loss (at action time) + timing penalty.

    Two training modes:
        - 'slack':   Phase 2 Slack model (no reconstruction loss)
        - 'tight':   Phase 1 model (with reconstruction loss → η minimised)
    """

    def __init__(
        self,
        model: AdjunctionModel,
        confidence_gate: ConfidenceGate,
        classifier: ShapeClassifier,
        device: torch.device = torch.device('cpu'),
        lr: float = 1e-4,
        confidence_threshold: float = 0.5,
        lambda_aff: float = 1.0,
        lambda_kl: float = 0.1,
        lambda_coherence: float = 0.1,
        lambda_timing: float = 0.05,
        lambda_cls: float = 1.0,
        mode: str = 'slack',
        lambda_recon: float = 1.0,
    ):
        """
        Args:
            model: AdjunctionModel instance
            confidence_gate: ConfidenceGate instance
            classifier: ShapeClassifier instance
            device: torch device
            lr: learning rate
            confidence_threshold: threshold for "act" decision
            lambda_aff: weight for affordance loss
            lambda_kl: weight for KL divergence
            lambda_coherence: weight for coherence regularization
            lambda_timing: weight for timing penalty (encourages waiting)
            lambda_cls: weight for classification loss
            mode: 'slack' (no L_recon) or 'tight' (with L_recon)
            lambda_recon: weight for reconstruction loss (only in 'tight' mode)
        """
        self.model = model.to(device)
        self.confidence_gate = confidence_gate.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.mode = mode

        # Loss weights
        self.lambda_aff = lambda_aff
        self.lambda_kl = lambda_kl
        self.lambda_coherence = lambda_coherence
        self.lambda_timing = lambda_timing
        self.lambda_cls = lambda_cls
        self.lambda_recon = lambda_recon

        # Loss functions
        self.aff_criterion = nn.MSELoss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.recon_criterion = nn.MSELoss()

        # Single optimizer for all parameters
        all_params = (
            list(model.parameters()) +
            list(confidence_gate.parameters()) +
            list(classifier.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=lr)

    # ------------------------------------------------------------------
    # Process a single time step
    # ------------------------------------------------------------------

    def _step(
        self,
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        agent_state: Dict[str, torch.Tensor],
        coherence_prev: torch.Tensor,
        coherence_spatial_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one forward pass of the adjunction model on a partial point cloud.

        Returns:
            results: full model output dict
            eta: (B, 1) unit slack
            eps: (B, 1) counit slack
            confidence: (B, 1) confidence gate output
            affordances_batched: (B, num_affordances) mean affordance per sample
        """
        N = pos.size(0)

        # In temporal sequences each step has a different number of points.
        # coherence_spatial_prev comes from the *previous* step with N_{t-1}
        # points, but the current step has N_t points.  We must provide a
        # spatial coherence vector of length N_t.  When the sizes differ we
        # simply reset to zeros — this is safe because the model uses
        # coherence_spatial only as an input signal, and the first step's
        # spatial coherence is always zero anyway.
        if coherence_spatial_prev is None or coherence_spatial_prev.size(0) != N:
            coherence_spatial_prev = torch.zeros(N, device=self.device)

        # The agent_state may carry 'priority_normalized' from the previous
        # time step, whose length matches the *previous* point count N_{t-1}.
        # Remove it so that AgentC falls back to uniform attention.
        state_clean = {k: v for k, v in agent_state.items()
                       if k not in ('priority_normalized',)}

        results = self.model(
            pos, batch_idx, state_clean, coherence_prev, coherence_spatial_prev
        )

        eta = results['coherence_signal']    # (B, 1)
        eps = results['counit_signal']       # (B, 1)
        context = results['context']         # (B, context_dim)

        confidence = self.confidence_gate(context)  # (B, 1)

        # Aggregate per-point affordances to per-sample
        affordances = results['affordances']  # (N, num_affordances)
        batch_size = batch_idx.max().item() + 1
        num_aff = affordances.size(-1)
        aff_batched = torch.zeros(batch_size, num_aff, device=self.device)
        for b in range(batch_size):
            mask = (batch_idx == b)
            if mask.sum() > 0:
                aff_batched[b] = affordances[mask].mean(dim=0)

        return results, eta, eps, confidence, aff_batched

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch over temporal sequences."""
        self.model.train()
        self.confidence_gate.train()
        self.classifier.train()

        accum = {
            'loss': 0.0, 'aff': 0.0, 'kl': 0.0, 'coherence': 0.0,
            'cls': 0.0, 'timing': 0.0, 'recon': 0.0,
            'mean_action_step': 0.0, 'accuracy': 0.0,
        }
        # Per-step η/ε accumulators (keyed by step index)
        eta_by_step: Dict[int, List[float]] = {}
        eps_by_step: Dict[int, List[float]] = {}
        conf_by_step: Dict[int, List[float]] = {}
        num_batches = 0

        for batch_data in dataloader:
            pts_seq = batch_data['points_sequence']   # list of T dicts
            affordances_gt = batch_data['affordances'].to(self.device)
            shape_types = batch_data['shape_types']
            ambiguity = batch_data['ambiguity_schedule'].to(self.device)

            T = len(pts_seq)
            B = affordances_gt.size(0)

            # Initialise agent state
            agent_state = self.model.initial_state(B, self.device)
            coherence_prev = torch.zeros(B, 1, device=self.device)
            coherence_spatial_prev = None

            # Accumulators for this sequence
            etas: List[torch.Tensor] = []
            epss: List[torch.Tensor] = []
            confs: List[torch.Tensor] = []
            aff_preds: List[torch.Tensor] = []
            kl_losses: List[torch.Tensor] = []

            # ---- Process each time step ----
            for t in range(T):
                pos_t = pts_seq[t]['points'].to(self.device)
                batch_t = pts_seq[t]['batch'].to(self.device)

                results, eta_t, eps_t, conf_t, aff_t = self._step(
                    pos_t, batch_t, agent_state, coherence_prev,
                    coherence_spatial_prev,
                )

                etas.append(eta_t)
                epss.append(eps_t)
                confs.append(conf_t)
                aff_preds.append(aff_t)

                # KL divergence
                rssm_info = results['rssm_info']
                kl_t = self.model.agent_c.rssm.kl_divergence(
                    rssm_info['posterior_mean'], rssm_info['posterior_std'],
                    rssm_info['prior_mean'], rssm_info['prior_std'],
                ).mean()
                kl_losses.append(kl_t)

                # Update carry-over state
                agent_state = results['agent_state']
                coherence_prev = eta_t.detach()
                coherence_spatial_prev = results['coherence_spatial'].detach()

                # Record per-step stats
                eta_by_step.setdefault(t, []).append(eta_t.mean().item())
                eps_by_step.setdefault(t, []).append(eps_t.mean().item())
                conf_by_step.setdefault(t, []).append(conf_t.mean().item())

            # ---- Stack temporal tensors ----
            etas_t = torch.stack(etas, dim=1)     # (B, T, 1)
            confs_t = torch.stack(confs, dim=1)   # (B, T, 1)
            aff_preds_t = torch.stack(aff_preds, dim=1)  # (B, T, num_aff)

            # ---- Determine action step per sample ----
            # "act" at first step where confidence > threshold
            conf_sq = confs_t.squeeze(-1)         # (B, T)
            acted = conf_sq > self.confidence_threshold
            # If never acted, default to last step
            action_steps = torch.full((B,), T - 1, dtype=torch.long,
                                      device=self.device)
            for b in range(B):
                indices = torch.where(acted[b])[0]
                if len(indices) > 0:
                    action_steps[b] = indices[0]

            # ---- Compute losses ----

            # 1. Affordance loss at the action step
            aff_at_action = aff_preds_t[
                torch.arange(B, device=self.device), action_steps]  # (B, A)
            # Ground truth: average affordance per sample
            B_gt, N_gt, A_gt = affordances_gt.shape
            aff_gt_mean = affordances_gt.mean(dim=1)  # (B, A)
            L_aff = self.aff_criterion(aff_at_action, aff_gt_mean)

            # 2. KL divergence (average over steps)
            L_kl = torch.stack(kl_losses).mean()

            # 3. Coherence regularization (prevent η collapse)
            eta_mean = etas_t.mean()
            L_coherence = -torch.log(eta_mean + 1e-8)

            # 4. Classification loss at action step
            logits = self.classifier(aff_at_action)
            targets = torch.tensor(shape_types, dtype=torch.long,
                                   device=self.device)
            L_cls = self.cls_criterion(logits, targets)

            # 5. Timing penalty: penalise acting too early (encourage waiting)
            # Reward = (action_step / T) — higher is better (waited longer)
            # Penalty = 1 - reward = fraction of unused steps
            # But we also penalise acting too late (time cost)
            # Net: quadratic around optimal timing
            normalised_step = action_steps.float() / (T - 1)
            # Ideal: act when ambiguity is low.  Use ambiguity at action step.
            amb_at_action = ambiguity[
                torch.arange(B, device=self.device), action_steps]
            # Penalty: high if acted while ambiguity is still high
            L_timing = (amb_at_action ** 2).mean()

            # 6. Reconstruction loss (only in 'tight' mode)
            L_recon = torch.tensor(0.0, device=self.device)
            if self.mode == 'tight':
                # Use final step's reconstructed vs original
                pts_final = batch_data['points_final']['points'].to(self.device)
                batch_final = batch_data['points_final']['batch'].to(self.device)
                # Re-run model on final points to get reconstruction
                with torch.no_grad():
                    final_state = self.model.initial_state(B, self.device)
                    final_coh = torch.zeros(B, 1, device=self.device)
                results_final = self.model(
                    pts_final, batch_final, final_state, final_coh)
                reconstructed = results_final['reconstructed']  # (B, P, 3)
                # Convert original to batched form for MSE
                pts_batched = []
                for b_i in range(B):
                    mask = (batch_final == b_i)
                    pts_batched.append(pts_final[mask])
                max_n = max(p.size(0) for p in pts_batched)
                pts_padded = torch.stack([
                    torch.cat([p, torch.zeros(max_n - p.size(0), 3,
                                              device=self.device)])
                    for p in pts_batched
                ])
                # Truncate to match
                min_n = min(pts_padded.size(1), reconstructed.size(1))
                L_recon = self.recon_criterion(
                    reconstructed[:, :min_n, :], pts_padded[:, :min_n, :])

            # ---- Total loss ----
            loss = (
                self.lambda_aff * L_aff
                + self.lambda_kl * L_kl
                + self.lambda_coherence * L_coherence
                + self.lambda_cls * L_cls
                + self.lambda_timing * L_timing
                + (self.lambda_recon * L_recon if self.mode == 'tight' else 0)
            )

            # ---- Backward ----
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) +
                list(self.confidence_gate.parameters()) +
                list(self.classifier.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            # ---- Accuracy ----
            preds = logits.argmax(dim=-1)
            correct = (preds == targets).float().mean().item()

            # ---- Accumulate ----
            accum['loss'] += loss.item()
            accum['aff'] += L_aff.item()
            accum['kl'] += L_kl.item()
            accum['coherence'] += L_coherence.item()
            accum['cls'] += L_cls.item()
            accum['timing'] += L_timing.item()
            accum['recon'] += L_recon.item()
            accum['mean_action_step'] += action_steps.float().mean().item()
            accum['accuracy'] += correct
            num_batches += 1

            if num_batches % 5 == 0:
                print(
                    f"  Batch {num_batches}: "
                    f"Loss={loss.item():.4f}, "
                    f"Cls={L_cls.item():.4f}, "
                    f"Acc={correct:.2f}, "
                    f"ActStep={action_steps.float().mean().item():.1f}/{T-1}"
                )

        # ---- Average metrics ----
        metrics = {k: v / max(num_batches, 1) for k, v in accum.items()}

        # Per-step η, ε, confidence
        metrics['eta_by_step'] = {
            t: float(np.mean(vals)) for t, vals in eta_by_step.items()}
        metrics['eps_by_step'] = {
            t: float(np.mean(vals)) for t, vals in eps_by_step.items()}
        metrics['conf_by_step'] = {
            t: float(np.mean(vals)) for t, vals in conf_by_step.items()}

        return metrics

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on temporal sequences (no gradient)."""
        self.model.eval()
        self.confidence_gate.eval()
        self.classifier.eval()

        accum = {
            'accuracy': 0.0, 'mean_action_step': 0.0,
            'unit_eta_final': 0.0, 'counit_eps_final': 0.0,
        }
        eta_by_step: Dict[int, List[float]] = {}
        eps_by_step: Dict[int, List[float]] = {}
        conf_by_step: Dict[int, List[float]] = {}
        num_batches = 0

        for batch_data in dataloader:
            pts_seq = batch_data['points_sequence']
            affordances_gt = batch_data['affordances'].to(self.device)
            shape_types = batch_data['shape_types']

            T = len(pts_seq)
            B = affordances_gt.size(0)

            agent_state = self.model.initial_state(B, self.device)
            coherence_prev = torch.zeros(B, 1, device=self.device)
            coherence_spatial_prev = None

            confs_list = []
            aff_list = []
            last_eta = None
            last_eps = None

            for t in range(T):
                pos_t = pts_seq[t]['points'].to(self.device)
                batch_t = pts_seq[t]['batch'].to(self.device)

                results, eta_t, eps_t, conf_t, aff_t = self._step(
                    pos_t, batch_t, agent_state, coherence_prev,
                    coherence_spatial_prev,
                )
                confs_list.append(conf_t)
                aff_list.append(aff_t)
                last_eta = eta_t
                last_eps = eps_t

                agent_state = results['agent_state']
                coherence_prev = eta_t
                coherence_spatial_prev = results['coherence_spatial']

                eta_by_step.setdefault(t, []).append(eta_t.mean().item())
                eps_by_step.setdefault(t, []).append(eps_t.mean().item())
                conf_by_step.setdefault(t, []).append(conf_t.mean().item())

            confs_t = torch.stack(confs_list, dim=1).squeeze(-1)  # (B, T)
            aff_preds_t = torch.stack(aff_list, dim=1)            # (B, T, A)

            acted = confs_t > self.confidence_threshold
            action_steps = torch.full((B,), T - 1, dtype=torch.long,
                                      device=self.device)
            for b in range(B):
                indices = torch.where(acted[b])[0]
                if len(indices) > 0:
                    action_steps[b] = indices[0]

            aff_at_action = aff_preds_t[
                torch.arange(B, device=self.device), action_steps]
            logits = self.classifier(aff_at_action)
            targets = torch.tensor(shape_types, dtype=torch.long,
                                   device=self.device)
            preds = logits.argmax(dim=-1)
            correct = (preds == targets).float().mean().item()

            accum['accuracy'] += correct
            accum['mean_action_step'] += action_steps.float().mean().item()
            accum['unit_eta_final'] += last_eta.mean().item()
            accum['counit_eps_final'] += last_eps.mean().item()
            num_batches += 1

        metrics = {k: v / max(num_batches, 1) for k, v in accum.items()}
        metrics['eta_by_step'] = {
            t: float(np.mean(vals)) for t, vals in eta_by_step.items()}
        metrics['eps_by_step'] = {
            t: float(np.mean(vals)) for t, vals in eps_by_step.items()}
        metrics['conf_by_step'] = {
            t: float(np.mean(vals)) for t, vals in conf_by_step.items()}
        return metrics


# ======================================================================
# Main experiment runner
# ======================================================================

def run_temporal_suspension_experiment(
    num_epochs: int = 50,
    num_samples: int = 100,
    num_points_final: int = 512,
    num_time_steps: int = 8,
    batch_size: int = 4,
    lr: float = 1e-4,
    confidence_threshold: float = 0.5,
    device: str = 'cpu',
):
    """
    Run the full temporal suspension experiment.

    Trains two models:
        (a) 'slack' — Phase 2 Slack (no reconstruction loss)
        (b) 'tight' — Phase 1 style (with reconstruction loss)

    Compares their temporal behaviour.
    """
    device = torch.device(device)
    print(f"Device: {device}")

    output_dir = Path("/home/ubuntu/adjunction-model/results/temporal_suspension")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    print(f"\nCreating temporal dataset: {num_samples} samples, "
          f"{num_time_steps} steps, {num_points_final} final points")
    dataset = TemporalShapeDataset(
        num_samples=num_samples,
        num_points_final=num_points_final,
        num_time_steps=num_time_steps,
        seed=42,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_temporal_batch,
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_temporal_batch,
    )

    all_histories = {}

    for mode in ['slack', 'tight']:
        print(f"\n{'=' * 80}")
        print(f"Training mode: {mode.upper()}")
        print(f"{'=' * 80}")

        # ---- Model ----
        model = AdjunctionModel(
            num_affordances=5,
            num_points=num_points_final,
            f_hidden_dim=64,
            g_hidden_dim=128,
            agent_hidden_dim=256,
            agent_latent_dim=64,
            context_dim=128,
            valence_dim=32,
            valence_decay=0.1,
            alpha_curiosity=0.0,
            beta_competence=0.6,
            gamma_novelty=0.4,
        )
        gate = ConfidenceGate(context_dim=128)
        classifier = ShapeClassifier(num_affordances=5, num_classes=3)

        trainer = TemporalSuspensionTrainer(
            model=model,
            confidence_gate=gate,
            classifier=classifier,
            device=device,
            lr=lr,
            confidence_threshold=confidence_threshold,
            mode=mode,
            lambda_aff=1.0,
            lambda_kl=0.1,
            lambda_coherence=0.1 if mode == 'slack' else 0.0,
            lambda_timing=0.05,
            lambda_cls=1.0,
            lambda_recon=1.0 if mode == 'tight' else 0.0,
        )

        history: Dict[str, list] = {
            'loss': [], 'aff': [], 'kl': [], 'coherence': [],
            'cls': [], 'timing': [], 'recon': [],
            'mean_action_step': [], 'accuracy': [],
            'val_accuracy': [], 'val_action_step': [],
            'val_eta_final': [], 'val_eps_final': [],
            'eta_by_step': [], 'eps_by_step': [], 'conf_by_step': [],
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs} [{mode}]")
            print("-" * 60)

            train_m = trainer.train_epoch(dataloader, epoch)
            val_m = trainer.validate(val_loader)

            # Record scalar metrics
            for key in ['loss', 'aff', 'kl', 'coherence', 'cls', 'timing',
                        'recon', 'mean_action_step', 'accuracy']:
                history[key].append(train_m[key])
            history['val_accuracy'].append(val_m['accuracy'])
            history['val_action_step'].append(val_m['mean_action_step'])
            history['val_eta_final'].append(val_m['unit_eta_final'])
            history['val_eps_final'].append(val_m['counit_eps_final'])

            # Record per-step metrics
            history['eta_by_step'].append(train_m['eta_by_step'])
            history['eps_by_step'].append(train_m['eps_by_step'])
            history['conf_by_step'].append(train_m['conf_by_step'])

            print(
                f"  Loss={train_m['loss']:.4f}  "
                f"Aff={train_m['aff']:.4f}  "
                f"Cls={train_m['cls']:.4f}  "
                f"Acc={train_m['accuracy']:.2f}  "
                f"ActStep={train_m['mean_action_step']:.1f}  "
                f"η_mean={np.mean(list(train_m['eta_by_step'].values())):.4f}"
            )
            print(
                f"  [Val] Acc={val_m['accuracy']:.2f}  "
                f"ActStep={val_m['mean_action_step']:.1f}  "
                f"η_final={val_m['unit_eta_final']:.4f}  "
                f"ε_final={val_m['counit_eps_final']:.4f}"
            )

        all_histories[mode] = history

        # Save per-mode results
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Convert history to JSON-serialisable form
        history_json = {}
        for k, v in history.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                # Per-step dicts: convert int keys to str
                history_json[k] = [
                    {str(kk): vv for kk, vv in d.items()} for d in v
                ]
            else:
                history_json[k] = v
        with open(mode_dir / 'metrics.json', 'w') as f:
            json.dump(history_json, f, indent=2)

        # Save model
        torch.save(model.state_dict(), mode_dir / 'model.pt')
        torch.save(gate.state_dict(), mode_dir / 'gate.pt')
        torch.save(classifier.state_dict(), mode_dir / 'classifier.pt')

        print(f"\n{mode} results saved to {mode_dir}")

    # ---- Save combined summary ----
    summary = {
        'slack': {
            'final_accuracy': all_histories['slack']['val_accuracy'][-1],
            'final_action_step': all_histories['slack']['val_action_step'][-1],
            'final_eta': all_histories['slack']['val_eta_final'][-1],
            'final_eps': all_histories['slack']['val_eps_final'][-1],
        },
        'tight': {
            'final_accuracy': all_histories['tight']['val_accuracy'][-1],
            'final_action_step': all_histories['tight']['val_action_step'][-1],
            'final_eta': all_histories['tight']['val_eta_final'][-1],
            'final_eps': all_histories['tight']['val_eps_final'][-1],
        },
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Experiment complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")

    return all_histories


# ======================================================================
# Entry point
# ======================================================================

if __name__ == '__main__':
    run_temporal_suspension_experiment(
        num_epochs=50,
        num_samples=100,
        num_points_final=512,
        num_time_steps=8,
        batch_size=4,
        lr=1e-4,
        confidence_threshold=0.5,
        device='cpu',
    )
