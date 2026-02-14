"""
Temporal Suspension Experiment — Active Point-Cloud Assembly

This experiment validates whether an agent with slack (Phase 2 trained) can
appropriately *defer* large displacements when the target shape is ambiguous,
and *commit* decisively once the shape becomes clear.

Theoretical basis:
    docs/docs/docs/01_temporal_suspension.md
    docs/theory/suspension_and_confidence.md

Experiment design:
    1. Points start randomly scattered.  A target shape (sphere / cube /
       cylinder) is chosen per episode but NOT explicitly told to the agent.
       A small fraction of initial points are placed near the target surface
       as a "hint".
    2. At each time step t = 0 … T-1 the agent processes the current cloud
       through the adjunction model and produces a per-point displacement
       vector via a DisplacementHead.
    3. Points are moved by the predicted displacement.  New points are
       revealed (progressive schedule) and appended to the cloud.
    4. Reward = negative Chamfer Distance between the assembled cloud and
       the target shape.

    Two conditions:
        (a) 'slack'  — Phase 2 Slack model (no reconstruction loss)
        (b) 'tight'  — Phase 1 style (with reconstruction loss → η minimised)

Key hypothesis:
    The slack model should produce *small* displacements early (exploration)
    and *large* displacements later (commitment), achieving lower final
    Chamfer Distance.  The tight model should commit early and fail to
    correct.

Metrics recorded per step:
    - η(t), ε(t)
    - displacement magnitude ‖Δx(t)‖
    - Chamfer Distance to target
    - affordance prediction quality

Implementation notes:
    - Follows the structure of phase2_slack_experiment.py
    - Reuses AdjunctionModel from existing codebase
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


# ======================================================================
# Displacement Head — maps per-point features to displacement vectors
# ======================================================================

class DisplacementHead(nn.Module):
    """
    Produces a per-point displacement vector Δx ∈ R³ from the agent's
    context and per-point affordance features.

    Architecture:
        context (B, context_dim) is broadcast to every point, concatenated
        with per-point affordance features (N, num_affordances), then
        mapped to (N, 3) through a small MLP.

    The output is *not* clamped — the loss landscape naturally encourages
    the model to learn appropriate magnitudes.
    """

    def __init__(self, context_dim: int = 128, num_affordances: int = 5):
        super().__init__()
        input_dim = context_dim + num_affordances
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(
        self,
        context: torch.Tensor,
        affordances: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context:     (B, context_dim)
            affordances: (N, num_affordances)  per-point
            batch_idx:   (N,) long

        Returns:
            displacement: (N, 3)
        """
        # Broadcast context to each point
        ctx_per_point = context[batch_idx]          # (N, context_dim)
        feat = torch.cat([ctx_per_point, affordances], dim=-1)  # (N, C+A)
        return self.net(feat)                       # (N, 3)


# ======================================================================
# Chamfer Distance (differentiable, batch-aware)
# ======================================================================

def chamfer_distance_graph(
    assembled: torch.Tensor,
    target: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Chamfer Distance between assembled (graph-format) and target
    (batched-format) point clouds.

    Args:
        assembled: (N, 3)  graph-format assembled cloud
        target:    (B, M, 3)  target clouds (all same M)
        batch_idx: (N,) long  batch assignment for assembled
        batch_size: B

    Returns:
        cd_per_sample: (B,)  Chamfer Distance per sample
        cd_mean:       scalar  mean over batch
    """
    cd_list = []
    for b in range(batch_size):
        mask = (batch_idx == b)
        pts_a = assembled[mask]            # (n_b, 3)
        pts_t = target[b]                  # (M, 3)

        if pts_a.size(0) == 0:
            cd_list.append(torch.tensor(0.0, device=assembled.device))
            continue

        dist = torch.cdist(pts_a, pts_t)   # (n_b, M)
        d_a2t = dist.min(dim=1)[0].mean()  # assembled → target
        d_t2a = dist.min(dim=0)[0].mean()  # target → assembled
        cd_list.append((d_a2t + d_t2a) / 2)

    cd_per_sample = torch.stack(cd_list)
    return cd_per_sample, cd_per_sample.mean()


# ======================================================================
# Temporal Suspension Trainer (Active Assembly)
# ======================================================================

class TemporalSuspensionTrainer:
    """
    Trainer for the active point-cloud assembly experiment.

    For each episode:
        1. Start with scattered initial points.
        2. At each step t:
           a. Run adjunction model on current cloud → context, affordances,
              η(t), ε(t).
           b. DisplacementHead produces Δx(t) per point.
           c. Move points: x(t+1) = x(t) + Δx(t).
           d. Reveal new points (from target surface + noise) and append.
        3. Loss = Chamfer Distance (final assembled vs target)
                + affordance loss + KL + coherence regularisation
                + (optional) reconstruction loss for 'tight' mode.

    Two modes:
        'slack':  No reconstruction loss → η preserved
        'tight':  With reconstruction loss → η minimised
    """

    def __init__(
        self,
        model: AdjunctionModel,
        displacement_head: DisplacementHead,
        device: torch.device = torch.device('cpu'),
        lr: float = 1e-4,
        lambda_chamfer: float = 1.0,
        lambda_aff: float = 0.5,
        lambda_kl: float = 0.1,
        lambda_coherence: float = 0.1,
        lambda_recon: float = 1.0,
        mode: str = 'slack',
    ):
        self.model = model.to(device)
        self.displacement_head = displacement_head.to(device)
        self.device = device
        self.mode = mode

        # Loss weights
        self.lambda_chamfer = lambda_chamfer
        self.lambda_aff = lambda_aff
        self.lambda_kl = lambda_kl
        self.lambda_coherence = lambda_coherence
        self.lambda_recon = lambda_recon

        # Loss functions
        self.aff_criterion = nn.MSELoss()
        self.recon_criterion = nn.MSELoss()

        # Single optimiser for all trainable parameters
        all_params = (
            list(model.parameters())
            + list(displacement_head.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=lr)

    # ------------------------------------------------------------------
    # Process one time step
    # ------------------------------------------------------------------

    def _step(
        self,
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        agent_state: Dict[str, torch.Tensor],
        coherence_prev: torch.Tensor,
        coherence_spatial_prev: Optional[torch.Tensor],
    ) -> Dict:
        """
        Run the adjunction model + displacement head on the current cloud.

        Returns a dict with:
            results     – raw model outputs
            displacement – (N, 3) per-point displacement
            eta         – (B, 1)
            eps         – (B, 1)
            aff_batched – (B, A) mean affordance per sample
        """
        N = pos.size(0)

        # Reset per-point tensors that may have stale sizes
        if coherence_spatial_prev is None or coherence_spatial_prev.size(0) != N:
            coherence_spatial_prev = torch.zeros(N, device=self.device)

        state_clean = {k: v for k, v in agent_state.items()
                       if k != 'priority_normalized'}

        results = self.model(
            pos, batch_idx, state_clean, coherence_prev,
            coherence_spatial_prev,
        )

        eta = results['coherence_signal']       # (B, 1)
        eps = results['counit_signal']          # (B, 1)
        context = results['context']            # (B, context_dim)
        affordances = results['affordances']    # (N, A)

        # Displacement from context + per-point affordances
        displacement = self.displacement_head(
            context, affordances, batch_idx)     # (N, 3)

        # Aggregate affordances per sample for logging / loss
        B = batch_idx.max().item() + 1
        A = affordances.size(-1)
        aff_batched = torch.zeros(B, A, device=self.device)
        for b in range(B):
            mask = (batch_idx == b)
            if mask.sum() > 0:
                aff_batched[b] = affordances[mask].mean(dim=0)

        return {
            'results': results,
            'displacement': displacement,
            'eta': eta,
            'eps': eps,
            'aff_batched': aff_batched,
        }

    # ------------------------------------------------------------------
    # Reveal new points at step t
    # ------------------------------------------------------------------

    @staticmethod
    def _reveal_points(
        current_pos: torch.Tensor,
        current_batch: torch.Tensor,
        target_points: torch.Tensor,
        n_new: int,
        batch_size: int,
        noise_std: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append *n_new* points per sample, sampled from the target surface
        with added noise (simulating progressive revelation).

        Returns updated (pos, batch_idx).
        """
        new_pts_list = []
        new_batch_list = []
        for b in range(batch_size):
            tgt = target_points[b]                          # (M, 3)
            idx = torch.randint(0, tgt.size(0), (n_new,))
            pts = tgt[idx] + torch.randn(n_new, 3, device=device) * noise_std
            new_pts_list.append(pts)
            new_batch_list.append(
                torch.full((n_new,), b, dtype=torch.long, device=device))

        pos_new = torch.cat([current_pos] + new_pts_list, dim=0)
        batch_new = torch.cat([current_batch] + new_batch_list, dim=0)
        return pos_new, batch_new

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch over episodes."""
        self.model.train()
        self.displacement_head.train()

        accum = {
            'loss': 0.0, 'chamfer': 0.0, 'aff': 0.0, 'kl': 0.0,
            'coherence': 0.0, 'recon': 0.0,
        }
        # Per-step accumulators
        eta_by_step: Dict[int, List[float]] = {}
        eps_by_step: Dict[int, List[float]] = {}
        disp_mag_by_step: Dict[int, List[float]] = {}
        cd_by_step: Dict[int, List[float]] = {}
        num_batches = 0

        for batch_data in dataloader:
            init_pts = batch_data['initial_points'].to(self.device)
            init_batch = batch_data['initial_batch'].to(self.device)
            target_pts = batch_data['target_points'].to(self.device)
            target_aff = batch_data['target_affordances'].to(self.device)
            rev_counts = batch_data['revelation_counts']    # (T,) long
            shape_types = batch_data['shape_types']

            T = rev_counts.size(0)
            B = target_pts.size(0)

            # Initialise
            agent_state = self.model.initial_state(B, self.device)
            coherence_prev = torch.zeros(B, 1, device=self.device)
            coherence_spatial_prev = None

            pos = init_pts.clone()
            batch_idx = init_batch.clone()

            kl_losses: List[torch.Tensor] = []
            aff_preds: List[torch.Tensor] = []

            # ---- Process each time step ----
            for t in range(T):
                step_out = self._step(
                    pos, batch_idx, agent_state, coherence_prev,
                    coherence_spatial_prev,
                )

                results = step_out['results']
                displacement = step_out['displacement']
                eta_t = step_out['eta']
                eps_t = step_out['eps']
                aff_t = step_out['aff_batched']

                # Record displacement magnitude
                disp_mag = displacement.norm(dim=-1).mean().item()
                disp_mag_by_step.setdefault(t, []).append(disp_mag)

                # Move points
                pos = pos + displacement

                # Chamfer Distance at this step
                _, cd_mean = chamfer_distance_graph(
                    pos, target_pts, batch_idx, B)
                cd_by_step.setdefault(t, []).append(cd_mean.item())

                # KL
                rssm_info = results['rssm_info']
                kl_t = self.model.agent_c.rssm.kl_divergence(
                    rssm_info['posterior_mean'], rssm_info['posterior_std'],
                    rssm_info['prior_mean'], rssm_info['prior_std'],
                ).mean()
                kl_losses.append(kl_t)
                aff_preds.append(aff_t)

                # Carry-over state
                agent_state = results['agent_state']
                coherence_prev = eta_t.detach()
                coherence_spatial_prev = results['coherence_spatial'].detach()

                # Record per-step stats
                eta_by_step.setdefault(t, []).append(eta_t.mean().item())
                eps_by_step.setdefault(t, []).append(eps_t.mean().item())

                # Reveal new points (except at last step)
                if t < T - 1:
                    n_current = int(rev_counts[t].item())
                    n_next = int(rev_counts[t + 1].item())
                    n_new = max(0, n_next - n_current)
                    if n_new > 0:
                        pos, batch_idx = self._reveal_points(
                            pos, batch_idx, target_pts, n_new, B,
                            noise_std=0.05, device=self.device,
                        )

            # ---- Losses ----

            # 1. Chamfer Distance (final assembled vs target)
            _, L_chamfer = chamfer_distance_graph(
                pos, target_pts, batch_idx, B)

            # 2. Affordance loss (average over steps)
            aff_gt_mean = target_aff.mean(dim=1)        # (B, A)
            L_aff = torch.stack([
                self.aff_criterion(a, aff_gt_mean) for a in aff_preds
            ]).mean()

            # 3. KL divergence
            L_kl = torch.stack(kl_losses).mean()

            # 4. Coherence regularisation (prevent η collapse)
            eta_vals = torch.stack(
                [step_out['eta'] for _ in range(1)])  # use last step
            # Collect all η values from this episode
            eta_mean = eta_t.mean()
            L_coherence = -torch.log(eta_mean + 1e-8)

            # 5. Reconstruction loss (only 'tight' mode)
            L_recon = torch.tensor(0.0, device=self.device)
            if self.mode == 'tight':
                # Use the last step's reconstructed output
                reconstructed = results['reconstructed']  # (B, P, 3)
                # Build padded version of assembled cloud for MSE
                pts_per_sample = []
                for b_i in range(B):
                    mask = (batch_idx == b_i)
                    pts_per_sample.append(pos[mask])
                max_n = max(p.size(0) for p in pts_per_sample)
                pts_padded = torch.stack([
                    torch.cat([p, torch.zeros(max_n - p.size(0), 3,
                                              device=self.device)])
                    for p in pts_per_sample
                ])
                min_n = min(pts_padded.size(1), reconstructed.size(1))
                L_recon = self.recon_criterion(
                    reconstructed[:, :min_n, :], pts_padded[:, :min_n, :])

            # ---- Total loss ----
            loss = (
                self.lambda_chamfer * L_chamfer
                + self.lambda_aff * L_aff
                + self.lambda_kl * L_kl
                + self.lambda_coherence * L_coherence
                * (1.0 if self.mode == 'slack' else 0.0)
                + self.lambda_recon * L_recon
                * (1.0 if self.mode == 'tight' else 0.0)
            )

            # ---- Backward ----
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters())
                + list(self.displacement_head.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            # ---- Accumulate ----
            accum['loss'] += loss.item()
            accum['chamfer'] += L_chamfer.item()
            accum['aff'] += L_aff.item()
            accum['kl'] += L_kl.item()
            accum['coherence'] += L_coherence.item()
            accum['recon'] += L_recon.item()
            num_batches += 1

            if num_batches % 5 == 0:
                print(
                    f"  Batch {num_batches}: "
                    f"Loss={loss.item():.4f}, "
                    f"CD={L_chamfer.item():.4f}, "
                    f"Aff={L_aff.item():.4f}, "
                    f"‖Δx‖={disp_mag:.4f}"
                )

        # ---- Average metrics ----
        metrics = {k: v / max(num_batches, 1) for k, v in accum.items()}
        metrics['eta_by_step'] = {
            t: float(np.mean(v)) for t, v in eta_by_step.items()}
        metrics['eps_by_step'] = {
            t: float(np.mean(v)) for t, v in eps_by_step.items()}
        metrics['disp_mag_by_step'] = {
            t: float(np.mean(v)) for t, v in disp_mag_by_step.items()}
        metrics['cd_by_step'] = {
            t: float(np.mean(v)) for t, v in cd_by_step.items()}

        return metrics

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on episodes (no gradient)."""
        self.model.eval()
        self.displacement_head.eval()

        accum = {
            'chamfer': 0.0, 'aff': 0.0,
        }
        eta_by_step: Dict[int, List[float]] = {}
        eps_by_step: Dict[int, List[float]] = {}
        disp_mag_by_step: Dict[int, List[float]] = {}
        cd_by_step: Dict[int, List[float]] = {}
        num_batches = 0

        for batch_data in dataloader:
            init_pts = batch_data['initial_points'].to(self.device)
            init_batch = batch_data['initial_batch'].to(self.device)
            target_pts = batch_data['target_points'].to(self.device)
            target_aff = batch_data['target_affordances'].to(self.device)
            rev_counts = batch_data['revelation_counts']

            T = rev_counts.size(0)
            B = target_pts.size(0)

            agent_state = self.model.initial_state(B, self.device)
            coherence_prev = torch.zeros(B, 1, device=self.device)
            coherence_spatial_prev = None

            pos = init_pts.clone()
            batch_idx = init_batch.clone()

            for t in range(T):
                step_out = self._step(
                    pos, batch_idx, agent_state, coherence_prev,
                    coherence_spatial_prev,
                )
                results = step_out['results']
                displacement = step_out['displacement']

                disp_mag = displacement.norm(dim=-1).mean().item()
                disp_mag_by_step.setdefault(t, []).append(disp_mag)

                pos = pos + displacement

                _, cd_mean = chamfer_distance_graph(
                    pos, target_pts, batch_idx, B)
                cd_by_step.setdefault(t, []).append(cd_mean.item())

                agent_state = results['agent_state']
                coherence_prev = step_out['eta']
                coherence_spatial_prev = results['coherence_spatial']

                eta_by_step.setdefault(t, []).append(
                    step_out['eta'].mean().item())
                eps_by_step.setdefault(t, []).append(
                    step_out['eps'].mean().item())

                if t < T - 1:
                    n_cur = int(rev_counts[t].item())
                    n_nxt = int(rev_counts[t + 1].item())
                    n_new = max(0, n_nxt - n_cur)
                    if n_new > 0:
                        pos, batch_idx = self._reveal_points(
                            pos, batch_idx, target_pts, n_new, B,
                            noise_std=0.05, device=self.device,
                        )

            # Final Chamfer Distance
            _, cd_final = chamfer_distance_graph(
                pos, target_pts, batch_idx, B)
            accum['chamfer'] += cd_final.item()

            # Affordance loss
            aff_gt_mean = target_aff.mean(dim=1)
            aff_pred = step_out['aff_batched']
            accum['aff'] += self.aff_criterion(aff_pred, aff_gt_mean).item()
            num_batches += 1

        metrics = {k: v / max(num_batches, 1) for k, v in accum.items()}
        metrics['eta_by_step'] = {
            t: float(np.mean(v)) for t, v in eta_by_step.items()}
        metrics['eps_by_step'] = {
            t: float(np.mean(v)) for t, v in eps_by_step.items()}
        metrics['disp_mag_by_step'] = {
            t: float(np.mean(v)) for t, v in disp_mag_by_step.items()}
        metrics['cd_by_step'] = {
            t: float(np.mean(v)) for t, v in cd_by_step.items()}
        return metrics


# ======================================================================
# Main experiment runner
# ======================================================================

def run_temporal_suspension_experiment(
    num_epochs: int = 50,
    num_samples: int = 100,
    num_points_final: int = 256,
    num_time_steps: int = 8,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = 'cpu',
):
    """
    Run the full active assembly experiment.

    Trains two models:
        (a) 'slack' — Phase 2 Slack (no reconstruction loss)
        (b) 'tight' — Phase 1 style (with reconstruction loss)

    Compares their temporal behaviour (displacement patterns, η(t), CD(t)).
    """
    device = torch.device(device)
    print(f"Device: {device}")

    output_dir = Path(
        "/home/ubuntu/adjunction-model/results/temporal_suspension")
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
        disp_head = DisplacementHead(context_dim=128, num_affordances=5)

        trainer = TemporalSuspensionTrainer(
            model=model,
            displacement_head=disp_head,
            device=device,
            lr=lr,
            lambda_chamfer=1.0,
            lambda_aff=0.5,
            lambda_kl=0.1,
            lambda_coherence=0.1 if mode == 'slack' else 0.0,
            lambda_recon=1.0 if mode == 'tight' else 0.0,
            mode=mode,
        )

        history: Dict[str, list] = {
            'loss': [], 'chamfer': [], 'aff': [], 'kl': [],
            'coherence': [], 'recon': [],
            'val_chamfer': [], 'val_aff': [],
            'eta_by_step': [], 'eps_by_step': [],
            'disp_mag_by_step': [], 'cd_by_step': [],
            'val_eta_by_step': [], 'val_eps_by_step': [],
            'val_disp_mag_by_step': [], 'val_cd_by_step': [],
        }

        for ep in range(num_epochs):
            print(f"\nEpoch {ep + 1}/{num_epochs} [{mode}]")
            print("-" * 60)

            train_m = trainer.train_epoch(dataloader, ep)
            val_m = trainer.validate(val_loader)

            # Scalar metrics
            for key in ['loss', 'chamfer', 'aff', 'kl', 'coherence',
                        'recon']:
                history[key].append(train_m[key])
            history['val_chamfer'].append(val_m['chamfer'])
            history['val_aff'].append(val_m['aff'])

            # Per-step metrics
            for key in ['eta_by_step', 'eps_by_step',
                        'disp_mag_by_step', 'cd_by_step']:
                history[key].append(train_m[key])
            for key in ['eta_by_step', 'eps_by_step',
                        'disp_mag_by_step', 'cd_by_step']:
                history[f'val_{key}'].append(val_m[key])

            # Print summary
            disp_vals = list(train_m['disp_mag_by_step'].values())
            disp_early = np.mean(disp_vals[:2]) if len(disp_vals) >= 2 else 0
            disp_late = np.mean(disp_vals[-2:]) if len(disp_vals) >= 2 else 0
            print(
                f"  Loss={train_m['loss']:.4f}  "
                f"CD={train_m['chamfer']:.4f}  "
                f"Aff={train_m['aff']:.4f}  "
                f"‖Δx‖_early={disp_early:.4f}  "
                f"‖Δx‖_late={disp_late:.4f}"
            )
            print(
                f"  [Val] CD={val_m['chamfer']:.4f}  "
                f"Aff={val_m['aff']:.4f}"
            )

        all_histories[mode] = history

        # Save per-mode results
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serialisable form
        history_json = {}
        for k, v in history.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                history_json[k] = [
                    {str(kk): vv for kk, vv in d.items()} for d in v
                ]
            else:
                history_json[k] = v
        with open(mode_dir / 'metrics.json', 'w') as f:
            json.dump(history_json, f, indent=2)

        # Save model weights
        torch.save(model.state_dict(), mode_dir / 'model.pt')
        torch.save(disp_head.state_dict(), mode_dir / 'displacement_head.pt')

        print(f"\n{mode} results saved to {mode_dir}")

    # ---- Combined summary ----
    summary = {}
    for mode in ['slack', 'tight']:
        h = all_histories[mode]
        summary[mode] = {
            'final_chamfer': h['val_chamfer'][-1],
            'final_aff': h['val_aff'][-1],
            'final_eta_by_step': h['val_eta_by_step'][-1],
            'final_disp_mag_by_step': h['val_disp_mag_by_step'][-1],
            'final_cd_by_step': h['val_cd_by_step'][-1],
        }
    # Convert int keys to str for JSON
    for mode in summary:
        for k in ['final_eta_by_step', 'final_disp_mag_by_step',
                   'final_cd_by_step']:
            summary[mode][k] = {
                str(kk): vv for kk, vv in summary[mode][k].items()}

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
        num_points_final=256,
        num_time_steps=8,
        batch_size=4,
        lr=1e-4,
        device='cpu',
    )
