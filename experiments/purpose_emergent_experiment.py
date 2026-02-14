"""
Purpose-Emergent Active Assembly Experiment
"""

import sys
import os
# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models.adjunction_model import AdjunctionModel
from src.data.purposeless_dataset import (
    PurposelessAssemblyDataset,
    collate_purposeless_batch,
)

class DisplacementHead(nn.Module):
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

    def forward(self, context: torch.Tensor, affordances: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        ctx_per_point = context[batch_idx]
        feat = torch.cat([ctx_per_point, affordances], dim=-1)
        return self.net(feat)

def chamfer_distance_graph(assembled: torch.Tensor, target: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cd_list = []
    for b in range(batch_size):
        mask = (batch_idx == b)
        pts_a = assembled[mask]
        if pts_a.size(0) == 0:
            cd_list.append(torch.tensor(0.0, device=assembled.device))
            continue
        dist = torch.cdist(pts_a, target)
        d_a2t = dist.min(dim=1)[0].mean()
        d_t2a = dist.min(dim=0)[0].mean()
        cd_list.append((d_a2t + d_t2a) / 2)
    cd_per_sample = torch.stack(cd_list)
    return cd_per_sample, cd_per_sample.mean()

def compute_purpose_loss(pos: torch.Tensor, batch_idx: torch.Tensor, batch_size: int, reference_shapes: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    cd_per_shape = {}
    for name, ref in reference_shapes.items():
        ref_dev = ref.to(device)
        cd_per, _ = chamfer_distance_graph(pos, ref_dev, batch_idx, batch_size)
        cd_per_shape[name] = cd_per
    all_cds = torch.stack(list(cd_per_shape.values()), dim=0)
    min_cd, _ = all_cds.min(dim=0)
    return min_cd.mean(), cd_per_shape

class PurposeEmergentTrainer:
    def __init__(self, model: AdjunctionModel, displacement_head: DisplacementHead, reference_shapes: Dict[str, torch.Tensor], device: torch.device = torch.device('cpu'), lr: float = 1e-4, lambda_purpose: float = 1.0, lambda_coherence: float = 0.1, lambda_kl: float = 0.1, num_steps: int = 8, condition: str = 'purpose_emergent'):
        self.model = model.to(device)
        self.displacement_head = displacement_head.to(device)
        self.reference_shapes = reference_shapes
        self.device = device
        self.condition = condition
        self.num_steps = num_steps
        self.lambda_purpose = lambda_purpose
        self.lambda_coherence = lambda_coherence
        self.lambda_kl = lambda_kl
        all_params = list(model.parameters()) + list(displacement_head.parameters())
        self.optimizer = optim.Adam(all_params, lr=lr)

    def _step(self, pos: torch.Tensor, batch_idx: torch.Tensor, agent_state: Dict[str, torch.Tensor], coherence_prev: torch.Tensor, coherence_spatial_prev: Optional[torch.Tensor]) -> Dict:
        N = pos.size(0)
        if coherence_spatial_prev is None or coherence_spatial_prev.size(0) != N:
            coherence_spatial_prev = torch.zeros(N, device=self.device)
        state_clean = {k: v for k, v in agent_state.items() if k != 'priority_normalized'}
        results = self.model(pos, batch_idx, state_clean, coherence_prev, coherence_spatial_prev)
        eta = results['coherence_signal']
        eps = results['counit_signal']
        context = results['context']
        affordances = results['affordances']
        displacement = self.displacement_head(context, affordances, batch_idx)
        return {'results': results, 'displacement': displacement, 'eta': eta, 'eps': eps}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        self.model.train()
        self.displacement_head.train()
        accum = {'loss': 0.0, 'purpose': 0.0, 'coherence': 0.0, 'kl': 0.0}
        eta_by_step, eps_by_step, disp_mag_by_step, curiosity_by_step = {}, {}, {}, {}
        cd_by_step_by_shape = {name: {} for name in self.reference_shapes}
        chosen_by_step = {}
        num_batches = 0
        shape_names = list(self.reference_shapes.keys())

        for batch_data in dataloader:
            init_pts = batch_data['initial_points'].to(self.device)
            init_batch = batch_data['initial_batch'].to(self.device)
            B = init_batch.max().item() + 1
            agent_state = self.model.initial_state(B, self.device)
            coherence_prev = torch.zeros(B, 1, device=self.device)
            coherence_spatial_prev = None
            pos = init_pts.clone()
            batch_idx = init_batch.clone()
            kl_losses, eta_history = [], []

            for t in range(self.num_steps):
                step_out = self._step(pos, batch_idx, agent_state, coherence_prev, coherence_spatial_prev)
                results = step_out['results']
                displacement = step_out['displacement']
                eta_t, eps_t = step_out['eta'], step_out['eps']
                disp_mag = displacement.norm(dim=-1).mean().item()
                disp_mag_by_step.setdefault(t, []).append(disp_mag)
                pos = pos + displacement
                for name, ref in self.reference_shapes.items():
                    ref_dev = ref.to(self.device)
                    _, cd_mean = chamfer_distance_graph(pos, ref_dev, batch_idx, B)
                    cd_by_step_by_shape[name].setdefault(t, []).append(cd_mean.item())
                step_cds = []
                for name in shape_names:
                    ref_dev = self.reference_shapes[name].to(self.device)
                    cd_per, _ = chamfer_distance_graph(pos, ref_dev, batch_idx, B)
                    step_cds.append(cd_per)
                step_cds_tensor = torch.stack(step_cds, dim=0)
                chosen = step_cds_tensor.argmin(dim=0)
                chosen_by_step.setdefault(t, []).extend(chosen.tolist())
                if t > 0:
                    R_curiosity = (eta_history[-1] - eta_t).mean().item() * 100
                    curiosity_by_step.setdefault(t, []).append(R_curiosity)
                rssm_info = results['rssm_info']
                kl_t = self.model.agent_c.rssm.kl_divergence(rssm_info['posterior_mean'], rssm_info['posterior_std'], rssm_info['prior_mean'], rssm_info['prior_std']).mean()
                kl_losses.append(kl_t)
                agent_state = results['agent_state']
                coherence_prev = eta_t.detach()
                coherence_spatial_prev = results['coherence_spatial'].detach()
                eta_by_step.setdefault(t, []).append(eta_t.mean().item())
                eps_by_step.setdefault(t, []).append(eps_t.mean().item())
                eta_history.append(eta_t)

            L_purpose, _ = compute_purpose_loss(pos, batch_idx, B, self.reference_shapes, self.device)
            L_coherence = -torch.log(eta_history[-1].mean() + 1e-8)
            L_kl = torch.stack(kl_losses).mean()
            loss = self.lambda_purpose * L_purpose + self.lambda_coherence * L_coherence + self.lambda_kl * L_kl
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.displacement_head.parameters()), max_norm=1.0)
            self.optimizer.step()
            accum['loss'] += loss.item()
            accum['purpose'] += L_purpose.item()
            accum['coherence'] += L_coherence.item()
            accum['kl'] += L_kl.item()
            num_batches += 1
            if num_batches % 5 == 0:
                print(f"  Batch {num_batches}: Loss={loss.item():.4f}, L_purpose={L_purpose.item():.4f}")

        metrics = {k: v / max(num_batches, 1) for k, v in accum.items()}
        metrics['eta_by_step'] = {t: float(np.mean(v)) for t, v in eta_by_step.items()}
        metrics['eps_by_step'] = {t: float(np.mean(v)) for t, v in eps_by_step.items()}
        metrics['disp_mag_by_step'] = {t: float(np.mean(v)) for t, v in disp_mag_by_step.items()}
        metrics['curiosity_by_step'] = {t: float(np.mean(v)) for t, v in curiosity_by_step.items()}
        for name in shape_names:
            metrics[f'cd_{name}_by_step'] = {t: float(np.mean(v)) for t, v in cd_by_step_by_shape[name].items()}
        metrics['chosen_by_step'] = {t: {shape_names[i]: choices.count(i) for i in range(len(shape_names))} for t, choices in chosen_by_step.items()}
        return metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        self.displacement_head.eval()
        eta_by_step, eps_by_step, disp_mag_by_step, curiosity_by_step = {}, {}, {}, {}
        cd_by_step_by_shape = {name: {} for name in self.reference_shapes}
        chosen_by_step, purpose_losses = {}, []
        num_batches = 0
        shape_names = list(self.reference_shapes.keys())

        for batch_data in dataloader:
            init_pts = batch_data['initial_points'].to(self.device)
            init_batch = batch_data['initial_batch'].to(self.device)
            B = init_batch.max().item() + 1
            agent_state = self.model.initial_state(B, self.device)
            coherence_prev = torch.zeros(B, 1, device=self.device)
            coherence_spatial_prev = None
            pos = init_pts.clone()
            batch_idx = init_batch.clone()
            eta_history = []

            for t in range(self.num_steps):
                step_out = self._step(pos, batch_idx, agent_state, coherence_prev, coherence_spatial_prev)
                results = step_out['results']
                displacement = step_out['displacement']
                eta_t, eps_t = step_out['eta'], step_out['eps']
                disp_mag_by_step.setdefault(t, []).append(displacement.norm(dim=-1).mean().item())
                pos = pos + displacement
                for name, ref in self.reference_shapes.items():
                    ref_dev = ref.to(self.device)
                    _, cd_mean = chamfer_distance_graph(pos, ref_dev, batch_idx, B)
                    cd_by_step_by_shape[name].setdefault(t, []).append(cd_mean.item())
                step_cds = []
                for name in shape_names:
                    ref_dev = self.reference_shapes[name].to(self.device)
                    cd_per, _ = chamfer_distance_graph(pos, ref_dev, batch_idx, B)
                    step_cds.append(cd_per)
                step_cds_tensor = torch.stack(step_cds, dim=0)
                chosen = step_cds_tensor.argmin(dim=0)
                chosen_by_step.setdefault(t, []).extend(chosen.tolist())
                if t > 0:
                    R_curiosity = (eta_history[-1] - eta_t).mean().item() * 100
                    curiosity_by_step.setdefault(t, []).append(R_curiosity)
                agent_state = results['agent_state']
                coherence_prev = eta_t
                coherence_spatial_prev = results['coherence_spatial']
                eta_by_step.setdefault(t, []).append(eta_t.mean().item())
                eps_by_step.setdefault(t, []).append(eps_t.mean().item())
                eta_history.append(eta_t)

            L_purpose, _ = compute_purpose_loss(pos, batch_idx, B, self.reference_shapes, self.device)
            purpose_losses.append(L_purpose.item())
            num_batches += 1

        metrics = {'purpose': float(np.mean(purpose_losses))}
        metrics['eta_by_step'] = {t: float(np.mean(v)) for t, v in eta_by_step.items()}
        metrics['eps_by_step'] = {t: float(np.mean(v)) for t, v in eps_by_step.items()}
        metrics['disp_mag_by_step'] = {t: float(np.mean(v)) for t, v in disp_mag_by_step.items()}
        metrics['curiosity_by_step'] = {t: float(np.mean(v)) for t, v in curiosity_by_step.items()}
        for name in shape_names:
            metrics[f'cd_{name}_by_step'] = {t: float(np.mean(v)) for t, v in cd_by_step_by_shape[name].items()}
        metrics['chosen_by_step'] = {t: {shape_names[i]: choices.count(i) for i in range(len(shape_names))} for t, choices in chosen_by_step.items()}
        return metrics

def run_purpose_emergent_experiment(num_epochs: int = 3, num_samples: int = 100, num_points: int = 256, num_steps: int = 8, batch_size: int = 2, lr: float = 1e-4, device: str = 'cpu'):
    device = torch.device(device)
    output_dir = Path("/home/ubuntu/adjunction-model/results/purpose_emergent")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = PurposelessAssemblyDataset(num_samples=num_samples, num_points=num_points, seed=42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_purposeless_batch)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_purposeless_batch)
    reference_shapes = dataset.reference_shapes
    all_histories = {}
    conditions = {'purpose_emergent': 0.3, 'baseline': 0.0}

    for condition, alpha_curiosity in conditions.items():
        print(f"\nCondition: {condition.upper()} (alpha_curiosity={alpha_curiosity})")
        model = AdjunctionModel(num_affordances=5, num_points=num_points, f_hidden_dim=64, g_hidden_dim=128, agent_hidden_dim=256, agent_latent_dim=64, context_dim=128, valence_dim=32, valence_decay=0.1, alpha_curiosity=alpha_curiosity, beta_competence=0.6, gamma_novelty=0.4)
        disp_head = DisplacementHead(context_dim=128, num_affordances=5)
        trainer = PurposeEmergentTrainer(model=model, displacement_head=disp_head, reference_shapes=reference_shapes, device=device, lr=lr, num_steps=num_steps, condition=condition)
        history = {k: [] for k in ['loss', 'purpose', 'coherence', 'kl', 'val_purpose', 'eta_by_step', 'eps_by_step', 'disp_mag_by_step', 'curiosity_by_step', 'chosen_by_step', 'val_eta_by_step', 'val_eps_by_step', 'val_disp_mag_by_step', 'val_curiosity_by_step', 'val_chosen_by_step']}
        shape_names = list(reference_shapes.keys())
        for name in shape_names:
            history[f'cd_{name}_by_step'], history[f'val_cd_{name}_by_step'] = [], []

        for ep in range(num_epochs):
            print(f"Epoch {ep + 1}/{num_epochs}")
            train_m = trainer.train_epoch(dataloader, ep)
            val_m = trainer.validate(val_loader)
            for key in ['loss', 'purpose', 'coherence', 'kl']: history[key].append(train_m[key])
            history['val_purpose'].append(val_m['purpose'])
            for key in ['eta_by_step', 'eps_by_step', 'disp_mag_by_step', 'curiosity_by_step', 'chosen_by_step']:
                history[key].append(train_m.get(key, {}))
                history[f'val_{key}'].append(val_m.get(key, {}))
            for name in shape_names:
                history[f'cd_{name}_by_step'].append(train_m.get(f'cd_{name}_by_step', {}))
                history[f'val_cd_{name}_by_step'].append(val_m.get(f'cd_{name}_by_step', {}))
            print(f"  Loss={train_m['loss']:.4f}, L_purpose={train_m['purpose']:.4f}, [Val] L_purpose={val_m['purpose']:.4f}")

        all_histories[condition] = history
        cond_dir = output_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        history_json = {k: ([{str(kk): vv for kk, vv in d.items()} for d in v] if isinstance(v, list) and v and isinstance(v[0], dict) else v) for k, v in history.items()}
        with open(cond_dir / 'metrics.json', 'w') as f: json.dump(history_json, f, indent=2)
        torch.save(model.state_dict(), cond_dir / 'model.pt')
        torch.save(disp_head.state_dict(), cond_dir / 'displacement_head.pt')

    summary = {cond: {'final_purpose': h['val_purpose'][-1], 'final_eta_by_step': {str(k): v for k, v in h['val_eta_by_step'][-1].items()}, 'final_disp_mag_by_step': {str(k): v for k, v in h['val_disp_mag_by_step'][-1].items()}} for cond, h in all_histories.items()}
    with open(output_dir / 'summary.json', 'w') as f: json.dump(summary, f, indent=2)
    print("\nExperiment complete!")

if __name__ == '__main__':
    run_purpose_emergent_experiment(num_epochs=3, batch_size=2)
