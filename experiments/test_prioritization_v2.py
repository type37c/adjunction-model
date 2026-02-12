"""
Prioritization Experiment V2 (Suspension Structure Validation 3 - with Agent C v2)

This experiment tests whether Agent C v2's priority-based attention mechanism
improves intentionality compared to v1.

Key design decision:
OnlineLearner is DISABLED. Instead of F/G weights being directly updated,
adaptation happens ONLY through Agent C's internal state change, which
modulates F and G via FiLM conditioning.

This is theoretically correct: in the Conditional Adjunction (F_C ⊣ G_C),
C is the sole parameter that changes F/G's behavior. Direct weight updates
bypass C and make it irrelevant.

Additionally, we now pass F's intermediate features as observations to Agent C,
so that posterior != prior and KL divergence becomes non-zero.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.conditional_adjunction_v2 import ConditionalAdjunctionModelV2
from src.data.synthetic_dataset import SyntheticAffordanceDataset
from src.training.train_phase2_v2 import Phase2TrainerV2
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


def forward_with_obs(model, pos, batch, agent_state, coherence_signal_prev, coherence_spatial_prev):
    """
    Forward pass that also extracts F's intermediate features as observations for Agent C.
    
    This is the key fix: Agent C needs to SEE the shape (via F's features)
    to form a meaningful posterior, so that KL(posterior || prior) > 0.
    """
    device = pos.device
    
    if batch is None:
        batch_size = 1
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=device)
    else:
        batch_size = batch.max().item() + 1
    
    N = pos.size(0)
    
    if agent_state is None:
        agent_state = model.initial_state(batch_size, device)
    if coherence_signal_prev is None:
        coherence_signal_prev = torch.zeros(batch_size, 1, device=device)
    if coherence_spatial_prev is None:
        coherence_spatial_prev = torch.zeros(N, device=device)
    
    # Step 1: Extract F's intermediate features as observations
    # Run the first layer of F's base model to get shape features
    with torch.no_grad():
        obs_features = model.F.base_f.input_embed(pos)  # (N, f_hidden_dim)
    
    # Step 2: Get context from Agent C v2 WITH observations
    dummy_action = torch.zeros(batch_size, model.num_affordances, device=device)
    
    agent_state_new, context, agent_info = model.agent_c(
        prev_state=agent_state,
        action=dummy_action,
        coherence_signal_scalar=coherence_signal_prev,
        coherence_signal_spatial=coherence_spatial_prev,
        batch=batch,
        obs=obs_features  # Now Agent C can SEE the shape
    )
    
    # Step 3: Apply F_C and G_C with context
    affordances = model.F(pos, batch, context)
    
    affordances_batched = torch.zeros(batch_size, model.num_affordances, device=device)
    for b in range(batch_size):
        mask = (batch == b)
        if mask.sum() > 0:
            affordances_batched[b] = affordances[mask].mean(dim=0)
    
    reconstructed = model.G(affordances_batched, model.num_points, context)
    
    # Step 4: Compute coherence signals
    coherence_signal, coherence_spatial = model._compute_coherence_signal(pos, reconstructed, batch)
    
    return {
        'affordances': affordances,
        'reconstructed': reconstructed,
        'coherence_signal': coherence_signal,
        'coherence_spatial': coherence_spatial,
        'agent_state': agent_state_new,
        'context': context,
        'rssm_info': agent_info,
        'priority': agent_info.get('priority', None),
        'priority_normalized': agent_info.get('priority_normalized', None),
        'uncertainty': agent_info.get('uncertainty', None)
    }


def run_prioritization_experiment(
    model: ConditionalAdjunctionModelV2,
    device: torch.device,
    num_trials: int = 10
) -> dict:
    """
    Run the prioritization experiment WITHOUT OnlineLearner.
    
    Adaptation happens only through Agent C's internal state change.
    F/G weights are frozen after training.
    """
    
    # Freeze F/G weights - only Agent C's state changes
    model.eval()
    
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
    coherence_spatial_prev = torch.zeros(model.num_points, device=device)
    
    # Track results
    cube_coherence = []
    torus_coherence = []
    cube_attention = []
    torus_attention = []
    cube_priority = []
    torus_priority = []
    cube_uncertainty = []
    torus_uncertainty = []
    cube_kl = []
    torus_kl = []
    
    print(f"  Running {num_trials} trials of alternating presentations...")
    print(f"  OnlineLearner: DISABLED (Agent C state change only)")
    print(f"  Observations: F's intermediate features passed to Agent C")
    
    for trial in range(num_trials):
        # Present cube
        h_before = agent_state['h'].clone()
        
        with torch.no_grad():
            results_cube = forward_with_obs(
                model, cube_tensor, None, agent_state,
                coherence_signal_prev, coherence_spatial_prev
            )
        
        h_after = results_cube['agent_state']['h']
        attention_cube = torch.norm(h_after - h_before).item()
        
        cube_coherence.append(results_cube['coherence_signal'].item())
        cube_attention.append(attention_cube)
        cube_priority.append(results_cube['priority'].mean().item() if results_cube['priority'] is not None else 0.0)
        cube_uncertainty.append(results_cube['uncertainty'].mean().item() if results_cube['uncertainty'] is not None else 0.0)
        
        # Compute KL divergence
        info = results_cube['rssm_info']
        kl = model.agent_c.rssm.kl_divergence(
            info['posterior_mean'], info['posterior_std'],
            info['prior_mean'], info['prior_std']
        ).mean().item()
        cube_kl.append(kl)
        
        agent_state = {k: v.detach() for k, v in results_cube['agent_state'].items()}
        coherence_signal_prev = results_cube['coherence_signal'].detach()
        coherence_spatial_prev = results_cube['coherence_spatial'].detach()
        
        # Present torus
        h_before = agent_state['h'].clone()
        
        with torch.no_grad():
            results_torus = forward_with_obs(
                model, torus_tensor, None, agent_state,
                coherence_signal_prev, coherence_spatial_prev
            )
        
        h_after = results_torus['agent_state']['h']
        attention_torus = torch.norm(h_after - h_before).item()
        
        torus_coherence.append(results_torus['coherence_signal'].item())
        torus_attention.append(attention_torus)
        torus_priority.append(results_torus['priority'].mean().item() if results_torus['priority'] is not None else 0.0)
        torus_uncertainty.append(results_torus['uncertainty'].mean().item() if results_torus['uncertainty'] is not None else 0.0)
        
        info = results_torus['rssm_info']
        kl = model.agent_c.rssm.kl_divergence(
            info['posterior_mean'], info['posterior_std'],
            info['prior_mean'], info['prior_std']
        ).mean().item()
        torus_kl.append(kl)
        
        agent_state = {k: v.detach() for k, v in results_torus['agent_state'].items()}
        coherence_signal_prev = results_torus['coherence_signal'].detach()
        coherence_spatial_prev = results_torus['coherence_spatial'].detach()
        
        if trial % 3 == 0 or trial == num_trials - 1:
            print(f"    Trial {trial + 1}:")
            print(f"      Cube:  Coh={results_cube['coherence_signal'].item():.4f}, "
                  f"Att={attention_cube:.4f}, "
                  f"Pri={cube_priority[-1]:.4f}, "
                  f"KL={cube_kl[-1]:.4f}")
            print(f"      Torus: Coh={results_torus['coherence_signal'].item():.4f}, "
                  f"Att={attention_torus:.4f}, "
                  f"Pri={torus_priority[-1]:.4f}, "
                  f"KL={torus_kl[-1]:.4f}")
    
    return {
        'cube_coherence': cube_coherence,
        'torus_coherence': torus_coherence,
        'cube_attention': cube_attention,
        'torus_attention': torus_attention,
        'cube_priority': cube_priority,
        'torus_priority': torus_priority,
        'cube_uncertainty': cube_uncertainty,
        'torus_uncertainty': torus_uncertainty,
        'cube_kl': cube_kl,
        'torus_kl': torus_kl
    }


def visualize_prioritization(results: dict, save_dir: Path):
    """Visualize the prioritization experiment results."""
    
    trials = np.arange(1, len(results['cube_coherence']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Coherence Signal
    axes[0, 0].plot(trials, results['cube_coherence'], 'o-', label='Cube (Known)',
                     linewidth=2, markersize=6, color='blue')
    axes[0, 0].plot(trials, results['torus_coherence'], 's-', label='Torus (Novel)',
                     linewidth=2, markersize=6, color='red')
    axes[0, 0].set_xlabel('Trial', fontsize=12)
    axes[0, 0].set_ylabel('Coherence Signal', fontsize=12)
    axes[0, 0].set_title('Coherence Signal', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Attention
    axes[0, 1].plot(trials, results['cube_attention'], 'o-', label='Cube (Known)',
                     linewidth=2, markersize=6, color='blue')
    axes[0, 1].plot(trials, results['torus_attention'], 's-', label='Torus (Novel)',
                     linewidth=2, markersize=6, color='red')
    axes[0, 1].set_xlabel('Trial', fontsize=12)
    axes[0, 1].set_ylabel('Attention (||Δh||)', fontsize=12)
    axes[0, 1].set_title('Attention Allocation', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Priority
    axes[0, 2].plot(trials, results['cube_priority'], 'o-', label='Cube (Known)',
                     linewidth=2, markersize=6, color='blue')
    axes[0, 2].plot(trials, results['torus_priority'], 's-', label='Torus (Novel)',
                     linewidth=2, markersize=6, color='red')
    axes[0, 2].set_xlabel('Trial', fontsize=12)
    axes[0, 2].set_ylabel('Priority (mean)', fontsize=12)
    axes[0, 2].set_title('Priority Scores', fontsize=14, fontweight='bold')
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Uncertainty
    axes[1, 0].plot(trials, results['cube_uncertainty'], 'o-', label='Cube (Known)',
                     linewidth=2, markersize=6, color='blue')
    axes[1, 0].plot(trials, results['torus_uncertainty'], 's-', label='Torus (Novel)',
                     linewidth=2, markersize=6, color='red')
    axes[1, 0].set_xlabel('Trial', fontsize=12)
    axes[1, 0].set_ylabel('Uncertainty', fontsize=12)
    axes[1, 0].set_title('Uncertainty', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: KL Divergence
    axes[1, 1].plot(trials, results['cube_kl'], 'o-', label='Cube (Known)',
                     linewidth=2, markersize=6, color='blue')
    axes[1, 1].plot(trials, results['torus_kl'], 's-', label='Torus (Novel)',
                     linewidth=2, markersize=6, color='red')
    axes[1, 1].set_xlabel('Trial', fontsize=12)
    axes[1, 1].set_ylabel('KL Divergence', fontsize=12)
    axes[1, 1].set_title('KL Divergence (posterior vs prior)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Summary bar chart
    metrics = ['Coherence', 'Attention', 'Priority', 'KL']
    cube_vals = [
        np.mean(results['cube_coherence']),
        np.mean(results['cube_attention']),
        np.mean(results['cube_priority']),
        np.mean(results['cube_kl'])
    ]
    torus_vals = [
        np.mean(results['torus_coherence']),
        np.mean(results['torus_attention']),
        np.mean(results['torus_priority']),
        np.mean(results['torus_kl'])
    ]
    
    # Compute ratios
    ratios = [t / max(c, 1e-8) for c, t in zip(cube_vals, torus_vals)]
    
    x = np.arange(len(metrics))
    axes[1, 2].bar(x, ratios, color=['green' if r > 1.2 else 'orange' if r > 1.0 else 'red' for r in ratios])
    axes[1, 2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].axhline(y=1.5, color='green', linestyle='--', alpha=0.3, label='Strong evidence (1.5x)')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics)
    axes[1, 2].set_ylabel('Ratio (Novel / Known)', fontsize=12)
    axes[1, 2].set_title('Summary: Novel/Known Ratios', fontsize=14, fontweight='bold')
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'prioritization_v2.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_dir / 'prioritization_v2.png'}")
    plt.close()


def train_with_obs(model, device, num_epochs=10):
    """
    Train the model with observations passed to Agent C.
    
    This ensures Agent C learns to use observations during training,
    so its posterior becomes meaningful.
    """
    dataset = SyntheticAffordanceDataset(num_samples=50, num_points=model.num_points)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    aff_criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_kl = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            points_list = batch_data['points']
            affordances_list = batch_data['affordances']
            
            pos = torch.cat([p for p in points_list], dim=0).to(device)
            affordances_gt = torch.cat([a for a in affordances_list], dim=0).to(device)
            
            batch_size = len(points_list)
            num_points = points_list[0].size(0)
            batch = torch.repeat_interleave(
                torch.arange(batch_size, device=device), num_points
            )
            
            N = pos.size(0)
            agent_state = model.initial_state(batch_size, device)
            coherence_signal_prev = torch.zeros(batch_size, 1, device=device)
            coherence_spatial_prev = torch.zeros(N, device=device)
            
            # Extract F's features as observations
            obs_features = model.F.base_f.input_embed(pos)  # (N, f_hidden_dim)
            
            # Agent C forward with observations
            dummy_action = torch.zeros(batch_size, model.num_affordances, device=device)
            agent_state_new, context, agent_info = model.agent_c(
                prev_state=agent_state,
                action=dummy_action,
                coherence_signal_scalar=coherence_signal_prev,
                coherence_signal_spatial=coherence_spatial_prev,
                batch=batch,
                obs=obs_features.detach()  # detach to avoid double backprop through F
            )
            
            # F_C and G_C with context
            affordances = model.F(pos, batch, context)
            
            affordances_batched = torch.zeros(batch_size, model.num_affordances, device=device)
            for b in range(batch_size):
                mask = (batch == b)
                if mask.sum() > 0:
                    affordances_batched[b] = affordances[mask].mean(dim=0)
            
            reconstructed = model.G(affordances_batched, model.num_points, context)
            coherence_signal, _ = model._compute_coherence_signal(pos, reconstructed, batch)
            
            # Losses
            L_recon = coherence_signal.mean()
            L_aff = aff_criterion(affordances, affordances_gt)
            
            L_kl = model.agent_c.rssm.kl_divergence(
                agent_info['posterior_mean'], agent_info['posterior_std'],
                agent_info['prior_mean'], agent_info['prior_std']
            ).mean()
            
            L_coherence = -torch.log(coherence_signal + 1e-8).mean()
            
            loss = L_recon + 1.0 * L_aff + 0.1 * L_kl + 0.1 * L_coherence
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_kl += L_kl.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_kl = total_kl / num_batches
        print(f"   Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}, KL={avg_kl:.4f}")
    
    return model


def main():
    print("=" * 80)
    print("PRIORITIZATION EXPERIMENT V2")
    print("(Agent C v2, NO OnlineLearner, with observations)")
    print("=" * 80)
    
    device = torch.device('cpu')
    save_dir = Path('logs/prioritization_test_v2')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create v2 model
    print("\n1. Creating ConditionalAdjunctionModelV2...")
    model = ConditionalAdjunctionModelV2(
        num_affordances=5,
        num_points=256,
        f_hidden_dim=32,
        g_hidden_dim=64,
        agent_hidden_dim=128,
        agent_latent_dim=32,
        context_dim=64,
        uncertainty_type='entropy',
        attention_temperature=1.0
    ).to(device)
    
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train with observations (so Agent C learns to use them)
    print("\n2. Training model (with observations passed to Agent C)...")
    model = train_with_obs(model, device, num_epochs=10)
    
    # Run prioritization experiment (NO OnlineLearner)
    print("\n3. Running prioritization experiment...")
    results = run_prioritization_experiment(model, device, num_trials=10)
    
    # Visualize
    print("\n4. Visualizing results...")
    visualize_prioritization(results, save_dir)
    
    # Analysis
    print("\n5. Analysis:")
    
    avg_cube_coherence = np.mean(results['cube_coherence'])
    avg_torus_coherence = np.mean(results['torus_coherence'])
    avg_cube_attention = np.mean(results['cube_attention'])
    avg_torus_attention = np.mean(results['torus_attention'])
    avg_cube_priority = np.mean(results['cube_priority'])
    avg_torus_priority = np.mean(results['torus_priority'])
    avg_cube_uncertainty = np.mean(results['cube_uncertainty'])
    avg_torus_uncertainty = np.mean(results['torus_uncertainty'])
    avg_cube_kl = np.mean(results['cube_kl'])
    avg_torus_kl = np.mean(results['torus_kl'])
    
    print(f"   Average Coherence Signal:")
    print(f"     Cube (known):  {avg_cube_coherence:.4f}")
    print(f"     Torus (novel): {avg_torus_coherence:.4f}")
    coherence_ratio = avg_torus_coherence / max(avg_cube_coherence, 1e-8)
    print(f"     Ratio: {coherence_ratio:.2f}x")
    
    print(f"   Average Attention (||Δh||):")
    print(f"     Cube (known):  {avg_cube_attention:.4f}")
    print(f"     Torus (novel): {avg_torus_attention:.4f}")
    attention_ratio = avg_torus_attention / max(avg_cube_attention, 1e-8)
    print(f"     Ratio: {attention_ratio:.2f}x")
    
    print(f"   Average Priority:")
    print(f"     Cube (known):  {avg_cube_priority:.4f}")
    print(f"     Torus (novel): {avg_torus_priority:.4f}")
    priority_ratio = avg_torus_priority / max(avg_cube_priority, 1e-8)
    print(f"     Ratio: {priority_ratio:.2f}x")
    
    print(f"   Average Uncertainty:")
    print(f"     Cube (known):  {avg_cube_uncertainty:.4f}")
    print(f"     Torus (novel): {avg_torus_uncertainty:.4f}")
    uncertainty_ratio = avg_torus_uncertainty / max(avg_cube_uncertainty, 1e-8)
    print(f"     Ratio: {uncertainty_ratio:.2f}x")
    
    print(f"   Average KL Divergence:")
    print(f"     Cube (known):  {avg_cube_kl:.4f}")
    print(f"     Torus (novel): {avg_torus_kl:.4f}")
    kl_ratio = avg_torus_kl / max(avg_cube_kl, 1e-8)
    print(f"     Ratio: {kl_ratio:.2f}x")
    
    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT:")
    print("=" * 80)
    
    if attention_ratio > 1.5:
        print(f"   STRONG EVIDENCE: Attention ratio = {attention_ratio:.2f}x")
    elif attention_ratio > 1.2:
        print(f"   MODERATE EVIDENCE: Attention ratio = {attention_ratio:.2f}x")
    else:
        print(f"   WEAK EVIDENCE: Attention ratio = {attention_ratio:.2f}x")
    
    print(f"\n   KL Divergence (is Agent C seeing the world?):")
    if avg_cube_kl > 0.01 or avg_torus_kl > 0.01:
        print(f"     YES: KL > 0 (Cube={avg_cube_kl:.4f}, Torus={avg_torus_kl:.4f})")
    else:
        print(f"     NO: KL ~ 0 (Agent C is still blind)")
    
    print(f"\n   Coherence persistence (is novelty maintained?):")
    if coherence_ratio > 1.2:
        print(f"     YES: Torus coherence remains higher ({coherence_ratio:.2f}x)")
    else:
        print(f"     NO: Coherence converged ({coherence_ratio:.2f}x)")
    
    print(f"\n   Comparison with v1:")
    print(f"     v1 attention ratio: 1.11x")
    print(f"     v2 attention ratio: {attention_ratio:.2f}x")
    improvement = (attention_ratio - 1.11) / 1.11 * 100
    print(f"     Improvement: {improvement:+.1f}%")
    
    # Save results
    np.save(save_dir / 'prioritization_v2_results.npy', results)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
