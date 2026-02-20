"""
Run Phase 0-2 Experiments

This script runs the complete experimental pipeline from the initial experiment note:
- Phase 0: Train F/G on known shapes (cube, cylinder, sphere)
- Phase 1: Test on unknown shapes (lever, button, knob)
- Phase 2: Test on known shapes with constraints (gravity, friction)

The key innovation is the suspension structure:
- When η > threshold, enter suspension mode
- Buffer observations and fine-tune F/G
- Exit suspension when η < threshold
- Agent uses proposal generation filtered by ε and η
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from core.models.bidirectional_fg import BidirectionalFG
from core.models.proposal_agent import ProposalAgent
from core.models.suspension import SuspensionStructure, finetune_fg_during_suspension
from core.envs.escape_room import EscapeRoomEnv
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


def run_phase_0(
    num_episodes: int = 1000,
    affordance_dim: int = 16,
    device: str = 'cpu'
):
    """
    Phase 0: Train on known shapes (cube, cylinder, sphere).
    
    Expected outcome:
    - Agent learns to map shapes to correct actions
    - F/G learns shape-action coherence
    - Success rate > 80%
    """
    print("\n" + "=" * 80)
    print("PHASE 0: Train on Known Shapes")
    print("=" * 80)
    
    # Load pre-trained F/G
    print("Loading pre-trained F/G...")
    fg_model = BidirectionalFG(
        point_dim=3,
        affordance_dim=affordance_dim,
        action_dim=3,
        hidden_dim=128,
        num_layers=3
    ).to(device)
    
    checkpoint_path = Path('/home/ubuntu/adjunction-model/results/phase0/best_bidirectional_fg.pt')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        fg_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded F/G from {checkpoint_path}")
    else:
        print("Warning: No pre-trained F/G found, using random initialization")
    
    fg_model.eval()  # F/G is frozen during Phase 0
    
    # Create agent
    print("Creating proposal agent...")
    agent = ProposalAgent(
        observation_dim=affordance_dim,
        action_dim=3,
        hidden_dim=128,
        num_proposals=10,
        epsilon_threshold=0.1
    ).to(device)
    
    agent_optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    # Create suspension structure
    suspension = SuspensionStructure(
        eta_threshold=0.1,
        buffer_size=100,
        min_buffer_for_update=10,
        fg_update_steps=10
    )
    
    # Training loop
    print(f"Training for {num_episodes} episodes...")
    
    history = {
        'episode': [],
        'success': [],
        'reward': [],
        'steps': [],
        'eta': [],
        'suspension_count': []
    }
    
    success_window = []
    window_size = 100
    
    for episode in tqdm(range(num_episodes), desc="Phase 0"):
        # Sample object type (known shapes only)
        object_type = np.random.choice([
            EscapeRoomEnv.CUBE,
            EscapeRoomEnv.CYLINDER,
            EscapeRoomEnv.SPHERE
        ])
        
        # Create environment
        env = EscapeRoomEnv(object_type=object_type, render=False, num_points=512)
        obs = env.reset()
        
        # Convert to torch
        pos = torch.FloatTensor(obs).to(device)
        
        # Get affordance from F/G
        with torch.no_grad():
            affordance = fg_model.get_affordance_from_shape(pos).mean(dim=0, keepdim=True)  # (1, affordance_dim)
            eta = fg_model.compute_eta(pos.unsqueeze(0))
        
        # Check suspension
        is_suspended = suspension.check_suspension(eta)
        
        if is_suspended:
            # Add to buffer
            suspension.add_to_buffer(pos, affordance=affordance, eta=eta)
            
            # Fine-tune F/G (not in Phase 0, but we buffer for later)
            # In Phase 1, we would call finetune_fg_during_suspension here
        
        # Select action using proposal agent
        # For Phase 0, we use simple policy without F/G filtering
        action_tensor, log_probs = agent.act(affordance, deterministic=False)
        action = action_tensor.item()
        
        # Take step
        next_obs, reward, done, info = env.step(action)
        
        # Update agent (simple policy gradient)
        loss = -log_probs * reward
        agent_optimizer.zero_grad()
        loss.backward()
        agent_optimizer.step()
        
        # Record
        history['episode'].append(episode)
        history['success'].append(info['success'])
        history['reward'].append(reward)
        history['steps'].append(info['steps'])
        history['eta'].append(eta.item())
        history['suspension_count'].append(suspension.suspension_count)
        
        success_window.append(info['success'])
        if len(success_window) > window_size:
            success_window.pop(0)
        
        # Log every 100 episodes
        if (episode + 1) % 100 == 0:
            success_rate = sum(success_window) / len(success_window)
            avg_eta = np.mean(history['eta'][-100:])
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Success rate (last {window_size}): {success_rate:.2%}")
            print(f"  Avg η: {avg_eta:.6f}")
            print(f"  Suspensions: {suspension.suspension_count}")
        
        env.close()
    
    # Save results
    results_dir = Path('/home/ubuntu/adjunction-model/results/phase0')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save agent
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'agent_optimizer_state_dict': agent_optimizer.state_dict(),
        'history': history
    }, results_dir / 'phase0_agent.pt')
    
    # Save history
    with open(results_dir / 'phase0_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot results
    plot_phase_results(history, results_dir, "Phase 0")
    
    print("\n" + "=" * 80)
    print("PHASE 0 COMPLETED")
    print(f"Final success rate: {sum(success_window) / len(success_window):.2%}")
    print("=" * 80)
    
    return agent, fg_model, history


def run_phase_1(
    agent: ProposalAgent,
    fg_model: BidirectionalFG,
    num_episodes: int = 500,
    device: str = 'cpu'
):
    """
    Phase 1: Test on unknown shapes (lever, button, knob).
    
    Expected outcome:
    - Agent encounters high η (unknown shapes)
    - Suspension is triggered
    - F/G adapts through fine-tuning
    - Agent learns to generalize
    """
    print("\n" + "=" * 80)
    print("PHASE 1: Test on Unknown Shapes")
    print("=" * 80)
    
    # Create suspension structure
    suspension = SuspensionStructure(
        eta_threshold=0.1,
        buffer_size=100,
        min_buffer_for_update=10,
        fg_update_steps=10
    )
    
    # F/G optimizer for fine-tuning
    fg_optimizer = optim.Adam(fg_model.parameters(), lr=1e-4)
    
    # Agent optimizer
    agent_optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    
    # Training loop
    print(f"Testing for {num_episodes} episodes...")
    
    history = {
        'episode': [],
        'success': [],
        'reward': [],
        'steps': [],
        'eta': [],
        'suspension_count': [],
        'fg_updates': []
    }
    
    success_window = []
    window_size = 100
    
    for episode in tqdm(range(num_episodes), desc="Phase 1"):
        # Sample object type (unknown shapes)
        object_type = np.random.choice([
            EscapeRoomEnv.LEVER,
            EscapeRoomEnv.BUTTON,
            EscapeRoomEnv.KNOB
        ])
        
        # Create environment
        env = EscapeRoomEnv(object_type=object_type, render=False, num_points=512)
        obs = env.reset()
        
        # Convert to torch
        pos = torch.FloatTensor(obs).to(device)
        
        # Get affordance from F/G
        with torch.no_grad():
            affordance = fg_model.get_affordance_from_shape(pos).mean(dim=0, keepdim=True)
            eta = fg_model.compute_eta(pos.unsqueeze(0))
        
        # Check suspension
        is_suspended = suspension.check_suspension(eta)
        
        fg_updated = False
        if is_suspended:
            # Add to buffer
            suspension.add_to_buffer(pos, affordance=affordance, eta=eta)
            
            # Fine-tune F/G
            if suspension.should_update_fg():
                avg_loss = finetune_fg_during_suspension(
                    fg_model, suspension, fg_optimizer, device, batch_size=32
                )
                fg_updated = True
        
        # Select action
        action_tensor, log_probs = agent.act(affordance, deterministic=False)
        action = action_tensor.item()
        
        # Take step
        next_obs, reward, done, info = env.step(action)
        
        # Update agent
        loss = -log_probs * reward
        agent_optimizer.zero_grad()
        loss.backward()
        agent_optimizer.step()
        
        # Record
        history['episode'].append(episode)
        history['success'].append(info['success'])
        history['reward'].append(reward)
        history['steps'].append(info['steps'])
        history['eta'].append(eta.item())
        history['suspension_count'].append(suspension.suspension_count)
        history['fg_updates'].append(fg_updated)
        
        success_window.append(info['success'])
        if len(success_window) > window_size:
            success_window.pop(0)
        
        # Log every 50 episodes
        if (episode + 1) % 50 == 0:
            success_rate = sum(success_window) / len(success_window) if success_window else 0
            avg_eta = np.mean(history['eta'][-50:])
            num_fg_updates = sum(history['fg_updates'][-50:])
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Success rate (last {min(len(success_window), window_size)}): {success_rate:.2%}")
            print(f"  Avg η: {avg_eta:.6f}")
            print(f"  Suspensions: {suspension.suspension_count}")
            print(f"  F/G updates (last 50): {num_fg_updates}")
        
        env.close()
    
    # Save results
    results_dir = Path('/home/ubuntu/adjunction-model/results/phase1')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'fg_state_dict': fg_model.state_dict(),
        'history': history
    }, results_dir / 'phase1_models.pt')
    
    # Save history
    with open(results_dir / 'phase1_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot results
    plot_phase_results(history, results_dir, "Phase 1")
    
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETED")
    print(f"Final success rate: {sum(success_window) / len(success_window) if success_window else 0:.2%}")
    print(f"Total F/G updates: {sum(history['fg_updates'])}")
    print("=" * 80)
    
    return history


def plot_phase_results(history: dict, save_dir: Path, phase_name: str):
    """Plot phase results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = history['episode']
    
    # Success rate (moving average)
    window = 50
    success_ma = np.convolve(history['success'], np.ones(window)/window, mode='valid')
    axes[0, 0].plot(episodes[window-1:], success_ma, linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title(f'{phase_name}: Success Rate (MA-{window})')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Reward
    reward_ma = np.convolve(history['reward'], np.ones(window)/window, mode='valid')
    axes[0, 1].plot(episodes[window-1:], reward_ma, linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title(f'{phase_name}: Reward (MA-{window})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # η
    eta_ma = np.convolve(history['eta'], np.ones(window)/window, mode='valid')
    axes[1, 0].plot(episodes[window-1:], eta_ma, linewidth=2, color='green')
    axes[1, 0].axhline(0.1, color='red', linestyle='--', label='Suspension threshold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('η')
    axes[1, 0].set_title(f'{phase_name}: Coherence Signal η (MA-{window})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Suspension count
    axes[1, 1].plot(episodes, history['suspension_count'], linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Suspensions')
    axes[1, 1].set_title(f'{phase_name}: Suspension Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{phase_name.lower().replace(' ', '_')}_results.png"
    plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Results plot saved to {save_dir / filename}")


if __name__ == '__main__':
    device = 'cpu'
    
    # Run Phase 0
    agent, fg_model, phase0_history = run_phase_0(
        num_episodes=1000,
        affordance_dim=16,
        device=device
    )
    
    # Run Phase 1
    phase1_history = run_phase_1(
        agent=agent,
        fg_model=fg_model,
        num_episodes=500,
        device=device
    )
    
    print("\n" + "=" * 80)
    print("ALL PHASES COMPLETED")
    print("=" * 80)
