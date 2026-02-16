"""
Phase 2 Slack Experiment Analysis Script

This script analyzes the results of Phase 2 Slack experiments, focusing on:
1. η (unit) and ε (counit) correlation and dynamics
2. Agent C's behavior patterns
3. Intrinsic motivation evolution
4. Emergence of suspension structures
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.signal import find_peaks
import seaborn as sns


def load_metrics(results_dir: str):
    """Load metrics from the experiment results directory."""
    results_path = Path(results_dir)
    
    with open(results_path / 'metrics.json', 'r') as f:
        metrics = json.load(f)
    
    with open(results_path / 'slack_analysis.json', 'r') as f:
        slack_analysis = json.load(f)
    
    return metrics, slack_analysis


def analyze_eta_epsilon_correlation(metrics, output_dir: Path):
    """
    Analyze the correlation between η (unit) and ε (counit).
    
    Key questions:
    - Are they correlated? Anti-correlated? Independent?
    - Does the agent learn to trade one for the other?
    - Do they co-vary over time?
    """
    # Handle both flat and nested metric structures
    if 'train' in metrics:
        eta = np.array(metrics['train']['unit_eta'])
        epsilon = np.array(metrics['val']['counit_eps']) if 'val' in metrics and 'counit_eps' in metrics['val'] else None
    else:
        eta = np.array(metrics['unit_eta'])
        epsilon = np.array(metrics['counit_eps']) if 'counit_eps' in metrics else None
    
    epochs = list(range(len(eta)))
    
    if epsilon is not None and len(epsilon) > 0:
        
        # Compute correlations
        pearson_corr, pearson_p = pearsonr(eta, epsilon)
        spearman_corr, spearman_p = spearmanr(eta, epsilon)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Time series of η and ε
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        ax1.plot(epochs, eta, 'b-', label='η (unit)', linewidth=2)
        ax1_twin.plot(epochs, epsilon, 'r-', label='ε (counit)', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('η (unit)', color='b')
        ax1_twin.set_ylabel('ε (counit)', color='r')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        ax1.set_title('η and ε Over Time')
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot with correlation
        ax2 = axes[0, 1]
        ax2.scatter(eta, epsilon, alpha=0.6, s=50)
        ax2.set_xlabel('η (unit)')
        ax2.set_ylabel('ε (counit)')
        ax2.set_title(f'η vs ε Correlation\nPearson: {pearson_corr:.3f} (p={pearson_p:.3e})\nSpearman: {spearman_corr:.3f} (p={spearman_p:.3e})')
        
        # Add trend line
        z = np.polyfit(eta, epsilon, 1)
        p = np.poly1d(z)
        ax2.plot(eta, p(eta), "r--", alpha=0.8, linewidth=2)
        ax2.grid(True, alpha=0.3)
        
        # 3. Rate of change
        ax3 = axes[1, 0]
        eta_diff = np.diff(eta)
        epsilon_diff = np.diff(epsilon)
        ax3.scatter(eta_diff, epsilon_diff, alpha=0.6, s=50)
        ax3.set_xlabel('Δη (change in unit)')
        ax3.set_ylabel('Δε (change in counit)')
        ax3.set_title('Rate of Change: Δη vs Δε')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # 4. Joint distribution
        ax4 = axes[1, 1]
        # Normalize to [0, 1] for comparison
        eta_norm = (eta - eta.min()) / (eta.max() - eta.min() + 1e-8)
        epsilon_norm = (epsilon - epsilon.min()) / (epsilon.max() - epsilon.min() + 1e-8)
        
        ax4.hist2d(eta_norm, epsilon_norm, bins=20, cmap='viridis')
        ax4.set_xlabel('η (normalized)')
        ax4.set_ylabel('ε (normalized)')
        ax4.set_title('Joint Distribution of η and ε')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'eta_epsilon_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Statistical summary
        analysis = {
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'eta_mean': float(eta.mean()),
            'eta_std': float(eta.std()),
            'epsilon_mean': float(epsilon.mean()),
            'epsilon_std': float(epsilon.std()),
            'eta_trend': 'increasing' if eta[-1] > eta[0] else 'decreasing',
            'epsilon_trend': 'increasing' if epsilon[-1] > epsilon[0] else 'decreasing',
        }
        
        return analysis
    else:
        print("Warning: ε (counit) data not available in validation metrics")
        return None


def analyze_slack_dynamics(metrics, output_dir: Path):
    """
    Analyze the dynamics of slack (η + ε) over time.
    
    Key questions:
    - Does total slack increase, decrease, or remain stable?
    - Are there critical points or phase transitions?
    - Does the agent learn to modulate slack?
    """
    # Handle both flat and nested metric structures
    if 'train' in metrics:
        eta = np.array(metrics['train']['unit_eta'])
        epsilon = np.array(metrics['val']['counit_eps']) if 'val' in metrics and 'counit_eps' in metrics['val'] else None
    else:
        eta = np.array(metrics['unit_eta'])
        epsilon = np.array(metrics['counit_eps']) if 'counit_eps' in metrics else None
    
    epochs = list(range(len(eta)))
    
    if epsilon is not None and len(epsilon) > 0:
        total_slack = eta + epsilon
        
        # Detect peaks and troughs
        peaks, _ = find_peaks(total_slack, distance=5)
        troughs, _ = find_peaks(-total_slack, distance=5)
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 1. Total slack over time
        ax1 = axes[0]
        ax1.plot(epochs, total_slack, 'g-', linewidth=2, label='Total Slack (η + ε)')
        ax1.plot(epochs, eta, 'b--', alpha=0.6, label='η (unit)')
        ax1.plot(epochs, epsilon, 'r--', alpha=0.6, label='ε (counit)')
        
        # Mark peaks and troughs
        if len(peaks) > 0:
            ax1.plot(peaks, total_slack[peaks], 'g^', markersize=10, label='Peaks')
        if len(troughs) > 0:
            ax1.plot(troughs, total_slack[troughs], 'gv', markersize=10, label='Troughs')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Slack Value')
        ax1.set_title('Total Slack Dynamics (η + ε)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Slack modulation (rate of change)
        ax2 = axes[1]
        slack_diff = np.diff(total_slack)
        ax2.plot(epochs[1:], slack_diff, 'purple', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('ΔSlack (change per epoch)')
        ax2.set_title('Slack Modulation Rate')
        ax2.grid(True, alpha=0.3)
        
        # Highlight periods of increasing vs decreasing slack
        increasing = slack_diff > 0
        ax2.fill_between(epochs[1:], 0, slack_diff, where=increasing, alpha=0.3, color='green', label='Increasing')
        ax2.fill_between(epochs[1:], 0, slack_diff, where=~increasing, alpha=0.3, color='red', label='Decreasing')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'slack_dynamics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        analysis = {
            'total_slack_initial': float(total_slack[0]),
            'total_slack_final': float(total_slack[-1]),
            'total_slack_change': float(total_slack[-1] - total_slack[0]),
            'total_slack_mean': float(total_slack.mean()),
            'total_slack_std': float(total_slack.std()),
            'num_peaks': int(len(peaks)),
            'num_troughs': int(len(troughs)),
            'slack_modulation_amplitude': float(np.abs(slack_diff).mean()),
        }
        
        return analysis
    else:
        return None


def analyze_agent_behavior(metrics, output_dir: Path):
    """
    Analyze Agent C's behavior patterns.
    
    Key questions:
    - How does the agent's policy evolve?
    - Are there emergent patterns in coherence, uncertainty, or valence?
    - Does the agent show signs of suspension structure?
    """
    # Handle both flat and nested metric structures
    if 'train' in metrics:
        aff_loss = np.array(metrics['train']['affordance'])
        kl_loss = np.array(metrics['train']['kl'])
        coherence = np.array(metrics['train']['coherence'])
        total_loss = np.array(metrics['train']['loss'])
    else:
        aff_loss = np.array(metrics['aff'])
        kl_loss = np.array(metrics['kl'])
        coherence = np.array(metrics['coherence'])
        total_loss = np.array(metrics['loss'])
    
    epochs = list(range(len(total_loss)))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss components
    ax1 = axes[0, 0]
    ax1.plot(epochs, aff_loss, label='Affordance Loss', linewidth=2)
    ax1.plot(epochs, kl_loss, label='KL Divergence', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Coherence evolution
    ax2 = axes[0, 1]
    ax2.plot(epochs, coherence, 'orange', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Coherence')
    ax2.set_title('Coherence Regularization')
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning rate (loss gradient)
    ax3 = axes[1, 0]
    loss_gradient = np.gradient(total_loss)
    ax3.plot(epochs, loss_gradient, 'purple', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Gradient')
    ax3.set_title('Learning Rate (Loss Change)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Phase diagram (Affordance vs Coherence)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(coherence, aff_loss, c=epochs, cmap='viridis', s=50, alpha=0.6)
    ax4.set_xlabel('Coherence')
    ax4.set_ylabel('Affordance Loss')
    ax4.set_title('Agent Behavior Phase Diagram')
    plt.colorbar(scatter, ax=ax4, label='Epoch')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'agent_behavior.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Detect potential suspension structure indicators
    # 1. Non-monotonic coherence (oscillation)
    coherence_peaks, _ = find_peaks(coherence, distance=5)
    coherence_troughs, _ = find_peaks(-coherence, distance=5)
    oscillation_detected = len(coherence_peaks) > 2 and len(coherence_troughs) > 2
    
    # 2. Plateau in loss (exploration phase)
    loss_diff = np.abs(np.diff(total_loss))
    plateau_threshold = loss_diff.mean() * 0.1
    plateau_epochs = np.sum(loss_diff < plateau_threshold)
    
    analysis = {
        'final_affordance_loss': float(aff_loss[-1]),
        'affordance_improvement': float(aff_loss[0] - aff_loss[-1]),
        'final_coherence': float(coherence[-1]),
        'coherence_stability': float(coherence.std()),
        'oscillation_detected': bool(oscillation_detected),
        'num_coherence_peaks': int(len(coherence_peaks)),
        'plateau_epochs': int(plateau_epochs),
        'suspension_structure_indicator': bool(oscillation_detected and plateau_epochs > 5),
    }
    
    return analysis


def generate_comprehensive_report(eta_eps_analysis, slack_analysis_data, agent_analysis, output_dir: Path):
    """Generate a comprehensive analysis report."""
    report = []
    report.append("=" * 80)
    report.append("PHASE 2 SLACK EXPERIMENT - COMPREHENSIVE ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    # η/ε Correlation Analysis
    if eta_eps_analysis:
        report.append("## 1. η/ε Correlation Analysis")
        report.append("-" * 80)
        report.append(f"Pearson Correlation: {eta_eps_analysis['pearson_correlation']:.4f} (p={eta_eps_analysis['pearson_p_value']:.2e})")
        report.append(f"Spearman Correlation: {eta_eps_analysis['spearman_correlation']:.4f} (p={eta_eps_analysis['spearman_p_value']:.2e})")
        report.append("")
        report.append(f"η Statistics:")
        report.append(f"  Mean: {eta_eps_analysis['eta_mean']:.4f}")
        report.append(f"  Std:  {eta_eps_analysis['eta_std']:.4f}")
        report.append(f"  Trend: {eta_eps_analysis['eta_trend']}")
        report.append("")
        report.append(f"ε Statistics:")
        report.append(f"  Mean: {eta_eps_analysis['epsilon_mean']:.4f}")
        report.append(f"  Std:  {eta_eps_analysis['epsilon_std']:.4f}")
        report.append(f"  Trend: {eta_eps_analysis['epsilon_trend']}")
        report.append("")
        
        # Interpretation
        if abs(eta_eps_analysis['pearson_correlation']) > 0.7:
            if eta_eps_analysis['pearson_correlation'] > 0:
                report.append("✅ STRONG POSITIVE CORRELATION: η and ε increase/decrease together.")
            else:
                report.append("✅ STRONG NEGATIVE CORRELATION: η and ε trade off (one increases as the other decreases).")
        elif abs(eta_eps_analysis['pearson_correlation']) < 0.3:
            report.append("✅ WEAK CORRELATION: η and ε are largely independent.")
        else:
            report.append("⚠️  MODERATE CORRELATION: η and ε show some relationship.")
        report.append("")
    
    # Slack Dynamics Analysis
    if slack_analysis_data:
        report.append("## 2. Slack Dynamics Analysis")
        report.append("-" * 80)
        report.append(f"Total Slack (η + ε):")
        report.append(f"  Initial: {slack_analysis_data['total_slack_initial']:.4f}")
        report.append(f"  Final:   {slack_analysis_data['total_slack_final']:.4f}")
        report.append(f"  Change:  {slack_analysis_data['total_slack_change']:+.4f} ({slack_analysis_data['total_slack_change']/slack_analysis_data['total_slack_initial']*100:+.1f}%)")
        report.append(f"  Mean:    {slack_analysis_data['total_slack_mean']:.4f}")
        report.append(f"  Std:     {slack_analysis_data['total_slack_std']:.4f}")
        report.append("")
        report.append(f"Modulation:")
        report.append(f"  Peaks:   {slack_analysis_data['num_peaks']}")
        report.append(f"  Troughs: {slack_analysis_data['num_troughs']}")
        report.append(f"  Amplitude: {slack_analysis_data['slack_modulation_amplitude']:.4f}")
        report.append("")
        
        # Interpretation
        if abs(slack_analysis_data['total_slack_change']) < slack_analysis_data['total_slack_initial'] * 0.1:
            report.append("✅ SLACK PRESERVED: Total slack remains stable (< 10% change).")
        else:
            report.append("⚠️  SLACK CHANGED: Total slack changed significantly.")
        
        if slack_analysis_data['num_peaks'] > 3 and slack_analysis_data['num_troughs'] > 3:
            report.append("✅ MODULATION DETECTED: Agent shows dynamic slack modulation.")
        report.append("")
    
    # Agent Behavior Analysis
    if agent_analysis:
        report.append("## 3. Agent C Behavior Analysis")
        report.append("-" * 80)
        report.append(f"Learning Performance:")
        report.append(f"  Final Affordance Loss: {agent_analysis['final_affordance_loss']:.4f}")
        report.append(f"  Improvement: {agent_analysis['affordance_improvement']:.4f}")
        report.append("")
        report.append(f"Coherence:")
        report.append(f"  Final Value: {agent_analysis['final_coherence']:.4f}")
        report.append(f"  Stability (Std): {agent_analysis['coherence_stability']:.4f}")
        report.append(f"  Oscillation Detected: {agent_analysis['oscillation_detected']}")
        report.append(f"  Number of Peaks: {agent_analysis['num_coherence_peaks']}")
        report.append("")
        report.append(f"Exploration:")
        report.append(f"  Plateau Epochs: {agent_analysis['plateau_epochs']}")
        report.append("")
        
        # Suspension structure indicator
        if agent_analysis['suspension_structure_indicator']:
            report.append("🎯 SUSPENSION STRUCTURE INDICATOR: Agent shows signs of suspension structure!")
            report.append("   - Non-monotonic coherence (oscillation)")
            report.append("   - Extended exploration phases (plateau)")
        else:
            report.append("⚠️  NO CLEAR SUSPENSION STRUCTURE: Agent behavior appears monotonic.")
        report.append("")
    
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_dir / 'comprehensive_analysis.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    
    # Save JSON
    full_analysis = {
        'eta_epsilon_correlation': eta_eps_analysis,
        'slack_dynamics': slack_analysis_data,
        'agent_behavior': agent_analysis,
    }
    
    with open(output_dir / 'comprehensive_analysis.json', 'w') as f:
        json.dump(full_analysis, f, indent=2)
    
    return report_text


def main(results_dir: str = '/home/ubuntu/adjunction-model/results/phase2_slack'):
    """Run comprehensive analysis on Phase 2 Slack experiment results."""
    output_dir = Path(results_dir)
    
    print("Loading metrics...")
    metrics, slack_analysis = load_metrics(results_dir)
    
    print("\n1. Analyzing η/ε correlation...")
    eta_eps_analysis = analyze_eta_epsilon_correlation(metrics, output_dir)
    
    print("\n2. Analyzing slack dynamics...")
    slack_dynamics = analyze_slack_dynamics(metrics, output_dir)
    
    print("\n3. Analyzing agent behavior...")
    agent_analysis = analyze_agent_behavior(metrics, output_dir)
    
    print("\n4. Generating comprehensive report...")
    generate_comprehensive_report(eta_eps_analysis, slack_dynamics, agent_analysis, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")


if __name__ == '__main__':
    main()
