"""
Temporal Suspension Experiment — Analysis Script

Analyses the results produced by temporal_suspension_experiment.py, focusing on:

1. η(t) time evolution: How does unit slack change over the revelation sequence?
2. Action timing comparison: Does the slack model wait longer than the tight model?
3. Accuracy vs timing trade-off: Does waiting improve classification accuracy?
4. Confidence gate dynamics: How does the learned confidence evolve?
5. Slack vs tight comparison: Overall performance comparison.

Outputs:
    - Visualisation PNGs in the results directory
    - analysis_report.txt with statistical summary
    - analysis.json with machine-readable metrics

Usage:
    python experiments/analyze_temporal_suspension.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional


# ======================================================================
# Data loading
# ======================================================================

def load_metrics(results_dir: str) -> Dict[str, Dict]:
    """Load metrics for both slack and tight modes."""
    base = Path(results_dir)
    data = {}
    for mode in ['slack', 'tight']:
        path = base / mode / 'metrics.json'
        if path.exists():
            with open(path) as f:
                data[mode] = json.load(f)
    # Load summary if available
    summary_path = base / 'summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            data['summary'] = json.load(f)
    return data


# ======================================================================
# 1. η(t) time evolution
# ======================================================================

def plot_eta_time_evolution(data: Dict, output_dir: Path):
    """
    Plot η as a function of time step, comparing slack and tight models.

    This is the core visualisation of the experiment: we expect the slack
    model to show a gradual decrease in η as the shape is revealed, while
    the tight model should have uniformly low η.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, mode in enumerate(['slack', 'tight']):
        if mode not in data:
            continue
        ax = axes[idx]
        metrics = data[mode]

        # eta_by_step is a list of dicts (one per epoch), each mapping
        # step_index (str) → mean η
        eta_steps = metrics.get('eta_by_step', [])
        if not eta_steps:
            continue

        num_epochs = len(eta_steps)
        T = max(int(k) for k in eta_steps[-1].keys()) + 1

        # Build matrix (epochs × steps)
        eta_matrix = np.zeros((num_epochs, T))
        for e, step_dict in enumerate(eta_steps):
            for t_str, val in step_dict.items():
                eta_matrix[e, int(t_str)] = val

        # Plot: early, mid, late epochs
        early = eta_matrix[:5].mean(axis=0)
        mid_start = max(0, num_epochs // 2 - 2)
        mid_end = min(num_epochs, num_epochs // 2 + 3)
        mid = eta_matrix[mid_start:mid_end].mean(axis=0)
        late = eta_matrix[-5:].mean(axis=0)

        steps = np.arange(T)
        ax.plot(steps, early, 'b--', label='Early (ep 1-5)', linewidth=2)
        ax.plot(steps, mid, 'g-', label=f'Mid (ep {mid_start+1}-{mid_end})',
                linewidth=2)
        ax.plot(steps, late, 'r-', label=f'Late (ep {num_epochs-4}-{num_epochs})',
                linewidth=2, marker='o')

        ax.set_xlabel('Time Step t', fontsize=12)
        ax.set_ylabel('η(t) — Unit Slack', fontsize=12)
        ax.set_title(f'η(t) Evolution — {mode.upper()} Model', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'eta_time_evolution.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved eta_time_evolution.png")


# ======================================================================
# 2. Confidence dynamics
# ======================================================================

def plot_confidence_dynamics(data: Dict, output_dir: Path):
    """
    Plot the confidence gate output c(t) over time steps.

    We expect the slack model's confidence to rise gradually (deferred
    commitment), while the tight model may commit earlier.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, mode in enumerate(['slack', 'tight']):
        if mode not in data:
            continue
        ax = axes[idx]
        metrics = data[mode]

        conf_steps = metrics.get('conf_by_step', [])
        if not conf_steps:
            continue

        num_epochs = len(conf_steps)
        T = max(int(k) for k in conf_steps[-1].keys()) + 1

        conf_matrix = np.zeros((num_epochs, T))
        for e, step_dict in enumerate(conf_steps):
            for t_str, val in step_dict.items():
                conf_matrix[e, int(t_str)] = val

        late = conf_matrix[-5:].mean(axis=0)
        steps = np.arange(T)

        ax.plot(steps, late, 'purple', linewidth=2, marker='s')
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5,
                    label='Threshold (0.5)')
        ax.fill_between(steps, 0, late, alpha=0.15, color='purple')

        ax.set_xlabel('Time Step t', fontsize=12)
        ax.set_ylabel('Confidence c(t)', fontsize=12)
        ax.set_title(f'Confidence Dynamics — {mode.upper()} Model',
                     fontsize=14)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_dynamics.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved confidence_dynamics.png")


# ======================================================================
# 3. Action timing comparison
# ======================================================================

def plot_action_timing(data: Dict, output_dir: Path):
    """
    Compare when each model decides to act across training.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for mode, color, label in [('slack', 'blue', 'Slack (Phase 2)'),
                                ('tight', 'red', 'Tight (Phase 1)')]:
        if mode not in data:
            continue
        metrics = data[mode]
        action_steps = metrics.get('mean_action_step', [])
        if action_steps:
            epochs = list(range(1, len(action_steps) + 1))
            ax.plot(epochs, action_steps, color=color, label=label,
                    linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Action Step', fontsize=12)
    ax.set_title('Action Timing Over Training', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'action_timing.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved action_timing.png")


# ======================================================================
# 4. Accuracy comparison
# ======================================================================

def plot_accuracy_comparison(data: Dict, output_dir: Path):
    """
    Compare classification accuracy between slack and tight models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training accuracy
    ax1 = axes[0]
    for mode, color, label in [('slack', 'blue', 'Slack'),
                                ('tight', 'red', 'Tight')]:
        if mode not in data:
            continue
        acc = data[mode].get('accuracy', [])
        if acc:
            ax1.plot(range(1, len(acc) + 1), acc, color=color, label=label,
                     linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training Accuracy', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Validation accuracy
    ax2 = axes[1]
    for mode, color, label in [('slack', 'blue', 'Slack'),
                                ('tight', 'red', 'Tight')]:
        if mode not in data:
            continue
        acc = data[mode].get('val_accuracy', [])
        if acc:
            ax2.plot(range(1, len(acc) + 1), acc, color=color, label=label,
                     linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved accuracy_comparison.png")


# ======================================================================
# 5. Comprehensive analysis figure
# ======================================================================

def plot_comprehensive_analysis(data: Dict, output_dir: Path):
    """
    8-panel figure summarising the experiment.
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for col, mode in enumerate(['slack', 'tight']):
        if mode not in data:
            continue
        metrics = data[mode]

        # Row 0, Col 0/1: η(t) at final epoch
        ax = axes[0, col]
        eta_steps = metrics.get('eta_by_step', [])
        if eta_steps:
            T = max(int(k) for k in eta_steps[-1].keys()) + 1
            late_eta = np.zeros(T)
            for step_dict in eta_steps[-5:]:
                for t_str, val in step_dict.items():
                    late_eta[int(t_str)] += val
            late_eta /= min(5, len(eta_steps))
            ax.plot(range(T), late_eta, 'b-o', linewidth=2)
        ax.set_title(f'η(t) — {mode.upper()}')
        ax.set_xlabel('Step')
        ax.set_ylabel('η')
        ax.grid(True, alpha=0.3)

        # Row 0, Col 2/3: ε(t) at final epoch
        ax = axes[0, col + 2]
        eps_steps = metrics.get('eps_by_step', [])
        if eps_steps:
            T = max(int(k) for k in eps_steps[-1].keys()) + 1
            late_eps = np.zeros(T)
            for step_dict in eps_steps[-5:]:
                for t_str, val in step_dict.items():
                    late_eps[int(t_str)] += val
            late_eps /= min(5, len(eps_steps))
            ax.plot(range(T), late_eps, 'r-s', linewidth=2)
        ax.set_title(f'ε(t) — {mode.upper()}')
        ax.set_xlabel('Step')
        ax.set_ylabel('ε')
        ax.grid(True, alpha=0.3)

        # Row 1, Col 0/1: Confidence at final epoch
        ax = axes[1, col]
        conf_steps = metrics.get('conf_by_step', [])
        if conf_steps:
            T = max(int(k) for k in conf_steps[-1].keys()) + 1
            late_conf = np.zeros(T)
            for step_dict in conf_steps[-5:]:
                for t_str, val in step_dict.items():
                    late_conf[int(t_str)] += val
            late_conf /= min(5, len(conf_steps))
            ax.plot(range(T), late_conf, 'purple', linewidth=2, marker='D')
            ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax.set_title(f'Confidence — {mode.upper()}')
        ax.set_xlabel('Step')
        ax.set_ylabel('c(t)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Row 1, Col 2/3: Loss over epochs
        ax = axes[1, col + 2]
        loss = metrics.get('loss', [])
        if loss:
            ax.plot(range(1, len(loss) + 1), loss, 'green', linewidth=2)
        ax.set_title(f'Total Loss — {mode.upper()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Temporal Suspension Experiment — Comprehensive Analysis',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_analysis.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved comprehensive_analysis.png")


# ======================================================================
# Text report
# ======================================================================

def generate_report(data: Dict, output_dir: Path):
    """Generate a human-readable analysis report."""
    lines = []
    lines.append("=" * 80)
    lines.append("Temporal Suspension Experiment — Analysis Report")
    lines.append("=" * 80)
    lines.append("")

    summary = data.get('summary', {})

    for mode in ['slack', 'tight']:
        if mode not in data:
            continue
        metrics = data[mode]
        s = summary.get(mode, {})

        lines.append(f"--- {mode.upper()} Model ---")
        lines.append("")

        # Final metrics
        acc = metrics.get('val_accuracy', [])
        act = metrics.get('val_action_step', [])
        eta_f = metrics.get('val_eta_final', [])
        eps_f = metrics.get('val_eps_final', [])

        if acc:
            lines.append(f"  Final Validation Accuracy:  {acc[-1]:.4f}")
        if act:
            lines.append(f"  Final Mean Action Step:     {act[-1]:.2f}")
        if eta_f:
            lines.append(f"  Final η (unit slack):       {eta_f[-1]:.4f}")
        if eps_f:
            lines.append(f"  Final ε (counit slack):     {eps_f[-1]:.4f}")

        # η(t) trajectory at final epoch
        eta_steps = metrics.get('eta_by_step', [])
        if eta_steps:
            last = eta_steps[-1]
            T = max(int(k) for k in last.keys()) + 1
            lines.append(f"  η(t) at final epoch:")
            for t in range(T):
                val = last.get(str(t), 0.0)
                lines.append(f"    Step {t}: η = {val:.4f}")

        lines.append("")

    # Comparison
    if 'slack' in summary and 'tight' in summary:
        lines.append("--- COMPARISON ---")
        lines.append("")
        s_acc = summary['slack'].get('final_accuracy', 0)
        t_acc = summary['tight'].get('final_accuracy', 0)
        s_step = summary['slack'].get('final_action_step', 0)
        t_step = summary['tight'].get('final_action_step', 0)
        s_eta = summary['slack'].get('final_eta', 0)
        t_eta = summary['tight'].get('final_eta', 0)

        lines.append(f"  Accuracy:     Slack={s_acc:.4f}  Tight={t_acc:.4f}  "
                     f"Diff={s_acc - t_acc:+.4f}")
        lines.append(f"  Action Step:  Slack={s_step:.2f}  Tight={t_step:.2f}  "
                     f"Diff={s_step - t_step:+.2f}")
        lines.append(f"  Final η:      Slack={s_eta:.4f}  Tight={t_eta:.4f}  "
                     f"Diff={s_eta - t_eta:+.4f}")
        lines.append("")

        # Interpretation
        lines.append("--- INTERPRETATION ---")
        lines.append("")
        if s_step > t_step:
            lines.append(
                "  The SLACK model waits longer before acting, consistent with")
            lines.append(
                "  the temporal suspension hypothesis: preserved slack enables")
            lines.append(
                "  the agent to defer action under ambiguity.")
        else:
            lines.append(
                "  The SLACK model does NOT wait longer. This may indicate that")
            lines.append(
                "  the confidence gate needs further tuning, or that the")
            lines.append(
                "  ambiguity schedule is not sufficiently challenging.")

        if s_acc > t_acc:
            lines.append(
                "  The SLACK model achieves higher accuracy, suggesting that")
            lines.append(
                "  deferred action leads to better classification.")
        elif s_acc < t_acc:
            lines.append(
                "  The TIGHT model achieves higher accuracy. This may be")
            lines.append(
                "  expected if the tight model commits quickly on easy cases.")

        if s_eta > t_eta:
            lines.append(
                "  The SLACK model preserves more η, confirming that slack")
            lines.append(
                "  preservation works in the temporal setting.")

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report)
    print("  Saved analysis_report.txt")
    print(report)


# ======================================================================
# Machine-readable analysis
# ======================================================================

def save_analysis_json(data: Dict, output_dir: Path):
    """Save machine-readable analysis results."""
    analysis = {}

    for mode in ['slack', 'tight']:
        if mode not in data:
            continue
        metrics = data[mode]
        entry = {}

        # Scalar summaries
        for key in ['accuracy', 'val_accuracy', 'mean_action_step',
                     'val_action_step', 'val_eta_final', 'val_eps_final',
                     'loss']:
            vals = metrics.get(key, [])
            if vals:
                entry[f'{key}_final'] = vals[-1]
                entry[f'{key}_mean'] = float(np.mean(vals))

        # η trajectory at final epoch
        eta_steps = metrics.get('eta_by_step', [])
        if eta_steps:
            entry['eta_trajectory_final'] = eta_steps[-1]

        analysis[mode] = entry

    with open(output_dir / 'analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print("  Saved analysis.json")


# ======================================================================
# Main
# ======================================================================

def main():
    results_dir = "/home/ubuntu/adjunction-model/results/temporal_suspension"
    output_dir = Path(results_dir)

    print("Loading metrics...")
    data = load_metrics(results_dir)

    if not data:
        print("ERROR: No metrics found. Run the experiment first:")
        print("  python experiments/temporal_suspension_experiment.py")
        return

    print(f"Found modes: {[k for k in data.keys() if k != 'summary']}")
    print("\nGenerating analysis...")

    plot_eta_time_evolution(data, output_dir)
    plot_confidence_dynamics(data, output_dir)
    plot_action_timing(data, output_dir)
    plot_accuracy_comparison(data, output_dir)
    plot_comprehensive_analysis(data, output_dir)
    generate_report(data, output_dir)
    save_analysis_json(data, output_dir)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
