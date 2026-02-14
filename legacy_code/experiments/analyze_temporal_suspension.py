"""
Temporal Suspension Experiment — Analysis Script (Active Assembly)

Analyses the results produced by temporal_suspension_experiment.py, focusing on:

1. Displacement magnitude ‖Δx(t)‖ over time: Does the slack model produce
   small displacements early (exploration) and large ones later (commitment)?
2. Chamfer Distance CD(t) over time: How quickly does each model converge
   to the target shape?
3. η(t) time evolution: How does unit slack change over the assembly process?
4. ε(t) time evolution: How does counit slack change?
5. Slack vs tight comparison: Overall performance and behavioural differences.

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
from typing import Dict


# ======================================================================
# Data loading
# ======================================================================

def load_metrics(results_dir: str) -> Dict:
    """Load metrics for both slack and tight modes."""
    base = Path(results_dir)
    data = {}
    for mode in ['slack', 'tight']:
        path = base / mode / 'metrics.json'
        if path.exists():
            with open(path) as f:
                data[mode] = json.load(f)
    summary_path = base / 'summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            data['summary'] = json.load(f)
    return data


# ======================================================================
# Helper: extract per-step trajectory from epoch list
# ======================================================================

def _extract_step_trajectory(
    epoch_list, epoch_range=None
) -> np.ndarray:
    """
    Given a list-of-dicts (one per epoch, each mapping str(step) → value),
    return a (num_epochs_selected, T) matrix.
    """
    if not epoch_list:
        return np.array([])
    if epoch_range is None:
        epoch_range = range(len(epoch_list))
    T = max(int(k) for k in epoch_list[-1].keys()) + 1
    mat = np.zeros((len(epoch_range), T))
    for i, e in enumerate(epoch_range):
        for t_str, val in epoch_list[e].items():
            mat[i, int(t_str)] = val
    return mat


# ======================================================================
# 1. Displacement magnitude over time
# ======================================================================

def plot_displacement_dynamics(data: Dict, output_dir: Path):
    """
    Plot ‖Δx(t)‖ as a function of time step for both models.

    Core visualisation: we expect the slack model to show small→large
    displacement pattern, while the tight model commits early.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, mode in enumerate(['slack', 'tight']):
        if mode not in data:
            continue
        ax = axes[idx]
        metrics = data[mode]

        disp_steps = metrics.get('disp_mag_by_step', [])
        if not disp_steps:
            continue

        num_ep = len(disp_steps)
        T = max(int(k) for k in disp_steps[-1].keys()) + 1

        early = _extract_step_trajectory(disp_steps, range(min(5, num_ep)))
        late = _extract_step_trajectory(
            disp_steps, range(max(0, num_ep - 5), num_ep))

        steps = np.arange(T)
        ax.plot(steps, early.mean(axis=0), 'b--',
                label='Early (ep 1-5)', linewidth=2)
        ax.plot(steps, late.mean(axis=0), 'r-',
                label=f'Late (ep {max(1,num_ep-4)}-{num_ep})',
                linewidth=2, marker='o')
        ax.fill_between(steps,
                        late.mean(axis=0) - late.std(axis=0),
                        late.mean(axis=0) + late.std(axis=0),
                        alpha=0.15, color='red')

        ax.set_xlabel('Time Step t', fontsize=12)
        ax.set_ylabel('‖Δx(t)‖ — Displacement Magnitude', fontsize=12)
        ax.set_title(f'Displacement Dynamics — {mode.upper()}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'displacement_dynamics.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved displacement_dynamics.png")


# ======================================================================
# 2. Chamfer Distance over time
# ======================================================================

def plot_chamfer_distance(data: Dict, output_dir: Path):
    """Plot CD(t) over time steps, comparing slack and tight."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for mode, color, label in [('slack', 'blue', 'Slack (Phase 2)'),
                                ('tight', 'red', 'Tight (Phase 1)')]:
        if mode not in data:
            continue
        cd_steps = data[mode].get('cd_by_step', [])
        if not cd_steps:
            continue

        num_ep = len(cd_steps)
        late = _extract_step_trajectory(
            cd_steps, range(max(0, num_ep - 5), num_ep))
        T = late.shape[1] if late.ndim == 2 else 0
        if T == 0:
            continue

        steps = np.arange(T)
        mean = late.mean(axis=0)
        ax.plot(steps, mean, color=color, label=label,
                linewidth=2, marker='o')
        ax.fill_between(steps,
                        mean - late.std(axis=0),
                        mean + late.std(axis=0),
                        alpha=0.1, color=color)

    ax.set_xlabel('Time Step t', fontsize=12)
    ax.set_ylabel('Chamfer Distance', fontsize=12)
    ax.set_title('Chamfer Distance Over Assembly Steps (Late Epochs)',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'chamfer_distance_over_time.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved chamfer_distance_over_time.png")


# ======================================================================
# 3. η(t) time evolution
# ======================================================================

def plot_eta_time_evolution(data: Dict, output_dir: Path):
    """Plot η as a function of time step."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, mode in enumerate(['slack', 'tight']):
        if mode not in data:
            continue
        ax = axes[idx]
        eta_steps = data[mode].get('eta_by_step', [])
        if not eta_steps:
            continue

        num_ep = len(eta_steps)
        T = max(int(k) for k in eta_steps[-1].keys()) + 1

        early = _extract_step_trajectory(eta_steps, range(min(5, num_ep)))
        mid_s = max(0, num_ep // 2 - 2)
        mid_e = min(num_ep, num_ep // 2 + 3)
        mid = _extract_step_trajectory(eta_steps, range(mid_s, mid_e))
        late = _extract_step_trajectory(
            eta_steps, range(max(0, num_ep - 5), num_ep))

        steps = np.arange(T)
        ax.plot(steps, early.mean(axis=0), 'b--',
                label='Early (ep 1-5)', linewidth=2)
        ax.plot(steps, mid.mean(axis=0), 'g-',
                label=f'Mid (ep {mid_s+1}-{mid_e})', linewidth=2)
        ax.plot(steps, late.mean(axis=0), 'r-',
                label=f'Late (ep {max(1,num_ep-4)}-{num_ep})',
                linewidth=2, marker='o')

        ax.set_xlabel('Time Step t', fontsize=12)
        ax.set_ylabel('η(t) — Unit Slack', fontsize=12)
        ax.set_title(f'η(t) Evolution — {mode.upper()}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'eta_time_evolution.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved eta_time_evolution.png")


# ======================================================================
# 4. Training curves
# ======================================================================

def plot_training_curves(data: Dict, output_dir: Path):
    """Plot loss and Chamfer Distance over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total loss
    ax1 = axes[0]
    for mode, color in [('slack', 'blue'), ('tight', 'red')]:
        if mode not in data:
            continue
        loss = data[mode].get('loss', [])
        if loss:
            ax1.plot(range(1, len(loss) + 1), loss, color=color,
                     label=mode.upper(), linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Validation Chamfer Distance
    ax2 = axes[1]
    for mode, color in [('slack', 'blue'), ('tight', 'red')]:
        if mode not in data:
            continue
        cd = data[mode].get('val_chamfer', [])
        if cd:
            ax2.plot(range(1, len(cd) + 1), cd, color=color,
                     label=mode.upper(), linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Chamfer Distance', fontsize=12)
    ax2.set_title('Validation Chamfer Distance', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved training_curves.png")


# ======================================================================
# 5. Comprehensive 8-panel figure
# ======================================================================

def plot_comprehensive_analysis(data: Dict, output_dir: Path):
    """8-panel figure summarising the experiment."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    for col, mode in enumerate(['slack', 'tight']):
        if mode not in data:
            continue
        m = data[mode]

        # Row 0, Col 0/1: ‖Δx(t)‖ at final epoch
        ax = axes[0, col]
        disp = m.get('disp_mag_by_step', [])
        if disp:
            T = max(int(k) for k in disp[-1].keys()) + 1
            late = _extract_step_trajectory(
                disp, range(max(0, len(disp) - 5), len(disp)))
            ax.plot(range(T), late.mean(axis=0), 'b-o', linewidth=2)
        ax.set_title(f'‖Δx(t)‖ — {mode.upper()}')
        ax.set_xlabel('Step')
        ax.set_ylabel('‖Δx‖')
        ax.grid(True, alpha=0.3)

        # Row 0, Col 2/3: CD(t) at final epoch
        ax = axes[0, col + 2]
        cd = m.get('cd_by_step', [])
        if cd:
            T = max(int(k) for k in cd[-1].keys()) + 1
            late = _extract_step_trajectory(
                cd, range(max(0, len(cd) - 5), len(cd)))
            ax.plot(range(T), late.mean(axis=0), 'g-o', linewidth=2)
        ax.set_title(f'CD(t) — {mode.upper()}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Chamfer Dist')
        ax.grid(True, alpha=0.3)

        # Row 1, Col 0/1: η(t) at final epoch
        ax = axes[1, col]
        eta = m.get('eta_by_step', [])
        if eta:
            T = max(int(k) for k in eta[-1].keys()) + 1
            late = _extract_step_trajectory(
                eta, range(max(0, len(eta) - 5), len(eta)))
            ax.plot(range(T), late.mean(axis=0), 'r-o', linewidth=2)
        ax.set_title(f'η(t) — {mode.upper()}')
        ax.set_xlabel('Step')
        ax.set_ylabel('η')
        ax.grid(True, alpha=0.3)

        # Row 1, Col 2/3: Loss over epochs
        ax = axes[1, col + 2]
        loss = m.get('loss', [])
        if loss:
            ax.plot(range(1, len(loss) + 1), loss, 'purple', linewidth=2)
        ax.set_title(f'Loss — {mode.upper()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Temporal Suspension Experiment — Active Assembly Analysis',
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
    lines.append("Temporal Suspension Experiment — Active Assembly Report")
    lines.append("=" * 80)
    lines.append("")

    summary = data.get('summary', {})

    for mode in ['slack', 'tight']:
        if mode not in data:
            continue
        m = data[mode]
        s = summary.get(mode, {})

        lines.append(f"--- {mode.upper()} Model ---")
        lines.append("")

        # Final Chamfer Distance
        val_cd = m.get('val_chamfer', [])
        if val_cd:
            lines.append(f"  Final Validation CD:  {val_cd[-1]:.4f}")

        # Displacement trajectory at final epoch
        disp = m.get('disp_mag_by_step', [])
        if disp:
            last = disp[-1]
            T = max(int(k) for k in last.keys()) + 1
            lines.append(f"  ‖Δx(t)‖ at final epoch:")
            for t in range(T):
                val = last.get(str(t), 0.0)
                lines.append(f"    Step {t}: ‖Δx‖ = {val:.4f}")

        # η trajectory at final epoch
        eta = m.get('eta_by_step', [])
        if eta:
            last = eta[-1]
            T = max(int(k) for k in last.keys()) + 1
            lines.append(f"  η(t) at final epoch:")
            for t in range(T):
                val = last.get(str(t), 0.0)
                lines.append(f"    Step {t}: η = {val:.4f}")

        # CD trajectory at final epoch
        cd = m.get('cd_by_step', [])
        if cd:
            last = cd[-1]
            T = max(int(k) for k in last.keys()) + 1
            lines.append(f"  CD(t) at final epoch:")
            for t in range(T):
                val = last.get(str(t), 0.0)
                lines.append(f"    Step {t}: CD = {val:.4f}")

        lines.append("")

    # Comparison
    if 'slack' in summary and 'tight' in summary:
        lines.append("--- COMPARISON ---")
        lines.append("")
        s_cd = summary['slack'].get('final_chamfer', 0)
        t_cd = summary['tight'].get('final_chamfer', 0)
        lines.append(
            f"  Final CD:  Slack={s_cd:.4f}  Tight={t_cd:.4f}  "
            f"Diff={s_cd - t_cd:+.4f}")

        # Displacement pattern comparison
        s_disp = summary['slack'].get('final_disp_mag_by_step', {})
        t_disp = summary['tight'].get('final_disp_mag_by_step', {})
        if s_disp and t_disp:
            T = max(int(k) for k in s_disp.keys()) + 1
            s_early = np.mean([s_disp.get(str(t), 0) for t in range(T // 2)])
            s_late = np.mean([s_disp.get(str(t), 0)
                              for t in range(T // 2, T)])
            t_early = np.mean([t_disp.get(str(t), 0) for t in range(T // 2)])
            t_late = np.mean([t_disp.get(str(t), 0)
                              for t in range(T // 2, T)])
            lines.append(
                f"  ‖Δx‖ early half:  Slack={s_early:.4f}  "
                f"Tight={t_early:.4f}")
            lines.append(
                f"  ‖Δx‖ late half:   Slack={s_late:.4f}  "
                f"Tight={t_late:.4f}")
            ratio_s = s_late / (s_early + 1e-8)
            ratio_t = t_late / (t_early + 1e-8)
            lines.append(
                f"  Late/Early ratio:  Slack={ratio_s:.2f}  "
                f"Tight={ratio_t:.2f}")

        lines.append("")
        lines.append("--- INTERPRETATION ---")
        lines.append("")

        if s_cd < t_cd:
            lines.append(
                "  The SLACK model achieves lower final Chamfer Distance,")
            lines.append(
                "  suggesting that preserved slack enables better assembly.")
        else:
            lines.append(
                "  The TIGHT model achieves lower final Chamfer Distance.")
            lines.append(
                "  This may indicate the task is too simple for slack to")
            lines.append(
                "  provide an advantage, or that more epochs are needed.")

        if s_disp and t_disp:
            if ratio_s > ratio_t:
                lines.append(
                    "  The SLACK model shows a stronger small→large "
                    "displacement pattern,")
                lines.append(
                    "  consistent with temporal suspension: cautious "
                    "exploration followed")
                lines.append(
                    "  by decisive commitment.")
            else:
                lines.append(
                    "  Both models show similar displacement patterns.")
                lines.append(
                    "  The ambiguity schedule may need to be more "
                    "challenging.")

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
        m = data[mode]
        entry = {}

        for key in ['loss', 'chamfer', 'val_chamfer', 'aff', 'val_aff']:
            vals = m.get(key, [])
            if vals:
                entry[f'{key}_final'] = vals[-1]
                entry[f'{key}_mean'] = float(np.mean(vals))

        # Final-epoch per-step trajectories
        for key in ['eta_by_step', 'disp_mag_by_step', 'cd_by_step']:
            steps = m.get(key, [])
            if steps:
                entry[f'{key}_final'] = steps[-1]

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

    plot_displacement_dynamics(data, output_dir)
    plot_chamfer_distance(data, output_dir)
    plot_eta_time_evolution(data, output_dir)
    plot_training_curves(data, output_dir)
    plot_comprehensive_analysis(data, output_dir)
    generate_report(data, output_dir)
    save_analysis_json(data, output_dir)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
