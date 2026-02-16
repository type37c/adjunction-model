"""
Phase 2.5 Valence Experiment Analysis

This script analyzes and compares the results from the three experimental conditions:
- Condition 1: Baseline (no valence updates)
- Condition 2: Emergent valence (AgentCV3)
- Condition 3: Designed valence (Priority computation)

It generates:
1. Comparative plots of η, ε, and L_aff across conditions
2. Statistical analysis of differences
3. Summary report
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def load_results(condition_dir: Path) -> List[Dict]:
    """Load results from a condition directory."""
    results_file = condition_dir / 'results.json'
    if not results_file.exists():
        print(f"Warning: {results_file} not found")
        return []
    
    with open(results_file, 'r') as f:
        return json.load(f)


def extract_metrics(results: List[Dict], metric_name: str) -> np.ndarray:
    """Extract a specific metric from results."""
    return np.array([r.get(metric_name, np.nan) for r in results])


def plot_comparison(
    results_1: List[Dict],
    results_2: List[Dict],
    results_3: List[Dict],
    output_dir: Path
):
    """Generate comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 2.5 Valence Experiment: Comparison of Conditions', fontsize=16)
    
    # Extract metrics
    epochs = np.arange(len(results_1))
    
    metrics = [
        ('aff_loss', 'Affordance Loss', axes[0, 0]),
        ('unit_mean', 'Unit η (Slack)', axes[0, 1]),
        ('counit_mean', 'Counit ε', axes[1, 0]),
        ('kl_loss', 'KL Divergence', axes[1, 1])
    ]
    
    for metric_name, title, ax in metrics:
        if results_1:
            values_1 = extract_metrics(results_1, metric_name)
            ax.plot(epochs, values_1, label='Condition 1: Baseline', linewidth=2)
        
        if results_2:
            values_2 = extract_metrics(results_2, metric_name)
            ax.plot(epochs, values_2, label='Condition 2: Emergent', linewidth=2)
        
        if results_3:
            values_3 = extract_metrics(results_3, metric_name)
            ax.plot(epochs, values_3, label='Condition 3: Designed', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_plot.png', dpi=150)
    print(f"Saved comparison plot to {output_dir / 'comparison_plot.png'}")


def generate_report(
    results_1: List[Dict],
    results_2: List[Dict],
    results_3: List[Dict],
    output_dir: Path
):
    """Generate a text report summarizing the results."""
    
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("PHASE 2.5 VALENCE EXPERIMENT: ANALYSIS REPORT")
    report_lines.append("="*60)
    report_lines.append("")
    
    # Condition 1
    if results_1:
        report_lines.append("CONDITION 1: Baseline (No Valence Updates)")
        report_lines.append("-" * 60)
        final = results_1[-1]
        report_lines.append(f"  Final L_aff: {final.get('aff_loss', 'N/A'):.4f}")
        report_lines.append(f"  Final η: {final.get('unit_mean', 'N/A'):.4f}")
        report_lines.append(f"  Final ε: {final.get('counit_mean', 'N/A'):.4f}")
        report_lines.append("")
    
    # Condition 2
    if results_2:
        report_lines.append("CONDITION 2: Emergent Valence (AgentCV3)")
        report_lines.append("-" * 60)
        final = results_2[-1]
        report_lines.append(f"  Final L_aff: {final.get('aff_loss', 'N/A'):.4f}")
        report_lines.append(f"  Final η: {final.get('unit_mean', 'N/A'):.4f}")
        report_lines.append(f"  Final ε: {final.get('counit_mean', 'N/A'):.4f}")
        report_lines.append("")
    else:
        report_lines.append("CONDITION 2: Not yet implemented")
        report_lines.append("")
    
    # Condition 3
    if results_3:
        report_lines.append("CONDITION 3: Designed Valence (Priority Computation)")
        report_lines.append("-" * 60)
        final = results_3[-1]
        report_lines.append(f"  Final L_aff: {final.get('aff_loss', 'N/A'):.4f}")
        report_lines.append(f"  Final η: {final.get('unit_mean', 'N/A'):.4f}")
        report_lines.append(f"  Final ε: {final.get('counit_mean', 'N/A'):.4f}")
        report_lines.append("")
    
    # Comparison
    report_lines.append("COMPARISON")
    report_lines.append("-" * 60)
    
    if results_1 and results_3:
        aff_1_final = results_1[-1].get('aff_loss', np.nan)
        aff_3_final = results_3[-1].get('aff_loss', np.nan)
        improvement = ((aff_1_final - aff_3_final) / aff_1_final) * 100
        report_lines.append(f"  L_aff improvement (Condition 3 vs 1): {improvement:.2f}%")
    
    report_lines.append("")
    report_lines.append("="*60)
    
    report_text = "\n".join(report_lines)
    
    # Print to console
    print(report_text)
    
    # Save to file
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\nSaved report to {output_dir / 'analysis_report.txt'}")


def main():
    """Main analysis function."""
    base_dir = Path('/home/ubuntu/adjunction-model/experiments/phase2_valence_experiment')
    
    print("Loading results...")
    results_1 = load_results(base_dir / 'condition_1')
    results_2 = load_results(base_dir / 'condition_2')
    results_3 = load_results(base_dir / 'condition_3')
    
    print(f"Condition 1: {len(results_1)} epochs")
    print(f"Condition 2: {len(results_2)} epochs")
    print(f"Condition 3: {len(results_3)} epochs")
    
    if results_1 or results_2 or results_3:
        print("\nGenerating plots...")
        plot_comparison(results_1, results_2, results_3, base_dir)
        
        print("\nGenerating report...")
        generate_report(results_1, results_2, results_3, base_dir)
    else:
        print("\nNo results found. Please run the experiment first.")


if __name__ == '__main__':
    main()
