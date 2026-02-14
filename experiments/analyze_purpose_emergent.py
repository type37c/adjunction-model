import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_results():
    base_path = Path("/home/ubuntu/adjunction-model/results/purpose_emergent")
    conditions = ["purpose_emergent", "baseline"]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for cond in conditions:
        metrics_path = base_path / cond / "metrics.json"
        if not metrics_path.exists():
            print(f"Warning: {metrics_path} not found")
            continue
            
        with open(metrics_path, "r") as f:
            data = json.load(f)
            
        # 1. Purpose Loss (Validation)
        axes[0, 0].plot(data["val_purpose"], label=cond)
        axes[0, 0].set_title("Validation Purpose Loss (CD)")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        
        # 2. Eta Trajectory (Final Epoch)
        final_eta = data["val_eta_by_step"][-1]
        steps = sorted([int(k) for k in final_eta.keys()])
        eta_vals = [final_eta[str(s)] for s in steps]
        axes[0, 1].plot(steps, eta_vals, marker='o', label=cond)
        axes[0, 1].set_title("Eta Trajectory (Final Epoch)")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("η")
        axes[0, 1].legend()
        
        # 3. Displacement Magnitude
        final_disp = data["val_disp_mag_by_step"][-1]
        disp_vals = [final_disp[str(s)] for s in steps]
        axes[1, 0].plot(steps, disp_vals, marker='s', label=cond)
        axes[1, 0].set_title("Displacement Magnitude (Final Epoch)")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("||Δx||")
        axes[1, 0].legend()
        
        # 4. Curiosity Reward (if available)
        if "val_curiosity_by_step" in data:
            final_cur = data["val_curiosity_by_step"][-1]
            cur_steps = sorted([int(k) for k in final_cur.keys()])
            cur_vals = [final_cur[str(s)] for s in cur_steps]
            axes[1, 1].plot(cur_steps, cur_vals, marker='^', label=cond)
            axes[1, 1].set_title("Curiosity Reward (Final Epoch)")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("R_curiosity")
            axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(base_path / "comparison_analysis.png")
    print(f"Analysis plot saved to {base_path / 'comparison_analysis.png'}")

    # Generate summary report
    with open(base_path / "analysis_report.md", "w") as f:
        f.write("# Purpose-Emergent Active Assembly Analysis Report\n\n")
        f.write("## Overview\n")
        f.write("This report compares Condition A (Purpose-Emergent) and Condition C (Baseline).\n\n")
        
        for cond in conditions:
            metrics_path = base_path / cond / "metrics.json"
            if not metrics_path.exists(): continue
            with open(metrics_path, "r") as m:
                data = json.load(m)
                f.write(f"### Condition: {cond}\n")
                f.write(f"- Final Val Purpose Loss: {data['val_purpose'][-1]:.4f}\n")
                f.write(f"- Final Step Eta: {data['val_eta_by_step'][-1][str(max(steps))]:.4f}\n\n")

if __name__ == "__main__":
    analyze_results()
