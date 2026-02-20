"""
Analyze results from Step 1 eta validation experiment.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data():
    """Load all CSV files."""
    data_dir = Path('/home/ubuntu/adjunction-model/experiments/step1_eta_validation/data')
    
    dfs = []
    for object_type in ['box', 'cup', 'bowl']:
        csv_files = list((data_dir / object_type).glob('*.csv'))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No data files found")
    
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_all)} data points from {len(dfs)} objects")
    return df_all


def compute_statistics(df):
    """Compute statistics by action type."""
    stats = df.groupby(['object', 'action'])['eta'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    print("\n" + "="*60)
    print("Statistics by Object and Action")
    print("="*60)
    print(stats.to_string(index=False))
    print()
    
    # Overall statistics
    print("="*60)
    print("Overall Statistics")
    print("="*60)
    print(f"Overall mean eta: {df['eta'].mean():.6f}")
    print(f"Overall std eta: {df['eta'].std():.6f}")
    print(f"Overall min eta: {df['eta'].min():.6f}")
    print(f"Overall max eta: {df['eta'].max():.6f}")
    print(f"Coefficient of variation: {df['eta'].std() / df['eta'].mean():.6f}")
    print()
    
    return stats


def plot_eta_by_action(df, output_dir):
    """Plot eta distribution by action type."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, object_type in enumerate(['box', 'cup', 'bowl']):
        df_obj = df[df['object'] == object_type]
        
        # Box plot
        ax = axes[idx]
        df_obj.boxplot(column='eta', by='action', ax=ax)
        ax.set_title(f'Eta by Action: {object_type.capitalize()}')
        ax.set_xlabel('Action')
        ax.set_ylabel('Eta (Reconstruction Error)')
        ax.get_figure().suptitle('')  # Remove default title
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = output_dir / 'eta_by_action_boxplot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_eta_timeseries(df, output_dir):
    """Plot eta time series for each action."""
    actions = df['action'].unique()
    
    fig, axes = plt.subplots(len(actions), 3, figsize=(18, 4*len(actions)))
    
    for action_idx, action in enumerate(sorted(actions)):
        for obj_idx, object_type in enumerate(['box', 'cup', 'bowl']):
            df_subset = df[(df['object'] == object_type) & (df['action'] == action)]
            
            if len(df_subset) == 0:
                continue
            
            ax = axes[action_idx, obj_idx] if len(actions) > 1 else axes[obj_idx]
            
            # Plot each trial
            for trial in df_subset['trial'].unique():
                df_trial = df_subset[df_subset['trial'] == trial]
                ax.plot(df_trial['frame'], df_trial['eta'], alpha=0.5, label=f'Trial {trial}')
            
            ax.set_title(f'{object_type.capitalize()} - {action}')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Eta')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'eta_timeseries.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_eta_vs_position(df, output_dir):
    """Plot eta vs object position."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, coord in enumerate(['pos_x', 'pos_y', 'pos_z']):
        ax = axes[idx]
        
        for object_type in ['box', 'cup', 'bowl']:
            df_obj = df[df['object'] == object_type]
            ax.scatter(df_obj[coord], df_obj['eta'], alpha=0.3, label=object_type, s=10)
        
        ax.set_xlabel(coord.replace('_', ' ').capitalize())
        ax.set_ylabel('Eta')
        ax.set_title(f'Eta vs {coord.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'eta_vs_position.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_variance(df):
    """Analyze variance in eta."""
    print("="*60)
    print("Variance Analysis")
    print("="*60)
    
    # Variance by object
    print("\nVariance by Object:")
    for object_type in ['box', 'cup', 'bowl']:
        df_obj = df[df['object'] == object_type]
        print(f"  {object_type}: mean={df_obj['eta'].mean():.6f}, std={df_obj['eta'].std():.6f}")
    
    # Variance by action
    print("\nVariance by Action:")
    for action in sorted(df['action'].unique()):
        df_action = df[df['action'] == action]
        print(f"  {action}: mean={df_action['eta'].mean():.6f}, std={df_action['eta'].std():.6f}")
    
    # Check if eta is constant
    unique_etas = df['eta'].nunique()
    print(f"\nNumber of unique eta values: {unique_etas}")
    
    if unique_etas <= 10:
        print("Unique eta values:")
        for val in sorted(df['eta'].unique()):
            count = (df['eta'] == val).sum()
            print(f"  {val:.6f}: {count} occurrences ({100*count/len(df):.1f}%)")
    
    print()


def main():
    """Main analysis function."""
    print("="*60)
    print("Step 1: Eta Validation - Results Analysis")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Create output directory
    output_dir = Path('/home/ubuntu/adjunction-model/experiments/step1_eta_validation/results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute statistics
    stats = compute_statistics(df)
    
    # Analyze variance
    analyze_variance(df)
    
    # Create plots
    print("="*60)
    print("Generating Plots")
    print("="*60)
    plot_eta_by_action(df, output_dir)
    plot_eta_timeseries(df, output_dir)
    plot_eta_vs_position(df, output_dir)
    
    # Save statistics to CSV
    stats_path = output_dir.parent / 'statistics.csv'
    stats.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")
    
    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)


if __name__ == '__main__':
    main()
