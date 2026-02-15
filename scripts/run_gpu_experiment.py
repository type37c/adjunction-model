#!/usr/bin/env python3
"""
GPU Experiment Runner for Purpose-Emergent Active Assembly

This script provides a convenient interface for running the Purpose-Emergent
experiment on GPU with various configurations.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from experiments.purpose_emergent_experiment import run_purpose_emergent_experiment


def main():
    parser = argparse.ArgumentParser(
        description='Run Purpose-Emergent Active Assembly experiment on GPU'
    )
    
    # Experiment parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples in dataset (default: 1000)')
    parser.add_argument('--points', type=int, default=256,
                        help='Number of points per cloud (default: 256)')
    parser.add_argument('--steps', type=int, default=8,
                        help='Number of time steps (default: 8)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use if multiple GPUs available (default: 0)')
    
    # Experiment presets
    parser.add_argument('--preset', type=str, default=None,
                        choices=['test', 'standard', 'paper'],
                        help='Use predefined parameter preset')
    
    args = parser.parse_args()
    
    # Apply preset if specified
    if args.preset == 'test':
        print("Using 'test' preset: quick validation run")
        args.epochs = 3
        args.samples = 100
        args.batch_size = 2
    elif args.preset == 'standard':
        print("Using 'standard' preset: default full experiment")
        args.epochs = 50
        args.samples = 1000
        args.batch_size = 8
    elif args.preset == 'paper':
        print("Using 'paper' preset: high-quality results for publication")
        args.epochs = 100
        args.samples = 2000
        args.batch_size = 16
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = f'cuda:{args.gpu_id}'
            print(f"Auto-detected GPU: {torch.cuda.get_device_name(args.gpu_id)}")
        else:
            device = 'cpu'
            print("No GPU detected, using CPU")
    else:
        device = args.device if args.device == 'cpu' else f'cuda:{args.gpu_id}'
    
    # Print configuration
    print("\n" + "="*60)
    print("Purpose-Emergent Active Assembly Experiment")
    print("="*60)
    print(f"Epochs:       {args.epochs}")
    print(f"Samples:      {args.samples}")
    print(f"Points:       {args.points}")
    print(f"Steps:        {args.steps}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device:       {device}")
    print("="*60 + "\n")
    
    # Confirm if running expensive experiment
    if args.epochs >= 50 and args.samples >= 1000:
        print("WARNING: This is a large-scale experiment.")
        if device.startswith('cuda'):
            gpu_name = torch.cuda.get_device_name(args.gpu_id)
            print(f"Estimated runtime on {gpu_name}: 8-16 hours")
            if 'A100' in gpu_name:
                print("Estimated cost on CoreWeave: $160-320")
            elif 'H100' in gpu_name:
                print("Estimated cost on CoreWeave: $400-800")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Experiment cancelled.")
            return
    
    # Run experiment
    try:
        run_purpose_emergent_experiment(
            num_epochs=args.epochs,
            num_samples=args.samples,
            num_points=args.points,
            num_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device
        )
        print("\n" + "="*60)
        print("Experiment completed successfully!")
        print("Results saved to: results/purpose_emergent/")
        print("="*60)
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print("Partial results may be saved in: results/purpose_emergent/")
    except Exception as e:
        print(f"\n\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
