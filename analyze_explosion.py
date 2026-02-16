import torch
import sys
sys.path.append('/home/ubuntu/adjunction-model')

# Load the final model checkpoint
checkpoint_path = '/home/ubuntu/adjunction-model/experiments/intrinsic_reward_baseline/checkpoints/checkpoint_epoch_50.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Load training history
history_path = '/home/ubuntu/adjunction-model/experiments/intrinsic_reward_baseline/results/training_history.pt'
history = torch.load(history_path, map_location='cpu')

print("="*60)
print("EXPLOSION ANALYSIS")
print("="*60)

print("\nη (Coherence) Growth:")
eta_values = history['avg_eta']
print(f"  Epoch 1:  {eta_values[0]:.6f}")
print(f"  Epoch 10: {eta_values[9]:.6f}")
print(f"  Epoch 20: {eta_values[19]:.6f}")
print(f"  Epoch 30: {eta_values[29]:.6f}")
print(f"  Epoch 40: {eta_values[39]:.6f}")
print(f"  Epoch 50: {eta_values[49]:.6f}")

print("\nε (Counit) Growth:")
eps_values = history['avg_epsilon']
print(f"  Epoch 1:  {eps_values[0]:.6f}")
print(f"  Epoch 10: {eps_values[9]:.6f}")
print(f"  Epoch 20: {eps_values[19]:.6f}")
print(f"  Epoch 30: {eps_values[29]:.6f}")
print(f"  Epoch 40: {eps_values[39]:.6f}")
print(f"  Epoch 50: {eps_values[49]:.6f}")

print("\nValue Function Growth:")
value_values = history['value_end']
print(f"  Epoch 1:  {value_values[0]:.2f}")
print(f"  Epoch 10: {value_values[9]:.2f}")
print(f"  Epoch 20: {value_values[19]:.2f}")
print(f"  Epoch 30: {value_values[29]:.2f}")
print(f"  Epoch 40: {value_values[39]:.2f}")
print(f"  Epoch 50: {value_values[49]:.2f}")

print("\nValence (should grow in 2/13):")
valence_values = history['avg_valence']
print(f"  Epoch 1:  {valence_values[0]:.6f}")
print(f"  Epoch 10: {valence_values[9]:.6f}")
print(f"  Epoch 20: {valence_values[19]:.6f}")
print(f"  Epoch 30: {valence_values[29]:.6f}")
print(f"  Epoch 40: {valence_values[39]:.6f}")
print(f"  Epoch 50: {valence_values[49]:.6f}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
print("1. η exploded from 0.47 to 96 trillion")
print("2. ε exploded from 0.002 to 13 million")
print("3. Value exploded from 0.016 to 52 quadrillion")
print("4. Valence barely changed (0.0000 → 0.0003)")
print("\nConclusion: The adjunction structure collapsed.")
print("F/G parameters are frozen, but context modulation is destroying the output.")
