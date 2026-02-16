import torch
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.adjunction_model import AdjunctionModel

# Create model
model = AdjunctionModel(
    num_affordances=5,
    num_points=256,
    f_hidden_dim=64,
    g_hidden_dim=128,
    agent_hidden_dim=256,
    agent_latent_dim=64,
    context_dim=128,
    valence_dim=32
)

print("Before freeze_fg():")
print(f"  F parameters requires_grad: {[p.requires_grad for p in model.F.parameters()][:5]}")
print(f"  G parameters requires_grad: {[p.requires_grad for p in model.G.parameters()][:5]}")

# Freeze F/G
model.freeze_fg()

print("\nAfter freeze_fg():")
print(f"  F parameters requires_grad: {[p.requires_grad for p in model.F.parameters()][:5]}")
print(f"  G parameters requires_grad: {[p.requires_grad for p in model.G.parameters()][:5]}")

# Test forward pass and check if F/G parameters change
device = torch.device('cpu')
model = model.to(device)

# Get initial F/G parameter values
f_params_before = [p.clone() for p in model.F.parameters()]
g_params_before = [p.clone() for p in model.G.parameters()]

# Forward pass
pos = torch.randn(256, 3)
batch = torch.zeros(256, dtype=torch.long)
results = model(pos, batch)

# Backward pass (simulate training)
loss = results['coherence_signal'].mean()
loss.backward()

# Check if F/G parameters have gradients
print("\nAfter backward pass:")
f_has_grad = [p.grad is not None for p in model.F.parameters()]
g_has_grad = [p.grad is not None for p in model.G.parameters()]
print(f"  F parameters have gradients: {f_has_grad[:5]}")
print(f"  G parameters have gradients: {g_has_grad[:5]}")

# Simulate optimizer step
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.step()

# Check if F/G parameters changed
f_params_after = [p.clone() for p in model.F.parameters()]
g_params_after = [p.clone() for p in model.G.parameters()]

f_changed = [(before - after).abs().max().item() > 1e-10 for before, after in zip(f_params_before, f_params_after)]
g_changed = [(before - after).abs().max().item() > 1e-10 for before, after in zip(g_params_before, g_params_after)]

print("\nAfter optimizer.step():")
print(f"  F parameters changed: {f_changed[:5]}")
print(f"  G parameters changed: {g_changed[:5]}")

if any(f_changed) or any(g_changed):
    print("\n❌ ERROR: F/G parameters changed despite being frozen!")
else:
    print("\n✓ SUCCESS: F/G parameters remain frozen")
