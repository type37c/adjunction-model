# Debugging and Testing Guide

This document provides a checklist of common bugs encountered during development and a systematic guide to testing. It is designed to help future agents quickly diagnose and resolve issues.

## 1. Common Errors and Solutions

This project, like any involving deep learning and complex data structures, has common failure points. Here are the ones encountered and solved during the MVP development.

### Error Type: Environment & Installation

| Symptom | Root Cause | Solution |
|:---|:---|:---|
| `pip install torch-geometric` fails with compilation errors. | PyTorch Geometric (PyG) has complex dependencies that need to be compiled against a specific PyTorch version. | **Do not use the standard pip install.** Install from pre-built wheels matching the installed PyTorch version. Refer to `docs/QUICKSTART.md` for the exact command using the `-f` flag. |
| `ModuleNotFoundError: No module named 'torch_cluster'` (or similar PyG modules). | Same as above. The core PyG library was installed, but its dependencies were not. | Re-run the installation command from the PyG pre-built wheels to install all required packages (`torch-scatter`, `torch-sparse`, etc.). |

### Error Type: Python Imports

| Symptom | Root Cause | Solution |
|:---|:---|:---|
| `ImportError: attempted relative import with no known parent package` | A script inside the `src` directory (e.g., `src/models/adjunction.py`) was run directly (`python3.11 src/models/adjunction.py`). Python doesn't recognize `src` as a package in this case. | **Run scripts as modules from the root directory.** This allows Python to correctly resolve relative imports. Use the `-m` flag. <br><br> **Correct:** `python3.11 -m src.models.adjunction` <br> **Incorrect:** `python3.11 src/models/adjunction.py` |

### Error Type: Tensor Shape Mismatches (`RuntimeError`)

This is the most common category of bug in this project.

| Symptom | Root Cause | Solution |
|:---|:---|:---|
| `RuntimeError: stack expects each tensor to be equal size, but got [512, 3] at entry 0 and [510, 3] at entry 1` | The `DataLoader` tried to stack samples with slightly different numbers of points into a single batch. This was traced back to the `synthetic_dataset.py` generator, where different shape generation algorithms produced slightly different point counts. | **Enforce a fixed size for all generated point clouds.** In `synthetic_dataset.py`, logic was added to every shape generation function to either truncate or pad the point cloud to ensure it has exactly `self.num_points`. |
| `RuntimeError: Tensors must have same number of dimensions: got 4 and 2` in `chamfer_distance` | The Chamfer distance function was called with a batched tensor (e.g., `[B, N, 3]`) and an unbatched tensor (e.g., `[N, 3]`), causing an error during padding/stacking. | **Standardize tensor dimensions before computation.** The `chamfer_distance` function was updated to be more robust. It now checks the dimensions of its inputs and explicitly batches any unbatched tensors before proceeding with the distance calculation. |

## 2. Systematic Testing Workflow

To avoid and quickly identify bugs, follow this bottom-up testing workflow. All components are designed to be testable in isolation.

### Step 1: Verify the Data Source

**Goal**: Ensure the data being fed to the models is correct and consistent.

```bash
# Run the data loader test
python3.11 -m src.data.synthetic_dataset
```

**What to check for**: 
- Does it run without errors?
- Does the output confirm that all generated point clouds have the *exact same* number of points?

### Step 2: Test Model Components Individually

**Goal**: Ensure each neural network module (F and G) can perform a forward pass without errors.

```bash
# Test Functor F
python3.11 -m src.models.functor_f

# Test Functor G
python3.11 -m src.models.functor_g
```

**What to check for**: 
- Do both scripts run to completion and print the "test passed" message?
- Are the output tensor shapes consistent with the model definitions?

### Step 3: Test the Integrated Model

**Goal**: Ensure F and G can be combined and the coherence signal can be computed.

```bash
# Test the full AdjunctionModel
python3.11 -m src.models.adjunction
```

**What to check for**: 
- Does it successfully compute a `coherence_signal`?
- This is the most likely place for tensor shape mismatches between F and G to appear.

### Step 4: Test the Training Loop

**Goal**: Ensure the model can train for at least one step without crashing.

```bash
# Test the Phase 1 trainer
python3.11 -m src.training.train_phase1
```

**What to check for**: 
- Does it complete at least one epoch?
- Does the loss decrease (even slightly)?
- This test will catch errors in the loss function computation or the `DataLoader` collation.

### The Golden Rule of Debugging

When a `RuntimeError` occurs, the most effective debugging tool is to **print the shape of your tensors**. 

```python
# Example
print(f"pc1 shape: {pc1.shape}")
print(f"pc2 shape: {pc2.shape}")
distance = self.chamfer_distance(pc1, pc2)
```

90% of the bugs in this MVP were resolved by identifying where tensor shapes became inconsistent.
