# Quickstart for Developers

This guide provides a streamlined process for setting up the development environment and running the core components of the Physical-Semantic Adjunction Model. It is intended for AI agents or human developers taking over the project.

## 1. Environment Setup

The project is developed in a standard Ubuntu 22.04 environment with Python 3.11.

### 1.1. Clone the Repository

First, ensure you have cloned the repository:

```bash
gh repo clone type37c/adjunction-model
cd adjunction-model
```

### 1.2. Install Dependencies

Dependencies are managed via `uv` and listed in `requirements.txt`. PyTorch and PyTorch Geometric require a specific installation process.

**Step 1: Install PyTorch**

```bash
sudo uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Step 2: Install PyTorch Geometric**

*Note: The standard `pip install torch-geometric` may fail. The recommended method is to install from the official pre-built wheels corresponding to the PyTorch version.* As of this writing, we use PyTorch 2.10.0+cpu.

```bash
# This command might fail due to URL changes. Refer to PyG documentation if needed.
sudo uv pip install --system torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
```

**Step 3: Install Remaining Dependencies**

```bash
sudo uv pip install --system -r requirements.txt
```

## 2. Running Key Components

All key components include a test routine within their respective files. This allows for quick, isolated verification.

### 2.1. Test Synthetic Data Generation

This script generates synthetic point clouds (cubes, cylinders, spheres) and ensures the data loader provides batches of the correct shape.

```bash
python3.11 -m src.data.synthetic_dataset
```

**Expected Output**: A series of messages indicating the shape of generated point clouds, followed by a success message.

### 2.2. Test Functor F (Shape → Action)

This tests the GNN-based affordance prediction model.

```bash
python3.11 -m src.models.functor_f
```

**Expected Output**: A summary of the model architecture and output tensor shapes, ending with "Functor F test passed!".

### 2.3. Test Functor G (Action → Shape)

This tests the conditional GNN decoder for shape reconstruction.

```bash
python3.11 -m src.models.functor_g
```

**Expected Output**: A summary of the model architecture and output tensor shapes, ending with "Functor G test passed!".

### 2.4. Test the Full Adjunction Model

This integrates F and G and computes the `coherence_signal`.

```bash
python3.11 -m src.models.adjunction
```

**Expected Output**: Model summary, intermediate tensor shapes, a calculated coherence signal value, and "Adjunction Model test passed!".

## 3. Running the Core Experiment

The primary experiment validates the core hypothesis that the coherence signal increases for novel shapes.

### 3.1. Execute the Experiment Script

This script will:
1.  Train a model for a few epochs on known shapes.
2.  Calculate the average coherence signal for known shapes.
3.  Calculate the average coherence signal for a novel shape (torus).
4.  Compare the results and generate a visualization.

```bash
python3.11 experiments/test_coherence_signal.py
```

### 3.2. Check the Results

-   **Log File**: The full console output is saved to `logs/experiment_output.log`.
-   **Visualization**: A plot comparing the coherence signal distributions is saved to `logs/coherence_test/coherence_comparison.png`.
-   **Summary**: A detailed analysis and interpretation of the results is in `logs/coherence_test/results_summary.md`.

After running the experiment, the key output to check for is the final conclusion:

```
✓ HYPOTHESIS CONFIRMED
  Novel shapes show XXX.X% higher coherence signal.
```

This confirms the MVP of the theory is working as expected.
