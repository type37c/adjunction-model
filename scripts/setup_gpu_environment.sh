#!/bin/bash
# GPU Environment Setup Script for Purpose-Emergent Experiment
# This script prepares a fresh GPU environment (CoreWeave or similar) for running the experiment

set -e

echo "=== Adjunction Model: GPU Environment Setup ==="
echo ""

# 1. System information
echo "[1/6] Checking system information..."
echo "  OS: $(uname -s)"
echo "  Kernel: $(uname -r)"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  CUDA: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}')"
else
    echo "  WARNING: nvidia-smi not found. GPU may not be available."
fi
echo ""

# 2. Install system dependencies
echo "[2/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    build-essential
echo "  ✓ System dependencies installed"
echo ""

# 3. Install Python dependencies
echo "[3/6] Installing Python dependencies..."
pip3 install --upgrade pip -q
pip3 install -q \
    torch==2.1.0 \
    torch-geometric==2.4.0 \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    numpy==1.24.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    tqdm==4.66.1
echo "  ✓ Python dependencies installed"
echo ""

# 4. Verify PyTorch GPU support
echo "[4/6] Verifying PyTorch GPU support..."
python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA device count: {torch.cuda.device_count()}'); print(f'  Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}'); print(f'  Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# 5. Clone repository (if not already present)
echo "[5/6] Checking repository..."
if [ ! -d "/workspace/adjunction-model" ]; then
    echo "  Cloning repository..."
    cd /workspace
    git clone https://github.com/type37c/adjunction-model.git
    cd adjunction-model
    echo "  ✓ Repository cloned"
else
    echo "  ✓ Repository already exists at /workspace/adjunction-model"
    cd /workspace/adjunction-model
    git pull origin master
    echo "  ✓ Repository updated"
fi
echo ""

# 6. Verify experiment script
echo "[6/6] Verifying experiment script..."
if [ -f "experiments/purpose_emergent_experiment.py" ]; then
    echo "  ✓ Experiment script found"
    python3 -c "from src.models.adjunction_model import AdjunctionModel; print('  ✓ AdjunctionModel import successful')"
else
    echo "  ✗ Experiment script not found!"
    exit 1
fi
echo ""

echo "=== Setup Complete ==="
echo ""
echo "To run the experiment:"
echo "  cd /workspace/adjunction-model"
echo "  python3 experiments/purpose_emergent_experiment.py"
echo ""
echo "To run with custom parameters:"
echo "  python3 -c \"from experiments.purpose_emergent_experiment import run_purpose_emergent_experiment; run_purpose_emergent_experiment(num_epochs=50, batch_size=8, device='cuda')\""
echo ""
