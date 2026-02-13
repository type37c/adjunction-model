# Kaggle GPU Experiment Setup

This directory contains files for running Phase 2 Slack experiments on Kaggle's free GPU.

## Quick Start

### Option 1: Direct GitHub Clone (Recommended)

1. Create a new Kaggle Notebook
2. Enable GPU: Settings → Accelerator → **GPU T4 x2**
3. Upload `phase2_slack_gpu_experiment.ipynb`
4. Run all cells

The notebook will automatically clone the repository from GitHub.

### Option 2: Upload as Kaggle Dataset

If you want to use a specific version or have made local changes:

1. **Prepare the code archive**:
   ```bash
   cd /home/ubuntu/adjunction-model
   ./kaggle/prepare_dataset.sh
   ```

2. **Upload to Kaggle**:
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload `kaggle/adjunction-model-code.zip`
   - Title: "Adjunction Model Code"
   - Make it private

3. **Create and run notebook**:
   - Create new notebook with GPU enabled
   - Add your dataset as input
   - Upload `phase2_slack_gpu_experiment.ipynb`
   - Modify the notebook to use dataset path instead of git clone
   - Run all cells

## Experiment Configuration

Default settings in the notebook:
- **Epochs**: 100
- **Shapes**: 100 (larger dataset on GPU)
- **Batch size**: 8 (larger batch on GPU)
- **Device**: CUDA (GPU)

You can modify these in the CONFIG cell.

## Expected Runtime

With GPU T4 x2:
- **100 epochs**: ~30-60 minutes
- **200 epochs**: ~1-2 hours

## Output Files

Results will be saved to `results/phase2_slack_gpu/`:
- `metrics.json`: Training metrics
- `model_final.pt`: Trained model
- `slack_signals.png`: η/ε visualization
- `training_losses.png`: Loss curves

## Downloading Results

After training completes:
1. Click "Output" tab in Kaggle
2. Download all files from `results/phase2_slack_gpu/`
3. Copy to local `results/` directory for analysis

## Troubleshooting

### GPU not available
- Check Settings → Accelerator is set to GPU
- Kaggle provides 30 hours/week of free GPU time

### Out of memory
- Reduce `batch_size` in CONFIG
- Reduce `num_shapes` in CONFIG

### Import errors
- Ensure the repository was cloned successfully
- Check that `sys.path.append()` points to correct directory

## Analysis

After downloading results, run local analysis:
```bash
cd /home/ubuntu/adjunction-model
python3.11 experiments/analyze_phase2_slack.py --results_dir results/phase2_slack_gpu
```

## Tips

- **Monitor progress**: Kaggle shows real-time output
- **Save frequently**: Kaggle auto-saves, but you can manually commit
- **GPU quota**: Check your remaining GPU hours at https://www.kaggle.com/settings
- **Longer experiments**: For 200+ epochs, consider Kaggle's paid tier or Colab Pro
