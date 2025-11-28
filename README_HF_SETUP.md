# I-JEPA HuggingFace Dataset Pretraining - Setup Complete!

## 🎯 Overview

I-JEPA has been successfully configured for pretraining on the HuggingFace dataset `tsbpp/fall2025_deeplearning` using a single A100 GPU (64GB).

## 📁 Files Created

### Dataset Loader
- **`src/datasets/huggingface_dataset.py`**: Custom HuggingFace dataset loader for I-JEPA

### Configuration Files
- **`configs/hf_vits16_ep300.yaml`**: ViT-Small config (~27M params, batch size 64) **[RECOMMENDED]**
- **`configs/hf_vitb16_ep300.yaml`**: ViT-Base config (~96M params, batch size 32)

### Launch Scripts
- **`run_hf_pretrain.sh`**: Bash script for local/interactive training
- **`submit_hf_pretrain.sbatch`**: SLURM batch script for cluster submission

### Verification Scripts
- **`test_hf_dataset.py`**: Test HuggingFace dataset loading
- **`check_model_size.py`**: Verify model parameter counts

### Modified Files
- **`src/train.py`**: Updated to support both ImageNet and HuggingFace datasets via config flag

---

## 🚀 Quick Start

### 1. Install Dependencies (IMPORTANT!)

First, you need to install PyTorch and other requirements in your `ijepa` conda environment:

```bash
conda activate ijepa
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml numpy opencv-python submitit datasets
```

### 2. Run Verification Tests

Test that everything works:

```bash
cd /gpfs/data/shenlab/aj4718/ijepa

# Check model sizes
python check_model_size.py --model vit_small
python check_model_size.py --model vit_base

# Test dataset loading
python test_hf_dataset.py
```

### 3. Launch Training

**Option A: Interactive Training (single GPU)**
```bash
# ViT-Small (recommended)
bash run_hf_pretrain.sh vit_small

# Or ViT-Base
bash run_hf_pretrain.sh vit_base
```

**Option B: SLURM Cluster Submit**
```bash
# ViT-Small (recommended)
sbatch submit_hf_pretrain.sbatch vit_small

# Or ViT-Base
sbatch submit_hf_pretrain.sbatch vit_base
```

---

## 📊 Model Specifications

### ViT-Small (Recommended)
- **Encoder**: 384 dim, 12 layers, 6 heads → ~22M params
- **Predictor**: 192 dim, 6 layers → ~5M params  
- **Total**: ~27M parameters
- **Batch Size**: 64
- **Learning Rate**: 0.0005 (scaled for batch size)
- **Memory**: ~20-25GB GPU

### ViT-Base
- **Encoder**: 768 dim, 12 layers, 12 heads → ~86M params
- **Predictor**: 384 dim, 6 layers → ~10M params
- **Total**: ~96M parameters
- **Batch Size**: 32
- **Learning Rate**: 0.00025 (scaled for smaller batch)
- **Memory**: ~40-50GB GPU

Both models are **under the 100M parameter limit**.

---

## ⚙️ Configuration Details

### Data Configuration
```yaml
dataset_type: huggingface  # Key setting!
dataset_name: tsbpp/fall2025_deeplearning
batch_size: 64  # (vit_small) or 32 (vit_base)
crop_size: 224
```

### Training Configuration
- **Epochs**: 300
- **Warmup**: 40 epochs
- **Mixed Precision**: bfloat16 enabled
- **EMA**: [0.996, 1.0]

### Masking Configuration
- **Context masks**: 1 per image (85-100% of image)
- **Target masks**: 4 per image (15-20% each)
- **Patch size**: 16x16
- **No overlap** between context and target

---

## 📝 Expected Training Time

On a single A100 (64GB):
- **Dataset size**: ~500,000 images
- **ViT-Small (batch 64)**: ~7,800 iterations/epoch → ~5-6 hours/epoch → ~75-90 days total
- **ViT-Base (batch 32)**: ~15,600 iterations/epoch → ~10-12 hours/epoch → ~125-150 days total

**Note**: For production training, you may want to reduce epochs or use distributed training across multiple GPUs.

---

##  Monitoring Training

### Logs Location
- **Experiment logs**: `/gpfs/data/shenlab/aj4718/ijepa/logs/hf_vits16_bs64_ep300/` (or `hf_vitb16_bs32_ep300/`)
- **SLURM logs**: `/gpfs/data/shenlab/aj4718/ijepa/logs/slurm-*.out`

### Checkpoints
- **Latest**: `jepa-latest.pth.tar` (saved every epoch)
- **Periodic**: `jepa-ep{epoch}.pth.tar` (saved every 50 epochs)

### Metrics Logged
- Loss (smooth L1 between predictions and targets)
- Learning rate
- Weight decay
- GPU memory usage
- Time per iteration

---

## 🔧 Troubleshooting

### PyTorch Installation Failed
The conda environment might have memory issues. Try installing PyTorch with pip:
```bash
conda activate ijepa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Dataset Column Issues
If the HuggingFace dataset doesn't have an `image` column, modify line 34 in `src/datasets/huggingface_dataset.py`:
```python
image = item['your_column_name_here']
```

### Out of Memory
- Use ViT-Small instead of ViT-Base
- Reduce batch size in config file
- Reduce number of workers

### Import Errors
Make sure you're in the ijepa directory when running scripts:
```bash
cd /gpfs/data/shenlab/aj4718/ijepa
python ...
```

---

## 📚 Next Steps

1. **Install dependencies** in the `ijepa` conda environment
2. **Run verification scripts** to ensure everything works
3. **Start with a short test run**: Modify config to use fewer epochs (e.g., 10) for testing
4. **Monitor the first few epochs** to ensure training is stable
5. **Launch full 300-epoch training** once validated

---

## 🎓 Key Differences from Original I-JEPA

1. **Dataset**: HuggingFace dataset instead of ImageNet
2. **Single GPU**: Optimized for 1x A100 instead of 16x A100
3. **Batch Size**: Reduced to 32-64 instead of 2048
4. **Learning Rates**: Scaled down proportionally
5. **Model Size**: ViT-S/ViT-B instead of ViT-H/ViT-G

---

## 📞 Support

If you encounter issues:
1. Check the logs in` /gpfs/data/shenlab/aj4718/ijepa/logs/`
2. Run verification scripts to isolate the problem
3. Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
