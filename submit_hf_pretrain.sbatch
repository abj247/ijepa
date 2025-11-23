#!/bin/bash
#SBATCH --job-name=ijepa-hf-pretrain
#SBATCH --output=/gpfs/data/shenlab/aj4718/ijepa/logs/slurm-%j.out
#SBATCH --error=/gpfs/data/shenlab/aj4718/ijepa/logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Set which model to use: vit_small or vit_base
MODEL=${1:-vit_small}

echo "========================================" 
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "Model: $MODEL"
echo "========================================"

# Navigate to project directory
cd /gpfs/data/shenlab/aj4718/ijepa

# Activate conda environment
source /gpfs/data/shenlab/aj4718/miniconda3/etc/profile.d/conda.sh
conda activate ijepa

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"
echo "========================================" 

# Run the training script
bash run_hf_pretrain.sh $MODEL

echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
