#SBATCH --output=/gpfs/data/shenlab/aj4718/ijepa/logs/slurm-%j.out
#SBATCH --error=/gpfs/data/shenlab/aj4718/ijepa/logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=320G
#SBATCH --time=480:00:00
#SBATCH --partition=a100_long

echo "========================================" 
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "Model: Scale-Conditioned I-JEPA (ViT-Small)"
echo "========================================"

# Navigate to project directory
cd /gpfs/data/shenlab/aj4718/ijepa

# Load CUDA module to fix cuDNN library issues
module load cuda/11.8

# Activate conda environment
source /gpfs/data/shenlab/aj4718/miniconda3/etc/profile.d/conda.sh
conda activate ijepa
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/data/shenlab/aj4718/miniconda3/envs/ijepa/lib

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"
echo "========================================" 

# Run the training script
# Using main_scale.py which calls train_scale.py
CONFIG="configs/scale_ijepa_vits8.yaml"

# Construct device list dynamically based on available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
DEVICES=""
for ((i=0;i<NUM_GPUS;i++)); do
    DEVICES="$DEVICES cuda:$i"
done

echo "Detected $NUM_GPUS GPUs: $DEVICES"

python main_scale.py \
    --fname $CONFIG \
    --devices $DEVICES

echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
