#!/bin/bash
# Launch script for I-JEPA pretraining on HuggingFace dataset
# Single GPU training

# Set which config to use: vit_small (recommended) or vit_base
MODEL=${1:-vit_small}  # Default to vit_small if no argument provided

if [ "$MODEL" == "vit_small" ]; then
    CONFIG="configs/hf_vits16_ep300.yaml"
    echo "Using ViT-Small configuration (~27M parameters)"
elif [ "$MODEL" == "vit_base" ]; then
    CONFIG="configs/hf_vitb16_ep300.yaml"
    echo "Using ViT-Base configuration (~96M parameters)"
else
    echo "Invalid model choice: $MODEL"
    echo "Usage: bash run_hf_pretrain.sh [vit_small|vit_base]"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG"
    exit 1
fi

echo "Starting I-JEPA pretraining with config: $CONFIG"
echo "Using single GPU (cuda:0)"
echo "---"

# Run training on single GPU
python main.py \
    --fname $CONFIG \
    --devices cuda:0

echo "---"
echo "Training complete!"
