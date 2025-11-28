
import torch
from datasets import load_dataset
import numpy as np
from PIL import Image

def inspect_dataset():
    dataset_name = "tsbpp/fall2025_deeplearning"
    print(f"Loading dataset: {dataset_name}")
    
    try:
        # Load the dataset (assuming 'train' split exists)
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        print("Dataset loaded successfully (streaming mode).")
        print("Inspecting first 10 samples...")
        
        sizes = []
        modes = []
        
        for i, sample in enumerate(dataset):
            if i >= 10:
                break
            
            # Assuming the image column is named 'image' or similar
            # Let's check keys first
            if i == 0:
                print(f"Sample keys: {sample.keys()}")
            
            if 'image' in sample:
                img = sample['image']
            elif 'img' in sample:
                img = sample['img']
            else:
                # Try to find image column
                for k, v in sample.items():
                    if isinstance(v, Image.Image):
                        img = v
                        break
                else:
                    print(f"Could not find image in sample {i}")
                    continue
            
            print(f"Sample {i}: Size={img.size}, Mode={img.mode}")
            sizes.append(img.size)
            modes.append(img.mode)
            
        # Analyze sizes
        sizes = np.array(sizes)
        print("\nSummary of first 10 samples:")
        print(f"Unique sizes: {np.unique(sizes, axis=0)}")
        print(f"Unique modes: {np.unique(modes)}")
        
    except Exception as e:
        print(f"Error loading or inspecting dataset: {e}")

if __name__ == "__main__":
    inspect_dataset()
