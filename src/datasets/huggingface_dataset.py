# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import logging

import numpy as np
import torch

from datasets import load_dataset

_GLOBAL_SEED = 0
logger = logging.getLogger()


def make_huggingface(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    dataset_name='tsbpp/fall2025_deeplearning',
    split='train',
    training=True,
    drop_last=True,
):
    """
    Create HuggingFace dataset and dataloader for I-JEPA pretraining.
    
    Args:
        transform: Transform to apply to images
        batch_size: Batch size per GPU
        collator: Collator function for batching
        pin_mem: Whether to pin memory
        num_workers: Number of data loading workers
        world_size: Number of distributed processes
        rank: Rank of current process
        dataset_name: Name of HuggingFace dataset
        split: Dataset split to use
        training: Whether this is training mode
        drop_last: Whether to drop the last incomplete batch
    
    Returns:
        dataset, data_loader, dist_sampler
    """
    dataset = HuggingFaceDataset(
        dataset_name=dataset_name,
        split=split,
        transform=transform,
        training=training
    )
    logger.info(f'HuggingFace dataset created: {dataset_name} ({len(dataset)} samples)')
    
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info(f'HuggingFace data loader created')

    return dataset, data_loader, dist_sampler


class HuggingFaceDataset(torch.utils.data.Dataset):
    """
    HuggingFace Dataset wrapper for I-JEPA pretraining.
    
    Loads images from a HuggingFace dataset and applies transforms.
    Assumes the dataset has an 'image' column containing PIL images.
    """
    
    def __init__(
        self,
        dataset_name='tsbpp/fall2025_deeplearning',
        split='train',
        transform=None,
        training=True,
    ):
        """
        Args:
            dataset_name: Name/path of HuggingFace dataset
            split: Dataset split to use ('train', 'validation', 'test')
            transform: Transform to apply to images
            training: Whether this is for training (affects logging)
        """
        logger.info(f'Loading HuggingFace dataset: {dataset_name}, split: {split}')
        
        # Load the dataset from HuggingFace
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transform
        self.dataset_name = dataset_name
        self.split = split
        
        logger.info(f'Loaded {len(self.dataset)} samples from {dataset_name}')
        
        # Log a sample to verify image column exists
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            logger.info(f'Dataset columns: {list(sample.keys())}')
            if 'image' not in sample:
                logger.warning(
                    f"'image' column not found in dataset. "
                    f"Available columns: {list(sample.keys())}"
                )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Returns:
            Transformed image and dummy target (0)
            For self-supervised learning, we don't use labels
        """
        item = self.dataset[index]
        
        # Get the image - assumes column name is 'image'
        # If your dataset uses a different column name, modify this
        image = item['image']
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Return image with dummy target (not used in I-JEPA pretraining)
        # The target is required for compatibility with the collator
        target = 0
        
        return image, target
