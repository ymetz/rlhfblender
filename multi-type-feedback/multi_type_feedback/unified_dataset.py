from typing import Any, Dict, List

import pytorch_lightning
import torch
from torch.utils.data import DataLoader, Dataset

from multi_type_feedback.feedback_dataset import BufferDataset

# Pairwise feedback types that support ResponseRank loss
_PAIRWISE_TYPES = {"comparative", "demonstrative", "corrective", "descriptive_preference"}


class UnifiedBufferDataset(Dataset):
    """
    Dataset that includes feedback type with the data.
    """

    def __init__(self, feedbacks_by_type: Dict[str, List[Any]]):
        """
        Initialize dataset with feedbacks organized by type.

        Args:
            feedbacks_by_type: Dictionary mapping feedback types to lists of feedback
        """
        self.data = []

        # Flatten all feedbacks, but keep track of their type
        for feedback_type, feedbacks in feedbacks_by_type.items():
            for feedback in feedbacks:
                self.data.append((feedback_type, feedback))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloaders_by_type(
    feedback_buffers: Dict[str, List[Any]], batch_size: int, val_split: float = 0.2
):
    """
    Create separate dataloaders for each feedback type.

    Args:
        feedback_buffers: Dictionary mapping feedback types to lists of feedback
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation

    Returns:
        Dictionary mapping feedback types to (train_loader, val_loader) tuples
    """
    dataloaders = {}

    def create_collate_fn(feedback_type_str):
        """Create a collate function that includes feedback type"""
        def collate_fn(batch):
            # Default collate the batch data
            batch_data = torch.utils.data.dataloader.default_collate(batch)
            # Add feedback type as the first element
            return (feedback_type_str, batch_data)
        return collate_fn

    for feedback_type, feedback_data in feedback_buffers.items():
        if not feedback_data:
            continue

        # Create dataset
        dataset = BufferDataset(feedback_data)

        # Split into train and validation
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        if train_size <= 0 or val_size <= 0:
            print(
                f"Skipping {feedback_type} - insufficient data ({len(dataset)} samples)"
            )
            continue

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create collate function for this feedback type
        collate_fn = create_collate_fn(feedback_type)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        dataloaders[feedback_type] = (train_loader, val_loader)

    return dataloaders


def _unified_collate_fn(batch, partition_size: int = 4):
    """
    Custom collate function for unified training.

    For pairwise feedback types (comparative, demonstrative, etc.), if items
    carry a return-difference third element, we inject ResponseRank metadata:
      - ranks:  negative return difference (higher diff = stronger preference = lower rank)
      - partition_ids: random partitions of size ``partition_size``

    The resulting batch is:
      (feedback_types, (pair_data, pref_indices, ranks, partition_ids))

    For scalar types or pairwise items without diff, the batch is unchanged:
      (feedback_types, collated_data)
    """
    feedback_types = []
    data_batch = []

    for feedback_type, data in batch:
        feedback_types.append(feedback_type)
        data_batch.append(data)

    # Check if this batch is a pairwise type with rank info (3-element tuples)
    fb_type_0 = feedback_types[0] if feedback_types else None
    is_pairwise = fb_type_0 in _PAIRWISE_TYPES
    has_diff = is_pairwise and len(data_batch) > 0 and len(data_batch[0]) == 3

    if has_diff and len(data_batch) > 1:
        # Separate pair_data, preference, diff
        pair_data_list = [d[0] for d in data_batch]
        pref_list = [d[1] for d in data_batch]
        diff_list = [d[2] for d in data_batch]

        # Collate trajectories and preferences
        collated_pairs = torch.utils.data.dataloader.default_collate(pair_data_list)
        collated_prefs = torch.utils.data.dataloader.default_collate(pref_list)

        # Build ranks: negative diff (lower = stronger preference)
        ranks = torch.tensor([-d for d in diff_list], dtype=torch.float32)

        # Random partition assignment
        n = len(data_batch)
        perm = torch.randperm(n)
        partition_ids = torch.zeros(n, dtype=torch.long)
        pid = 0
        for i in range(0, n, partition_size):
            for j in perm[i : i + partition_size]:
                partition_ids[j] = pid
            pid += 1

        collated_data = (collated_pairs, collated_prefs, ranks, partition_ids)
    else:
        # Standard collation (scalar feedback, or single-item batch, or no diff)
        collated_data = torch.utils.data.dataloader.default_collate(data_batch)

    return feedback_types, collated_data


def create_unified_dataloaders(
    feedback_buffers: Dict[str, List[Any]],
    batch_size: int,
    val_split: float = 0.2,
    partition_size: int = 4,
):
    """
    Create unified dataloaders that include feedback type with the data.

    Args:
        feedback_buffers: Dictionary mapping feedback types to lists of feedback
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        partition_size: Partition size for ResponseRank PL loss grouping
    """

    def collate_fn(batch):
        return _unified_collate_fn(batch, partition_size=partition_size)

    # Create unified dataset
    dataset = UnifiedBufferDataset(feedback_buffers)

    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"Insufficient data ({len(dataset)} samples)")

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader