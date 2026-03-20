from typing import Any, Dict, List

import pytorch_lightning
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
            drop_last=False,
            collate_fn=collate_fn,
        )

        dataloaders[feedback_type] = (train_loader, val_loader)

    return dataloaders


def _collate_pairwise_group(items, feedback_types, partition_size):
    """Collate a group of pairwise feedback items."""
    obs1_list, act1_list, mask1_list = [], [], []
    obs2_list, act2_list, mask2_list = [], [], []
    pref_list = []
    diff_list = []
    for d in items:
        pair_data = d[0]
        pref_list.append(d[1])
        diff_list.append(d[2] if len(d) == 3 else 0.0)

        (o1, a1, m1), (o2, a2, m2) = pair_data
        obs1_list.append(o1)
        act1_list.append(a1)
        mask1_list.append(m1)
        obs2_list.append(o2)
        act2_list.append(a2)
        mask2_list.append(m2)

    collated_pairs = (
        (torch.stack(obs1_list), torch.stack(act1_list), torch.stack(mask1_list)),
        (torch.stack(obs2_list), torch.stack(act2_list), torch.stack(mask2_list)),
    )
    collated_prefs = torch.utils.data.dataloader.default_collate(pref_list)
    ranks = torch.tensor([-d for d in diff_list], dtype=torch.float32)

    n = len(items)
    perm = torch.randperm(n)
    partition_ids = torch.zeros(n, dtype=torch.long)
    pid = 0
    for i in range(0, n, partition_size):
        for j in perm[i : i + partition_size]:
            partition_ids[j] = pid
        pid += 1

    return feedback_types, (collated_pairs, collated_prefs, ranks, partition_ids)


def _collate_scalar_group(items, feedback_types):
    """Collate a group of scalar feedback items, sub-grouped by type.

    Different scalar types can have incompatible tensor shapes (e.g.
    evaluative [150, obs_dim] vs descriptive [1, obs_dim]), so we collate
    each feedback type independently and return a list of sub-batches.
    """
    from collections import defaultdict

    by_type = defaultdict(list)
    for fb_type, item in zip(feedback_types, items):
        by_type[fb_type].append(item)

    sub_batches = []
    for fb_type, type_items in by_type.items():
        collated = torch.utils.data.dataloader.default_collate(type_items)
        sub_batches.append(([fb_type] * len(type_items), collated))

    # If only one scalar type, return it directly
    if len(sub_batches) == 1:
        return sub_batches[0]
    return sub_batches


def _unified_collate_fn(batch, partition_size: int = 4):
    """
    Custom collate function for unified training.

    Separates batch items by structural type (pairwise vs scalar) and collates
    each group independently.  Returns a **list** of sub-batches so that
    ``training_step`` can sum losses across groups.

    Each sub-batch is a tuple:
      (feedback_types_list, collated_data)

    For pairwise groups, collated_data is:
      (pair_data, pref_indices, ranks, partition_ids)

    For scalar groups, collated_data is the standard default_collate output.
    """
    # Separate items by structural type
    pairwise_types = []
    pairwise_items = []
    scalar_types = []
    scalar_items = []

    for feedback_type, data in batch:
        if feedback_type in _PAIRWISE_TYPES:
            pairwise_types.append(feedback_type)
            pairwise_items.append(data)
        else:
            scalar_types.append(feedback_type)
            scalar_items.append(data)

    sub_batches = []

    if pairwise_items:
        sub_batches.append(
            _collate_pairwise_group(pairwise_items, pairwise_types, partition_size)
        )

    if scalar_items:
        scalar_result = _collate_scalar_group(scalar_items, scalar_types)
        # _collate_scalar_group returns a list of sub-batches when multiple
        # scalar types with incompatible shapes are present
        if isinstance(scalar_result, list):
            sub_batches.extend(scalar_result)
        else:
            sub_batches.append(scalar_result)

    # If only one group, return it directly (backward-compatible single-batch path)
    if len(sub_batches) == 1:
        return sub_batches[0]

    return sub_batches


def _reward_weight(item, eps: float = 0.05) -> float:
    """Return a sampling weight for one UnifiedBufferDataset entry.

    For scalar feedback types (supervised, evaluative, descriptive) the weight
    is proportional to |reward| so that rare high-reward steps are oversampled
    relative to the majority near-zero steps.  Pairwise types get weight 1.0.
    eps is the minimum weight assigned to zero-reward steps so they are never
    fully excluded.
    """
    feedback_type, data = item
    if feedback_type not in _PAIRWISE_TYPES:
        try:
            reward = data[1]
            if hasattr(reward, "item"):
                reward = float(reward.item())
            else:
                reward = float(reward)
            return abs(reward) + eps
        except Exception:
            pass
    return 1.0


def create_unified_dataloaders(
    feedback_buffers: Dict[str, List[Any]],
    batch_size: int,
    val_split: float = 0.2,
    partition_size: int = 4,
    oversample_rewards: bool = False,
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
    if oversample_rewards:
        # Weight each training sample by |reward| + eps so rare high-reward
        # steps are seen more often than the majority near-zero steps.
        weights = torch.tensor(
            [_reward_weight(train_dataset.dataset.data[i]) for i in train_dataset.indices],
            dtype=torch.float32,
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
        )
    else:
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
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader