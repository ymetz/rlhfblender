import torch
import pytorch_lightning
from multi_type_feedback.feedback_dataset import BufferDataset
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any

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
    feedback_buffers: Dict[str, List[Any]], 
    batch_size: int,
    val_split: float = 0.2
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
    
    for feedback_type, feedback_data in feedback_buffers.items():
        if not feedback_data:
            continue
            
        # Create dataset
        dataset = BufferDataset(feedback_data)
        
        # Split into train and validation
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        if train_size <= 0 or val_size <= 0:
            print(f"Skipping {feedback_type} - insufficient data ({len(dataset)} samples)")
            continue
            
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
        
        dataloaders[feedback_type] = (train_loader, val_loader)
    
    return dataloaders


def create_unified_dataloaders(
    feedback_buffers: Dict[str, List[Any]], 
    batch_size: int,
    val_split: float = 0.2
):
    """
    Create unified dataloaders that include feedback type with the data.
    
    Args:
        feedback_buffers: Dictionary mapping feedback types to lists of feedback
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        ensemble_count: Number of ensemble models
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, val_loader


class MultiHeadDataModule(pytorch_lightning.LightningDataModule):
    """
    DataModule for multi-head network training.
    Allows training with mixed batches of different feedback types.
    """
    
    def __init__(
        self,
        feedback_buffers: Dict[str, List[Any]],
        batch_size: int = 32,
        val_split: float = 0.2,
        ensemble_count: int = 4
    ):
        super().__init__()
        self.feedback_buffers = feedback_buffers
        self.batch_size = batch_size
        self.val_split = val_split
        self.ensemble_count = ensemble_count
        
    def setup(self, stage=None):
        """Prepare datasets for training and validation."""
        # Create datasets for each feedback type
        self.datasets_by_type = {}
        self.train_datasets = []
        self.val_datasets = []
        
        for feedback_type, feedback_data in self.feedback_buffers.items():
            if not feedback_data:
                continue
                
            # Create dataset with feedback type included
            dataset = [(feedback_type, data) for data in feedback_data]
            
            # Split into train and validation
            val_size = int(len(dataset) * self.val_split)
            train_size = len(dataset) - val_size
            
            if train_size <= 0 or val_size <= 0:
                print(f"Skipping {feedback_type} - insufficient data ({len(dataset)} samples)")
                continue
                
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            
    def train_dataloader(self):
        """Create training dataloader with balanced batches."""
        # Combine all training datasets
        combined_train = torch.utils.data.ConcatDataset(self.train_datasets)
        
        return DataLoader(
            combined_train,
            batch_size=self.ensemble_count,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Create validation dataloader with balanced batches."""
        # Combine all validation datasets
        combined_val = torch.utils.data.ConcatDataset(self.val_datasets)
        
        return DataLoader(
            combined_val,
            batch_size=self.ensemble_count,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )