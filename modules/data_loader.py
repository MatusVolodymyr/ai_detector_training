"""
Dataset loader module - handles loading datasets from local or HuggingFace
"""

import os
from typing import Dict, Any
from datasets import load_dataset, load_from_disk, Dataset


class DatasetLoader:
    """Loads datasets from various sources"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dataset_config = config['dataset']
    
    def load(self) -> Dataset:
        """
        Load dataset from configured source.
        
        Returns:
            Dataset object with 'text' and 'label' columns
        """
        source = self.dataset_config['source']
        path = self.dataset_config['path']
        
        if source == 'local':
            dataset = self._load_local(path)
        elif source == 'huggingface':
            dataset = self._load_huggingface(path)
        else:
            raise ValueError(f"Unknown dataset source: {source}")
        
        # Validate schema
        self._validate_schema(dataset)
        
        return dataset
    
    def _load_local(self, path: str) -> Dataset:
        """Load dataset from local file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        
        # Check file extension
        if path.endswith('.arrow'):
            return load_from_disk(path)
        else:
            # Try loading as various formats
            try:
                return load_dataset('arrow', data_files=path)['train']
            except:
                try:
                    return load_dataset('json', data_files=path)['train']
                except:
                    return load_dataset('csv', data_files=path)['train']
    
    def _load_huggingface(self, dataset_id: str) -> Dataset:
        """Load dataset from HuggingFace Hub"""
        dataset = load_dataset(dataset_id)
        
        # If dataset has splits, use train split
        if isinstance(dataset, dict):
            if 'train' in dataset:
                return dataset['train']
            else:
                # Use first split
                return list(dataset.values())[0]
        
        return dataset
    
    def _validate_schema(self, dataset: Dataset):
        """Validate that dataset has required columns"""
        text_col = self.dataset_config['text_column']
        label_col = self.dataset_config['label_column']
        
        if text_col not in dataset.column_names:
            raise ValueError(
                f"Text column '{text_col}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )
        
        if label_col not in dataset.column_names:
            raise ValueError(
                f"Label column '{label_col}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )
        
        # Check if labels are binary (0/1)
        unique_labels = set(dataset[label_col])
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"Labels must be binary (0 or 1). Found: {unique_labels}"
            )
