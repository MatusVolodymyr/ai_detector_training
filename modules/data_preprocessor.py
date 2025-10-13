"""
Data preprocessor - handles sentence splitting and window creation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Dict, Any, List
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import numpy as np

# Import existing stylometric extraction module
from modules.stylometric_extraction import (
    create_windows,
    initialize_models as init_stylometric_models
)


class DataPreprocessor:
    """Preprocesses raw text dataset into windowed sentences"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.windowing_config = config['windowing']
        self.split_config = config['data_split']
        
        # Initialize stylometric models for sentence splitting
        init_stylometric_models()
    
    def process(self, dataset: Dataset) -> Dict[str, Dataset]:
        """
        Process dataset: split into sentences and create windows.
        
        Args:
            dataset: Raw dataset with 'text' and 'label' columns
        
        Returns:
            Dictionary with 'train', 'val', 'test' splits of windowed sentences
        """
        text_col = self.config['dataset']['text_column']
        label_col = self.config['dataset']['label_column']
        
        print(f"\n📝 Processing {len(dataset)} texts into sentences...")
        
        # Convert to windowed sentences
        all_sentences = []
        
        for i, example in enumerate(tqdm(dataset, desc="Creating windows")):
            text = example[text_col]
            label = example[label_col]
            
            # Create windows for this text
            windows_dict = create_windows(
                text,
                max_sentences=self.windowing_config['max_sentences_per_text']
            )
            
            # Each sentence becomes a separate example
            num_sentences = len(windows_dict['window_1'])
            
            for sent_idx in range(num_sentences):
                sentence_example = {
                    'window_1': windows_dict['window_1'][sent_idx],
                    'window_3': windows_dict['window_3'][sent_idx],
                    'window_5': windows_dict['window_5'][sent_idx],
                    'language': windows_dict['language'][sent_idx],
                    'label': label,
                    'doc_id': f"doc_{i}",
                    'sentence_index': sent_idx,
                }
                all_sentences.append(sentence_example)
        
        print(f"✓ Created {len(all_sentences)} sentences from {len(dataset)} texts")
        
        # Create dataset from sentences
        sentence_dataset = Dataset.from_list(all_sentences)
        
        # Split into train/val/test
        splits = self._split_dataset(sentence_dataset)
        
        return splits
    
    def _split_dataset(self, dataset: Dataset) -> Dict[str, Dataset]:
        """
        Split dataset into train/val/test.
        
        Args:
            dataset: Full windowed sentence dataset
        
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        train_size = self.split_config['deberta_train']
        val_size = self.split_config['deberta_val']
        test_size = self.split_config['test']
        seed = self.split_config['seed']
        stratify = self.split_config['stratify']
        
        # Validate split sizes
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split sizes must sum to 1.0, got {total}")
        
        print(f"\n📊 Splitting dataset (train={train_size}, val={val_size}, test={test_size})...")
        
        # First split: separate test set
        train_val = dataset.train_test_split(
            test_size=test_size,
            seed=seed,
            stratify_by_column='label' if stratify else None
        )
        
        # Second split: separate train and val
        remaining_train_size = train_size / (train_size + val_size)
        train_val_split = train_val['train'].train_test_split(
            train_size=remaining_train_size,
            seed=seed,
            stratify_by_column='label' if stratify else None
        )
        
        splits = {
            'train': train_val_split['train'],
            'val': train_val_split['test'],
            'test': train_val['test'],
        }
        
        print(f"✓ Train: {len(splits['train'])} sentences")
        print(f"✓ Val:   {len(splits['val'])} sentences")
        print(f"✓ Test:  {len(splits['test'])} sentences")
        
        return splits
