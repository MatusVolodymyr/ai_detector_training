"""
Feature extractor - orchestrates stylometric and prediction feature extraction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Dict, Any
from datasets import Dataset, concatenate_datasets
import numpy as np

# Import existing modules
from modules.incremental_processor import create_processor


class FeatureExtractor:
    """Extracts features for XGBoost training"""
    
    def __init__(self, config: Dict[str, Any], logger):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.feature_config = config['feature_extraction']
    
    def extract_features(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        deberta_models: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract features for XGBoost training.
        
        Args:
            train_dataset: Training sentence dataset
            val_dataset: Validation sentence dataset
            test_dataset: Test sentence dataset
            deberta_models: Dictionary mapping window to model path
        
        Returns:
            Dictionary with train/val/test datasets and feature columns
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("FEATURE EXTRACTION FOR XGBOOST")
        self.logger.info("="*60)
        
        # Step 1: Sample data for XGBoost training
        xgb_train_data = self._sample_for_xgboost(train_dataset, val_dataset)
        
        # Step 2: Extract features
        model_config = {
            "s1": (deberta_models['s1'], "window_1"),
            "s3": (deberta_models['s3'], "window_3"),
            "s5": (deberta_models['s5'], "window_5"),
        }
        
        self.logger.info(f"\nExtracting features for {len(xgb_train_data)} training examples...")
        xgb_train_features = self._extract_features_incremental(
            xgb_train_data,
            model_config,
            output_subdir="xgboost_train"
        )
        
        self.logger.info(f"\nExtracting features for {len(test_dataset)} test examples...")
        test_features = self._extract_features_incremental(
            test_dataset,
            model_config,
            output_subdir="xgboost_test"
        )
        
        # Step 3: Split train data into train/val for XGBoost
        xgb_splits = self._split_for_xgboost(xgb_train_features)
        
        # Get feature columns (exclude non-feature columns)
        NON_FEATURES = {
            "label", "language", "window_1", "window_3", "window_5",
            "doc_id", "sentence_index"
        }
        feature_columns = [
            col for col in xgb_train_features.column_names
            if col not in NON_FEATURES
        ]
        
        self.logger.info(f"\n✓ Feature extraction complete!")
        self.logger.info(f"  Train: {len(xgb_splits['train'])} examples")
        self.logger.info(f"  Val:   {len(xgb_splits['val'])} examples")
        self.logger.info(f"  Test:  {len(test_features)} examples")
        self.logger.info(f"  Features: {len(feature_columns)} columns")
        
        return {
            'train': xgb_splits['train'],
            'val': xgb_splits['val'],
            'test': test_features,
            'feature_columns': feature_columns,
        }
    
    def _sample_for_xgboost(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset
    ) -> Dataset:
        """
        Sample data for XGBoost training.
        
        Samples from train set + optionally uses full val set.
        """
        sample_size = self.feature_config['sample_from_train']
        use_val = self.feature_config['use_deberta_val']
        stratify = self.config['data_split']['stratify']
        
        self.logger.info(f"\n📊 Sampling data for XGBoost training...")
        self.logger.info(f"  Sampling {sample_size} from {len(train_dataset)} train examples")
        
        # Sample from training set
        if sample_size >= len(train_dataset):
            sampled_train = train_dataset
        else:
            # Stratified sampling
            if stratify and self.feature_config['sampling_strategy'] == 'stratified':
                # Split by label and sample from each
                label_0 = train_dataset.filter(lambda x: x['label'] == 0)
                label_1 = train_dataset.filter(lambda x: x['label'] == 1)
                
                # Proportional sampling
                n0 = int(sample_size * len(label_0) / len(train_dataset))
                n1 = sample_size - n0
                
                sample_0 = label_0.shuffle(seed=self.config['system']['seed']).select(range(min(n0, len(label_0))))
                sample_1 = label_1.shuffle(seed=self.config['system']['seed']).select(range(min(n1, len(label_1))))
                
                sampled_train = concatenate_datasets([sample_0, sample_1])
                sampled_train = sampled_train.shuffle(seed=self.config['system']['seed'])
            else:
                # Random sampling
                sampled_train = train_dataset.shuffle(
                    seed=self.config['system']['seed']
                ).select(range(sample_size))
        
        self.logger.info(f"  ✓ Sampled: {len(sampled_train)} examples")
        
        # Optionally add validation set
        if use_val:
            self.logger.info(f"  Adding {len(val_dataset)} validation examples")
            combined = concatenate_datasets([sampled_train, val_dataset])
            self.logger.info(f"  ✓ Total: {len(combined)} examples for XGBoost")
            return combined
        
        return sampled_train
    
    def _extract_features_incremental(
        self,
        dataset: Dataset,
        model_config: Dict[str, tuple],
        output_subdir: str
    ) -> Dataset:
        """
        Extract features using incremental processor.
        
        Args:
            dataset: Input dataset
            model_config: Model configuration
            output_subdir: Subdirectory for output
        
        Returns:
            Dataset with extracted features
        """
        output_dir = os.path.join(
            self.feature_config['output_dir'],
            output_subdir
        )
        
        # Create incremental processor
        processor = create_processor(
            model_config=model_config,
            chunk_size=self.feature_config['chunk_size'],
            enable_perplexity=self.feature_config['enable_perplexity']
        )
        
        # Process dataset
        final_path = processor.process_dataset(
            dataset=dataset,
            output_dir=output_dir,
            include_stylometric=self.feature_config['enable_stylometric'],
            include_predictions=self.feature_config['enable_predictions'],
            batch_size=self.feature_config['batch_size'],
            resume=self.feature_config['resume']
        )
        
        # Load final dataset
        from datasets import load_from_disk
        return load_from_disk(final_path)
    
    def _split_for_xgboost(self, dataset: Dataset) -> Dict[str, Dataset]:
        """Split dataset for XGBoost internal train/val"""
        val_split = self.config['xgboost']['validation_split']
        
        self.logger.info(f"\n📊 Splitting for XGBoost (val={val_split})...")
        
        split = dataset.train_test_split(
            test_size=val_split,
            seed=self.config['system']['seed'],
            stratify_by_column='label' if self.config['data_split']['stratify'] else None
        )
        
        return {
            'train': split['train'],
            'val': split['test'],
        }
