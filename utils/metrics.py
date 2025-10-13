"""
Metrics and evaluation utilities
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import xgboost as xgb
from typing import Dict, Any, List
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from datetime import datetime


class FinalEvaluator:
    """Performs final evaluation on test set"""
    
    def __init__(self, config: Dict[str, Any], logger):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
    
    def evaluate_all(
        self,
        test_dataset: Dataset,
        deberta_models: Dict[str, str],
        xgboost_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation on test set.
        
        Args:
            test_dataset: Test dataset (raw sentences)
            deberta_models: Dictionary of DeBERTa model paths
            xgboost_model: XGBoost model info
        
        Returns:
            Comprehensive evaluation report
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("FINAL EVALUATION")
        self.logger.info("="*60)
        
        # Note: For now, we need features to be pre-extracted
        # In a full implementation, we would extract features here
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(test_dataset),
            'models': {
                'deberta_s1': deberta_models.get('s1', 'N/A'),
                'deberta_s3': deberta_models.get('s3', 'N/A'),
                'deberta_s5': deberta_models.get('s5', 'N/A'),
                'xgboost': xgboost_model.get('metrics', {}),
            },
            'feature_importance': xgboost_model.get('importance', {}),
        }
        
        self.logger.info("✓ Evaluation report generated")
        
        return report


def generate_final_report(
    deberta_metrics: Dict[str, Any],
    xgboost_metrics: Dict[str, Any],
    feature_importance: Dict[str, float],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate comprehensive final training report.
    
    Args:
        deberta_metrics: Metrics from DeBERTa training
        xgboost_metrics: Metrics from XGBoost training
        feature_importance: Feature importance scores
        config: Configuration dictionary
    
    Returns:
        Complete training report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'deberta_models': deberta_metrics,
        'xgboost_model': xgboost_metrics,
        'feature_importance_top_20': dict(list(feature_importance.items())[:20]),
        'summary': {
            'pipeline_version': '1.0.0',
            'status': 'completed',
        }
    }
    
    return report


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
    
    return metrics


def per_language_metrics(
    dataset: Dataset,
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per language.
    
    Args:
        dataset: Dataset with 'language' column
        predictions: Predicted labels
        labels: True labels
    
    Returns:
        Dictionary mapping language to metrics
    """
    if 'language' not in dataset.column_names:
        return {}
    
    languages = dataset['language']
    unique_langs = set(languages)
    
    lang_metrics = {}
    for lang in unique_langs:
        mask = [l == lang for l in languages]
        lang_preds = predictions[mask]
        lang_labels = labels[mask]
        
        if len(lang_labels) > 0:
            lang_metrics[lang] = compute_classification_metrics(
                lang_labels,
                lang_preds
            )
    
    return lang_metrics
