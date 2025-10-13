"""
DeBERTa trainer module - trains multiple DeBERTa models for different window sizes
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


class DeBERTaTrainer:
    """Trains DeBERTa models for different window sizes"""
    
    def __init__(self, config: Dict[str, Any], logger):
        """
        Initialize DeBERTa trainer.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.deberta_config = config['deberta']
        self.output_dir = config['output']['base_dir']
    
    def train_all_models(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        resume: bool = False,
        checkpoint_manager: Optional[Any] = None
    ) -> Dict[str, str]:
        """
        Train all three DeBERTa models (s1, s3, s5).
        
        Args:
            train_dataset: Training dataset with window columns
            val_dataset: Validation dataset
            resume: Resume from checkpoints if available
            checkpoint_manager: Checkpoint manager instance
        
        Returns:
            Dictionary mapping window size to model path
        """
        models = {}
        
        window_configs = [
            ('s1', 'window_1', self.deberta_config['model_names']['s1']),
            ('s3', 'window_3', self.deberta_config['model_names']['s3']),
            ('s5', 'window_5', self.deberta_config['model_names']['s5']),
        ]
        
        for window_id, window_col, model_name in window_configs:
            checkpoint_key = f'phase2_deberta_{window_id}'
            
            # Check if already trained
            model_path = os.path.join(self.output_dir, model_name)
            if resume and os.path.exists(model_path):
                self.logger.info(f"✓ DeBERTa {window_id} already trained. Skipping...")
                models[window_id] = model_path
                continue
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training DeBERTa {window_id.upper()} on {window_col}")
            self.logger.info(f"{'='*60}")
            
            # Train model
            trained_model_path = self._train_single_model(
                window_id=window_id,
                window_col=window_col,
                model_name=model_name,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
            
            models[window_id] = trained_model_path
            
            # Mark checkpoint as complete
            if checkpoint_manager:
                checkpoint_manager.mark_phase_complete(checkpoint_key)
            
            self.logger.info(f"✓ DeBERTa {window_id} training complete!")
        
        return models
    
    def _train_single_model(
        self,
        window_id: str,
        window_col: str,
        model_name: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> str:
        """Train a single DeBERTa model"""
        
        # Load tokenizer and model
        base_model = self.deberta_config['base_model']
        self.logger.info(f"Loading base model: {base_model}")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=self.deberta_config['num_labels']
        )
        
        # Tokenize datasets
        self.logger.info(f"Tokenizing datasets...")
        
        def tokenize_function(examples):
            return tokenizer(
                examples[window_col],
                truncation=True,
                max_length=512,  # Standard max length
            )
        
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing train"
        )
        
        tokenized_val = val_dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing val"
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        
        # Training arguments
        output_path = os.path.join(self.output_dir, f"{model_name}_checkpoints")
        
        training_args = TrainingArguments(
            output_dir=output_path,
            eval_strategy=self.deberta_config['output']['eval_strategy'],
            save_strategy=self.deberta_config['output']['save_strategy'],
            learning_rate=self.deberta_config['training']['learning_rate'],
            per_device_train_batch_size=self.deberta_config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.deberta_config['training']['per_device_eval_batch_size'],
            num_train_epochs=self.deberta_config['training']['num_train_epochs'],
            weight_decay=self.deberta_config['training']['weight_decay'],
            load_best_model_at_end=self.deberta_config['output']['load_best_model_at_end'],
            metric_for_best_model=self.deberta_config['early_stopping']['metric'],
            save_total_limit=self.deberta_config['output']['save_total_limit'],
            fp16=self.deberta_config['training']['fp16'] and torch.cuda.is_available(),
            seed=self.config['system']['seed'],
            logging_steps=100,
            report_to='none',  # Disable wandb, tensorboard, etc.
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.deberta_config['early_stopping']['patience']
                )
            ],
        )
        
        # Train
        self.logger.info(f"Starting training...")
        trainer.train()
        
        # Evaluate on validation set
        self.logger.info(f"Evaluating on validation set...")
        metrics = trainer.evaluate()
        
        self.logger.info(f"Validation metrics:")
        for key, value in metrics.items():
            if key.startswith('eval_'):
                self.logger.info(f"  {key}: {value:.4f}")
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, model_name)
        self.logger.info(f"Saving model to: {final_model_path}")
        
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        # Save metrics
        import json
        metrics_path = os.path.join(final_model_path, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Clean up checkpoint directory
        import shutil
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        return final_model_path
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions),
        }
