"""
XGBoost trainer module - trains meta-classifier with Optuna optimization
"""

import os
import json
import numpy as np
import xgboost as xgb
import optuna
from typing import Dict, Any, Tuple
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report, accuracy_score, roc_auc_score
from collections import Counter


class XGBoostTrainer:
    """Trains XGBoost meta-classifier"""
    
    def __init__(self, config: Dict[str, Any], logger):
        """
        Initialize XGBoost trainer.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.xgb_config = config['xgboost']
        self.output_dir = self.xgb_config['output_dir']
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train(self, xgboost_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train XGBoost meta-classifier.
        
        Args:
            xgboost_data: Dictionary with train/val datasets and feature columns
        
        Returns:
            Dictionary with trained model and metrics
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("XGBOOST META-CLASSIFIER TRAINING")
        self.logger.info("="*60)
        
        train_ds = xgboost_data['train']
        val_ds = xgboost_data['val']
        feature_cols = xgboost_data['feature_columns']
        
        self.logger.info(f"Train size: {len(train_ds)}")
        self.logger.info(f"Val size: {len(val_ds)}")
        self.logger.info(f"Features: {len(feature_cols)}")
        
        # Convert to numpy
        X_train, y_train = self._dataset_to_numpy(train_ds, feature_cols)
        X_val, y_val = self._dataset_to_numpy(val_ds, feature_cols)
        
        # Apply stochastic expert dropout augmentation
        if self.xgb_config['augmentation']['enabled']:
            self.logger.info("\n🔄 Applying stochastic expert dropout augmentation...")
            X_train_aug, y_train_aug = self._apply_augmentation(
                X_train, y_train, feature_cols
            )
        else:
            X_train_aug, y_train_aug = X_train, y_train
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_aug, label=y_train_aug)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Hyperparameter optimization
        self.logger.info("\n🔍 Starting hyperparameter optimization with Optuna...")
        best_params = self._optimize_hyperparameters(dtrain, dval, y_val)
        
        # Train final model
        self.logger.info("\n🚀 Training final XGBoost model...")
        final_model = self._train_final_model(dtrain, dval, best_params)
        
        # Evaluate
        metrics = self._evaluate_model(final_model, dval, y_val)
        
        # Feature importance
        importance = self._compute_feature_importance(final_model, feature_cols)
        
        # Fragility analysis
        if self.config['evaluation']['fragility_analysis']:
            fragility = self._fragility_score(final_model, X_val, y_val, feature_cols)
            metrics['fragility'] = fragility
        
        # Save model and metadata
        self._save_model(final_model, feature_cols, best_params, metrics, importance)
        
        self.logger.info("\n✓ XGBoost training complete!")
        
        return {
            'model': final_model,
            'feature_columns': feature_cols,
            'metrics': metrics,
            'params': best_params,
            'importance': importance,
        }
    
    def _dataset_to_numpy(
        self,
        dataset: Dataset,
        feature_cols: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert dataset to numpy arrays"""
        X = np.column_stack([
            np.asarray(dataset[col], dtype=np.float32)
            for col in feature_cols
        ])
        y = np.asarray(dataset['label'], dtype=np.float32)
        return X, y
    
    def _apply_augmentation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply stochastic expert dropout augmentation"""
        aug_config = self.xgb_config['augmentation']
        
        X_aug = self._stochastic_expert_drop_np(
            X,
            feature_cols,
            drop_rate=aug_config['drop_rate'],
            neutral=aug_config['neutral_value'],
            bias_weights=tuple(aug_config['bias_weights']),
            seed=aug_config['seed'],
            mode=aug_config['mode']
        )
        
        if aug_config['mode'] == 'concat':
            y_aug = np.concatenate([y, y])
        else:
            y_aug = y
        
        self.logger.info(f"  Augmented dataset size: {len(X_aug)} (from {len(X)})")
        return X_aug, y_aug
    
    def _stochastic_expert_drop_np(
        self,
        X: np.ndarray,
        feature_cols: list,
        drop_rate: float = 0.30,
        neutral: float = 0.50,
        bias_weights: Tuple[int, int, int] = (1, 1, 2),
        seed: int = 42,
        mode: str = "concat"
    ) -> np.ndarray:
        """
        Stochastically drop expert predictions to reduce over-reliance.
        
        Args:
            X: Feature matrix
            feature_cols: List of feature column names
            drop_rate: Probability of dropping an expert
            neutral: Value to replace dropped predictions
            bias_weights: Weights for [s1, s3, s5] - higher = more likely to drop
            seed: Random seed
            mode: 'concat' (double dataset) or 'replace' (in-place)
        
        Returns:
            Augmented feature matrix
        """
        rng = np.random.RandomState(seed)
        n, d = X.shape
        
        expert_names = ["prob_ai_s1", "prob_ai_s3", "prob_ai_s5"]
        expert_idx = np.array([
            feature_cols.index(name) for name in expert_names
            if name in feature_cols
        ], dtype=int)
        
        if len(expert_idx) == 0:
            self.logger.warning("No expert columns found for augmentation")
            return X
        
        # Determine which samples to augment
        mask = rng.rand(n) < drop_rate
        k = int(mask.sum())
        
        # Normalize weights
        probs = np.array(bias_weights, dtype=float)
        probs = probs / probs.sum()
        
        if mode == "replace":
            X_out = X.copy()
            if k > 0:
                which = rng.choice(len(expert_idx), size=k, p=probs)
                rows = np.where(mask)[0]
                X_out[rows, expert_idx[which]] = neutral
            return X_out
        
        # mode == "concat": original + modified copy
        X_out = np.vstack([X, X.copy()])
        if k > 0:
            which = rng.choice(len(expert_idx), size=k, p=probs)
            rows2 = np.where(mask)[0] + n  # modify the second half
            X_out[rows2, expert_idx[which]] = neutral
        
        return X_out
    
    def _optimize_hyperparameters(
        self,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            opt_config = self.xgb_config['optuna']
            
            params = {
                "objective": self.xgb_config['objective'],
                "eval_metric": self.xgb_config['eval_metric'],
                "tree_method": self.xgb_config['tree_method'],
                "device": self.xgb_config['device'],
                "max_depth": trial.suggest_int("max_depth", *opt_config['max_depth']),
                "learning_rate": trial.suggest_float("learning_rate", *opt_config['learning_rate'], log=True),
                "subsample": trial.suggest_float("subsample", *opt_config['subsample']),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *opt_config['colsample_bytree']),
                "gamma": trial.suggest_float("gamma", *opt_config['gamma']),
                "min_child_weight": trial.suggest_int("min_child_weight", *opt_config['min_child_weight']),
            }
            
            num_boost_round = trial.suggest_int("num_boost_round", *opt_config['num_boost_round'], step=50)
            
            bst = xgb.train(
                params,
                dtrain,
                evals=[(dval, "validation")],
                num_boost_round=num_boost_round,
                early_stopping_rounds=self.xgb_config['early_stopping_rounds'],
                verbose_eval=False,
            )
            
            probas = bst.predict(dval)
            preds = (probas >= 0.5).astype(int)
            f1 = f1_score(y_val, preds)
            
            return f1
        
        study = optuna.create_study(direction=self.xgb_config['optuna']['direction'])
        study.optimize(
            objective,
            n_trials=self.xgb_config['optuna']['n_trials'],
            show_progress_bar=True
        )
        
        self.logger.info(f"✓ Best F1 score: {study.best_value:.4f}")
        self.logger.info(f"✓ Best parameters: {study.best_params}")
        
        return study.best_params
    
    def _train_final_model(
        self,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        best_params: Dict[str, Any]
    ) -> xgb.Booster:
        """Train final model with best parameters"""
        
        # Extract num_boost_round
        num_boost_round = best_params.pop('num_boost_round')
        
        # Add base config
        final_params = {
            "objective": self.xgb_config['objective'],
            "eval_metric": self.xgb_config['eval_metric'],
            "tree_method": self.xgb_config['tree_method'],
            "device": self.xgb_config['device'],
            **best_params
        }
        
        model = xgb.train(
            final_params,
            dtrain,
            evals=[(dval, "validation")],
            num_boost_round=num_boost_round,
            early_stopping_rounds=self.xgb_config['early_stopping_rounds'],
            verbose_eval=True
        )
        
        # Restore num_boost_round to params
        best_params['num_boost_round'] = num_boost_round
        
        return model
    
    def _evaluate_model(
        self,
        model: xgb.Booster,
        dval: xgb.DMatrix,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate model on validation set"""
        
        y_pred_prob = model.predict(dval)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1": float(f1_score(y_val, y_pred)),
            "roc_auc": float(roc_auc_score(y_val, y_pred_prob)),
        }
        
        self.logger.info(f"\n📊 Validation Metrics:")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        self.logger.info(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
        
        # Classification report
        report = classification_report(y_val, y_pred, digits=4)
        self.logger.info(f"\n{report}")
        
        return metrics
    
    def _compute_feature_importance(
        self,
        model: xgb.Booster,
        feature_cols: list
    ) -> Dict[str, float]:
        """Compute feature importance"""
        importances = model.get_score(importance_type='gain')
        
        # Map feature keys to names
        importance_dict = {}
        for fkey, score in importances.items():
            fname = feature_cols[int(fkey[1:])]
            importance_dict[fname] = float(score)
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Log top features
        top_n = self.config['evaluation']['feature_importance_top_n']
        self.logger.info(f"\n🎯 Top {top_n} Features:")
        for i, (fname, score) in enumerate(list(sorted_importance.items())[:top_n], 1):
            self.logger.info(f"  {i:2d}. {fname:40s}  (gain: {score:.6f})")
        
        return sorted_importance
    
    def _fragility_score(
        self,
        model: xgb.Booster,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: list,
        thr: float = 0.5,
        neutral: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute model fragility score.
        
        Measures how much predictions change when neutralizing each expert.
        """
        self.logger.info("\n🔬 Computing fragility scores...")
        
        base_preds = model.predict(xgb.DMatrix(X))
        base = (base_preds >= thr).astype(int)
        
        def flip_rate(col_name):
            if col_name not in feature_cols:
                return 0.0
            X2 = X.copy()
            j = feature_cols.index(col_name)
            X2[:, j] = neutral
            p2 = model.predict(xgb.DMatrix(X2))
            pred2 = (p2 >= thr).astype(int)
            return float((pred2 != base).mean())
        
        rates = {
            "s1_flip_rate": flip_rate("prob_ai_s1"),
            "s3_flip_rate": flip_rate("prob_ai_s3"),
            "s5_flip_rate": flip_rate("prob_ai_s5"),
        }
        
        self.logger.info("  Prediction flip rates when neutralizing:")
        for expert, rate in rates.items():
            self.logger.info(f"    {expert}: {rate:.4f}")
        
        return rates
    
    def _save_model(
        self,
        model: xgb.Booster,
        feature_cols: list,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        importance: Dict[str, float]
    ):
        """Save model and metadata"""
        
        # Save model
        model_path = os.path.join(self.output_dir, 'xgboost_model.json')
        model.save_model(model_path)
        self.logger.info(f"\n💾 Model saved: {model_path}")
        
        # Save feature columns
        if self.xgb_config['save_feature_list']:
            feature_path = os.path.join(self.output_dir, 'feature_columns.json')
            with open(feature_path, 'w') as f:
                json.dump(feature_cols, f, indent=2)
            self.logger.info(f"💾 Features saved: {feature_path}")
        
        # Save model info
        info = {
            'params': params,
            'metrics': metrics,
            'num_features': len(feature_cols),
            'top_20_features': dict(list(importance.items())[:20]),
        }
        
        info_path = os.path.join(self.output_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        self.logger.info(f"💾 Model info saved: {info_path}")
