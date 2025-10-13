"""
AI Detector Training Pipeline - Main Orchestrator

This script orchestrates the complete training pipeline:
1. Data preparation (sentence splitting & windowing)
2. DeBERTa model training (3 models for different window sizes)
3. Feature extraction (stylometric + model predictions)
4. XGBoost meta-classifier training
5. Final evaluation and reporting
"""

import os
import sys
import argparse
import yaml
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import pipeline modules
from modules.data_loader import DatasetLoader
from modules.data_preprocessor import DataPreprocessor
from modules.deberta_trainer import DeBERTaTrainer
from modules.feature_extractor import FeatureExtractor
from modules.xgboost_trainer import XGBoostTrainer
from utils.logger import setup_logger, log_phase_start, log_phase_complete
from utils.checkpoint_manager import CheckpointManager
from utils.metrics import generate_final_report


class TrainingPipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, config_path: str, resume: bool = False, 
                 skip_deberta: bool = False, only_xgboost: bool = False):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to configuration YAML file
            resume: Resume from checkpoints if available
            skip_deberta: Skip DeBERTa training (use existing models)
            only_xgboost: Only train XGBoost (requires existing DeBERTa models and features)
        """
        self.config = self._load_config(config_path)
        self.resume = resume
        self.skip_deberta = skip_deberta
        self.only_xgboost = only_xgboost
        
        # Setup logging
        self.logger = setup_logger(self.config)
        
        # Create output directories
        self._create_directories()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.config['output']['checkpoint_dir']
        )
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        self.logger.info("=" * 80)
        self.logger.info("AI DETECTOR TRAINING PIPELINE INITIALIZED")
        self.logger.info("=" * 80)
        self.logger.info(f"Config loaded from: {config_path}")
        self.logger.info(f"Resume mode: {resume}")
        self.logger.info(f"Skip DeBERTa: {skip_deberta}")
        self.logger.info(f"Only XGBoost: {only_xgboost}")
        self.logger.info(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        self.logger.info("=" * 80)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _create_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.config['output']['base_dir'],
            self.config['output']['checkpoint_dir'],
            self.config['output']['logs_dir'],
            self.config['feature_extraction']['output_dir'],
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config['system']['seed']
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if self.config['system']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def run(self):
        """Run the complete training pipeline"""
        start_time = datetime.now()
        
        try:
            # Phase 0: Validation
            self._validate_setup()
            
            # Phase 1: Data Preparation
            if not self.only_xgboost:
                windowed_data = self._phase1_data_preparation()
            else:
                self.logger.info("Skipping Phase 1 (only_xgboost mode)")
                windowed_data = None
            
            # Phase 2: DeBERTa Training
            if not self.skip_deberta and not self.only_xgboost:
                deberta_models = self._phase2_deberta_training(windowed_data)
            else:
                self.logger.info(f"Skipping Phase 2 (skip_deberta={self.skip_deberta}, only_xgboost={self.only_xgboost})")
                deberta_models = self._load_existing_deberta_models()
            
            # Phase 3: Feature Extraction
            if not self.only_xgboost:
                xgboost_data = self._phase3_feature_extraction(windowed_data, deberta_models)
            else:
                self.logger.info("Loading existing features for XGBoost training")
                xgboost_data = self._load_existing_features()
            
            # Phase 4: XGBoost Training
            xgboost_model = self._phase4_xgboost_training(xgboost_data)
            
            # Phase 5: Final Evaluation
            report = self._phase5_final_evaluation(windowed_data, deberta_models, xgboost_model)
            
            # Save final report
            self._save_report(report)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            self.logger.info(f"Total training time: {duration}")
            self.logger.info(f"Models saved to: {self.config['output']['base_dir']}")
            self.logger.info(f"Report saved to: {self.config['output']['base_dir']}/{self.config['output']['report_name']}")
            self.logger.info("=" * 80)
            
            # Cleanup if configured
            if self.config['output']['cleanup_on_success']:
                self._cleanup_intermediate_files()
            
            return report
            
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error(f"❌ PIPELINE FAILED: {str(e)}")
            self.logger.error("=" * 80)
            self.logger.exception("Full traceback:")
            raise
    
    def _validate_setup(self):
        """Validate configuration and environment"""
        log_phase_start(self.logger, "Phase 0", "Setup Validation")
        
        # Check dataset source
        if self.config['dataset']['source'] == 'local':
            if not os.path.exists(self.config['dataset']['path']):
                raise FileNotFoundError(f"Dataset not found: {self.config['dataset']['path']}")
        
        # Check GPU availability for training
        if not torch.cuda.is_available():
            self.logger.warning("⚠️  No GPU detected. Training will be significantly slower!")
        
        # Check if skipping phases makes sense
        if self.only_xgboost and not self._check_features_exist():
            raise ValueError("Cannot use --only-xgboost without existing feature extraction!")
        
        if self.skip_deberta and not self._check_deberta_models_exist():
            raise ValueError("Cannot use --skip-deberta without existing DeBERTa models!")
        
        log_phase_complete(self.logger, "Phase 0", {"status": "validated"})
    
    def _phase1_data_preparation(self) -> Dict:
        """Phase 1: Load and preprocess dataset"""
        log_phase_start(self.logger, "Phase 1", "Data Preparation")
        
        # Check for checkpoint
        if self.resume and self.checkpoint_manager.check_phase_completed('phase1'):
            self.logger.info("✓ Phase 1 already completed. Loading from checkpoint...")
            return self.checkpoint_manager.load_checkpoint('phase1')
        
        # Load dataset
        loader = DatasetLoader(self.config)
        raw_dataset = loader.load()
        self.logger.info(f"Loaded dataset: {len(raw_dataset)} examples")
        
        # Preprocess (sentence splitting and windowing)
        preprocessor = DataPreprocessor(self.config)
        windowed_data = preprocessor.process(raw_dataset)
        
        self.logger.info(f"Created windowed dataset: {sum(len(v) for v in windowed_data.values())} total sentences")
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint('phase1', windowed_data)
        
        metrics = {
            "original_texts": len(raw_dataset),
            "total_sentences": sum(len(v) for v in windowed_data.values()),
            "train_sentences": len(windowed_data['train']),
            "val_sentences": len(windowed_data['val']),
            "test_sentences": len(windowed_data['test']),
        }
        
        log_phase_complete(self.logger, "Phase 1", metrics)
        return windowed_data
    
    def _phase2_deberta_training(self, windowed_data: Dict) -> Dict:
        """Phase 2: Train DeBERTa models"""
        log_phase_start(self.logger, "Phase 2", "DeBERTa Training")
        
        trainer = DeBERTaTrainer(self.config, self.logger)
        models = trainer.train_all_models(
            windowed_data['train'],
            windowed_data['val'],
            resume=self.resume,
            checkpoint_manager=self.checkpoint_manager
        )
        
        log_phase_complete(self.logger, "Phase 2", {"models_trained": len(models)})
        return models
    
    def _phase3_feature_extraction(self, windowed_data: Dict, deberta_models: Dict) -> Dict:
        """Phase 3: Extract features for XGBoost"""
        log_phase_start(self.logger, "Phase 3", "Feature Extraction")
        
        # Check for checkpoint
        if self.resume and self.checkpoint_manager.check_phase_completed('phase3'):
            self.logger.info("✓ Phase 3 already completed. Loading from checkpoint...")
            return self.checkpoint_manager.load_checkpoint('phase3')
        
        extractor = FeatureExtractor(self.config, self.logger)
        xgboost_data = extractor.extract_features(
            windowed_data['train'],
            windowed_data['val'],
            windowed_data['test'],
            deberta_models
        )
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint('phase3', xgboost_data)
        
        metrics = {
            "total_features": len(xgboost_data['train']),
            "num_feature_columns": len(xgboost_data['feature_columns']),
        }
        
        log_phase_complete(self.logger, "Phase 3", metrics)
        return xgboost_data
    
    def _phase4_xgboost_training(self, xgboost_data: Dict) -> Any:
        """Phase 4: Train XGBoost meta-classifier"""
        log_phase_start(self.logger, "Phase 4", "XGBoost Training")
        
        trainer = XGBoostTrainer(self.config, self.logger)
        model_info = trainer.train(xgboost_data)
        
        log_phase_complete(self.logger, "Phase 4", model_info['metrics'])
        return model_info
    
    def _phase5_final_evaluation(self, windowed_data: Dict, deberta_models: Dict, xgboost_model: Any) -> Dict:
        """Phase 5: Final evaluation and reporting"""
        log_phase_start(self.logger, "Phase 5", "Final Evaluation")
        
        from utils.metrics import FinalEvaluator
        
        evaluator = FinalEvaluator(self.config, self.logger)
        report = evaluator.evaluate_all(
            test_dataset=windowed_data['test'],
            deberta_models=deberta_models,
            xgboost_model=xgboost_model
        )
        
        log_phase_complete(self.logger, "Phase 5", {"report_generated": True})
        return report
    
    def _save_report(self, report: Dict):
        """Save final training report"""
        report_path = os.path.join(
            self.config['output']['base_dir'],
            self.config['output']['report_name']
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved to: {report_path}")
    
    def _load_existing_deberta_models(self) -> Dict:
        """Load existing DeBERTa models from disk"""
        models = {}
        base_dir = self.config['output']['base_dir']
        
        for window, name in self.config['deberta']['model_names'].items():
            model_path = os.path.join(base_dir, name)
            if os.path.exists(model_path):
                models[window] = model_path
                self.logger.info(f"Found existing model: {model_path}")
            else:
                raise FileNotFoundError(f"DeBERTa model not found: {model_path}")
        
        return models
    
    def _load_existing_features(self) -> Dict:
        """Load existing feature extraction results"""
        feature_path = os.path.join(
            self.config['feature_extraction']['output_dir'],
            'xgboost_features.arrow'
        )
        
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Features not found: {feature_path}")
        
        from datasets import load_from_disk
        dataset = load_from_disk(feature_path)
        
        self.logger.info(f"Loaded existing features: {len(dataset)} examples")
        return {'dataset': dataset}
    
    def _check_deberta_models_exist(self) -> bool:
        """Check if all DeBERTa models exist"""
        base_dir = self.config['output']['base_dir']
        for name in self.config['deberta']['model_names'].values():
            if not os.path.exists(os.path.join(base_dir, name)):
                return False
        return True
    
    def _check_features_exist(self) -> bool:
        """Check if feature extraction is complete"""
        feature_path = os.path.join(
            self.config['feature_extraction']['output_dir'],
            'xgboost_features.arrow'
        )
        return os.path.exists(feature_path)
    
    def _cleanup_intermediate_files(self):
        """Clean up intermediate files after successful training"""
        self.logger.info("Cleaning up intermediate files...")
        
        import shutil
        
        # Remove checkpoints
        if os.path.exists(self.config['output']['checkpoint_dir']):
            shutil.rmtree(self.config['output']['checkpoint_dir'])
        
        # Remove feature extraction chunks
        chunks_dir = self.config['feature_extraction']['output_dir']
        if os.path.exists(chunks_dir):
            for item in os.listdir(chunks_dir):
                if item.startswith('chunk_'):
                    os.remove(os.path.join(chunks_dir, item))
        
        self.logger.info("Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI Detector Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from local dataset
  python main.py --config config.yaml
  
  # Train from HuggingFace dataset
  python main.py --config config.yaml --dataset-id NeksoN/my_dataset
  
  # Resume interrupted training
  python main.py --config config.yaml --resume
  
  # Skip DeBERTa training (use existing models)
  python main.py --config config.yaml --skip-deberta
  
  # Only train XGBoost
  python main.py --config config.yaml --only-xgboost
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--dataset-id',
        type=str,
        help='HuggingFace dataset ID (overrides config file)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoints if available'
    )
    
    parser.add_argument(
        '--skip-deberta',
        action='store_true',
        help='Skip DeBERTa training and use existing models'
    )
    
    parser.add_argument(
        '--only-xgboost',
        action='store_true',
        help='Only train XGBoost (requires existing DeBERTa models and features)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory (overrides config file)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without training'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        config_path=args.config,
        resume=args.resume,
        skip_deberta=args.skip_deberta,
        only_xgboost=args.only_xgboost
    )
    
    # Override config with command-line args
    if args.dataset_id:
        pipeline.config['dataset']['source'] = 'huggingface'
        pipeline.config['dataset']['path'] = args.dataset_id
    
    if args.output_dir:
        pipeline.config['output']['base_dir'] = args.output_dir
    
    # Dry run mode
    if args.dry_run:
        print("✓ Configuration validated successfully")
        print(f"  Dataset: {pipeline.config['dataset']['path']}")
        print(f"  Output: {pipeline.config['output']['base_dir']}")
        sys.exit(0)
    
    # Run pipeline
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💡 You can resume training with: python main.py --config config.yaml --resume")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
