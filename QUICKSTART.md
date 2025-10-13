# AI Detector Training Pipeline - Quick Reference

## Project Structure

```
ai_detector_training/
├── main.py                          # Main orchestrator
├── config.yaml                      # Configuration file
├── requirements.txt                 # Dependencies
├── setup.sh                         # Setup script
├── create_sample_dataset.py         # Sample data generator
├── README.md                        # Full documentation
│
├── modules/
│   ├── data_loader.py              # Dataset loading
│   ├── data_preprocessor.py        # Sentence splitting & windowing
│   ├── deberta_trainer.py          # DeBERTa training
│   ├── feature_extractor.py        # Feature extraction orchestration
│   └── xgboost_trainer.py          # XGBoost training with Optuna
│
└── utils/
    ├── logger.py                    # Logging utilities
    ├── checkpoint_manager.py        # Checkpoint management
    ├── metrics.py                   # Evaluation metrics
    └── helpers.py                   # Helper functions
```

## Quick Start Commands

```bash
# 1. Setup (one-time)
./setup.sh

# 2. Create sample dataset for testing
python create_sample_dataset.py

# 3. Run training
python main.py --config config.yaml

# 4. Resume if interrupted
python main.py --config config.yaml --resume
```

## Key Configuration Options

### Essential Settings

```yaml
# Dataset
dataset:
  path: "training_data.arrow"     # Your dataset path

# Data splits
data_split:
  deberta_train: 0.80              # 80% for DeBERTa
  deberta_val: 0.10                # 10% for validation
  test: 0.10                       # 10% for testing

# Feature extraction
feature_extraction:
  sample_from_train: 80000         # Sample size for XGBoost
  use_deberta_val: true            # Include val set

# XGBoost optimization
xgboost:
  optuna:
    n_trials: 30                   # Hyperparameter search trials
```

## Pipeline Phases

| Phase | Description | Time (est.) | Checkpoint |
|-------|-------------|-------------|------------|
| 1 | Data preparation & windowing | ~10 min | phase1 |
| 2 | Train 3 DeBERTa models | 6-12 hrs | phase2_deberta_s1/s3/s5 |
| 3 | Feature extraction | 3-4 hrs | phase3 |
| 4 | XGBoost training + Optuna | 2-4 hrs | phase4 |
| 5 | Final evaluation | ~10 min | - |

**Total:** ~12-18 hours on single GPU

## Common Commands

```bash
# Validate config without training
python main.py --config config.yaml --dry-run

# Skip DeBERTa (use existing models)
python main.py --config config.yaml --skip-deberta

# Only train XGBoost
python main.py --config config.yaml --only-xgboost

# Custom output directory
python main.py --config config.yaml --output-dir ./my_models

# Use HuggingFace dataset
python main.py --config config.yaml --dataset-id NeksoN/my_dataset
```

## Monitoring Training

### Check logs
```bash
tail -f logs/training.log
```

### Check checkpoint status
```bash
cat checkpoints/status.json
```

### Monitor GPU usage
```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` in config (try 16 or 8)
- Reduce `feature_extraction.batch_size` (try 50)

### Training Slow
- Verify GPU is being used: `nvidia-smi`
- Check `fp16: true` is enabled in config
- Monitor GPU utilization

### Feature Extraction Stuck
- Check disk space (features require ~10GB per 100k examples)
- Verify models are accessible
- Check logs for specific errors

## Output Files

After successful training:

```
trained_models/
├── deberta_s1/                 # Window size 1 model
├── deberta_s3/                 # Window size 3 model  
├── deberta_s5/                 # Window size 5 model
├── xgboost_meta/               # Meta-classifier
│   ├── xgboost_model.json
│   ├── feature_columns.json
│   └── model_info.json
└── training_report.json        # Complete metrics
```

## Performance Tips

1. **Use SSD** for faster data loading
2. **Mixed precision** training enabled by default (fp16)
3. **Chunk processing** automatically handles memory
4. **Resume feature** saves time on interruptions
5. **Stratified sampling** ensures balanced training

## Expected Metrics

With good quality data:
- DeBERTa models: F1 ~0.92-0.96
- XGBoost ensemble: F1 ~0.94-0.98

## Integration with Existing Code

This pipeline reuses your existing modules:
- `modules/stylometric_extraction.py` - For feature extraction
- `modules/incremental_processor.py` - For chunked processing
- `modules/prediction_service_dataset.py` - For model predictions

## Notes

- Training is resumable at any checkpoint
- All intermediate results are saved
- Models are saved in HuggingFace format
- Full metrics logged to training_report.json
- Stochastic expert dropout prevents over-reliance on DeBERTa predictions

## Support

For detailed documentation, see [README.md](README.md)

For issues:
1. Check logs: `logs/training.log`
2. Check checkpoints: `checkpoints/status.json`
3. Verify config: `python main.py --config config.yaml --dry-run`
