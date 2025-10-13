# AI Detector Training Pipeline

Automated end-to-end training pipeline for AI text detection ensemble system.

## Overview

This pipeline trains a complete AI text detection system consisting of:
- **3 DeBERTa models** trained on different context window sizes (1, 3, 5 sentences)
- **Stylometric feature extraction** (POS tags, perplexity, lexical richness, etc.)
- **XGBoost meta-classifier** that combines DeBERTa predictions and stylometric features

## Features

✅ **Fully Automated** - One command from raw data to trained models  
✅ **Resumable** - Interrupt and resume training at any point  
✅ **Configurable** - All hyperparameters in YAML config  
✅ **GPU Optimized** - Efficient training with CUDA support  
✅ **Comprehensive Logging** - Detailed progress tracking  
✅ **Checkpointing** - Automatic checkpoint management  

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required spaCy models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download xx_sent_ud_sm
# Add other languages as needed:
# python -m spacy download de_core_news_sm
# python -m spacy download fr_core_news_sm
# python -m spacy download es_core_news_sm
# python -m spacy download pt_core_news_sm
```

## Quick Start

### 1. Prepare Your Dataset

Your dataset should be in Arrow format with at least two columns:
- `text`: Raw text content
- `label`: Binary label (0 = human, 1 = AI-generated)

Example:
```python
from datasets import Dataset

data = {
    'text': ["This is a human written text...", "AI generated content..."],
    'label': [0, 1]
}
dataset = Dataset.from_dict(data)
dataset.save_to_disk('training_data.arrow')
```

### 2. Configure Training

Edit `config.yaml` to set your dataset path and adjust hyperparameters if needed:

```yaml
dataset:
  source: "local"
  path: "training_data.arrow"  # Your dataset path
  text_column: "text"
  label_column: "label"
```

### 3. Run Training

```bash
python main.py --config config.yaml
```

That's it! The pipeline will:
1. Split your texts into sentences with multiple window sizes
2. Train 3 DeBERTa models (6-12 hours on GPU)
3. Extract stylometric and prediction features (3-4 hours)
4. Train XGBoost meta-classifier with Optuna optimization (2-4 hours)
5. Generate comprehensive evaluation report

**Total time:** ~12-18 hours on single GPU

## Usage Examples

### Basic Training
```bash
# Train from local dataset
python main.py --config config.yaml

# Train from HuggingFace dataset
python main.py --config config.yaml --dataset-id NeksoN/my_dataset
```

### Resume Interrupted Training
```bash
# Resume from last checkpoint
python main.py --config config.yaml --resume
```

### Advanced Options
```bash
# Skip DeBERTa training (use existing models)
python main.py --config config.yaml --skip-deberta

# Only train XGBoost (requires existing DeBERTa models and features)
python main.py --config config.yaml --only-xgboost

# Custom output directory
python main.py --config config.yaml --output-dir ./my_models

# Validate configuration without training
python main.py --config config.yaml --dry-run
```

## Configuration Guide

### Data Splitting

```yaml
data_split:
  deberta_train: 0.80    # 80% for DeBERTa training
  deberta_val: 0.10      # 10% for DeBERTa validation
  test: 0.10             # 10% held out for final evaluation
  stratify: true         # Stratify splits by label
```

### DeBERTa Training

```yaml
deberta:
  base_model: "microsoft/mdeberta-v3-base"
  training:
    learning_rate: 5.0e-5
    per_device_train_batch_size: 32
    num_train_epochs: 2
    weight_decay: 0.04
    fp16: true
```

### Feature Extraction

```yaml
feature_extraction:
  sample_from_train: 80000      # Sample 80k from training set
  use_deberta_val: true          # Add validation set
  enable_stylometric: true       # Extract stylometric features
  enable_perplexity: true        # Calculate perplexity
  enable_predictions: true       # Get DeBERTa predictions
```

### XGBoost Training

```yaml
xgboost:
  augmentation:
    enabled: true
    drop_rate: 0.30              # Stochastic expert dropout
    bias_weights: [1, 1, 2]      # Bias towards window_5
  
  optuna:
    n_trials: 30                 # Hyperparameter optimization trials
```

## Output Structure

After training, you'll find:

```
trained_models/
├── deberta_s1/              # Window size 1 model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── metrics.json
├── deberta_s3/              # Window size 3 model
├── deberta_s5/              # Window size 5 model
├── xgboost_meta/            # Meta-classifier
│   ├── xgboost_model.json
│   ├── feature_columns.json
│   ├── model_info.json
│   └── metrics.json
└── training_report.json     # Comprehensive report
```

## Pipeline Architecture

### Phase 1: Data Preparation (~10 min)
- Load raw dataset
- Split texts into sentences
- Create multiple context windows (1, 3, 5 sentences)
- Split into train/val/test sets

### Phase 2: DeBERTa Training (~8-14 hours)
- Train 3 separate DeBERTa models
- Each specializes in different context window size
- Early stopping based on validation F1 score

### Phase 3: Feature Extraction (~4-6 hours)
- Sample 80k sentences from training + 50k from validation
- Extract stylometric features (POS, perplexity, lexical richness)
- Get predictions from all 3 DeBERTa models
- Process in chunks with automatic resumption

### Phase 4: XGBoost Training (~1 hours)
- Apply stochastic expert dropout augmentation
- Optimize hyperparameters with Optuna (30 trials)
- Train final ensemble classifier
- Compute feature importance

### Phase 5: Final Evaluation (~10 min)
- Test on held-out test set
- Generate comprehensive metrics
- Analyze model fragility
- Save detailed report

## Troubleshooting

### Out of Memory (OOM) Errors

Reduce batch sizes in `config.yaml`:
```yaml
deberta:
  training:
    per_device_train_batch_size: 16  # Reduce from 32
    per_device_eval_batch_size: 16

feature_extraction:
  batch_size: 50  # Reduce from 100
```

### Training Too Slow

Check GPU is being used:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Feature Extraction Interrupted

The pipeline automatically saves progress in chunks. Simply re-run:
```bash
python main.py --config config.yaml --resume
```

## Advanced: Using Trained Models

### Load and Use DeBERTa Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "trained_models/deberta_s3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

text = "Your text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model(**inputs)
prob_ai = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
print(f"AI probability: {prob_ai:.4f}")
```

### Load and Use XGBoost Model

```python
import xgboost as xgb
import json
import numpy as np

# Load model and feature list
model = xgb.Booster()
model.load_model("trained_models/xgboost_meta/xgboost_model.json")

with open("trained_models/xgboost_meta/feature_columns.json") as f:
    feature_cols = json.load(f)

# Prepare features (you need to extract them first)
# X should be a numpy array with shape (1, len(feature_cols))
X = np.array([...])  # Your features here

# Predict
dmatrix = xgb.DMatrix(X)
prob_ai = model.predict(dmatrix)[0]
print(f"AI probability: {prob_ai:.4f}")
```

## Performance Expectations

### Dataset Size Impact

| Original Texts | Sentences | DeBERTa Train Time | Feature Extract Time |
|----------------|-----------|-------------------|---------------------|
| 10k            | ~100k     | ~2-3 hours        | ~1 hour             |
| 50k            | ~500k     | ~6-8 hours        | ~3 hours            |
| 100k           | ~1M       | ~12-16 hours      | ~6 hours            |

### Expected Metrics

With good quality training data, expect:
- **DeBERTa models**: F1 ~0.92-0.96
- **XGBoost ensemble**: F1 ~0.94-0.98

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{ai_detector_training_pipeline,
  title = {AI Detector Training Pipeline},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/ai_detector}
}
```

## License

MIT License

## Support

For issues and questions:
- Open an issue on GitHub
- Check logs in `logs/training.log`
- Review checkpoint status in `checkpoints/status.json`

## Acknowledgments

- Microsoft DeBERTa: https://github.com/microsoft/DeBERTa
- XGBoost: https://github.com/dmlc/xgboost
- HuggingFace Transformers: https://github.com/huggingface/transformers
