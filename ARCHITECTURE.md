# AI Detector Training Pipeline - Visual Flow

## Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT: training_data.arrow                       │
│                         (50k texts with labels)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA PREPARATION (~10 min)                                     │
│  ────────────────────────────────────────────────────────────────────   │
│  1. Load dataset (data_loader.py)                                        │
│  2. Split into sentences (data_preprocessor.py)                          │
│  3. Create windows (window_1, window_3, window_5)                        │
│  4. Split train/val/test (80/10/10)                                      │
│                                                                           │
│  Output: 500k sentences split into:                                      │
│    ├─ Train: 400k (80%)                                                  │
│    ├─ Val:    50k (10%)                                                  │
│    └─ Test:   50k (10%)                                                  │
│                                                                           │
│  Checkpoint: phase1 ✓                                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: DEBERTA TRAINING (~6-12 hours)                                 │
│  ────────────────────────────────────────────────────────────────────   │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Model 1: DeBERTa S1 (window_1)                              │        │
│  │  ───────────────────────────────────────────────────────     │        │
│  │  • Train on 400k sentences (1-sentence context)              │        │
│  │  • Validate on 50k                                           │        │
│  │  • Early stopping on F1                                      │        │
│  │  • Save: trained_models/deberta_s1/                          │        │
│  │  Checkpoint: phase2_deberta_s1 ✓                             │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Model 2: DeBERTa S3 (window_3)                              │        │
│  │  ───────────────────────────────────────────────────────     │        │
│  │  • Train on 400k sentences (3-sentence context)              │        │
│  │  • Validate on 50k                                           │        │
│  │  • Early stopping on F1                                      │        │
│  │  • Save: trained_models/deberta_s3/                          │        │
│  │  Checkpoint: phase2_deberta_s3 ✓                             │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Model 3: DeBERTa S5 (window_5)                              │        │
│  │  ───────────────────────────────────────────────────────     │        │
│  │  • Train on 400k sentences (5-sentence context)              │        │
│  │  • Validate on 50k                                           │        │
│  │  • Early stopping on F1                                      │        │
│  │  • Save: trained_models/deberta_s5/                          │        │
│  │  Checkpoint: phase2_deberta_s5 ✓                             │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: FEATURE EXTRACTION (~3-4 hours)                                │
│  ────────────────────────────────────────────────────────────────────   │
│                                                                           │
│  Step 1: Sample for XGBoost                                              │
│    ├─ Sample 80k from 400k train (stratified)                            │
│    └─ Add 50k validation set                                             │
│    Total: 130k sentences                                                 │
│                                                                           │
│  Step 2: Extract Features (incremental_processor)                        │
│    For each sentence:                                                    │
│      ├─ Stylometric features (stylometric_extraction)                    │
│      │   ├─ POS tags (PUNCT, X, AUX, PRON, etc.)                         │
│      │   ├─ Perplexity (DistilGPT2)                                      │
│      │   ├─ Lexical richness (TTR, MTLD)                                 │
│      │   └─ ~180 stylometric features                                    │
│      │                                                                    │
│      └─ Model predictions (prediction_service_dataset)                   │
│          ├─ prob_ai_s1 (from DeBERTa S1)                                 │
│          ├─ prob_ai_s3 (from DeBERTa S3)                                 │
│          └─ prob_ai_s5 (from DeBERTa S5)                                 │
│                                                                           │
│  Process in chunks (50k per chunk) with auto-resume                      │
│                                                                           │
│  Output: 130k × ~200 features                                            │
│  Checkpoint: phase3 ✓                                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: XGBOOST TRAINING (~2-4 hours)                                  │
│  ────────────────────────────────────────────────────────────────────   │
│                                                                           │
│  Step 1: Split 130k features                                             │
│    ├─ XGB Train: 110k (85%)                                              │
│    └─ XGB Val:    20k (15%)                                              │
│                                                                           │
│  Step 2: Stochastic Expert Dropout Augmentation                          │
│    ├─ Randomly neutralize DeBERTa predictions (30% samples)              │
│    ├─ Bias towards S5 (weights: 1, 1, 2)                                 │
│    └─ Double dataset: 110k → 220k                                        │
│    Purpose: Prevent over-reliance on DeBERTa                             │
│                                                                           │
│  Step 3: Hyperparameter Optimization (Optuna)                            │
│    ├─ 30 trials                                                          │
│    ├─ Optimize: max_depth, learning_rate, subsample, etc.               │
│    ├─ Objective: Maximize F1 on validation                               │
│    └─ GPU acceleration                                                   │
│                                                                           │
│  Step 4: Train Final Model                                               │
│    ├─ Use best parameters from Optuna                                    │
│    ├─ Train on augmented dataset                                         │
│    └─ Early stopping on validation                                       │
│                                                                           │
│  Step 5: Analysis                                                        │
│    ├─ Feature importance (top 50)                                        │
│    ├─ Fragility scores (expert dropout impact)                           │
│    └─ Validation metrics                                                 │
│                                                                           │
│  Output: trained_models/xgboost_meta/                                    │
│  Checkpoint: phase4 ✓                                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: FINAL EVALUATION (~10 min)                                     │
│  ────────────────────────────────────────────────────────────────────   │
│                                                                           │
│  Evaluate on held-out test set (50k sentences):                          │
│    ├─ DeBERTa S1 performance                                             │
│    ├─ DeBERTa S3 performance                                             │
│    ├─ DeBERTa S5 performance                                             │
│    └─ XGBoost Ensemble performance                                       │
│                                                                           │
│  Metrics computed:                                                       │
│    ├─ Accuracy, Precision, Recall, F1, ROC-AUC                           │
│    ├─ Confusion matrix                                                   │
│    ├─ Per-language breakdown (if multilingual)                           │
│    ├─ Feature importance                                                 │
│    └─ Training time statistics                                           │
│                                                                           │
│  Output: training_report.json                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          FINAL OUTPUT                                     │
│  ────────────────────────────────────────────────────────────────────   │
│                                                                           │
│  trained_models/                                                         │
│  ├─ deberta_s1/          (DeBERTa window_1 specialist)                   │
│  ├─ deberta_s3/          (DeBERTa window_3 specialist)                   │
│  ├─ deberta_s5/          (DeBERTa window_5 specialist)                   │
│  ├─ xgboost_meta/        (Ensemble meta-classifier)                      │
│  └─ training_report.json (Complete metrics & analysis)                   │
│                                                                           │
│  logs/training.log       (Detailed execution log)                        │
│  checkpoints/            (Resumable checkpoints)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────┐
│  Raw Texts   │ 50k texts
└──────┬───────┘
       │
       │ sentence splitting
       ▼
┌──────────────┐
│  Sentences   │ 500k sentences
└──────┬───────┘
       │
       │ 80/10/10 split
       │
       ├─────────────────────┬──────────────────┐
       ▼                     ▼                  ▼
┌──────────────┐      ┌──────────────┐   ┌──────────────┐
│ Train: 400k  │      │  Val: 50k    │   │ Test: 50k    │
└──────┬───────┘      └──────┬───────┘   └──────┬───────┘
       │                     │                   │
       │ DeBERTa training    │                   │
       ▼                     ▼                   │
┌──────────────────────────────────────┐        │
│  3 DeBERTa Models (s1, s3, s5)       │        │
└──────┬───────────────────────────────┘        │
       │                     │                   │
       │ sample 80k          │ use all 50k       │
       │                     │                   │
       └──────────┬──────────┘                   │
                  │                              │
                  │ feature extraction           │
                  ▼                              │
         ┌────────────────┐                      │
         │ 130k features  │                      │
         └────────┬───────┘                      │
                  │                              │
                  │ XGBoost training             │
                  ▼                              │
         ┌────────────────┐                      │
         │ XGBoost Meta   │                      │
         └────────┬───────┘                      │
                  │                              │
                  │ final evaluation             │
                  │                              │
                  └──────────────────────────────┤
                                                 ▼
                                    ┌──────────────────────┐
                                    │  Evaluation Report   │
                                    └──────────────────────┘
```

## Checkpoint Resume Flow

```
Start Training
      │
      ▼
  ┌─────────────────┐
  │ Check Phase 1?  │──── Exists? ──▶ Load & Skip
  └────────┬────────┘
           │ Not exists
           ▼
  ┌─────────────────┐
  │ Execute Phase 1 │
  │ Save Checkpoint │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Check Phase 2?  │──── Exists? ──▶ Load & Skip
  └────────┬────────┘
           │ Not exists
           ▼
  ┌─────────────────┐
  │ Execute Phase 2 │
  │ Save Checkpoint │
  └────────┬────────┘
           │
          ...

  (continues for all phases)
```

## Feature Flow in XGBoost

```
Input Sentence
      │
      ├──────────────────────────────┬────────────────────────┐
      │                              │                        │
      ▼                              ▼                        ▼
┌──────────────┐            ┌──────────────┐       ┌──────────────┐
│ Stylometric  │            │  DeBERTa S1  │       │  DeBERTa S3  │
│  Extraction  │            │ (window_1)   │       │ (window_3)   │
└──────┬───────┘            └──────┬───────┘       └──────┬───────┘
       │                           │                       │
       │ ~180 features             │ prob_ai_s1            │ prob_ai_s3
       │                           │                       │
       └───────────┬───────────────┴───────────────────────┘
                   │                                      │
                   │                          ┌───────────┴──────┐
                   │                          │   DeBERTa S5     │
                   │                          │   (window_5)     │
                   │                          └───────┬──────────┘
                   │                                  │ prob_ai_s5
                   │                                  │
                   └──────────────┬───────────────────┘
                                  │
                                  │ ~200 features total
                                  ▼
                         ┌─────────────────┐
                         │  XGBoost Meta   │
                         │  Classifier     │
                         └────────┬────────┘
                                  │
                                  ▼
                              Prediction
                           (AI probability)
```

This visual documentation helps understand the complete data flow and architecture!
