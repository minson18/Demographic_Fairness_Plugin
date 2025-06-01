# CARCA (PyTorch Migration) - Fintech Recommendation System

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/sequential-recommendation-on-amazon-men)](https://paperswithcode.com/sota/sequential-recommendation-on-amazon-men?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-games)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-games?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-fashion)](https://paperswithcode.com/sota/sequential-recommendation-on-amazon-fashion?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-beauty)](https://paperswithcode.com/sota/sequential-recommendation-on-amazon-beauty?p=carca-context-and-attribute-aware-next-item)

---

## About This Repository

This is a comprehensive PyTorch implementation of the CARCA (Context- and Attribute-aware Recommendation with Contextualized Attention) system for sequential recommendation, extended with advanced fairness-aware training capabilities. Originally migrated from the [CARCA](https://github.com/ahmedrashed57/CARCA) repository, this implementation supports both Amazon datasets and MovieLens 1M with unified data loading, preprocessing, and comprehensive fairness evaluation.

### Key Features
- **Context- and Attribute-aware Sequential Recommendation**: Multi-modal recommendation using user features, item features, and contextual information
- **Fairness-aware Training (CUFRL)**: Universal fairness regularization with controllable demographic parity constraints
- **Multi-GPU Grid Search**: Distributed hyperparameter optimization with automatic experiment management
- **Comprehensive Evaluation**: Standard metrics (NDCG, Hit Rate, MRR) plus fairness metrics (Demographic Parity, Delta NDCG)
- **Unified Data Pipeline**: Support for both Amazon and MovieLens datasets with consistent preprocessing

---

## Setup & Usage

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Prepare Data

#### MovieLens 1M (Recommended for Testing)
**Option A: Download and Preprocess**
1. Download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and place in `RawData/ml-1m/`
2. Run preprocessing:
   ```bash
   python RawData/preprocess_ml1m.py
   ```
   This creates `Data/movielens_preprocessed/` with all necessary files.

**Option B: Use Preprocessed Data**
Download preprocessed data from [Google Drive](https://drive.google.com/drive/folders/1Cy1c3vGwSKgjLT0u-8ERqBVq_Y5bauaM?usp=sharing), unzip `movielens_preprocessed.zip`, and place in `Data/` directory.

#### Amazon Datasets
Download/preprocess as in the original CARCA repository and place in `Data/`.

---

## Training, Testing, and Experiment Management

### **Quick Start with main.py (Recommended)**

`main.py` provides a unified interface for all training and testing operations:

```bash
# Train only
python main.py --mode train --dataset ml-1m --maxlen 100 --batch_size 128 --num_epochs 20

# Test only
python main.py --mode test --dataset ml-1m --maxlen 100 --model_dir saved_models/ml-1m

# Train and test sequentially
python main.py --mode both --dataset ml-1m --maxlen 100 --batch_size 128 --num_epochs 20
```

### **Direct Script Usage**

#### Standard Training
```bash
python train.py --dataset ml-1m --maxlen 100 --batch_size 128 --num_epochs 20 --lr 0.0001 --hidden_units 90
```

#### Fairness-aware Training
Enable universal fairness regularization with the CUFRL approach:
```bash
python train.py --dataset ml-1m --maxlen 100 --batch_size 128 --num_epochs 20 \
    --use_fairness --fairness_lambda 0.1 --sensitive_indices 0 1
```

**Fairness Parameters:**
- `--use_fairness`: Enable fairness regularization
- `--fairness_lambda`: Weight for fairness loss (0.0 = disabled, >0 = enabled)
- `--sensitive_indices`: Indices of sensitive attributes in user features (0=gender, 1=age, 3+=occupation)

#### Testing Models
```bash
# Test with model directory
python test.py --dataset ml-1m --maxlen 100 --model_dir saved_models/ml-1m

# Test with specific model file
python test.py --dataset ml-1m --maxlen 100 --model_path saved_models/ml-1m/best_model.pth
```

### **Fairness Experimentation**

Use `fair_train.py` for systematic fairness experiments:

```bash
# Single fairness experiment
python fair_train.py --lambdas 0.1 --dataset ml-1m --epochs 20

# Compare multiple fairness levels
python fair_train.py --lambdas 0.0 0.1 0.5 1.0 --dataset ml-1m --epochs 20
```

This automatically:
- Trains models with different fairness regularization strengths
- Evaluates both accuracy and fairness metrics
- Generates comparative analysis and summary tables
- Saves results with timestamps for reproducibility

### **Hyperparameter Optimization**

#### Standard Grid Search
```bash
python grid_search.py
```

#### Multi-GPU Grid Search
For faster hyperparameter tuning with multiple GPUs:
```bash
python grid_search_multi_gpu.py
```

**Grid Search Features:**
- Automatic experiment timestamping and organization
- Best model retraining and testing
- Comprehensive results saving (JSON + CSV)
- All models and metrics preserved for analysis
- Automatic test evaluation on best hyperparameters

**Grid Search Results Structure:**
```
saved_models/gridsearch_YYYYMMDD_HHMMSS/
├── grid_0/, grid_1/, ..., grid_N/  # Individual parameter combinations
├── best_retrain/                   # Best model retrained and tested
├── grid_search_results.json        # Detailed results
└── grid_search_results.csv         # Summary table
```

---

## Evaluation Metrics

### Standard Recommendation Metrics
- **NDCG@k**: Normalized Discounted Cumulative Gain (k=1,5,10,20)
- **Hit@k**: Hit Rate - fraction of users with ≥1 correct item in top-k
- **MRR@k**: Mean Reciprocal Rank - average reciprocal rank of first correct item

### Fairness Metrics
- **Distance (Demographic Parity)**: Measures recommendation differences when swapping sensitive attributes
  - **Distance_gender**: Binary gender swap evaluation
  - **Distance_age**: Random age group swap evaluation (averaged over multiple runs)
  - **Distance_occupation**: Random occupation swap evaluation (averaged over multiple runs)
- **Delta NDCG@k**: NDCG difference between original and swapped attribute predictions

### Example Evaluation Output
```
Validation metrics:
  |   k   | NDCG  | Hit   |  MRR  |
  |-------|-------|-------|-------|
  | 1     | 0.0124 | 0.0100 | 0.0100 |
  | 5     | 0.0136 | 0.0300 | 0.0150 |
  | 10    | 0.0150 | 0.0500 | 0.0170 |
  | 20    | 0.0151 | 0.0800 | 0.0180 |
  |-------|-------|-------|-------|
  Distance (gender): 0.1029
  Distance (age): 0.7267
  Distance (occupation): 0.2154
  Delta NDCG (gender)@20: 0.0112
  Delta NDCG (age)@20: 0.0098
  Delta NDCG (occupation)@20: 0.0123
```

---

## Model Architecture & Features

### CARCA Model Components
- **Multi-modal Embeddings**: User features, item features, and contextual information
- **Positional Encoding**: Sequence position awareness
- **Transformer Encoder**: Multi-head attention with residual connections
- **Context-aware Attention**: Item-sequence attention incorporating contextual factors

### Fairness Extensions (CUFRL)
- **Sensitive Attribute Discriminators**: Neural networks estimating mutual information between representations and sensitive attributes
- **Universal Fairness Loss**: Adversarial training to reduce demographic disparities
- **Flexible Attribute Support**: Configurable sensitive attributes (gender, age, occupation)
- **Model Selection Integration**: Option to incorporate fairness into model selection criteria

---

## Data Format & Structure

### MovieLens 1M Preprocessed Format
After preprocessing, `Data/movielens_preprocessed/` contains:

**Core Data Files:**
- `user_train.pkl`: User interaction sequences (Dict[user_id → List[item_ids]] sorted by timestamp)
- `user_train_split.pkl`: Training sequences (excludes last 2 items per user)
- `user_valid.pkl`: Validation targets (second-to-last item per user)
- `user_test.pkl`: Test targets (last item per user)

**Feature Files:**
- `user_features.npy`: `(num_users, 3+num_occupations)` - `[gender, normalized_age, zip_hash, one_hot_occupation]`
- `item_features.npy`: `(num_items, 384+num_genres)` - `[title_embedding, genre_multi_hot]`
- `item2title_emb.npy`: `(num_items, 384)` - L2-normalized title embeddings

**Context & Mapping Files:**
- `cxtdict.pkl`: Context vectors - Dict[(user_id, item_id) → [timestamp_features, normalized_rating]]
- `ratings_matrix.npy`: `(num_users, num_items)` - Full rating matrix
- `itemid2idx.pkl`: MovieLens item ID → matrix index mapping
- `userid2idx.pkl`: MovieLens user ID → matrix index mapping
- `cxtsize.txt`: Context vector dimensionality
- `genre_map.json`: Genre name → index mapping

**Important Notes:**
- MovieLens uses 1-based IDs, mapped to 0-based indices internally
- Missing contexts default to zero vectors
- Negative sampling handles users who've seen all items gracefully

### Usage Example: Accessing Rating Data
```python
import numpy as np, pickle

# Load data
ratings_matrix = np.load('Data/movielens_preprocessed/ratings_matrix.npy')
with open('Data/movielens_preprocessed/userid2idx.pkl', 'rb') as f:
    userid2idx = pickle.load(f)
with open('Data/movielens_preprocessed/itemid2idx.pkl', 'rb') as f:
    itemid2idx = pickle.load(f)

# Get rating for user 123, item 4567
user_idx = userid2idx[123]
item_idx = itemid2idx[4567]
rating = ratings_matrix[user_idx, item_idx]
print(f'User 123 rated item 4567: {rating}')
```

---

## File Structure & Components

### Core Implementation
- `model.py`: CARCA model with Transformer architecture + fairness discriminators
- `data_utils.py`: Data loading and feature extraction utilities
- `dataset.py`: PyTorch Dataset/DataLoader with negative sampling
- `train.py`: Training loop with fairness regularization support
- `test.py`: Comprehensive evaluation with fairness metrics
- `evaluate.py`: Efficient batch evaluation with demographic parity testing
- `metrics.py`: Implementation of all recommendation and fairness metrics

### Experiment Management
- `main.py`: Unified controller for training/testing workflows
- `grid_search.py`: Hyperparameter optimization with experiment tracking
- `grid_search_multi_gpu.py`: Multi-GPU distributed grid search
- `fair_train.py`: Fairness experimentation and comparison utilities

### Data Processing
- `RawData/preprocess_ml1m.py`: MovieLens 1M preprocessing pipeline
- `RawData/DataProcessing.py`: General data processing utilities

---

## Advanced Features

### Efficient Evaluation Pipeline
- **Batch Processing**: Efficient user-batch and candidate-chunk processing
- **Context Caching**: Per-user context caching for repeated evaluations
- **Memory Management**: Automatic GPU memory management during evaluation
- **Full Score Matrix**: All metrics computed using complete candidate scoring (not just top-k)

### Experiment Reproducibility
- **Timestamped Directories**: All experiments automatically timestamped
- **Complete Artifact Saving**: Models, metrics, configurations, and logs preserved
- **JSON + CSV Output**: Results in both structured (JSON) and tabular (CSV) formats
- **Grid Search Provenance**: Full hyperparameter exploration history maintained

### Fairness Implementation Details
- **Multi-attribute Support**: Simultaneous fairness across gender, age, and occupation
- **Robust Evaluation**: Multiple random swaps for categorical attributes (age, occupation)
- **Configurable Regularization**: Adjustable fairness-accuracy trade-offs
- **Adversarial Training**: Discriminator networks minimize mutual information with sensitive attributes

---

## Model Selection & Best Practices

### Standard Training
- Models selected based on validation NDCG@20
- Early stopping at 50 epochs with evaluation every 10 epochs
- Best model automatically saved and used for final testing

### Fairness-aware Training
- Option to incorporate fairness into model selection criteria
- Combined score: `NDCG@20 - λ × fairness_score`
- Balances recommendation accuracy with demographic parity
- Configurable trade-off via `fairness_lambda` parameter

### Recommended Hyperparameters
Based on grid search results for MovieLens 1M:
```bash
python train.py --dataset ml-1m --maxlen 100 --batch_size 128 --lr 0.0001 \
    --hidden_units 90 --num_blocks 3 --dropout_rate 0.5 --num_heads 1
```

---

## Citations & Credits

### Original Work
- PyTorch migration of [CARCA](https://github.com/ahmedrashed57/CARCA) by Ahmed Rashed et al.
- CARCA: Context- and Attribute-aware Next-item Recommendation via Cross-attention

### Fairness Extensions
- Universal fairness regularization based on CUFRL (Counterfactual Unfairness in Recommendation via Latent factor models)
- Demographic parity evaluation methodology
- Multi-attribute fairness evaluation framework

### Implementation
- Migration and modernization for PyTorch 2.0+
- Fairness-aware training implementation
- Multi-GPU optimization and experiment management systems

---

## Implementation Notes

### Dataset Handling
- **Robust Negative Sampling**: Handles edge cases where users have seen all items
- **Efficient ID Mapping**: Optimized mapping between original IDs and internal indices  
- **Context Alignment**: Correct temporal alignment of contexts with sequence positions
- **Memory Optimization**: Tensor caching and chunked processing for large-scale evaluation

### Fairness Evaluation Robustness
- **Multiple Random Swaps**: Age and occupation fairness averaged over multiple random attribute swaps
- **Binary vs Categorical**: Optimized evaluation for binary (gender) vs categorical (age, occupation) attributes
- **Full Score Evaluation**: Fairness metrics computed using complete candidate scoring, ensuring accuracy
- **Configurable Sensitive Attributes**: Flexible framework supporting various demographic attributes

### Performance Optimizations
- **Batch Candidate Scoring**: Efficient vectorized scoring of all candidates for multiple users
- **Context Caching**: Per-user context tensors cached to avoid repeated computation
- **GPU Memory Management**: Automatic memory cleanup during evaluation phases
- **Chunked Processing**: Large candidate sets processed in configurable chunks to manage memory

