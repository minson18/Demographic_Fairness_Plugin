# CARCA (PyTorch Migration)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/sequential-recommendation-on-amazon-men)](https://paperswithcode.com/sota/sequential-recommendation-on-amazon-men?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-games)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-games?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-fashion)](https://paperswithcode.com/sota/sequential-recommendation-on-amazon-fashion?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-beauty)](https://paperswithcode.com/sota/sequential-recommendation-on-amazon-beauty?p=carca-context-and-attribute-aware-next-item)

---

## About This Repository

This is a PyTorch migration of the official [CARCA](https://github.com/ahmedrashed57/CARCA) repository for context- and attribute-aware sequential recommendation. It supports both Amazon datasets and MovieLens 1M, with unified data loading and preprocessing.

---

## Setup & Usage

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
- **Amazon datasets:** Download/preprocess as in the original CARCA repo and place in `Data/`.
- **MovieLens 1M:**
  1. **Option A:** Download [ml-1m](https://grouplens.org/datasets/movielens/1m/) and place in `RawData/ml-1m/`, then preprocess:
     ```bash
     python RawData/preprocess_ml1m.py
     ```
     This creates `Data/movielens_preprocessed/` with all necessary files.
  2. **Option B:** Download preprocessed data for MovieLens 1M from [Google Drive](https://drive.google.com/drive/folders/1Cy1c3vGwSKgjLT0u-8ERqBVq_Y5bauaM?usp=sharing). Download the `movielens_preprocessed.zip` file, unzip it, and place the resulting folder inside the `Data/` directory.

---

## Training, Testing, and Experiment Management

The training loss is a weighted sum of three components: BCE Loss, Quantile Loss, and Ranking Quantile Loss. You can control the weights of quantile and ranking quantile loss using the `--alpha` and `--beta` arguments. The loss ratio is `1-alpha-beta : alpha : beta`.

**Example:**
```bash
python main.py --mode train --dataset ml-1m --maxlen 100 --batch_size 128 --num_epochs 20 --alpha 0.05 --beta 0.2
```

### **Recommended: Use `main.py` as a Controller**

`main.py` provides a unified interface to run training, testing, or both:

- **Train:**
  ```bash
  python main.py --mode train --dataset ml-1m --maxlen 100 --batch_size 128 --num_epochs 20
  ```
- **Test:**
  ```bash
  python main.py --mode test --dataset ml-1m --maxlen 100 --model_dir saved_models/ml-1m
  ```
- **Train and Test Sequentially:**
  ```bash
  python main.py --mode both --dataset ml-1m --maxlen 100 --batch_size 128 --num_epochs 20
  ```

All additional arguments are passed to the respective train or test scripts.

### **Direct Usage of train.py and test.py**

- **Training only:**
  ```bash
  python train.py --dataset ml-1m --maxlen 100 --batch_size 128 --num_epochs 20
  ```
- **Testing only:**
  ```bash
  python test.py --dataset ml-1m --maxlen 100 --model_dir saved_models/ml-1m
  ```
  - You can also use `--model_path` to specify a direct path to a model file.

### **Grid Search (Hyperparameter Tuning)**

- **Run grid search:**
  ```bash
  python grid_search.py
  ```
- Each grid search run creates a unique timestamped directory under `saved_models/` (e.g., `saved_models/gridsearch_20240610_153000/`).
- All models, validation metrics, and results for that search are saved in this directory.
- The best hyperparameter combination is automatically retrained, and its test results are saved in a `best_retrain` subdirectory within the same folder.
- Results are saved as both `grid_search_results.json` and `grid_search_results.csv` for easy review.

#### **To test a specific model from grid search:**
```bash
python test.py --model_dir saved_models/gridsearch_YYYYMMDD_HHMMSS/best_retrain --dataset ml-1m
```

---

## Evaluation Metrics

After training, the following metrics are reported for validation and test sets:

- **NDCG@k**: Normalized Discounted Cumulative Gain at k (k=1,5,10,20)
- **Hit@k**: Hit Rate at k (fraction of users with at least one correct item in top-k)
- **MRR@k**: Mean Reciprocal Rank at k (average reciprocal rank of the first correct item in top-k)
- **DP_gender**: Fairness metric (distance and delta NDCG when swapping user gender)
- **DP_age**: Fairness metric (distance and delta NDCG when swapping user age group)
- **DP_occupation**: Fairness metric (distance and delta NDCG when swapping user occupation, if present)

### Example Output
```
Test set metrics:
  |   k   | NDCG  | Hit   |  MRR  |
  |-------|-------|-------|-------|
  | 1     | 0.0124 | 0.0100 | 0.0100 |
  | 5     | 0.0136 | 0.0300 | 0.0150 |
  | 10    | 0.0150 | 0.0500 | 0.0170 |
  | 20    | 0.0151 | 0.0800 | 0.0180 |
  |-------|-------|-------|-------|
  DP_gender: 0.1029
  DP_age:    0.7267
  DP_occupation: 0.2154
  Delta NDCG (gender)@20: 0.0112
  Delta NDCG (age)@20: 0.0098
  Delta NDCG (occupation)@20: 0.0123
```

---

## Experiment Results Structure
- Each training or grid search run saves its results in a unique directory under `saved_models/`.
- For grid search, all models and results for a run are grouped together, and the best retrained model and its test results are in a `best_retrain` subdirectory.
- Validation and test metrics are saved as JSON and CSV for easy review and reproducibility.

---

## MovieLens 1M Preprocessed Data Format

After preprocessing, `Data/movielens_preprocessed/` contains:
- **user_train.pkl**: Dict mapping user ID to list of positively rated item IDs (sorted by timestamp).
- **user_features.npy**: `(num_users, 3 + num_occupations)` array: `[gender, min-max normalized age, zip_hash, one-hot occupation]` (occupation fairness supported if present).
- **item_features.npy**: `(num_items, 384 + num_genres)` array: `[L2-normalized title_embedding (384), genre_multi_hot (num_genres)]`.
- **cxtdict.pkl**: Dict mapping `(user_id, item_id)` to context vector `[timestamp_features (3), min-max normalized rating (1)]`.
- **itemid2idx.pkl**: Dict mapping MovieID to row index in `item_features.npy`.
- **ratings_matrix.npy**: `(num_users, num_items)` array, each entry is the rating (0 if not rated).
- **userid2idx.pkl**: Dict mapping UserID to row index in `ratings_matrix.npy`.
- **cxtsize.txt**: Length of context vector.
- **genre_map.json**: Genre name to index mapping.
- **item2title_emb.npy**: `(num_items, 384)` array of L2-normalized title embeddings.

> **Note:** MovieLens 1M uses 1-based user/item IDs, mapped to 0-based indices internally.

#### Example: Using the Ratings Matrix
```python
import numpy as np, pickle, json

# Load ratings matrix and user mapping
ratings_matrix = np.load('Data/movielens_preprocessed/ratings_matrix.npy')
with open('Data/movielens_preprocessed/userid2idx.pkl', 'rb') as f:
    userid2idx = pickle.load(f)
with open('Data/movielens_preprocessed/itemid2idx.pkl', 'rb') as f:
    itemid2idx = pickle.load(f)

# To get the rating for a MovieLens user and item (e.g., user 123, item 4567):
user_id = 123
item_id = 4567
user_row = userid2idx[user_id]  # Map MovieLens user ID to row
item_col = itemid2idx[item_id]  # Map MovieLens item ID to column
rating = ratings_matrix[user_row, item_col]
print(f'User {user_id} rated item {item_id} as {rating}')

# If you want to get all ratings for a user:
user_ratings = ratings_matrix[user_row, :]
# Or all ratings for an item:
item_ratings = ratings_matrix[:, item_col]
```

---

## File Structure
- `data_utils.py`: Data loading and feature extraction
- `dataset.py`: Unified PyTorch Dataset/DataLoader
- `model.py`: Model definition
- `train.py`: Training loop
- `test.py`: Test set evaluation
- `main.py`: Unified controller for training/testing
- `grid_search.py`: Hyperparameter grid search and experiment management
- `RawData/preprocess_ml1m.py`: MovieLens 1M preprocessing
- `requirements.txt`: Dependencies

---

## Credits
- PyTorch migration of [CARCA](https://github.com/ahmedrashed57/CARCA) by Ahmed Rashed et al.
- Migration and modernization by [your name or organization].

## Implementation Notes

### Dataset Negative Sampling and ID Mapping
- **Negative Sampling:**
  - During training, for each positive interaction, a negative item is sampled that the user has not interacted with. If no valid negative can be found (i.e., the user has interacted with all items), the pad index (`0`) is used for the negative item and its features/contexts.
  - Negative sampling is efficient due to per-user caching of seen items.
- **Item ID Mapping:**
  - For MovieLens, all item IDs are mapped using `itemid2idx`.
  - For Amazon datasets, if no mapping is provided, item IDs are assumed to be 1-based and are converted to 0-based indices internally.
- **Context Padding:**
  - When a context is missing (e.g., for a padded or negative item), a shared zero vector is used for context features.
- The dataset logic is robust and correct for both MovieLens and Amazon domains, with correct sequence truncation, context alignment, and negative sampling.

### Evaluation Pipeline

- All evaluation and fairness metrics (NDCG, Hit, MRR, and demographic parity) are computed using the full score matrix for each user, not just the top-k predictions. This ensures correctness and comparability with standard benchmarks.

