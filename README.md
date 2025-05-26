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

## Training and Testing

You can specify dataset, batch size, and other parameters for both training and testing:

### Training
```bash
python train.py --dataset ml-1m --maxlen 50 --batch_size 128 --num_epochs 50
```
- `--dataset`: Dataset name (e.g., ml-1m, Beauty, Men, Fashion, Video_Games)
- `--maxlen`: Maximum sequence length
- `--batch_size`: Training batch size
- `--num_epochs`: Number of training epochs

### Testing
```bash
python test.py --dataset ml-1m --maxlen 50 --batch_size 32 --model_path saved_models/ml-1m/best_model.pth
```
- `--dataset`: Dataset name
- `--maxlen`: Maximum sequence length
- `--batch_size`: Evaluation batch size
- `--model_path`: Path to the trained model (default: saved_models/{dataset}/best_model.pth)

---

## Evaluation Metrics

After training, the following metrics are reported for validation and test sets:

- **NDCG@k**: Normalized Discounted Cumulative Gain at k (k=1,5,10,20)
- **Hit@k**: Hit Rate at k (fraction of users with at least one correct item in top-k)
- **MRR@k**: Mean Reciprocal Rank at k (average reciprocal rank of the first correct item in top-k)
- **DP_gender**: Fairness metric (difference in recommendations when swapping user gender)
- **DP_age**: Fairness metric (difference in recommendations when swapping user age group)

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
```

---

## MovieLens 1M Preprocessed Data Format

After preprocessing, `Data/movielens_preprocessed/` contains:
- **user_train.pkl**: Dict mapping user ID to list of positively rated item IDs (sorted by timestamp).
- **user_features.npy**: `(num_users, 4)` array: `[gender, age, occupation, zip_hash]`.
- **item_features.npy**: `(num_items, 405)` array: `[title_embedding (384), genre_multi_hot (21)]`.
- **cxtdict.pkl**: Dict mapping `(user_id, item_id)` to context vector `[timestamp_features (3), title_embedding (384), genre_multi_hot (21), rating (1)]`.
- **itemid2idx.pkl**: Dict mapping MovieID to row index in `item_features.npy`.
- **ratings_matrix.npy**: `(num_users, num_items)` array, each entry is the rating (0 if not rated).
- **userid2idx.pkl**: Dict mapping UserID to row index in `ratings_matrix.npy`.
- **cxtsize.txt**: Length of context vector.
- **genre_map.json**: Genre name to index mapping.
- **item2title_emb.npy**: `(num_items, 384)` array of title embeddings.

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
- `RawData/preprocess_ml1m.py`: MovieLens 1M preprocessing
- `requirements.txt`: Dependencies

---

## Credits
- PyTorch migration of [CARCA](https://github.com/ahmedrashed57/CARCA) by Ahmed Rashed et al.
- Migration and modernization by [your name or organization].
