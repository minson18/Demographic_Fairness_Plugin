# CARCA (PyTorch Migration)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/sequential-recommendation-on-amazon-men)](https://paperswithcode.com/sota/sequential-recommendation-on-amazon-men?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-games)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-games?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-fashion)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-fashion?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-beauty)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-beauty?p=carca-context-and-attribute-aware-next-item)

---

## About This Repository

**This codebase is modified from the official [CARCA](https://github.com/ahmedrashed57/CARCA) repository, but has been fully migrated from TensorFlow to PyTorch. All model logic, data processing, and training routines are now implemented using PyTorch 2.x, with support for Python 3.12 and NumPy >2.0.**

Original paper: [Context and Attribute-Aware Sequential Recommendation via Cross-Attention (RecSys 2022)](https://dl.acm.org/doi/10.1145/3523227.3546777)

If you use this code or datasets, please cite the original paper.

---

## Setup

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

1. Download preprocessed data from [Google Drive](https://drive.google.com/drive/folders/1a_u52mIEUA-1WrwsNZZa-aoGJcMmVugs?usp=sharing) or the raw data from [Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/).
2. Place the data files inside the `Data/` folder (which is ignored by git).
3. MovieLens 1M Dataset can be downloaded here [ml-1m](https://grouplens.org/datasets/movielens/1m/).

---

## MovieLens 1M Preprocessing & Usage

### 1. Preprocess the MovieLens 1M dataset

First, ensure you have downloaded the [ml-1m dataset](https://grouplens.org/datasets/movielens/1m/) and placed it in `RawData/ml-1m/`.

Install requirements (if not already):
```bash
pip install -r requirements.txt
```

Run the preprocessing script to generate CARCA-ready files (with title embeddings, genre, and timestamp context features):
```bash
python RawData/preprocess_ml1m.py
```
This will create a `movielens_preprocessed/` directory with all necessary files.

> **Note:** MovieLens 1M uses 1-based user and item IDs (e.g., UserID 1-6040). Internally, these are mapped to 0-based indices for numpy arrays and PyTorch tensors. This mapping is handled automatically in the code, but if you inspect the data, be aware of this conversion.

### 2. Load the preprocessed data in your code

Use the unified loader function in `dataset.py`:

```python
from dataset import load_dataset, get_dataloader

# Load MovieLens 1M data (maxlen is the sequence length you want)
user_train, user_features, itemnum, cxtdict, cxtsize, maxlen, item_features, usernum, itemid2idx = load_dataset('ml-1m', maxlen=50)

# Create a DataLoader
batch_size = 128
loader = get_dataloader(user_train, user_features, itemnum, cxtdict, cxtsize, maxlen, batch_size, item_features, itemid2idx=itemid2idx)

# Iterate over batches
for batch in loader:
    # Your training code here
    pass
```

- The context features for each (user, item) pair include: timestamp (hour, day, normalized), title embedding, and genre multi-hot vector.
- The item features are a concatenation of the title embedding and genre.

---

## Unified Training Pipeline Usage (All Datasets)

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

- **For MovieLens 1M:**
    1. Download the [ml-1m dataset](https://grouplens.org/datasets/movielens/1m/) and place it in `others/ml-1m/`.
    2. Run the preprocessing script:
    ```bash
    python preprocess_ml1m.py
    ```
    This will create a `movielens_preprocessed/` directory with all necessary files.

- **For Amazon Datasets (Beauty, Men, Fashion, Video_Games):**
    1. Download or preprocess the data as described in the original CARCA instructions.
    2. Place the data files in the appropriate `Data/` folder.

### 3. Train the Model

Run the following command, replacing `DATASET_NAME` with one of: `ml-1m`, `Beauty`, `Men`, `Fashion`, `Video_Games`.

```bash
python train.py --dataset DATASET_NAME --maxlen 50 --batch_size 128
```

- All dataset-specific logic is now handled automatically by the unified loader in `dataset.py`.
- You can adjust other hyperparameters as needed (see `train.py` for options).

#### Example: Train on MovieLens 1M
```bash
python train.py --dataset ml-1m --maxlen 50 --batch_size 128
```

#### Example: Train on Amazon Beauty
```bash
python train.py --dataset Beauty --maxlen 75 --batch_size 128
```

---

- The codebase is now fully unified: you do not need to change any code to switch datasets.
- To add new datasets, simply extend the `load_dataset` function in `dataset.py`.

---

## Features
- PyTorch 2.x, Python 3.12, NumPy >2.0 compatible
- User feature support (see `get_UserData*` in `data_utils.py`)
- Transformer-based sequential recommendation
- Context and item features
- Modern, modular code structure

---

## File Structure
- `data_utils.py`: Data loading and feature extraction
- `dataset.py`: PyTorch Dataset/DataLoader
- `model.py`: Model definition (PyTorch)
- `train.py`: Training loop
- `requirements.txt`: Dependencies
- `.gitignore`: Ignores the `Data/` folder

---

## Credits
- This repository is a PyTorch migration of the official [CARCA](https://github.com/ahmedrashed57/CARCA) codebase.
- Original authors: Ahmed Rashed, et al.
- Migration and modernization by [your name or organization].

---

**Note:** This repo is a direct migration from TensorFlow 1.x to PyTorch 2.x, with modern idioms and user feature support. For preprocessing raw Amazon reviews or image features, refer to the original CARCA repo and adapt as needed.
