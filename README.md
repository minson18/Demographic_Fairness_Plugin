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

## Usage

Train the model on a dataset (e.g., Beauty):

```bash
python train.py --dataset Beauty
```

You can adjust hyperparameters via command-line arguments (see `train.py`).

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
