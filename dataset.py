import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle


class SequentialRecDataset(Dataset):
    """
    PyTorch Dataset for sequential recommendation.
    For MovieLens 1M, context vector for each (user, item) is:
    [hour/23.0, day_of_week/6.0, normalized_timestamp, title_embedding (384), genre_multi_hot (21), rating]
    """

    def __init__(
        self,
        user_train,
        user_features,
        itemnum,
        cxtdict,
        cxtsize,
        maxlen,
        item_features,
        itemid2idx=None,
    ):
        self.user_train = user_train
        self.user_features = user_features
        self.itemnum = itemnum
        self.cxtdict = cxtdict
        self.cxtsize = cxtsize
        self.maxlen = maxlen
        self.users = list(user_train.keys())
        self.item_features = item_features
        self.itemid2idx = itemid2idx
        # Create user ID to index mapping (for 1-based user IDs)
        self.userid2idx = {
            uid: idx for idx, uid in enumerate(range(1, len(user_features) + 1))
        }
        # Cache each user's set of seen items for efficient negative sampling
        self.user_seen_items = {u: set(seq) for u, seq in user_train.items()}
        # Shared zero context for padding
        self.zero_cxt = np.zeros(self.cxtsize, dtype=np.float32)

    def _map_userid(self, userid):
        return self.userid2idx.get(userid, 0)

    def __len__(self):
        return len(self.users)

    def _map_itemid(self, itemid):
        if self.itemid2idx is not None:
            return self.itemid2idx.get(itemid, 0)
        # For Amazon datasets, assume itemid is 1-based and convert to 0-based
        # If your preprocessing already makes them 0-based, remove the -1
        return itemid - 1

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        seqcxt = np.zeros([self.maxlen, self.cxtsize], dtype=np.float32)
        poscxt = np.zeros([self.maxlen, self.cxtsize], dtype=np.float32)
        negcxt = np.zeros([self.maxlen, self.cxtsize], dtype=np.float32)
        user_seq = self.user_train[user]
        ts = self.user_seen_items[user]
        nxt = user_seq[-1]
        idx_ = self.maxlen - 1
        for i in reversed(user_seq[:-1]):
            seq[idx_] = self._map_itemid(i)
            pos[idx_] = self._map_itemid(nxt)
            # Negative sampling with max attempts
            max_attempts = 100
            for _ in range(max_attempts):
                neg_i_raw = np.random.randint(1, self.itemnum + 1)
                if neg_i_raw not in ts:
                    break
            else:
                neg_i_raw = None  # use None to signal padding
            if neg_i_raw is None:
                neg_idx = 0  # pad index for both ML and Amazon
                neg_item_for_cxt = 0  # use 0 for context key, will default to zero_cxt
            else:
                neg_idx = self._map_itemid(neg_i_raw)
                neg_item_for_cxt = neg_i_raw
            neg[idx_] = neg_idx
            seqcxt[idx_] = self.cxtdict.get((user, i), self.zero_cxt)
            poscxt[idx_] = self.cxtdict.get((user, nxt), self.zero_cxt)
            negcxt[idx_] = self.cxtdict.get((user, neg_item_for_cxt), self.zero_cxt)
            nxt = i
            idx_ -= 1
            if idx_ == -1:
                break
        user_feat = (
            self.user_features[self._map_userid(user)]
            if len(self.user_features) > 0
            else np.zeros(self.user_features.shape[1], dtype=np.float32)
        )
        seq_feat = self.item_features[seq]
        pos_feat = self.item_features[pos]
        neg_feat = self.item_features[neg]
        return {
            "user": user,
            "user_feat": torch.tensor(user_feat, dtype=torch.float32),
            "seq": torch.tensor(seq, dtype=torch.long),
            "seq_feat": torch.tensor(seq_feat, dtype=torch.float32),
            "seqcxt": torch.tensor(seqcxt, dtype=torch.float32),
            "pos": torch.tensor(pos, dtype=torch.long),
            "pos_feat": torch.tensor(pos_feat, dtype=torch.float32),
            "poscxt": torch.tensor(poscxt, dtype=torch.float32),
            "neg": torch.tensor(neg, dtype=torch.long),
            "neg_feat": torch.tensor(neg_feat, dtype=torch.float32),
            "negcxt": torch.tensor(negcxt, dtype=torch.float32),
        }


def get_dataloader(
    user_train,
    user_features,
    itemnum,
    cxtdict,
    cxtsize,
    maxlen,
    batch_size,
    item_features,
    itemid2idx=None,
    shuffle=True,
    num_workers=0,
):
    dataset = SequentialRecDataset(
        user_train,
        user_features,
        itemnum,
        cxtdict,
        cxtsize,
        maxlen,
        item_features,
        itemid2idx,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def load_dataset(dataset_name, maxlen=50, data_dir=None, cxt_size=6):
    """
    Unified loader for CARCA datasets.
    dataset_name: 'ml-1m', 'Beauty', 'Men', 'Fashion', 'Video_Games'
    maxlen: sequence length
    data_dir: path to preprocessed data (for ml-1m)
    cxt_size: context feature size for Amazon datasets
    Returns: user_train, user_features, itemnum, cxtdict, cxtsize, maxlen, item_features, usernum
    """

    if dataset_name == "ml-1m":
        if data_dir is None:
            data_dir = "Data/movielens_preprocessed"
        with open(os.path.join(data_dir, "user_train.pkl"), "rb") as f:
            user_train = pickle.load(f)
        user_features = np.load(os.path.join(data_dir, "user_features.npy"))
        item_features = np.load(os.path.join(data_dir, "item_features.npy"))
        with open(os.path.join(data_dir, "cxtdict.pkl"), "rb") as f:
            cxtdict = pickle.load(f)
        with open(os.path.join(data_dir, "cxtsize.txt")) as f:
            cxtsize = int(f.read().strip())
        with open(os.path.join(data_dir, "itemid2idx.pkl"), "rb") as f:
            itemid2idx = pickle.load(f)
        itemnum = item_features.shape[0]
        usernum = user_features.shape[0]
        return (
            user_train,
            user_features,
            itemnum,
            cxtdict,
            cxtsize,
            maxlen,
            item_features,
            usernum,
            itemid2idx,
        )
    elif dataset_name == "Beauty":
        from data_utils import (
            data_partition,
            get_ItemDataBeauty,
            get_UserDataBeauty,
            load_data,
        )

        dataset = data_partition("Beauty")
        user_train, user_valid, user_test, usernum, itemnum = dataset
        item_features = get_ItemDataBeauty(itemnum)
        user_features = get_UserDataBeauty(usernum)
        cxtdict = load_data("./Data/CXTDictSasRec_Beauty.dat")
        cxtsize = cxt_size
        return (
            user_train,
            user_features,
            itemnum,
            cxtdict,
            cxtsize,
            maxlen,
            item_features,
            usernum,
            None,
        )
    elif dataset_name == "Men":
        from data_utils import (
            data_partition,
            get_ItemDataMen,
            get_UserDataMen,
            load_data,
        )

        dataset = data_partition("Men")
        user_train, user_valid, user_test, usernum, itemnum = dataset
        item_features = get_ItemDataMen(itemnum)
        user_features = get_UserDataMen(usernum)
        cxtdict = load_data("./Data/CXTDictSasRec_Men.dat")
        cxtsize = cxt_size
        return (
            user_train,
            user_features,
            itemnum,
            cxtdict,
            cxtsize,
            maxlen,
            item_features,
            usernum,
            None,
        )
    elif dataset_name == "Fashion":
        from data_utils import (
            data_partition,
            get_ItemDataFashion,
            get_UserDataFashion,
            load_data,
        )

        dataset = data_partition("Fashion")
        user_train, user_valid, user_test, usernum, itemnum = dataset
        item_features = get_ItemDataFashion(itemnum)
        user_features = get_UserDataFashion(usernum)
        cxtdict = load_data("./Data/CXTDictSasRec_Fashion.dat")
        cxtsize = cxt_size
        return (
            user_train,
            user_features,
            itemnum,
            cxtdict,
            cxtsize,
            maxlen,
            item_features,
            usernum,
            None,
        )
    elif dataset_name == "Video_Games":
        from data_utils import (
            data_partition,
            get_ItemDataGames,
            get_UserDataGames,
            load_data,
        )

        dataset = data_partition("Video_Games")
        user_train, user_valid, user_test, usernum, itemnum = dataset
        item_features = get_ItemDataGames(itemnum)
        user_features = get_UserDataGames(usernum)
        cxtdict = load_data("./Data/CXTDictSasRec_Games.dat")
        cxtsize = cxt_size
        return (
            user_train,
            user_features,
            itemnum,
            cxtdict,
            cxtsize,
            maxlen,
            item_features,
            usernum,
            None,
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))
