import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SequentialRecDataset(Dataset):
    def __init__(
        self,
        user_train,
        user_features,
        itemnum,
        cxtdict,
        cxtsize,
        maxlen,
        item_features,
        sensitive_attributes_map,
    ):
        self.user_train = user_train
        self.user_features = user_features
        self.itemnum = itemnum
        self.cxtdict = cxtdict
        self.cxtsize = cxtsize
        self.maxlen = maxlen
        self.users = list(user_train.keys())
        self.item_features = (
            item_features  # numpy array, shape: (itemnum+1, feature_dim)
        )
        self.sensitive_attributes_map = sensitive_attributes_map

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        seqcxt = np.zeros([self.maxlen, self.cxtsize], dtype=np.float32)
        poscxt = np.zeros([self.maxlen, self.cxtsize], dtype=np.float32)
        negcxt = np.zeros([self.maxlen, self.cxtsize], dtype=np.float32)
        user_seq = self.user_train[user]
        ts = set(user_seq)
        nxt = user_seq[-1]
        idx_ = self.maxlen - 1
        for i in reversed(user_seq[:-1]):
            seq[idx_] = i
            pos[idx_] = nxt
            neg_i = 0
            if nxt != 0:
                neg_i = np.random.randint(1, self.itemnum + 1)
                while neg_i in ts:
                    neg_i = np.random.randint(1, self.itemnum + 1)
                neg[idx_] = neg_i
            seqcxt[idx_] = self.cxtdict.get(
                (user, i), np.zeros(self.cxtsize, dtype=np.float32)
            )
            poscxt[idx_] = self.cxtdict.get(
                (user, nxt), np.zeros(self.cxtsize, dtype=np.float32)
            )
            negcxt[idx_] = self.cxtdict.get(
                (user, neg_i if neg_i != 0 else nxt),
                np.zeros(self.cxtsize, dtype=np.float32)
            )
            nxt = i
            idx_ -= 1
            if idx_ == -1:
                break
        user_feat = (
            self.user_features[user]
            if len(self.user_features) > 0 and user < len(self.user_features)
            else np.zeros(self.user_features.shape[1] if self.user_features.ndim > 1 else self.user_features.shape[0] if self.user_features.ndim ==1 and self.user_features.shape[0]>0 else 1, dtype=np.float32)
        )
        seq_feat = self.item_features[seq]
        pos_feat = self.item_features[pos]
        neg_feat = self.item_features[neg]
        
        sensitive_attribute = self.sensitive_attributes_map.get(user, 0)

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
            "sensitive_attribute": torch.tensor(sensitive_attribute, dtype=torch.long)
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
    sensitive_attributes_map,
    shuffle=True,
    num_workers=0,
):
    dataset = SequentialRecDataset(
        user_train, user_features, itemnum, cxtdict, cxtsize, maxlen, item_features,
        sensitive_attributes_map
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
