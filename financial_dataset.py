import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FinancialRecDataset(Dataset):
    def __init__(
        self,
        user_train,
        user_features,
        itemnum,
        cxtdict,
        cxtsize,
        maxlen,
        item_features,
        financial_features=None,
    ):
        self.user_train = user_train
        self.user_features = user_features
        self.itemnum = itemnum
        self.cxtdict = cxtdict
        self.cxtsize = cxtsize
        self.maxlen = maxlen
        self.users = list(user_train.keys())
        self.item_features = item_features
        self.financial_features = financial_features

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
        risk_weights = np.ones([self.maxlen], dtype=np.float32)
        
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
            
            # Get context with financial information
            current_ctx = self.cxtdict.get(
                (user, i), np.zeros(self.cxtsize, dtype=np.float32)
            )
            seqcxt[idx_] = current_ctx
            
            next_ctx = self.cxtdict.get(
                (user, nxt), np.zeros(self.cxtsize, dtype=np.float32)
            )
            poscxt[idx_] = next_ctx
            negcxt[idx_] = next_ctx
            
            # If financial context available, calculate risk weight based on financial metrics
            if len(current_ctx) > 0 and len(next_ctx) > 0:
                # Simplified risk weight based on context variance
                # In a real application, this would be based on financial metrics
                ctx_diff = np.abs(current_ctx - next_ctx).mean()
                risk_weights[idx_] = 1.0 + ctx_diff  # Higher difference means higher risk weight
            
            nxt = i
            idx_ -= 1
            if idx_ == -1:
                break
        
        # Get user features
        user_feat = (
            self.user_features[user]
            if len(self.user_features) > 0
            else np.zeros(self.user_features.shape[1], dtype=np.float32)
        )
        
        # Get financial features if available
        financial_feat = (
            self.financial_features[user] 
            if self.financial_features is not None and len(self.financial_features) > 0
            else None
        )
        
        # Get item features
        seq_feat = self.item_features[seq]
        pos_feat = self.item_features[pos]
        neg_feat = self.item_features[neg]
        
        # Prepare return dictionary
        result = {
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
            "risk_weights": torch.tensor(risk_weights, dtype=torch.float32),
        }
        
        # Add financial features if available
        if financial_feat is not None:
            result["financial_features"] = torch.tensor(financial_feat, dtype=torch.float32)
        
        return result


def get_financial_dataloader(
    user_train,
    user_features,
    itemnum,
    cxtdict,
    cxtsize,
    maxlen,
    batch_size,
    item_features,
    financial_features=None,
    shuffle=True,
    num_workers=0,
):
    dataset = FinancialRecDataset(
        user_train, 
        user_features, 
        itemnum, 
        cxtdict, 
        cxtsize, 
        maxlen, 
        item_features,
        financial_features,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


# Helper functions to generate dummy financial data for testing
def get_finance_item_features(itemnum, feature_dim=50):
    """Generate dummy financial item features for testing"""
    # In a real application, these would be loaded from real financial data
    np.random.seed(42)
    item_features = np.random.normal(0, 1, (itemnum + 1, feature_dim))
    item_features[0] = 0  # padding item
    return item_features


def get_finance_user_features(usernum, feature_dim=30):
    """Generate dummy financial user features for testing"""
    # In a real application, these would be loaded from real financial data
    np.random.seed(43)
    user_features = np.random.normal(0, 1, (usernum + 1, feature_dim))
    user_features[0] = 0  # padding user
    return user_features


def get_financial_risk_profiles(usernum, feature_dim=10):
    """Generate dummy financial risk profiles for testing"""
    # In a real application, these would be calculated from real financial data
    np.random.seed(44)
    risk_profiles = np.random.normal(0, 1, (usernum + 1, feature_dim))
    risk_profiles[0] = 0  # padding user
    return risk_profiles 