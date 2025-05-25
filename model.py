import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class MINE(nn.Module):
    """
    MINE: Mutual Information Neural Estimation
    使用 Donsker–Varadhan 下界：
      I(X;Y) ≥ E_p[T(x,y)] - log(E_{p(x)p(y)}[e^{T(x,y)}])
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def _net(self, inp):
        h = F.relu(self.fc1(inp))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def forward(self, x, y):
        # x, y: (batch, hidden_units)
        joint = torch.cat([x, y], dim=1)            # (batch, 2*d)
        # 造 marginals: 對 y 隨機打亂
        y_perm = y[torch.randperm(y.size(0))]
        marginal = torch.cat([x, y_perm], dim=1)

        t_joint = self._net(joint)                 # T(x,y)
        t_marginal = self._net(marginal)           # T(x,y')

        # 取最大值作為基底，避免 exp overflow
        M = torch.max(t_marginal)
        exp_term = torch.exp(t_marginal - M)
        mi = torch.mean(t_joint) - (torch.log(torch.mean(exp_term) + 1e-8) + M)
        # mi = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)) + 1e-8)
        return mi


# 假設 sens_feat tensor 裡的欄位順序是 [gender, age, occupation]
SENS_MAPPING = {"gender": 0, "age": 1, "occupation": 2, "zip": 3, "0": 0, "1": 1, "2": 2, "3": 3} #
class CARCA(nn.Module):
    def __init__(
        self, usernum, itemnum, args, item_feature_dim, user_feature_dim, sens_feature_dim, selected_sens: List[str], cxt_size
    ):
        super().__init__()
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen
        self.cxt_size = cxt_size
        self.user_embedding = nn.Linear(user_feature_dim, self.hidden_units)
        # 新增 sensitive embedding
        # 只保留你要的那些敏感屬性 index
        self.sens_indices = [SENS_MAPPING[str(attr)] for attr in selected_sens]
        self.sens_embedding = nn.Linear(len(self.sens_indices), self.hidden_units)
        # 初始化 MINE
        self.mine = MINE(input_dim=self.hidden_units*2, hidden_dim=self.hidden_units)
        self.item_embedding = nn.Embedding(
            itemnum + 1, self.hidden_units, padding_idx=0
        )
        self.item_feat_proj = nn.Linear(
            item_feature_dim + cxt_size, self.hidden_units * 5
        )
        self.emb_comp = nn.Linear(
            self.hidden_units + self.hidden_units * 5, self.hidden_units
        )
        self.pos_encoder = PositionalEncoding(self.hidden_units, max_len=self.maxlen)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.num_blocks = args.num_blocks
        self.attn_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_units,
                    nhead=args.num_heads,
                    dim_feedforward=self.hidden_units * 4,
                    dropout=args.dropout_rate,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.final_linear = nn.Linear(self.hidden_units, 1)
        self.attn = nn.MultiheadAttention(
            self.hidden_units,
            num_heads=args.num_heads,
            batch_first=True,
        )

    def forward(
        self,
        user_feat,
        sens_feat_all,
        seq,
        seq_feat,
        seq_cxt,
        pos,
        pos_feat,
        pos_cxt,
        neg,
        neg_feat,
        neg_cxt,
    ):
        # user_feat: (batch, user_feature_dim)
        # seq: (batch, maxlen)
        # seq_feat: (batch, maxlen, item_feature_dim)
        # seq_cxt: (batch, maxlen, cxt_size)
        # pos/neg: (batch, maxlen)
        # pos_feat/neg_feat: (batch, maxlen, item_feature_dim)
        # pos_cxt/neg_cxt: (batch, maxlen, cxt_size)
        batch_size = seq.size(0)
        # 1) 計算 user embedding
        user_emb = self.user_embedding(user_feat)  # (batch, hidden_units)
        # 2) 計算 sensitive embedding
        # 把 sens_feat_all slice 出來
        sens_feat = sens_feat_all[:, self.sens_indices]
        sens_emb = self.sens_embedding(sens_feat)       # (batch, d)
        # 3) Mutual Information lower‐bound
        mi_estimate = self.mine(user_emb, sens_emb)     # 標量或 (batch,)
        seq_emb = self.item_embedding(seq)  # (batch, maxlen, hidden_units)
        seq_feat_in = torch.cat(
            [seq_feat, seq_cxt], dim=-1
        )  # (batch, maxlen, item_feature_dim + cxt_size)
        seq_feat_emb = self.item_feat_proj(
            seq_feat_in
        )  # (batch, maxlen, hidden_units*5)
        seq_concat = torch.cat(
            [seq_emb, seq_feat_emb], dim=-1
        )  # (batch, maxlen, hidden_units + hidden_units*5)
        seq_out = self.emb_comp(seq_concat)  # (batch, maxlen, hidden_units)
        seq_out = self.pos_encoder(seq_out)
        seq_out = self.dropout(seq_out)
        mask = (seq != 0).unsqueeze(-1).float()
        seq_out = seq_out * mask
        for attn in self.attn_layers:
            seq_out = attn(seq_out)
            seq_out = seq_out * mask
        # Positive and negative item embeddings
        pos_emb = self.item_embedding(pos)
        pos_feat_in = torch.cat([pos_feat, pos_cxt], dim=-1)
        pos_feat_emb = self.item_feat_proj(pos_feat_in)
        pos_concat = torch.cat([pos_emb, pos_feat_emb], dim=-1)
        pos_out = self.emb_comp(pos_concat)
        neg_emb = self.item_embedding(neg)
        neg_feat_in = torch.cat([neg_feat, neg_cxt], dim=-1)
        neg_feat_emb = self.item_feat_proj(neg_feat_in)
        neg_concat = torch.cat([neg_emb, neg_feat_emb], dim=-1)
        neg_out = self.emb_comp(neg_concat)
        user_emb_exp = user_emb.unsqueeze(1).expand(-1, self.maxlen, -1)
        seq_out = seq_out + user_emb_exp
        pos_out = pos_out + user_emb_exp
        neg_out = neg_out + user_emb_exp
        # Use the registered attention layer
        pos_attn_out, _ = self.attn(pos_out, seq_out, seq_out)
        pos_logits = self.final_linear(pos_attn_out).squeeze(-1)
        neg_attn_out, _ = self.attn(neg_out, seq_out, seq_out)
        neg_logits = self.final_linear(neg_attn_out).squeeze(-1)
        return pos_logits, neg_logits, mi_estimate
