import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class SequentialRecModel(nn.Module):
    def __init__(
        self, usernum, itemnum, args, item_feature_dim, user_feature_dim, cxt_size
    ):
        super().__init__()
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen
        self.cxt_size = cxt_size
        self.user_embedding = nn.Linear(user_feature_dim, self.hidden_units)
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

    def forward(
        self,
        user_feat,
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
        user_emb = self.user_embedding(user_feat)  # (batch, hidden_units)
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
        # Optionally, add user embedding to sequence/item representations
        # (broadcast user_emb to (batch, maxlen, hidden_units) and add)
        user_emb_exp = user_emb.unsqueeze(1).expand(-1, self.maxlen, -1)
        seq_out = seq_out + user_emb_exp
        pos_out = pos_out + user_emb_exp
        neg_out = neg_out + user_emb_exp
        # Compute logits (dot product or MLP)
        pos_logits = (seq_out * pos_out).sum(-1)
        neg_logits = (seq_out * neg_out).sum(-1)
        return pos_logits, neg_logits
