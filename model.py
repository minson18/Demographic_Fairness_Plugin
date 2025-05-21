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


class MutualInformationNetwork(nn.Module):
    def __init__(self, representation_dim, sensitive_embedding_dim, mi_hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(representation_dim + sensitive_embedding_dim, mi_hidden_dim)
        self.fc2 = nn.Linear(mi_hidden_dim, mi_hidden_dim // 2)
        self.fc3 = nn.Linear(mi_hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, user_representation, sensitive_attribute_embedding):
        """
        估計用戶表示和敏感屬性之間的互信息。
        
        Args:
            user_representation: 用戶表示，形狀為(batch, representation_dim)
            sensitive_attribute_embedding: 敏感屬性嵌入，形狀為(batch, sensitive_embedding_dim)
            
        Returns:
            估計的互信息，形狀為(batch, 1)
        """
        # 連接用戶表示和敏感屬性嵌入
        combined_input = torch.cat([user_representation, sensitive_attribute_embedding], dim=-1)
        
        # 通過多層感知器進行互信息估計
        hidden1 = F.relu(self.fc1(combined_input))
        hidden1 = self.dropout(hidden1)
        hidden2 = F.relu(self.fc2(hidden1))
        output = self.fc3(hidden2)
        
        # 輸出可以視為表示用戶表示和敏感屬性之間的互信息
        # 訓練中我們希望最小化這個值以實現公平性
        return output


class SequentialRecModel(nn.Module):
    def __init__(
        self,
        usernum,
        itemnum,
        args,
        item_feature_dim,
        user_feature_dim,
        cxt_size,
        num_sensitive_categories,
        sensitive_embedding_dim=16,
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

        # Fairness components
        self.num_sensitive_categories = num_sensitive_categories
        self.sensitive_embedding_dim = sensitive_embedding_dim
        self.sensitive_attribute_embedding = nn.Embedding(
            self.num_sensitive_categories,
            self.sensitive_embedding_dim,
            padding_idx=None
        )
        self.mi_network = MutualInformationNetwork(
            self.hidden_units, self.sensitive_embedding_dim
        )

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
        sensitive_attribute,
    ):
        # user_feat: (batch, user_feature_dim)
        # seq: (batch, maxlen)
        # seq_feat: (batch, maxlen, item_feature_dim)
        # seq_cxt: (batch, maxlen, cxt_size)
        # pos/neg: (batch, maxlen)
        # pos_feat/neg_feat: (batch, maxlen, item_feature_dim)
        # pos_cxt/neg_cxt: (batch, maxlen, cxt_size)
        # sensitive_attribute: (batch)

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
        seq_out_candidate = self.emb_comp(seq_concat)  # (batch, maxlen, hidden_units)
        seq_out_candidate = self.pos_encoder(seq_out_candidate)
        seq_out_candidate = self.dropout(seq_out_candidate)
        
        padding_mask = (seq != 0) # (batch, maxlen), True for non-padded items
        seq_out_candidate = seq_out_candidate * padding_mask.unsqueeze(-1).float() # Apply padding mask before transformer

        for attn_layer in self.attn_layers: # Corrected loop variable name
            seq_out_candidate = attn_layer(seq_out_candidate) # Transformer expects (batch, seq_len, features)
            seq_out_candidate = seq_out_candidate * padding_mask.unsqueeze(-1).float() # Ensure padding is zeroed out after each layer

        # seq_out_candidate is now the Z from the diagram (sequence of item representations)
        # To get a user-level Z (Z_user), we average non-padded item representations from seq_out_candidate.
        masked_seq_out = seq_out_candidate * padding_mask.unsqueeze(-1).float()
        summed_seq_out = masked_seq_out.sum(dim=1) # (batch, hidden_units)
        num_non_padded_items = padding_mask.sum(dim=1, keepdim=True).float().clamp(min=1) # (batch, 1)
        user_representation_z = summed_seq_out / num_non_padded_items # (batch, hidden_units)

        # Sensitive attribute processing
        # sensitive_attribute is (batch_size), ensure it's long for embedding
        s_emb = self.sensitive_attribute_embedding(sensitive_attribute.long()) # (batch, sensitive_embedding_dim)

        # Mutual Information Network forward pass
        fairness_output = self.mi_network(user_representation_z, s_emb) # (batch, 1)

        # Positive and negative item embeddings for recommendation task
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
        
        # Logits calculation uses seq_out_candidate (output of transformer for each position)
        # and user_emb (initial user features)
        # The CUFRL diagram shows Z (user representations) going to downstream tasks.
        # Here, seq_out_candidate represents the Z for each item in the sequence.
        # pos_out/neg_out are representations of target items.

        # Let's use the transformer output directly for item prediction, as per original SASRec-like models.
        # The 'user_representation_z' is specifically for the MI network.
        
        # Add user embedding to the sequence output that's used for prediction (seq_out_candidate)
        # and to target item representations (pos_out, neg_out)
        predictive_seq_out = seq_out_candidate + user_emb_exp
        pos_out = pos_out + user_emb_exp
        neg_out = neg_out + user_emb_exp

        pos_logits = (predictive_seq_out * pos_out).sum(-1) # (batch, maxlen)
        neg_logits = (predictive_seq_out * neg_out).sum(-1) # (batch, maxlen)
        
        return pos_logits, neg_logits, fairness_output
