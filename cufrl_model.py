import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MutualInformationEstimationNetwork(nn.Module):
    def __init__(self, user_feature_dim, sensitive_attribute_dim, hidden_dim, representation_dim):
        super().__init__()
        # Encoder: Process user_data (x) and sensitive_attributes (C^U)
        self.encoder_user = nn.Linear(user_feature_dim, hidden_dim)
        self.encoder_sensitive = nn.Linear(sensitive_attribute_dim, hidden_dim)
        self.encoder_combined = nn.Linear(hidden_dim * 2, representation_dim) # Output Z

        # Decoder: Reconstruct sensitive_attributes from Z
        self.decoder = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sensitive_attribute_dim)
        )
        
        # For fairness objectives (predicting sensitive attributes from Z)
        self.fairness_predictor = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sensitive_attribute_dim)
        )

    def forward(self, user_data, sensitive_attributes):
        # Encode
        user_encoded = F.relu(self.encoder_user(user_data))
        sensitive_encoded = F.relu(self.encoder_sensitive(sensitive_attributes))
        combined = torch.cat((user_encoded, sensitive_encoded), dim=-1)
        z = self.encoder_combined(combined) # User representations Z

        # Decode (for reconstruction loss)
        reconstructed_sensitive_attributes = self.decoder(z)
        
        # Fairness prediction (for fairness loss)
        predicted_sensitive_attributes = self.fairness_predictor(z)

        return z, reconstructed_sensitive_attributes, predicted_sensitive_attributes


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


class FinancialContextAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)
        
    def forward(self, query, key_value, mask=None):
        # query: (batch, seq_len, hidden)
        # key_value: (batch, seq_len, hidden)
        q = self.query(query)
        k = self.key(key_value)
        v = self.value(key_value)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class RiskAwareLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.risk_projection = nn.Linear(hidden_size, hidden_size)
        self.risk_gate = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x, risk_profile):
        # x: (batch, seq_len, hidden)
        # risk_profile: (batch, hidden)
        risk_profile_expanded = risk_profile.unsqueeze(1).expand(-1, x.size(1), -1)
        risk_proj = self.risk_projection(risk_profile_expanded)
        gate_input = torch.cat([x, risk_proj], dim=-1)
        gate = torch.sigmoid(self.risk_gate(gate_input))
        output = x * gate + risk_proj * (1 - gate)
        return output


class CUFRLModel(nn.Module):
    def __init__(
        self, usernum, itemnum, args, item_feature_dim, user_feature_dim, cxt_size, 
        financial_feature_dim=0,
        sensitive_attribute_dim=1 # Assuming sensitive_attribute_dim is 1 by default
    ):
        super().__init__()
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen
        self.cxt_size = cxt_size
        self.user_feature_dim = user_feature_dim # Store user_feature_dim

        # Mutual Information Estimation Network
        self.mi_estimation_network = MutualInformationEstimationNetwork(
            user_feature_dim=user_feature_dim,
            sensitive_attribute_dim=sensitive_attribute_dim,
            hidden_dim=args.hidden_units, # Or another hyperparameter
            representation_dim=args.hidden_units # Output Z should match hidden_units for further processing
        )
        
        # User embeddings with financial profile
        # The user_embedding will now be derived from Z (output of MI Estimation Network)
        # self.user_embedding = nn.Linear(user_feature_dim, self.hidden_units) 
        
        self.financial_profile_encoder = nn.Linear(
            financial_feature_dim if financial_feature_dim > 0 else user_feature_dim, 
            self.hidden_units
        )
        
        # Item embeddings
        self.item_embedding = nn.Embedding(
            itemnum + 1, self.hidden_units, padding_idx=0
        )
        
        # Financial context projection
        self.financial_context_proj = nn.Linear(cxt_size, self.hidden_units)
        
        # Item feature projection
        self.item_feat_proj = nn.Linear(
            item_feature_dim, self.hidden_units * 3
        )
        
        # Combined feature projection
        self.context_feat_proj = nn.Linear(
            cxt_size, self.hidden_units * 2
        )
        
        # Embedding combination
        self.emb_comp = nn.Linear(
            self.hidden_units + self.hidden_units * 3 + self.hidden_units * 2, 
            self.hidden_units
        )
        
        # Risk-aware layer
        self.risk_aware_layer = RiskAwareLayer(self.hidden_units)
        
        # Positional encoding and dropout
        self.pos_encoder = PositionalEncoding(self.hidden_units, max_len=self.maxlen)
        self.dropout = nn.Dropout(args.dropout_rate)
        
        # Financial context attention
        self.fin_ctx_attention = FinancialContextAttention(self.hidden_units)
        
        # Transformer layers
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
        
        # Final prediction layer
        self.final_linear = nn.Linear(self.hidden_units, 1)
        
    def forward(
        self,
        user_feat, # This is the original user data (x)
        sensitive_attrs, # Sensitive attributes (C^U)
        seq,
        seq_feat,
        seq_cxt,
        pos,
        pos_feat,
        pos_cxt,
        neg,
        neg_feat,
        neg_cxt,
        financial_features=None,
    ):
        batch_size = seq.size(0)

        # Get user representations (Z) and other outputs from MI Estimation Network
        # user_feat is x, sensitive_attrs is C^U
        user_representations_z, reconstructed_sensitive_attrs, fairness_predicted_sensitive_attrs = \
            self.mi_estimation_network(user_feat, sensitive_attrs)
        
        # user_emb is now user_representations_z
        user_emb = user_representations_z # (batch, hidden_units) 
        
        # If financial features are provided, use them; otherwise, derive from user features
        # This part might need adjustment if financial_profile is also part of Z
        # For now, let's assume financial_profile is still processed separately or can be derived from user_feat
        if financial_features is not None:
            financial_profile = self.financial_profile_encoder(financial_features)
        else:
            # If financial_features are not directly provided, 
            # consider if they should be part of user_feat fed into MI network,
            # or if a separate path is needed. For now, using original user_feat.
            financial_profile = self.financial_profile_encoder(user_feat) 
        
        # Item sequence embeddings
        seq_emb = self.item_embedding(seq)  # (batch, maxlen, hidden_units)
        
        # Project item features and context features separately
        seq_feat_emb = self.item_feat_proj(seq_feat)  # (batch, maxlen, hidden_units*3)
        seq_cxt_emb = self.context_feat_proj(seq_cxt)  # (batch, maxlen, hidden_units*2)
        
        # Combine embeddings
        seq_concat = torch.cat(
            [seq_emb, seq_feat_emb, seq_cxt_emb], dim=-1
        )  # (batch, maxlen, hidden_units + hidden_units*3 + hidden_units*2)
        
        seq_out = self.emb_comp(seq_concat)  # (batch, maxlen, hidden_units)
        
        # Apply risk-aware layer
        seq_out = self.risk_aware_layer(seq_out, financial_profile)
        
        # Positional encoding and dropout
        seq_out = self.pos_encoder(seq_out)
        seq_out = self.dropout(seq_out)
        
        # Apply financial context attention
        fin_ctx = self.financial_context_proj(seq_cxt)
        seq_out, _ = self.fin_ctx_attention(seq_out, fin_ctx)
        
        # Masking for padding
        mask = (seq != 0).unsqueeze(-1).float()
        seq_out = seq_out * mask
        
        # Transformer layers
        for attn in self.attn_layers:
            seq_out = attn(seq_out)
            seq_out = seq_out * mask
        
        # Process positive items
        pos_emb = self.item_embedding(pos)
        pos_feat_emb = self.item_feat_proj(pos_feat)
        pos_cxt_emb = self.context_feat_proj(pos_cxt)
        pos_concat = torch.cat([pos_emb, pos_feat_emb, pos_cxt_emb], dim=-1)
        pos_out = self.emb_comp(pos_concat)
        pos_out = self.risk_aware_layer(pos_out, financial_profile)
        
        # Process negative items
        neg_emb = self.item_embedding(neg)
        neg_feat_emb = self.item_feat_proj(neg_feat)
        neg_cxt_emb = self.context_feat_proj(neg_cxt)
        neg_concat = torch.cat([neg_emb, neg_feat_emb, neg_cxt_emb], dim=-1)
        neg_out = self.emb_comp(neg_concat)
        neg_out = self.risk_aware_layer(neg_out, financial_profile)
        
        # Add user embedding to sequence/item representations
        user_emb_exp = user_emb.unsqueeze(1).expand(-1, self.maxlen, -1)
        seq_out = seq_out + user_emb_exp
        pos_out = pos_out + user_emb_exp
        neg_out = neg_out + user_emb_exp
        
        # Compute logits
        pos_logits = (seq_out * pos_out).sum(-1)
        neg_logits = (seq_out * neg_out).sum(-1)
        
        return pos_logits, neg_logits, reconstructed_sensitive_attrs, fairness_predicted_sensitive_attrs 