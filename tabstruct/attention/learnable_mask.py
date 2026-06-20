import torch
from torch import nn


class LearnableMaskPredictor(nn.Module):
    def __init__(self, hidden_dim=64, max_rows=50, max_cols=50, d_model=768, num_layers=6):
        super().__init__()
        self.row_embed = nn.Embedding(max_rows, hidden_dim)
        self.col_embed = nn.Embedding(max_cols, hidden_dim)
        self.seg_embed = nn.Embedding(2, hidden_dim)
        self.hidden_proj = nn.Linear(d_model, hidden_dim)

        feat_dim = hidden_dim * 3
        self.token_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.layer_embed = nn.Embedding(num_layers, 8)
        self.bias_predictor = nn.Linear(8, 1)

    def forward(self, token_type, hidden_states=None, layer_idx=0):
        B, L = token_type.shape[0], token_type.shape[1]

        row_ids = token_type[:, :, 2].clamp(0, self.row_embed.num_embeddings - 1)
        col_ids = token_type[:, :, 1].clamp(0, self.col_embed.num_embeddings - 1)
        seg_ids = token_type[:, :, 0].long()

        row = self.row_embed(row_ids)
        col = self.col_embed(col_ids)
        seg = self.seg_embed(seg_ids)
        tok_feat = torch.cat([row, col, seg], dim=-1)

        if hidden_states is not None:
            h_proj = self.hidden_proj(hidden_states)
            tok_feat = torch.cat([tok_feat, h_proj], dim=-1)

        scores = self.token_mlp(tok_feat).squeeze(-1)
        layer_emb = self.layer_embed(torch.tensor(layer_idx, device=scores.device))
        bias = self.bias_predictor(layer_emb).squeeze(-1)

        logits = scores.unsqueeze(2) + scores.unsqueeze(1) + bias
        mask_prob = torch.sigmoid(logits)
        return mask_prob.unsqueeze(1)
