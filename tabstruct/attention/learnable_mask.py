import torch
from torch import nn


class LearnableMaskPredictor(nn.Module):
    def __init__(self, hidden_dim=64, max_rows=50, max_cols=50, d_model=768, num_layers=6):
        super().__init__()
        self.row_embed = nn.Embedding(max_rows, hidden_dim)
        self.col_embed = nn.Embedding(max_cols, hidden_dim)
        self.seg_embed = nn.Embedding(2, hidden_dim)
        self.hidden_proj = nn.Linear(d_model, hidden_dim)
        self.layer_embed = nn.Embedding(num_layers, 8)

        input_dim = hidden_dim * 8 + 8
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.scorer[-1].bias, 5.0)

    def forward(self, token_type, hidden_states=None, layer_idx=0):
        B, L = token_type.shape[0], token_type.shape[1]

        row_ids = token_type[:, :, 2].clamp(0, self.row_embed.num_embeddings - 1)
        col_ids = token_type[:, :, 1].clamp(0, self.col_embed.num_embeddings - 1)
        seg_ids = token_type[:, :, 0].long()

        row = self.row_embed(row_ids)
        col = self.col_embed(col_ids)
        seg = self.seg_embed(seg_ids)

        def expand(x):
            return x.unsqueeze(2).expand(-1, -1, L, -1)

        def expand_j(x):
            return x.unsqueeze(1).expand(-1, L, -1, -1)

        pair_feat = torch.cat([
            expand(row), expand_j(row),
            expand(col), expand_j(col),
            expand(seg), expand_j(seg),
        ], dim=-1)

        if hidden_states is not None:
            h_proj = self.hidden_proj(hidden_states)
            h_feat = torch.cat([expand(h_proj), expand_j(h_proj)], dim=-1)
            pair_feat = torch.cat([pair_feat, h_feat], dim=-1)

        layer_emb = self.layer_embed(torch.tensor(layer_idx, device=pair_feat.device))
        layer_feat = layer_emb.view(1, 1, 1, 8).expand(B, L, L, -1)
        pair_feat = torch.cat([pair_feat, layer_feat], dim=-1)

        scores = self.scorer(pair_feat)
        scores = scores.squeeze(-1).unsqueeze(1)
        mask_prob = torch.sigmoid(scores)

        return mask_prob
