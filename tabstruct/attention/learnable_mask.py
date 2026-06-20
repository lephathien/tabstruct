import torch
from torch import nn


class LearnableMaskPredictor(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=12, max_rows=50, max_cols=50, d_model=768):
        super().__init__()
        self.row_embed = nn.Embedding(max_rows, hidden_dim)
        self.col_embed = nn.Embedding(max_cols, hidden_dim)
        self.seg_embed = nn.Embedding(2, hidden_dim)

        self.hidden_proj = nn.Linear(d_model, hidden_dim)

        input_dim = hidden_dim * 8
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.gate = nn.Parameter(torch.tensor(1.0))
        nn.init.constant_(self.scorer[-1].bias, 5.0)

    def forward(self, token_type, hidden_states=None):
        B, L = token_type.shape[0], token_type.shape[1]
        device = token_type.device

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

        scores = self.scorer(pair_feat)
        scores = scores.permute(0, 3, 1, 2)

        if self.training:
            noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-4) + 1e-4)
            gumbel = (scores + noise) / self.gate
        else:
            gumbel = scores / self.gate

        mask_prob = torch.sigmoid(gumbel)

        diversity_loss = -torch.var(mask_prob, dim=1).mean()

        return mask_prob, scores, diversity_loss
