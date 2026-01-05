import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        logits = self.scorer(x)
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        
        w = torch.softmax(logits, dim=1)
        return torch.sum(x * w, dim=1)

class RMPred_Linear(nn.Module):
    def __init__(
        self,
        d_llm_rna: int,
        c_onehot_rna: int,
        d_pssm_rna: int,
        d_llm_mole: int,
        c_onehot_mole: int,
        d_model_fusion: int = 512,
        dropout: float = 0.2,
        d_model_inner: int = 256,
        fusion_layers: int = 2,
        fusion_heads: int = 4,
        rna_max_len: int = 1024,
        rna_gnn_layers: int = 4,
        rna_gnn_heads: int = 4,
        mole_gnn_layers: int = 4,
        mole_gnn_heads: int = 4,
        mole_num_edge_types: int = 8,
    ):
        super().__init__()

        self.rna_input_dim = d_llm_rna + c_onehot_rna + d_pssm_rna
        self.rna_proj = nn.Sequential(
            nn.Linear(self.rna_input_dim, d_model_fusion),
            nn.LayerNorm(d_model_fusion),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.mole_input_dim = d_llm_mole + c_onehot_mole
        self.mole_proj = nn.Sequential(
            nn.Linear(self.mole_input_dim, d_model_fusion),
            nn.LayerNorm(d_model_fusion),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.rna_pool = AttentionPooling(d_model_fusion)
        self.mole_pool = AttentionPooling(d_model_fusion)

        self.predictor = nn.Sequential(
            nn.Linear(d_model_fusion * 2, 1),  # mu only
        )

    def forward(
        self,
        rna_llm, rna_onehot, rna_edges, rna_pssm, rna_mask,
        mole_llm, mole_onehot, mole_edges, mole_mask,
    ):
        rna_feat = torch.cat([rna_llm, rna_onehot, rna_pssm], dim=-1)
        x_rna = self.rna_proj(rna_feat) 
        mole_feat = torch.cat([mole_llm, mole_onehot], dim=-1)
        x_mole = self.mole_proj(mole_feat)

        pool_rna = self.rna_pool(x_rna, rna_mask)
        pool_mole = self.mole_pool(x_mole, mole_mask)

        combined = torch.cat([pool_rna, pool_mole], dim=1)

        mu = self.predictor(combined).squeeze(-1)
        
        return mu