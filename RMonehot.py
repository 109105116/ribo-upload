import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ctx_norm = nn.LayerNorm(d_model)
        self.cross_gate = nn.Parameter(torch.zeros(1))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _to_kpm(self, mask):
        if mask is None: return None
        return (mask == 0)

    def _apply_query_mask(self, x, mask):
        if mask is None: return x
        return x * mask.unsqueeze(-1)

    def forward(self, x, context, x_mask=None, context_mask=None):
        if x_mask is not None: x_mask = x_mask.float()
        if context_mask is not None: context_mask = context_mask.float()
        
        x_kpm = self._to_kpm(x_mask)
        ctx_kpm = self._to_kpm(context_mask)

        residual = x
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2, key_padding_mask=x_kpm, need_weights=False)
        x = residual + self.dropout(x2)
        x = self._apply_query_mask(x, x_mask)

        residual = x
        q = self.norm2(x)
        kv = self.ctx_norm(context)
        x2, _ = self.cross_attn(query=q, key=kv, value=kv, key_padding_mask=ctx_kpm, need_weights=False)
        x = residual + torch.tanh(self.cross_gate) * self.dropout(x2)
        x = self._apply_query_mask(x, x_mask)

        residual = x
        x2 = self.norm3(x)
        x2 = self.ffn(x2)
        x = residual + self.dropout(x2)
        x = self._apply_query_mask(x, x_mask)

        return x

class CoAttentionStack(nn.Module):
    def __init__(self, d_model: int = 512, n_layers: int = 2, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "rna_block": GatedCrossAttentionBlock(d_model, nhead, dropout),
                "mole_block": GatedCrossAttentionBlock(d_model, nhead, dropout),
            })
            for _ in range(n_layers)
        ])

    def forward(self, rna_feat, mole_feat, rna_mask, mole_mask):
        for layer in self.layers:
            new_rna = layer["rna_block"](rna_feat, mole_feat, rna_mask, mole_mask)
            new_mole = layer["mole_block"](mole_feat, rna_feat, mole_mask, rna_mask)
            rna_feat, mole_feat = new_rna, new_mole
        return rna_feat, mole_feat

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


class RMPred_OnehotOnly(nn.Module):
    def __init__(
        self,
        d_llm_rna: int,
        c_onehot_rna: int,
        d_pssm_rna: int,
        d_llm_mole: int,
        c_onehot_mole: int,
        d_model_inner: int = 256,
        d_model_fusion: int = 512,
        dropout: float = 0.2,
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

        self.rna_embed = nn.Sequential(
            nn.Linear(c_onehot_rna, d_model_fusion),
            nn.LayerNorm(d_model_fusion),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.mole_embed = nn.Sequential(
            nn.Linear(c_onehot_mole, d_model_fusion),
            nn.LayerNorm(d_model_fusion),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.fusion = CoAttentionStack(
            d_model=d_model_fusion,
            n_layers=fusion_layers,
            nhead=fusion_heads,
            dropout=dropout,
        )

        self.rna_pool = AttentionPooling(d_model_fusion)
        self.mole_pool = AttentionPooling(d_model_fusion)

        self.predictor = nn.Sequential(
            nn.Linear(d_model_fusion * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),  # mu only
        )

    def forward(
        self,
        rna_llm, rna_onehot, rna_edges, rna_pssm, rna_mask,
        mole_llm, mole_onehot, mole_edges, mole_mask,
    ):
        x_rna = self.rna_embed(rna_onehot)
        x_mole = self.mole_embed(mole_onehot)
        x_rna_fused, x_mole_fused = self.fusion(x_rna, x_mole, rna_mask, mole_mask)
        pool_rna = self.rna_pool(x_rna_fused, rna_mask)
        pool_mole = self.mole_pool(x_mole_fused, mole_mask)
        
        combined = torch.cat([pool_rna, pool_mole], dim=1)
        mu = self.predictor(combined).squeeze(-1)
        
        return mu