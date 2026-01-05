
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Literal
import math
import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import TransformerConv, GATv2Conv
    from torch_geometric.nn.norm import GraphNorm
except ImportError as e:
    raise ImportError("This module requires torch_geometric. Install PyG first.") from e


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, L: int, device: torch.device) -> torch.Tensor:
        if L > self.max_len:
            raise ValueError(f"L={L} exceeds max_len={self.max_len}. Increase max_len.")
        return self.pe[:L].to(device)


def build_positional_module(pos_type: str, max_len: int, d_model: int) -> Optional[nn.Module]:
    if pos_type == "none":
        return None
    if pos_type == "sinusoidal":
        return SinusoidalPositionalEncoding(max_len, d_model)
    raise ValueError(f"Unknown pos_type: {pos_type}")


class PSSMResCNN(nn.Module):
    def __init__(self, d_pssm: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(d_pssm, d_model, kernel_size=1)

        def block(dilation: int):
            return nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation, dilation=dilation, groups=1),
                nn.GroupNorm(1, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation, dilation=dilation, groups=1),
                nn.GroupNorm(1, d_model),
            )

        self.blocks = nn.ModuleList([block(1), block(2), block(4)])
        self.out_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pssm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = pssm.transpose(1, 2).contiguous()
        x = self.in_proj(x)

        for blk in self.blocks:
            res = x
            x = blk(x)
            x = F.gelu(x + res)

        x = self.dropout(x)
        x = x.transpose(1, 2).contiguous()
        x = self.out_ln(x)
        x = x * mask.unsqueeze(-1)
        return x


@dataclass
class BatchedGraph:
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    lengths: torch.Tensor
    N: int


def _bucket_distance(dist: torch.Tensor, num_buckets: int = 16, max_dist: int = 1024) -> torch.Tensor:
    dist = dist.clamp(min=1, max=max_dist).float()
    b = torch.floor(torch.log2(dist)).long()
    return b.clamp(min=0, max=num_buckets - 1)


def build_batched_graph_from_edgelist(
    bp_edges_list: List[torch.Tensor],
    mask: torch.Tensor,               
    edge_type_emb: nn.Embedding,       
    dist_emb: nn.Embedding,            
    add_backbone: bool = True,
    undirected: bool = True,
) -> BatchedGraph:
    device = mask.device
    B, L = mask.shape
    lengths = mask.sum(dim=1).long()

    ptr = torch.zeros(B + 1, device=device, dtype=torch.long)
    ptr[1:] = torch.cumsum(lengths, dim=0)
    N = int(ptr[-1].item())

    batch = torch.empty(N, device=device, dtype=torch.long)
    for b in range(B):
        s, e = int(ptr[b].item()), int(ptr[b + 1].item())
        if e > s:
            batch[s:e] = b

    src_all, dst_all, et_all, dist_all = [], [], [], []

    for b in range(B):
        lb = int(lengths[b].item())
        if lb <= 0:
            continue
        offset = int(ptr[b].item())

        edges = bp_edges_list[b]
        if not isinstance(edges, torch.Tensor):
            edges = torch.tensor(edges)
        edges = edges.to(device)

        if edges.numel() > 0:
            u_orig = edges[:, 0].long()
            v_orig = edges[:, 1].long()
            valid = (u_orig >= 0) & (v_orig >= 0) & (u_orig < lb) & (v_orig < lb) & (u_orig != v_orig)
            u = u_orig[valid]
            v = v_orig[valid]
            if u.numel() > 0:
                src = u + offset
                dst = v + offset
                src_all.append(src); dst_all.append(dst)
                et_all.append(torch.ones_like(src))           
                dist_all.append(torch.abs(u - v))

                if undirected:
                    src_all.append(dst); dst_all.append(src)
                    et_all.append(torch.ones_like(dst))
                    dist_all.append(torch.abs(u - v))

        if add_backbone and lb > 1:
            i = torch.arange(lb - 1, device=device, dtype=torch.long)
            j = i + 1
            src = i + offset
            dst = j + offset
            src_all.append(src); dst_all.append(dst)
            et_all.append(torch.zeros_like(src))              
            dist_all.append(torch.ones_like(i))               

            if undirected:
                src_all.append(dst); dst_all.append(src)
                et_all.append(torch.zeros_like(dst))
                dist_all.append(torch.ones_like(i))

    edge_dim = edge_type_emb.embedding_dim
    if len(src_all) == 0:
        edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
        edge_attr = torch.empty((0, edge_dim), device=device, dtype=torch.float32)
        return BatchedGraph(edge_index=edge_index, edge_attr=edge_attr, batch=batch, ptr=ptr, lengths=lengths, N=N)

    src = torch.cat(src_all, dim=0)
    dst = torch.cat(dst_all, dim=0)
    et = torch.cat(et_all, dim=0)
    dist = torch.cat(dist_all, dim=0)

    et_feat = edge_type_emb(et)
    dist_bucket = _bucket_distance(dist, num_buckets=dist_emb.num_embeddings)
    dist_feat = dist_emb(dist_bucket)

    edge_attr = et_feat + dist_feat
    edge_index = torch.stack([src, dst], dim=0)
    return BatchedGraph(edge_index=edge_index, edge_attr=edge_attr, batch=batch, ptr=ptr, lengths=lengths, N=N)


def dense_from_flat(node_x: torch.Tensor, ptr: torch.Tensor, L: int) -> torch.Tensor:
    device = node_x.device
    B = ptr.numel() - 1
    out = torch.zeros((B, L, node_x.size(-1)), device=device, dtype=node_x.dtype)
    for b in range(B):
        s = int(ptr[b].item())
        e = int(ptr[b + 1].item())
        lb = e - s
        if lb > 0:
            out[b, :lb] = node_x[s:e]
    return out


class RNAGraphEncoder(nn.Module):
    def __init__(
        self,
        c_onehot: int,
        d_model: int = 256,
        edge_dim: int = 64,
        gnn_type: Literal["transformer", "gatv2"] = "transformer",
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        dist_buckets: int = 16,
        add_backbone_edges: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.add_backbone_edges = add_backbone_edges

        self.node_proj = nn.Linear(c_onehot, d_model)

        self.edge_type_emb = nn.Embedding(2, edge_dim)          
        self.dist_emb = nn.Embedding(dist_buckets, edge_dim)    

        convs, gnorms = [], []
        for _ in range(layers):
            if gnn_type == "transformer":
                conv = TransformerConv(
                    in_channels=d_model,
                    out_channels=d_model,
                    heads=heads,
                    concat=False,
                    beta=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            elif gnn_type == "gatv2":
                conv = GATv2Conv(
                    in_channels=d_model,
                    out_channels=d_model,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    share_weights=True,
                )
            else:
                raise ValueError(f"Unknown gnn_type={gnn_type}")

            convs.append(conv)
            gnorms.append(GraphNorm(d_model))

        self.convs = nn.ModuleList(convs)
        self.gn = nn.ModuleList(gnorms)
        self.drop = nn.Dropout(dropout)

    def forward(self, onehot: torch.Tensor, bp_edges_list: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        device = onehot.device
        B, L, _ = onehot.shape
        lengths = mask.sum(dim=1).long()

        g = build_batched_graph_from_edgelist(
            bp_edges_list=bp_edges_list,
            mask=mask,
            edge_type_emb=self.edge_type_emb,
            dist_emb=self.dist_emb,
            add_backbone=self.add_backbone_edges,
            undirected=True,
        )

        nodes = []
        for b in range(B):
            lb = int(lengths[b].item())
            if lb > 0:
                nodes.append(onehot[b, :lb].float())
        node_feat = torch.cat(nodes, dim=0) if len(nodes) else torch.zeros((0, onehot.size(-1)), device=device)

        node_x = self.node_proj(node_feat)

        for conv, gn in zip(self.convs, self.gn):
            res = node_x
            if g.edge_index.numel() > 0 and node_x.numel() > 0:
                node_x = conv(node_x, g.edge_index, g.edge_attr)
            node_x = F.gelu(node_x)
            node_x = self.drop(node_x)
            node_x = node_x + res
            if node_x.numel() > 0:
                node_x = gn(node_x, g.batch)

        x2 = dense_from_flat(node_x, g.ptr, L)
        x2 = x2 * mask.unsqueeze(-1)
        return x2


class RNASeqEncoder(nn.Module):
    def __init__(
        self,
        d_llm: int,
        c_onehot: int,
        d_model: int = 256,
        max_len: int = 1024,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        pos_type: Literal["none", "sinusoidal"] = "sinusoidal",
    ):
        super().__init__()
        self.llm_proj = nn.Linear(d_llm, d_model)
        self.oh_proj = nn.Linear(c_onehot, d_model)
        self.pos = build_positional_module(pos_type, max_len=max_len, d_model=d_model)
        self.in_ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, llm_embed: torch.Tensor, onehot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device = llm_embed.device
        B, L, _ = llm_embed.shape
        x = self.llm_proj(llm_embed) + self.oh_proj(onehot.float())
        if self.pos is not None:
            x = x + self.pos(L, device).unsqueeze(0)
        x = self.drop(self.in_ln(x))
        key_padding_mask = (mask == 0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.out_ln(x)
        x = x * mask.unsqueeze(-1)
        return x


class ModalityRouterFusion(nn.Module):
    def __init__(self, d_model: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mod_emb = nn.Embedding(3, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=heads, dropout=dropout, batch_first=True)

        self.q_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, mask: torch.Tensor):
        device = x1.device
        B, L, D = x1.shape

        mods = torch.stack([x1, x2, x3], dim=2)
        mod_ids = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
        mods = mods + self.mod_emb(mod_ids).view(1, 1, 3, D)
        mods_flat = mods.view(B * L, 3, D)

        q = self.q_proj(torch.cat([x1, x2, x3], dim=-1)).view(B * L, 1, D)
        fused_attn, _ = self.attn(q, mods_flat, mods_flat, need_weights=False)
        fused_attn = fused_attn.view(B, L, D)

        gate_logits = self.gate_mlp(torch.cat([x1, x2, x3], dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        fused_gate = gate[..., 0:1] * x1 + gate[..., 1:2] * x2 + gate[..., 2:3] * x3

        fused = self.ln(fused_attn + fused_gate)
        fused = fused * mask.unsqueeze(-1)
        return fused, gate


class RNAFeatureExtraction(nn.Module):
    def __init__(
        self,
        d_llm: int,
        c_onehot: int,
        d_pssm: int,
        max_len: int = 1024,
        d_model: int = 256,
        d_out: int = 512,
        gnn_type: Literal["transformer", "gatv2"] = "transformer",
        gnn_layers: int = 4,
        gnn_heads: int = 4,
        edge_dim: int = 64,
        pssm_dropout: float = 0.1,
        fuse_heads: int = 8,
        final_layers: int = 2,
        dropout: float = 0.1,
        add_backbone_edges: bool = True,
    ):
        super().__init__()
        self.seq = RNASeqEncoder(
            d_llm=d_llm,
            c_onehot=c_onehot,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            pos_type="sinusoidal",
        )

        self.graph = RNAGraphEncoder(
            c_onehot=c_onehot,
            d_model=d_model,
            edge_dim=edge_dim,
            gnn_type=gnn_type,
            layers=gnn_layers,
            heads=gnn_heads,
            dropout=dropout,
            add_backbone_edges=add_backbone_edges,
        )

        self.pssm = PSSMResCNN(d_pssm=d_pssm, d_model=d_model, dropout=pssm_dropout)
        self.fuse = ModalityRouterFusion(d_model=d_model, heads=fuse_heads, dropout=dropout)

        self.to_out = nn.Linear(d_model, d_out)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_out,
            nhead=8,
            dim_feedforward=4 * d_out,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.final = nn.TransformerEncoder(enc_layer, num_layers=final_layers)
        self.out_ln = nn.LayerNorm(d_out)

    def forward(
        self,
        llm_embed: torch.Tensor,           
        onehot: torch.Tensor,              
        bp_edges_list: List[torch.Tensor], 
        pssm: torch.Tensor,                
        mask: Optional[torch.Tensor] = None,
        return_gate: bool = True,
    ):
        device = llm_embed.device
        B, L, _ = llm_embed.shape
        if mask is None:
            mask = torch.ones((B, L), device=device, dtype=torch.float32)
        else:
            mask = mask.float().to(device)

        x1 = self.seq(llm_embed, onehot, mask)
        x2 = self.graph(onehot, bp_edges_list, mask)
        x3 = self.pssm(pssm, mask)

        fused, gate = self.fuse(x1, x2, x3, mask)

        x = self.to_out(fused)
        key_padding_mask = (mask == 0)
        x = self.final(x, src_key_padding_mask=key_padding_mask)
        x = self.out_ln(x)
        x = x * mask.unsqueeze(-1)

        if return_gate:
            return x, x1, x2, x3, gate
        return x, x1, x2, x3


__all__ = ["RNAFeatureExtraction"]
