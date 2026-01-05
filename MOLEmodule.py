
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import TransformerConv
    from torch_geometric.nn.norm import GraphNorm
except Exception as e:
    raise ImportError("MOLEmodule requires torch_geometric. Install PyG first.") from e


class BottleneckAdapter(nn.Module):
    def __init__(self, d: int, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        self.drop = nn.Dropout(dropout)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.down(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.up(h)
        return x + self.drop(h)


class MoleSeqAdapter(nn.Module):
    def __init__(self, d_llm: int, c_onehot: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.llm_proj = nn.Sequential(
            nn.LayerNorm(d_llm),
            nn.Linear(d_llm, d_model),
        )
        self.oh_proj = nn.Linear(c_onehot, d_model, bias=False)
        self.adapter = BottleneckAdapter(d_model, bottleneck=max(64, d_model // 4), dropout=dropout)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, atom_llm: torch.Tensor, atom_onehot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.llm_proj(atom_llm) + self.oh_proj(atom_onehot)
        y = self.adapter(y)
        y = self.out_ln(y)
        return y * mask.unsqueeze(-1)


@dataclass
class BatchedGraph:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    lengths: torch.Tensor
    N: int


def _auto_fix_1_based(edges: torch.Tensor, L: int) -> torch.Tensor:
    if edges.numel() == 0:
        return edges
    uv = edges[:, :2]
    mn = int(uv.min().item())
    mx = int(uv.max().item())
    if mn >= 1 and mx <= L:
        edges = edges.clone()
        edges[:, :2] = edges[:, :2] - 1
    return edges


def _filter_invalid_edges(edges: torch.Tensor, L: int) -> torch.Tensor:
    if edges.numel() == 0:
        return edges
    u = edges[:, 0]
    v = edges[:, 1]
    keep = (u >= 0) & (v >= 0) & (u < L) & (v < L) & (u != v)
    return edges[keep]


def build_batched_graph_from_edge_lists(
    edge_lists: List[torch.Tensor],
    mask: torch.Tensor,
    device: torch.device,
    *,
    undirected: bool = True,
    num_edge_types: int = 8,
) -> BatchedGraph:
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

    src_all, dst_all, et_all = [], [], []
    for b in range(B):
        lb = int(lengths[b].item())
        if lb <= 0:
            continue
        offset = int(ptr[b].item())

        edges = edge_lists[b]
        if not isinstance(edges, torch.Tensor):
            edges = torch.tensor(edges)
        edges = edges.to(device).long()
        if edges.numel() == 0:
            continue

        edges = _auto_fix_1_based(edges, lb)
        edges = _filter_invalid_edges(edges, lb)
        if edges.numel() == 0:
            continue

        u = edges[:, 0]
        v = edges[:, 1]
        if edges.shape[1] >= 3:
            et = edges[:, 2].clamp(min=0, max=num_edge_types - 1)
        else:
            et = torch.zeros_like(u)

        src = u + offset
        dst = v + offset

        src_all.append(src); dst_all.append(dst); et_all.append(et)
        if undirected:
            src_all.append(dst); dst_all.append(src); et_all.append(et)

    if len(src_all) == 0:
        edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
        edge_type = torch.empty((0,), device=device, dtype=torch.long)
        return BatchedGraph(edge_index=edge_index, edge_type=edge_type, batch=batch, ptr=ptr, lengths=lengths, N=N)

    src = torch.cat(src_all, dim=0)
    dst = torch.cat(dst_all, dim=0)
    et = torch.cat(et_all, dim=0)
    edge_index = torch.stack([src, dst], dim=0)
    return BatchedGraph(edge_index=edge_index, edge_type=et, batch=batch, ptr=ptr, lengths=lengths, N=N)


def densify_nodes(x_nodes: torch.Tensor, ptr: torch.Tensor, L: int) -> torch.Tensor:
    B = ptr.numel() - 1
    D = x_nodes.shape[-1]
    out = x_nodes.new_zeros((B, L, D))
    for b in range(B):
        s = int(ptr[b].item())
        e = int(ptr[b + 1].item())
        lb = e - s
        if lb > 0:
            out[b, :lb] = x_nodes[s:e]
    return out


class MoleGraphEncoder(nn.Module):
    def __init__(
        self,
        d_llm: int,
        c_onehot: int,
        d_model: int = 256,
        layers: int = 4,
        heads: int = 4,
        edge_dim: int = 64,
        dropout: float = 0.1,
        num_edge_types: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_edge_types = num_edge_types

        self.llm_proj = nn.Sequential(
            nn.LayerNorm(d_llm),
            nn.Linear(d_llm, d_model),
        )
        self.oh_proj = nn.Linear(c_onehot, d_model, bias=False)

        self.edge_emb = nn.Embedding(num_edge_types, edge_dim)
        self.edge_ln = nn.LayerNorm(edge_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(
                TransformerConv(
                    in_channels=d_model,
                    out_channels=d_model // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    beta=True,
                )
            )
            self.norms.append(GraphNorm(d_model))

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.out_ln = nn.LayerNorm(d_model)

        for m in self.ff.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)

    def forward(
        self,
        atom_llm: torch.Tensor,
        atom_onehot: torch.Tensor,
        edge_lists: List[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        device = atom_llm.device
        B, L, _ = atom_llm.shape

        lengths = mask.sum(dim=1).long()
        ptr = torch.zeros(B + 1, device=device, dtype=torch.long)
        ptr[1:] = torch.cumsum(lengths, dim=0)

        x = self.llm_proj(atom_llm) + self.oh_proj(atom_onehot)
        x = x * mask.unsqueeze(-1)

        x_list = []
        for b in range(B):
            lb = int(lengths[b].item())
            if lb > 0:
                x_list.append(x[b, :lb])
        h = torch.cat(x_list, dim=0) if len(x_list) else x.new_zeros((0, self.d_model))

        bg = build_batched_graph_from_edge_lists(
            edge_lists=edge_lists,
            mask=mask,
            device=device,
            undirected=True,
            num_edge_types=self.num_edge_types,
        )
        edge_index = bg.edge_index
        edge_attr = self.edge_ln(self.edge_emb(bg.edge_type)) if bg.edge_type.numel() > 0 else None

        for conv, gn in zip(self.convs, self.norms):
            if edge_attr is not None:
                h2 = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h2 = conv(h, edge_index)
            h = gn(h + h2, bg.batch)
            h = F.gelu(h)

        h = self.out_ln(h + self.ff(h))
        h_dense = densify_nodes(h, bg.ptr, L) * mask.unsqueeze(-1)
        return h_dense


class SimpleGatedFusion(nn.Module):
    def __init__(self, d_model: int = 256, d_out: int = 512, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.out_ln = nn.LayerNorm(d_out)

    def forward(self, y1: torch.Tensor, y2: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.sigmoid(self.gate(torch.cat([y1, y2], dim=-1)))
        fused_256 = (1.0 - g) * y1 + g * y2
        y = torch.cat([fused_256, y2], dim=-1)
        y = self.out_ln(y) * mask.unsqueeze(-1)
        return y, g


class MoleFeatureExtraction(nn.Module):
    def __init__(
        self,
        d_llm: int,
        c_onehot: int,
        d_model: int = 256,
        d_out: int = 512,
        gnn_layers: int = 4,
        gnn_heads: int = 4,
        edge_dim: int = 64,
        dropout: float = 0.1,
        num_edge_types: int = 8,
    ):
        super().__init__()
        self.seq = MoleSeqAdapter(d_llm=d_llm, c_onehot=c_onehot, d_model=d_model, dropout=dropout)
        self.graph = MoleGraphEncoder(
            d_llm=d_llm,
            c_onehot=c_onehot,
            d_model=d_model,
            layers=gnn_layers,
            heads=gnn_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            num_edge_types=num_edge_types,
        )
        self.fuse = SimpleGatedFusion(d_model=d_model, d_out=d_out, dropout=dropout)

    def forward(
        self,
        atom_llm: torch.Tensor,
        atom_onehot: torch.Tensor,
        edge_lists: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        return_gate: bool = True,
    ):
        device = atom_llm.device
        B, L, _ = atom_llm.shape
        if mask is None:
            mask = torch.ones((B, L), device=device, dtype=torch.float32)
        else:
            mask = mask.float().to(device)

        y1 = self.seq(atom_llm, atom_onehot, mask)
        y2 = self.graph(atom_llm, atom_onehot, edge_lists, mask)
        y, gate = self.fuse(y1, y2, mask)

        if return_gate:
            return y, y1, y2, gate
        return y, y1, y2
