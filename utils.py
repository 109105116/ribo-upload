
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from rdkit import Chem
from RNAdataset import *
import json
from collections import defaultdict

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    if preds.numel() == 0:
        return {"pearson": float("nan"), "rmse": float("nan"), "count": 0}
    return {
        "pearson": pearson_corr(preds, targets),
        "rmse": rmse(preds, targets),
        "count": preds.numel()
    }
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    if "rna_edges" in out:
        out["rna_edges"] = [e.to(device) for e in out["rna_edges"]]
    if "mole_edges" in out:
        out["mole_edges"] = [e.to(device) for e in out["mole_edges"]]
    return out

@torch.no_grad()
def rmse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((x.float() - y.float()) ** 2)).item())

def mse_loss(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((mu - y) ** 2)

def ccc_loss(mu: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mu = mu.float().view(-1)
    y = y.float().view(-1)
    if mu.numel() < 2:
        return torch.mean((mu - y) ** 2)
    mu_mean, y_mean = mu.mean(), y.mean()
    mu_var, y_var = mu.var(unbiased=False), y.var(unbiased=False)
    cov = ((mu - mu_mean) * (y - y_mean)).mean()
    ccc = (2.0 * cov) / (mu_var + y_var + (mu_mean - y_mean).pow(2) + eps)
    return 1.0 - torch.clamp(ccc, -1.0, 1.0)

@torch.no_grad()
def pearson_corr(x, y):
    if x.numel() < 2: return float("nan")
    x, y = x.float(), y.float()
    x, y = x - x.mean(), y - y.mean()
    denom = (x.pow(2).mean().sqrt() * y.pow(2).mean().sqrt()).item()
    if denom == 0.0: return float("nan")
    raw_r = (x * y).mean().item() / denom
    return float(max(-1.0, min(1.0, raw_r)))

def smart_fetch(d: Dict, keys_to_try: List[Any]):
    for k in keys_to_try:
        if k in d:
            return d[k]
    return None