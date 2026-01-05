import os
import math
import argparse
import random
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import KFold

from RNAdataset import (
    load_global_stores, RMPredDataset, collate_rmpred_batch,
    GlobalStores, seq_to_onehot_rna, SmartRDKitEncoder,
    _as_float_tensor, _as_long_edges, _filter_edges, _auto_to_0_based
)
from RMPred import RMPred
from utils import *

from config import BASE_DIR, PSSM_DIR, DICT_PATH


def load_rna_type_map(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    id_to_type = {}
    for k, v in data.items():
        if isinstance(v, dict):
            id_to_type[str(k)] = v.get("RNA_type", "Unknown")
        else:
            id_to_type[str(k)] = str(v)
    return id_to_type


def numpy_pearson(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    denom = np.sqrt((pred_centered ** 2).mean()) * np.sqrt((target_centered ** 2).mean())
    if denom == 0:
        return float("nan")
    r = (pred_centered * target_centered).mean() / denom
    return float(np.clip(r, -1.0, 1.0))


def numpy_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


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


def get_blind_rna_kfold_indices(stores, entry_ids, n_splits, seed):
    all_sample_rna_ids = []
    for eid in entry_ids:
        if eid in stores.entry_binding:
            all_sample_rna_ids.append(stores.entry_binding[eid]['rna_id'])
        else:
            all_sample_rna_ids.append("UNKNOWN")

    unique_rnas = np.array(sorted(list(set(all_sample_rna_ids))))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    rna_to_sample_indices = defaultdict(list)
    for idx, r_id in enumerate(all_sample_rna_ids):
        rna_to_sample_indices[r_id].append(idx)
        
    folds = []
    print(f"\n[Split Strategy] Blind RNA (Scaffold Split) | Unique RNAs: {len(unique_rnas)} | Folds: {n_splits}")
    
    for i, (train_rna_indices, val_rna_indices) in enumerate(kf.split(unique_rnas)):
        train_rnas = set(unique_rnas[train_rna_indices])
        val_rnas = set(unique_rnas[val_rna_indices])
        
        train_idx, val_idx = [], []
        for r_id in train_rnas:
            train_idx.extend(rna_to_sample_indices[r_id])
        for r_id in val_rnas:
            val_idx.extend(rna_to_sample_indices[r_id])
            
        random.Random(seed + i).shuffle(train_idx)
        random.Random(seed + i).shuffle(val_idx)
        
        folds.append((np.array(train_idx), np.array(val_idx)))
        print(f"  Fold {i}: Train Samples={len(train_idx)} (RNAs={len(train_rnas)}) | Val Samples={len(val_idx)} (RNAs={len(val_rnas)})")
    
    return folds


class PseudoLabelDataset(Dataset):
    
    def __init__(
        self,
        stores: GlobalStores,
        pseudo_entries: List[Dict],
        max_rna_len: int = 1024,
        max_mole_len: int = 2048,
    ):
        self.stores = stores
        self.pseudo_entries = pseudo_entries
        self.max_rna_len = max_rna_len
        self.max_mole_len = max_mole_len
        self.encoder = SmartRDKitEncoder()
        self._mole_onehot_cache = {}
    
    def __len__(self):
        return len(self.pseudo_entries)
    
    def __getitem__(self, idx):
        entry = self.pseudo_entries[idx]
        rna_id = entry['rna_id']
        mol_id = entry['mol_id']
        pseudo_label = entry['pseudo_label']
        weight = entry.get('weight', 1.0)
        
        try:
            rna_name = self.stores.rna_map.get(rna_id)
            if not rna_name:
                return None
            
            r_llm = smart_fetch(self.stores.rna_embed, [rna_name, rna_id])
            if r_llm is None:
                return None
            r_llm = _as_float_tensor(r_llm)
            
            r_seq = smart_fetch(self.stores.rna_seqs, [rna_id, rna_name])
            if not r_seq:
                return None
            r_oh = seq_to_onehot_rna(r_seq)
            
            r_edges_raw = smart_fetch(self.stores.rna_graph, [rna_id, rna_name])
            r_edges = _as_long_edges(r_edges_raw)
            
            pssm_path = os.path.join(self.stores.pssm_dir, f"{rna_id}.npy")
            if not os.path.exists(pssm_path):
                return None
            p = np.load(pssm_path)
            if p.ndim != 2:
                p = p.reshape(p.shape[0], -1)
            r_pssm = torch.from_numpy(p).float()
            
            Lr = min(r_llm.shape[0], r_oh.shape[0], r_pssm.shape[0])
            if self.max_rna_len and Lr > self.max_rna_len:
                Lr = self.max_rna_len
            r_llm, r_oh, r_pssm = r_llm[:Lr], r_oh[:Lr], r_pssm[:Lr]
            r_edges = _filter_edges(_auto_to_0_based(r_edges, Lr), Lr)
            
            mol_name = self.stores.mol_map.get(mol_id)
            if not mol_name:
                return None
            
            m_llm = smart_fetch(self.stores.mole_embed, [mol_name, mol_id])
            if m_llm is None:
                return None
            m_llm = _as_float_tensor(m_llm)
            Lm = int(m_llm.shape[0])
            
            smiles = smart_fetch(self.stores.mole_smiles, [mol_id, mol_name])
            if not smiles:
                return None
            
            if mol_id in self._mole_onehot_cache:
                m_oh, status = self._mole_onehot_cache[mol_id]
                if m_oh.shape[0] != Lm:
                    m_oh, status = self.encoder.get_onehot(smiles, target_len=Lm)
            else:
                m_oh, status = self.encoder.get_onehot(smiles, target_len=Lm)
            
            if m_oh is None:
                return None
            self._mole_onehot_cache[mol_id] = (m_oh, status)
            
            if status == "Explicit_H":
                from rdkit import Chem
                mol_h = Chem.AddHs(Chem.MolFromSmiles(smiles))
                m_edges = self.encoder.bonds_as_edges(mol_h)
            else:
                m_edges_raw = smart_fetch(self.stores.mole_graph, [mol_name, mol_id])
                m_edges = _as_long_edges(m_edges_raw) if m_edges_raw is not None else torch.empty((0, 2), dtype=torch.long)
            
            m_edges = _filter_edges(_auto_to_0_based(m_edges, Lm), Lm)
            
            return {
                "entry_id": f"pseudo_{rna_id}_{mol_id}",
                "rna_id": rna_id,
                "mol_id": mol_id,
                "rna_llm": r_llm,
                "rna_onehot": r_oh,
                "rna_edges": r_edges,
                "rna_pssm": r_pssm,
                "mole_llm": m_llm,
                "mole_onehot": m_oh,
                "mole_edges": m_edges,
                "label": pseudo_label,
                "sample_weight": weight,
            }
        except Exception:
            return None


def collate_pseudo_batch(batch_list: List[Optional[dict]]):
    batch_list = [b for b in batch_list if b is not None]
    if len(batch_list) == 0:
        return None
    
    B = len(batch_list)
    max_rna_len = max(d["rna_llm"].shape[0] for d in batch_list)
    max_mole_len = max(d["mole_llm"].shape[0] for d in batch_list)
    
    d_rna_llm = batch_list[0]["rna_llm"].shape[1]
    d_mole_llm = batch_list[0]["mole_llm"].shape[1]
    c_mole_oh = batch_list[0]["mole_onehot"].shape[1]
    d_rna_pssm = batch_list[0]["rna_pssm"].shape[1]
    
    rna_llm = torch.zeros(B, max_rna_len, d_rna_llm)
    rna_oh = torch.zeros(B, max_rna_len, 5)
    rna_pssm = torch.zeros(B, max_rna_len, d_rna_pssm)
    rna_mask = torch.zeros(B, max_rna_len)
    rna_edges = []
    
    mole_llm = torch.zeros(B, max_mole_len, d_mole_llm)
    mole_oh = torch.zeros(B, max_mole_len, c_mole_oh)
    mole_mask = torch.zeros(B, max_mole_len)
    mole_edges = []
    
    entry_ids, rna_ids, mol_ids = [], [], []
    labels, weights = [], []
    
    for i, item in enumerate(batch_list):
        entry_ids.append(item["entry_id"])
        rna_ids.append(item["rna_id"])
        mol_ids.append(item["mol_id"])
        labels.append(item["label"])
        weights.append(item.get("sample_weight", 1.0))
        
        Lr = item["rna_llm"].shape[0]
        rna_llm[i, :Lr] = item["rna_llm"]
        rna_oh[i, :Lr] = item["rna_onehot"]
        rna_pssm[i, :Lr] = item["rna_pssm"]
        rna_mask[i, :Lr] = 1.0
        rna_edges.append(item["rna_edges"])
        
        Lm = item["mole_llm"].shape[0]
        mole_llm[i, :Lm] = item["mole_llm"]
        mole_oh[i, :Lm] = item["mole_onehot"]
        mole_mask[i, :Lm] = 1.0
        mole_edges.append(item["mole_edges"])
    
    return {
        "entry_ids": entry_ids,
        "rna_ids": rna_ids,
        "mol_ids": mol_ids,
        "rna_llm": rna_llm,
        "rna_onehot": rna_oh,
        "rna_edges": rna_edges,
        "rna_pssm": rna_pssm,
        "rna_mask": rna_mask,
        "mole_llm": mole_llm,
        "mole_onehot": mole_oh,
        "mole_edges": mole_edges,
        "mole_mask": mole_mask,
        "pkd": torch.tensor(labels, dtype=torch.float32),
        "sample_weights": torch.tensor(weights, dtype=torch.float32),
    }


@torch.no_grad()
def generate_pseudo_labels(
    models: List[nn.Module],
    stores: GlobalStores,
    existing_pairs: set,
    device: torch.device,
    max_samples: int = 50,
    confidence_threshold: float = 0.25,
    seed: int = 42,
) -> List[Dict]:
    print("\n=== Generating Pseudo-Labels ===")
    
    for m in models:
        m.eval()
    
    rna_ids = list(stores.rna_map.keys())
    mol_ids = list(stores.mol_map.keys())
    rng = np.random.RandomState(seed)
    
    valid_rna_ids = [rid for rid in rna_ids if os.path.exists(os.path.join(stores.pssm_dir, f"{rid}.npy"))]
    print(f"  Valid RNAs with PSSM: {len(valid_rna_ids)}/{len(rna_ids)}")
    
    candidate_pairs = []
    max_attempts = max_samples * 20
    attempts = 0
    
    while len(candidate_pairs) < max_samples * 5 and attempts < max_attempts:
        rid = rng.choice(valid_rna_ids)
        mid = rng.choice(mol_ids)
        if (rid, mid) not in existing_pairs and (rid, mid) not in [(p['rna_id'], p['mol_id']) for p in candidate_pairs]:
            candidate_pairs.append({'rna_id': rid, 'mol_id': mid})
        attempts += 1
    
    print(f"  Candidate pairs sampled: {len(candidate_pairs)}")
    
    temp_dataset = PseudoLabelDataset(stores, [{'rna_id': p['rna_id'], 'mol_id': p['mol_id'], 'pseudo_label': 0.0} for p in candidate_pairs])
    temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_pseudo_batch)
    
    all_predictions, all_rna_ids, all_mol_ids = [], [], []
    
    for batch in temp_loader:
        if batch is None:
            continue
        batch = move_batch_to_device(batch, device)
        batch_preds = []
        
        for model in models:
            with autocast():
                mu = model(
                    rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                    rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                    mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                    mole_mask=batch["mole_mask"],
                )
            batch_preds.append(mu.squeeze(-1).detach().cpu().numpy())
        
        batch_preds = np.stack(batch_preds, axis=0)
        all_predictions.append(batch_preds)
        all_rna_ids.extend(batch["rna_ids"])
        all_mol_ids.extend(batch["mol_ids"])
    
    if not all_predictions:
        print("  No valid candidates found!")
        return []
    
    all_predictions = np.concatenate(all_predictions, axis=1)
    ensemble_mean = np.mean(all_predictions, axis=0)
    ensemble_std = np.std(all_predictions, axis=0)
    
    print(f"  Predictions obtained: {len(ensemble_mean)}")
    print(f"  Ensemble std: mean={np.mean(ensemble_std):.4f}, min={np.min(ensemble_std):.4f}, max={np.max(ensemble_std):.4f}")
    
    confident_mask = ensemble_std < confidence_threshold
    confident_indices = np.where(confident_mask)[0]
    print(f"  High-confidence samples (std < {confidence_threshold}): {len(confident_indices)}")
    
    if len(confident_indices) == 0:
        top_k = max(10, len(ensemble_std) // 10)
        confident_indices = np.argsort(ensemble_std)[:top_k]
        print(f"  Fallback: using top {len(confident_indices)} most confident samples")
    
    if len(confident_indices) > max_samples:
        sorted_by_confidence = confident_indices[np.argsort(ensemble_std[confident_indices])]
        confident_indices = sorted_by_confidence[:max_samples]
    
    pseudo_entries = []
    for idx in confident_indices:
        std_val = ensemble_std[idx]
        weight = np.exp(-std_val)
        pseudo_entries.append({
            'rna_id': all_rna_ids[idx],
            'mol_id': all_mol_ids[idx],
            'pseudo_label': float(ensemble_mean[idx]),
            'ensemble_std': float(std_val),
            'weight': float(weight),
        })
    
    print(f"  Final pseudo-labels: {len(pseudo_entries)}")
    if pseudo_entries:
        stds = [e['ensemble_std'] for e in pseudo_entries]
        print(f"  Selected std range: {min(stds):.4f} - {max(stds):.4f}")
    
    return pseudo_entries


@torch.no_grad()
def collect_ensemble_predictions(
    models: List[nn.Module],
    loader: DataLoader,
    device: torch.device,
    type_map: Dict[str, str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    for m in models:
        m.eval()
    
    num_models = len(models)
    all_ys = []
    all_types = []
    member_preds = [[] for _ in range(num_models)]
    
    for batch in loader:
        if batch is None:
            continue
        batch_ids = batch.get("entry_ids") or batch.get("entry_id")
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None:
            continue
        y = y.to(device)
        keep = torch.isfinite(y)
        if keep.sum().item() == 0:
            continue
        
        all_ys.append(y[keep].detach().cpu())
        
        if type_map is not None and batch_ids is not None:
            keep_cpu = keep.cpu().tolist()
            valid_ids = [bid for bid, k in zip(batch_ids, keep_cpu) if k]
            all_types.extend([type_map.get(str(eid), "Unknown") for eid in valid_ids])
        
        for i, m in enumerate(models):
            with autocast():
                mu, _ = m(
                    rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                    rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                    mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                    mole_mask=batch["mole_mask"],
                )
            member_preds[i].append(mu.squeeze(-1)[keep].detach().cpu().numpy())
    
    if not all_ys:
        return np.array([]), np.array([]), []
    
    y_all = torch.cat(all_ys, dim=0).numpy()
    preds_matrix = np.column_stack([np.concatenate(preds) for preds in member_preds])
    
    return preds_matrix, y_all, all_types


def evaluate_ensemble_simple_avg(
    preds_matrix: np.ndarray,
    y_all: np.ndarray,
    all_types: List[str]
) -> Dict[str, Any]:
    results = {}
    
    simple_preds = np.mean(preds_matrix, axis=1)
    results["simple_avg"] = {
        "rmse": numpy_rmse(simple_preds, y_all),
        "pearson": numpy_pearson(simple_preds, y_all),
        "count": len(y_all)
    }
    
    if all_types and len(all_types) == len(simple_preds):
        type_buckets = defaultdict(lambda: {"preds": [], "ys": []})
        for pred, target, rna_type in zip(simple_preds, y_all, all_types):
            type_buckets[rna_type]["preds"].append(pred)
            type_buckets[rna_type]["ys"].append(target)
        
        for rna_type, data in type_buckets.items():
            preds_arr = np.array(data["preds"])
            ys_arr = np.array(data["ys"])
            results[rna_type] = {
                "rmse": numpy_rmse(preds_arr, ys_arr),
                "pearson": numpy_pearson(preds_arr, ys_arr),
                "count": len(preds_arr)
            }
    
    return results


def train_one_epoch_semi(
    model: nn.Module,
    labeled_loader: DataLoader,
    pseudo_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    grad_clip: float = 1.0,
    ccc_weight: float = 1.0,
    pseudo_weight: float = 0.05,
) -> Dict[str, float]:
    model.train()
    total_loss, total_mse, total_ccc, total_n = 0.0, 0.0, 0.0, 0
    total_pseudo_loss, total_pseudo_n = 0.0, 0
    
    for batch in labeled_loader:
        if batch is None:
            continue
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None:
            continue
        y = y.to(device)
        keep = torch.isfinite(y)
        if keep.sum().item() == 0:
            continue
        
        yy = y[keep]
        
        with autocast():
            mu = model(
                rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                mole_mask=batch["mole_mask"],
            )
            mu = mu.squeeze(-1)[keep]
            mse = mse_loss(mu, yy)
            ccc_l = ccc_loss(mu, yy)
            loss = 0.5 * (mse + ccc_weight * ccc_l)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        bs = int(yy.numel())
        total_loss += loss.item() * bs
        total_mse += mse.item() * bs
        total_ccc += ccc_l.item() * bs
        total_n += bs
    
    if pseudo_loader is not None:
        for batch in pseudo_loader:
            if batch is None:
                continue
            batch = move_batch_to_device(batch, device)
            y = batch.get("pkd")
            if y is None:
                continue
            y = y.to(device)
            sample_weights = batch.get("sample_weights")
            if sample_weights is not None:
                sample_weights = sample_weights.to(device)
            
            keep = torch.isfinite(y)
            if keep.sum().item() == 0:
                continue
            
            yy = y[keep]
            sw = sample_weights[keep] if sample_weights is not None else torch.ones_like(yy)
            
            with autocast():
                mu = model(
                    rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                    rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                    mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                    mole_mask=batch["mole_mask"],
                )
                mu = mu.squeeze(-1)[keep]
                weighted_mse = ((mu - yy) ** 2 * sw).mean()
                loss = pseudo_weight * weighted_mse
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            bs = int(yy.numel())
            total_pseudo_loss += loss.item() * bs
            total_pseudo_n += bs
    
    denom = max(1, total_n)
    denom_p = max(1, total_pseudo_n)
    return {
        "loss": total_loss / denom,
        "mse": total_mse / denom,
        "ccc_loss": total_ccc / denom,
        "pseudo_loss": total_pseudo_loss / denom_p if pseudo_loader else 0.0,
    }


def make_subset_loader(dataset, indices, batch_size, shuffle, num_workers, seed):
    subset = Subset(dataset, indices)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_rmpred_batch, generator=g)


def load_teacher_ensemble(fold_dir: str, device: torch.device, dims: Dict) -> List[nn.Module]:
    models = []
    for m in range(5):
        ckpt_path = os.path.join(fold_dir, f"member_{m:02d}_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        sd = torch.load(ckpt_path, map_location=device)
        model = RMPred(
            d_llm_rna=dims["dim_rna_llm"],
            c_onehot_rna=dims["c_onehot_rna"],
            d_pssm_rna=dims["d_pssm"],
            d_llm_mole=dims["dim_mole_llm"],
            c_onehot_mole=dims["c_onehot_mole"],
            d_model_inner=256, d_model_fusion=512, dropout=0.2,
            fusion_layers=2, fusion_heads=4, rna_gnn_layers=4, rna_gnn_heads=4,
            mole_gnn_layers=4, mole_gnn_heads=4, mole_num_edge_types=8, rna_max_len=1024,
        ).to(device)
        model.load_state_dict(sd["model_state"], strict=True)
        model.eval()
        models.append(model)
    
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="ensemble size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--teacher_dir", type=str, default="ckpt_blind_rna_replication", help="teacher checkpoint dir")
    ap.add_argument("--out_dir", type=str, default="ckpt_semi_blind_rna")
    ap.add_argument("--max_rna_len", type=int, default=1024)
    ap.add_argument("--max_mole_len", type=int, default=2048)
    ap.add_argument("--early_patience", type=int, default=20)
    ap.add_argument("--label_key", type=str, default="pkd")
    ap.add_argument("--folds", type=int, default=5, help="number of folds (must match teacher)")
    ap.add_argument("--run_folds", type=int, default=-1, help="run only first N folds (-1 for all)")
    ap.add_argument("--metric", type=str, default="pearson", choices=["rmse", "pearson"])
    
    ap.add_argument("--pseudo_samples", type=int, default=50, help="max pseudo-labels per fold")
    ap.add_argument("--confidence_threshold", type=float, default=0.25, help="max std for confidence")
    ap.add_argument("--pseudo_weight", type=float, default=0.05, help="weight for pseudo-label loss")
    
    args = ap.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Metric: {args.metric.upper()}")
    print(f"Semi-supervised settings: pseudo_samples={args.pseudo_samples}, confidence_threshold={args.confidence_threshold}, pseudo_weight={args.pseudo_weight}")
    os.makedirs(args.out_dir, exist_ok=True)
    
    stores = load_global_stores(
        ids_path=os.path.join(BASE_DIR, "all_processed_v4_ids.pkl"),
        rna_embed_path=os.path.join(BASE_DIR, "rna_embed.pkl"),
        rna_graph_path=os.path.join(BASE_DIR, "rna_graph_edges.pkl"),
        mole_embed_path=os.path.join(BASE_DIR, "mole_embeddings_v2.pkl"),
        mole_edge_path=os.path.join(BASE_DIR, "mole_edges.pkl"),
        pssm_dir=PSSM_DIR,
    )
    
    type_map = None
    if os.path.exists(DICT_PATH):
        type_map = load_rna_type_map(DICT_PATH)
    
    dataset = RMPredDataset(
        stores, strict=True, max_rna_len=args.max_rna_len,
        max_mole_len=(None if args.max_mole_len == 0 else args.max_mole_len),
        truncate_if_exceed=False, label_key=args.label_key,
    )
    
    n_total = len(dataset)
    print(f"Total labeled samples: {n_total}")
    
    set_seed(args.seed)
    
    existing_pairs = set()
    for eid, entry in stores.entry_binding.items():
        rid = entry.get("rna_id")
        mid = entry.get("mole_id") or entry.get("mol_id")
        if rid and mid:
            existing_pairs.add((rid, mid))
    print(f"Existing RNA-ligand pairs: {len(existing_pairs)}")
    
    temp_loader = DataLoader(Subset(dataset, [0]), batch_size=1, collate_fn=collate_rmpred_batch)
    batch0 = next(iter(temp_loader))
    dims = {
        "dim_rna_llm": batch0["rna_llm"].shape[-1],
        "dim_mole_llm": batch0["mole_llm"].shape[-1],
        "c_onehot_rna": batch0["rna_onehot"].shape[-1],
        "c_onehot_mole": batch0["mole_onehot"].shape[-1],
        "d_pssm": batch0["rna_pssm"].shape[-1],
    }
    
    entry_ids = getattr(dataset, "keys", None)
    if entry_ids is None:
        entry_ids = list(stores.entry_binding.keys())
    
    folds = get_blind_rna_kfold_indices(stores, entry_ids, n_splits=args.folds, seed=args.seed)
    
    if args.run_folds > 0 and args.run_folds < len(folds):
        folds = folds[:args.run_folds]
        print(f"Running only first {args.run_folds} folds")
    
    teacher_results = []
    student_results = []
    
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_id}/{len(folds)-1} | train={len(train_idx)} val={len(val_idx)}")
        print(f"{'='*60}")
        
        fold_dir = os.path.join(args.out_dir, f"fold_{fold_id:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        
        val_loader = make_subset_loader(
            dataset, val_idx.tolist(), batch_size=args.val_batch_size,
            shuffle=False, num_workers=0, seed=args.seed + 999 + fold_id
        )
        
        teacher_fold_dir = os.path.join(args.teacher_dir, f"fold_{fold_id:02d}")
        print(f"\nLoading teacher ensemble from {teacher_fold_dir}...")
        teacher_models = load_teacher_ensemble(teacher_fold_dir, device, dims)
        
        print("Evaluating teacher ensemble on test set (fresh)...")
        teacher_preds, teacher_y, teacher_types = collect_ensemble_predictions(
            teacher_models, val_loader, device, type_map
        )
        teacher_ens = evaluate_ensemble_simple_avg(teacher_preds, teacher_y, teacher_types)
        
        print(f"\n[TEACHER FOLD {fold_id:02d} RESULTS] (Simple Avg for generalizability)")
        print(f"  Simple Avg: RMSE={teacher_ens['simple_avg']['rmse']:.4f} P={teacher_ens['simple_avg']['pearson']:.4f} N={teacher_ens['simple_avg']['count']}")
        
        for key, metrics in teacher_ens.items():
            if key != "simple_avg":
                print(f"    Type {key}: RMSE={metrics['rmse']:.4f} P={metrics['pearson']:.4f} N={metrics['count']}")
        
        teacher_results.append(teacher_ens)
        
        pseudo_entries = generate_pseudo_labels(
            teacher_models, stores, existing_pairs, device,
            max_samples=args.pseudo_samples,
            confidence_threshold=args.confidence_threshold,
            seed=args.seed + fold_id * 100,
        )
        
        pseudo_loader = None
        if pseudo_entries:
            pseudo_dataset = PseudoLabelDataset(stores, pseudo_entries, args.max_rna_len, args.max_mole_len)
            pseudo_loader = DataLoader(
                pseudo_dataset, batch_size=8, shuffle=True, num_workers=0,
                collate_fn=collate_pseudo_batch
            )
            print(f"  Pseudo-label loader: {len(pseudo_entries)} samples")
        
        print(f"\n--- Training Student Ensemble ---")
        student_ckpts = []
        
        for m in range(args.k):
            member_seed = args.seed + 2000 * (fold_id * args.k + m)
            set_seed(member_seed)
            
            train_loader = make_subset_loader(
                dataset, train_idx.tolist(), batch_size=args.batch_size,
                shuffle=True, num_workers=0, seed=member_seed
            )
            
            model = RMPred(
                d_llm_rna=dims["dim_rna_llm"], c_onehot_rna=dims["c_onehot_rna"], d_pssm_rna=dims["d_pssm"],
                d_llm_mole=dims["dim_mole_llm"], c_onehot_mole=dims["c_onehot_mole"],
                d_model_inner=256, d_model_fusion=512, dropout=0.2,
                fusion_layers=2, fusion_heads=4, rna_gnn_layers=4, rna_gnn_heads=4,
                mole_gnn_layers=4, mole_gnn_heads=4, mole_num_edge_types=8, rna_max_len=args.max_rna_len,
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scaler = GradScaler()
            
            best_score = float("-inf") if args.metric == "pearson" else float("inf")
            best_path = os.path.join(fold_dir, f"student_{m:02d}_best.pt")
            patience = 0
            
            print(f"\n--- Student member {m+1}/{args.k} ---")
            
            for epoch in range(1, args.epochs + 1):
                tr = train_one_epoch_semi(
                    model, train_loader, pseudo_loader, optimizer, device, scaler,
                    grad_clip=1.0, ccc_weight=1.0, pseudo_weight=args.pseudo_weight
                )
                
                model.eval()
                student_preds_list, student_ys_list = [], []
                
                with torch.no_grad():
                    for batch in val_loader:
                        if batch is None:
                            continue
                        batch = move_batch_to_device(batch, device)
                        y = batch.get("pkd", batch.get("labels"))
                        if y is None:
                            continue
                        y = y.to(device)
                        keep = torch.isfinite(y)
                        if keep.sum().item() == 0:
                            continue
                        
                        with autocast():
                            mu = model(
                                rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                                rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                                mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                                mole_mask=batch["mole_mask"],
                            )
                        student_preds_list.append(mu.squeeze(-1)[keep].detach().cpu())
                        student_ys_list.append(y[keep].detach().cpu())
                
                if student_preds_list:
                    s_preds = torch.cat(student_preds_list, dim=0).numpy()
                    s_ys = torch.cat(student_ys_list, dim=0).numpy()
                    current_rmse = numpy_rmse(s_preds, s_ys)
                    current_pearson = numpy_pearson(s_preds, s_ys)
                else:
                    current_rmse, current_pearson = float("nan"), float("nan")
                
                current_score = current_pearson if args.metric == "pearson" else current_rmse
                
                if epoch % 5 == 0 or epoch == 1:
                    print(f"[Ep {epoch:03d}] Loss={tr['loss']:.4f} PseudoL={tr['pseudo_loss']:.4f} | RMSE={current_rmse:.4f} P={current_pearson:.4f}")
                
                is_best = False
                if math.isfinite(current_score):
                    if args.metric == "pearson":
                        is_best = current_score > best_score
                    else:
                        is_best = current_score < best_score
                
                if is_best:
                    best_score = current_score
                    patience = 0
                    torch.save({
                        "fold": fold_id, "member": m, "epoch": epoch,
                        "model_state": model.state_dict(),
                        "best_score": best_score, "metric": args.metric,
                        "dims": dims,
                    }, best_path)
                else:
                    patience += 1
                    if patience >= args.early_patience:
                        print(f"  -> Early stop. Best {args.metric}={best_score:.4f}")
                        break
            
            print(f"Student {m:02d} finished. Best {args.metric}={best_score:.4f}")
            student_ckpts.append(best_path)
        
        print(f"\nEvaluating Student Ensemble for Fold {fold_id}...")
        student_models = []
        for ckpt in student_ckpts:
            sd = torch.load(ckpt, map_location=device)
            model = RMPred(
                d_llm_rna=dims["dim_rna_llm"], c_onehot_rna=dims["c_onehot_rna"], d_pssm_rna=dims["d_pssm"],
                d_llm_mole=dims["dim_mole_llm"], c_onehot_mole=dims["c_onehot_mole"],
                d_model_inner=256, d_model_fusion=512, dropout=0.2,
                fusion_layers=2, fusion_heads=4, rna_gnn_layers=4, rna_gnn_heads=4,
                mole_gnn_layers=4, mole_gnn_heads=4, mole_num_edge_types=8, rna_max_len=args.max_rna_len,
            ).to(device)
            model.load_state_dict(sd["model_state"], strict=True)
            student_models.append(model)
        
        student_preds, student_y, student_types = collect_ensemble_predictions(
            student_models, val_loader, device, type_map
        )
        student_ens = evaluate_ensemble_simple_avg(student_preds, student_y, student_types)
        
        print(f"\n[STUDENT FOLD {fold_id:02d} RESULTS] (Simple Avg for generalizability)")
        print(f"  Simple Avg: RMSE={student_ens['simple_avg']['rmse']:.4f} P={student_ens['simple_avg']['pearson']:.4f} N={student_ens['simple_avg']['count']}")
        
        for key, metrics in student_ens.items():
            if key != "simple_avg":
                print(f"    Type {key}: RMSE={metrics['rmse']:.4f} P={metrics['pearson']:.4f} N={metrics['count']}")
        
        student_results.append(student_ens)
        
        t_rmse = teacher_ens['simple_avg']['rmse']
        s_rmse = student_ens['simple_avg']['rmse']
        t_pear = teacher_ens['simple_avg']['pearson']
        s_pear = student_ens['simple_avg']['pearson']
        
        rmse_delta = s_rmse - t_rmse
        pear_delta = s_pear - t_pear
        
        print(f"\n[FOLD {fold_id:02d} COMPARISON] (Simple Avg)")
        print(f"  Teacher: RMSE={t_rmse:.4f} P={t_pear:.4f}")
        print(f"  Student: RMSE={s_rmse:.4f} P={s_pear:.4f}")
        print(f"  Delta:   RMSE={rmse_delta:+.4f} P={pear_delta:+.4f}")
        if pear_delta > 0:
            print(f"  >>> Student IMPROVED! <<<")
        elif pear_delta < -0.01:
            print(f"  >>> Student DEGRADED <<<")
        else:
            print(f"  >>> Maintained performance <<<")
    
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY - BLIND RNA ({args.folds} FOLDS)")
    print(f"{'='*60}")
    
    teacher_rmses = [r['simple_avg']['rmse'] for r in teacher_results]
    teacher_pears = [r['simple_avg']['pearson'] for r in teacher_results]
    student_rmses = [r['simple_avg']['rmse'] for r in student_results]
    student_pears = [r['simple_avg']['pearson'] for r in student_results]
    
    print(f"\nTeacher (Simple Avg - Fresh Evaluation):")
    print(f"  RMSE:    mean={np.mean(teacher_rmses):.4f} std={np.std(teacher_rmses):.4f}")
    print(f"  Pearson: mean={np.mean(teacher_pears):.4f} std={np.std(teacher_pears):.4f}")
    print(f"  Per-fold RMSE:    {[f'{r:.4f}' for r in teacher_rmses]}")
    print(f"  Per-fold Pearson: {[f'{p:.4f}' for p in teacher_pears]}")
    
    print(f"\nStudent (Simple Avg - Semi-supervised):")
    print(f"  RMSE:    mean={np.mean(student_rmses):.4f} std={np.std(student_rmses):.4f}")
    print(f"  Pearson: mean={np.mean(student_pears):.4f} std={np.std(student_pears):.4f}")
    print(f"  Per-fold RMSE:    {[f'{r:.4f}' for r in student_rmses]}")
    print(f"  Per-fold Pearson: {[f'{p:.4f}' for p in student_pears]}")
    
    rmse_improvement = np.mean(teacher_rmses) - np.mean(student_rmses)
    pear_improvement = np.mean(student_pears) - np.mean(teacher_pears)
    
    print(f"\nImprovement (Student vs Teacher):")
    print(f"  RMSE: {rmse_improvement:+.4f} (positive = student better)")
    print(f"  Pearson: {pear_improvement:+.4f} (positive = student better)")
    
    if pear_improvement > 0.005:
        print(f"\n>>> SEMI-SUPERVISION IMPROVED GENERALIZABILITY! <<<")
    elif pear_improvement > -0.005:
        print(f"\n>>> SEMI-SUPERVISION MAINTAINED GENERALIZABILITY <<<")
    else:
        print(f"\n>>> CAUTION: Generalizability degraded <<<")


if __name__ == "__main__":
    main()
