import os
import sys
import argparse
import random
import json
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from typing import Dict, List, Set, Tuple, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RNAdataset import (
    load_global_stores, RMPredDataset, collate_rmpred_batch,
    GlobalStores, seq_to_onehot_rna, SmartRDKitEncoder,
    _as_float_tensor, _as_long_edges, _filter_edges, _auto_to_0_based
)
from RMPred import RMPred
from utils import *
from config import BASE_DIR, PSSM_DIR, DICT_PATH


try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available, similarity features disabled")


def compute_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048):
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto_similarity(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def batch_compute_fingerprints(smiles_dict: Dict[str, str], radius: int = 2, nbits: int = 2048) -> Dict[str, any]:
    fps = {}
    for mol_id, smiles in smiles_dict.items():
        fp = compute_fingerprint(smiles, radius, nbits)
        if fp is not None:
            fps[mol_id] = fp
    return fps


def find_similar_molecules(
    query_mol_id: str,
    query_fp,
    all_fps: Dict[str, any],
    exclude_ids: Set[str] = None,
    top_k: int = 5,
    min_similarity: float = 0.3,
) -> List[Tuple[str, float]]:
    if query_fp is None:
        return []
    
    exclude_ids = exclude_ids or set()
    similarities = []
    
    for mol_id, fp in all_fps.items():
        if mol_id == query_mol_id or mol_id in exclude_ids:
            continue
        sim = tanimoto_similarity(query_fp, fp)
        if sim >= min_similarity:
            similarities.append((mol_id, sim))
    
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


class MoleculeSimilarityIndex:
    def __init__(self, smiles_dict: Dict[str, str], radius: int = 2, nbits: int = 2048):
        print(f"[SimilarityIndex] Building fingerprint index for {len(smiles_dict)} molecules...")
        self.smiles_dict = smiles_dict
        self.fps = batch_compute_fingerprints(smiles_dict, radius, nbits)
        print(f"[SimilarityIndex] Successfully indexed {len(self.fps)} molecules")
    
    def get_similar(
        self,
        mol_id: str,
        exclude_ids: Set[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> List[Tuple[str, float]]:
        if mol_id not in self.fps:
            return []
        return find_similar_molecules(
            mol_id, self.fps[mol_id], self.fps,
            exclude_ids, top_k, min_similarity
        )
    
    def get_similarity(self, mol_id1: str, mol_id2: str) -> float:
        fp1 = self.fps.get(mol_id1)
        fp2 = self.fps.get(mol_id2)
        return tanimoto_similarity(fp1, fp2)


def build_binding_index(entry_binding: Dict, rna_key: str = "rna_id", mole_key: str = "mole_id") -> Dict:
    rna_to_binders = defaultdict(list)
    
    for eid, entry in entry_binding.items():
        rna_id = entry.get(rna_key)
        mol_id = entry.get(mole_key) or entry.get("mol_id") or entry.get("ligand_name")
        pkd = entry.get("pkd") or entry.get("pKd")
        
        if rna_id and mol_id and pkd is not None:
            try:
                pkd_val = float(pkd)
                if np.isfinite(pkd_val):
                    rna_to_binders[rna_id].append((mol_id, pkd_val))
            except (ValueError, TypeError):
                pass
    
    for rna_id in rna_to_binders:
        rna_to_binders[rna_id].sort(key=lambda x: -x[1])
    
    return dict(rna_to_binders)


def generate_similarity_based_pseudolabels(
    similarity_index: MoleculeSimilarityIndex,
    rna_to_binders: Dict[str, List[Tuple[str, float]]],
    train_mol_ids: Set[str],
    existing_pairs: Set[Tuple[str, str]],
    max_samples: int = 100,
    min_similarity: float = 0.4,
    min_pkd: float = 4.0,
    seed: int = 42,
) -> List[Dict]:
    rng = np.random.RandomState(seed)
    candidates = []
    
    for rna_id, binders in rna_to_binders.items():
        for known_mol_id, known_pkd in binders:
            if known_pkd < min_pkd:
                continue
            
            similar = similarity_index.get_similar(
                known_mol_id,
                exclude_ids=train_mol_ids,
                top_k=10,
                min_similarity=min_similarity,
            )
            
            for sim_mol_id, similarity in similar:
                if (rna_id, sim_mol_id) in existing_pairs:
                    continue
                
                pseudo_pkd = known_pkd * similarity + (1 - similarity) * 4.0
                
                candidates.append({
                    'rna_id': rna_id,
                    'mol_id': sim_mol_id,
                    'pseudo_label': pseudo_pkd,
                    'reference_mol_id': known_mol_id,
                    'reference_pkd': known_pkd,
                    'similarity': similarity,
                    'weight': similarity ** 2,
                    'source': 'similarity',
                })
    
    pair_to_best = {}
    for c in candidates:
        key = (c['rna_id'], c['mol_id'])
        if key not in pair_to_best or c['similarity'] > pair_to_best[key]['similarity']:
            pair_to_best[key] = c
    
    candidates = list(pair_to_best.values())
    
    if len(candidates) > max_samples:
        candidates.sort(key=lambda x: -(x['similarity'] * 0.5 + x['pseudo_label'] / 10 * 0.5))
        candidates = candidates[:max_samples * 2]
        rng.shuffle(candidates)
        candidates = candidates[:max_samples]
    
    print(f"[SimilarityPseudolabels] Generated {len(candidates)} candidates")
    if candidates:
        sims = [c['similarity'] for c in candidates]
        pkds = [c['pseudo_label'] for c in candidates]
        print(f"  Similarity: min={min(sims):.3f}, max={max(sims):.3f}, mean={np.mean(sims):.3f}")
        print(f"  Pseudo-pKd: min={min(pkds):.2f}, max={max(pkds):.2f}, mean={np.mean(pkds):.2f}")
    
    return candidates


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


def load_rna_type_map(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {str(k): v.get("RNA_type", "Unknown") if isinstance(v, dict) else str(v) 
            for k, v in data.items()}


def numpy_pearson(pred, target):
    if len(pred) < 2:
        return float("nan")
    pred, target = pred.astype(np.float64), target.astype(np.float64)
    p_c, t_c = pred - pred.mean(), target - target.mean()
    denom = np.sqrt((p_c**2).mean()) * np.sqrt((t_c**2).mean())
    return float(np.clip((p_c * t_c).mean() / denom, -1, 1)) if denom > 0 else float("nan")


def numpy_rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target)**2)))


def ccc_loss(mu, y, eps=1e-8):
    mu, y = mu.float().view(-1), y.float().view(-1)
    if mu.numel() < 2:
        return torch.mean((mu - y)**2)
    mu_m, y_m = mu.mean(), y.mean()
    mu_v, y_v = mu.var(unbiased=False), y.var(unbiased=False)
    cov = ((mu - mu_m) * (y - y_m)).mean()
    ccc = (2.0 * cov) / (mu_v + y_v + (mu_m - y_m).pow(2) + eps)
    return 1.0 - torch.clamp(ccc, -1.0, 1.0)


def get_blind_mole_folds(stores, entry_ids, n_splits, seed):
    sample_entry = stores.entry_binding[entry_ids[0]]
    mole_key = 'mole_id' if 'mole_id' in sample_entry else 'ligand_name'
    
    all_mole_ids = []
    for eid in entry_ids:
        entry = stores.entry_binding.get(eid, {})
        all_mole_ids.append(entry.get(mole_key, "UNKNOWN"))
    
    unique_moles = np.array(sorted(set(all_mole_ids)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    mole_to_idx = defaultdict(list)
    for idx, mid in enumerate(all_mole_ids):
        mole_to_idx[mid].append(idx)
    
    folds = []
    print(f"\n[Split] Blind Molecule | Unique Moles: {len(unique_moles)} | Folds: {n_splits}")
    
    for i, (train_m, val_m) in enumerate(kf.split(unique_moles)):
        train_moles, val_moles = set(unique_moles[train_m]), set(unique_moles[val_m])
        train_idx = [idx for m in train_moles for idx in mole_to_idx[m]]
        val_idx = [idx for m in val_moles for idx in mole_to_idx[m]]
        random.Random(seed + i).shuffle(train_idx)
        random.Random(seed + i).shuffle(val_idx)
        folds.append((np.array(train_idx), np.array(val_idx), train_moles, val_moles))
        print(f"  Fold {i}: Train={len(train_idx)} ({len(train_moles)} moles) | Val={len(val_idx)} ({len(val_moles)} moles)")
    
    return folds


def load_teacher_ensemble(fold_dir, device, dims):
    models = []
    for m in range(5):
        ckpt = os.path.join(fold_dir, f"member_{m:02d}_best.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing: {ckpt}")
        sd = torch.load(ckpt, map_location=device)
        model = RMPred(
            d_llm_rna=dims["dim_rna_llm"], c_onehot_rna=dims["c_onehot_rna"],
            d_pssm_rna=dims["d_pssm"], d_llm_mole=dims["dim_mole_llm"],
            c_onehot_mole=dims["c_onehot_mole"],
            d_model_inner=256, d_model_fusion=512, dropout=0.2,
            fusion_layers=2, fusion_heads=4, rna_gnn_layers=4, rna_gnn_heads=4,
            mole_gnn_layers=4, mole_gnn_heads=4, mole_num_edge_types=8, rna_max_len=1024,
        ).to(device)
        model.load_state_dict(sd["model_state"], strict=True)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def evaluate_ensemble(models, loader, device):
    for m in models:
        m.eval()
    all_preds, all_ys = [], []
    
    for batch in loader:
        if batch is None:
            continue
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None:
            continue
        keep = torch.isfinite(y)
        if keep.sum() == 0:
            continue
        
        preds = []
        for model in models:
            with autocast():
                mu = model(
                    rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"],
                    rna_edges=batch["rna_edges"], rna_pssm=batch["rna_pssm"],
                    rna_mask=batch["rna_mask"], mole_llm=batch["mole_llm"],
                    mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                    mole_mask=batch["mole_mask"],
                )
            preds.append(mu.squeeze(-1)[keep].cpu().numpy())
        
        all_preds.append(np.mean(preds, axis=0))
        all_ys.append(y[keep].cpu().numpy())
    
    if not all_ys:
        return 0.0, 0.0, 0
    
    preds = np.concatenate(all_preds)
    ys = np.concatenate(all_ys)
    return numpy_rmse(preds, ys), numpy_pearson(preds, ys), len(ys)


def train_epoch_semi(model, labeled_loader, pseudo_loader, optimizer, device, scaler,
                     grad_clip=1.0, ccc_weight=1.0, pseudo_weight=0.05):
    model.train()
    total_loss, total_n = 0.0, 0
    total_pseudo_loss, total_pseudo_n = 0.0, 0
    
    for batch in labeled_loader:
        if batch is None:
            continue
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None:
            continue
        keep = torch.isfinite(y)
        if keep.sum() == 0:
            continue
        yy = y[keep]
        
        with autocast():
            mu = model(
                rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"],
                rna_edges=batch["rna_edges"], rna_pssm=batch["rna_pssm"],
                rna_mask=batch["rna_mask"], mole_llm=batch["mole_llm"],
                mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                mole_mask=batch["mole_mask"],
            )
            mu = mu.squeeze(-1)[keep]
            mse = mse_loss(mu, yy)
            ccc_l = ccc_loss(mu, yy)
            loss = 0.5 * (mse + ccc_weight * ccc_l)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        bs = int(yy.numel())
        total_loss += loss.item() * bs
        total_n += bs
    
    if pseudo_loader:
        for batch in pseudo_loader:
            if batch is None:
                continue
            batch = move_batch_to_device(batch, device)
            y = batch.get("pkd")
            if y is None:
                continue
            sw = batch.get("sample_weights", torch.ones_like(y)).to(device)
            keep = torch.isfinite(y)
            if keep.sum() == 0:
                continue
            yy, sw = y[keep], sw[keep]
            
            with autocast():
                mu = model(
                    rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"],
                    rna_edges=batch["rna_edges"], rna_pssm=batch["rna_pssm"],
                    rna_mask=batch["rna_mask"], mole_llm=batch["mole_llm"],
                    mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                    mole_mask=batch["mole_mask"],
                )
                mu = mu.view(-1)
                if mu.numel() != keep.numel():
                    continue
                mu = mu[keep]
                weighted_mse = ((mu - yy)**2 * sw).mean()
                loss = pseudo_weight * weighted_mse
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            bs = int(yy.numel())
            total_pseudo_loss += loss.item() * bs
            total_pseudo_n += bs
    
    return {
        "loss": total_loss / max(1, total_n),
        "pseudo_loss": total_pseudo_loss / max(1, total_pseudo_n) if pseudo_loader else 0.0
    }


def make_loader(dataset, indices, batch_size, shuffle, seed):
    subset = Subset(dataset, indices)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=collate_rmpred_batch, generator=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--teacher_dir", type=str, default="ckpt_blind_mole_replication")
    ap.add_argument("--out_dir", type=str, default="ckpt_semi_blind_mole_v3")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--run_folds", type=int, default=-1)
    ap.add_argument("--early_patience", type=int, default=20)
    
    ap.add_argument("--max_pseudo_total", type=int, default=40, help="Max total pseudolabels")
    ap.add_argument("--max_similarity", type=int, default=25, help="Max from similarity")
    ap.add_argument("--max_ensemble", type=int, default=25, help="Max from ensemble")
    ap.add_argument("--min_similarity", type=float, default=0.5, help="Higher similarity threshold")
    ap.add_argument("--confidence_threshold", type=float, default=0.20, help="Lower std threshold")
    ap.add_argument("--min_pkd", type=float, default=0.5, help="Only high binders (normalized)")
    ap.add_argument("--pseudo_weight", type=float, default=0.03, help="Lower pseudo weight")
    
    args = ap.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"V3 Conservative: max_pseudo={args.max_pseudo_total}, weight={args.pseudo_weight}")
    print(f"  similarity: max={args.max_similarity}, min_sim={args.min_similarity}")
    print(f"  ensemble: max={args.max_ensemble}, conf_thresh={args.confidence_threshold}, min_pkd={args.min_pkd}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    stores = load_global_stores(
        ids_path=os.path.join(BASE_DIR, "all_processed_v4_ids.pkl"),
        rna_embed_path=os.path.join(BASE_DIR, "rna_embed.pkl"),
        rna_graph_path=os.path.join(BASE_DIR, "rna_graph_edges.pkl"),
        mole_embed_path=os.path.join(BASE_DIR, "mole_embeddings_v2.pkl"),
        mole_edge_path=os.path.join(BASE_DIR, "mole_edges.pkl"),
        pssm_dir=PSSM_DIR,
    )
    
    dataset = RMPredDataset(stores, strict=True, max_rna_len=1024,
                            max_mole_len=2048, truncate_if_exceed=False, label_key="pkd")
    print(f"Total samples: {len(dataset)}")
    
    set_seed(args.seed)
    
    existing_pairs = set()
    for eid, entry in stores.entry_binding.items():
        rid = entry.get("rna_id")
        mid = entry.get("mole_id") or entry.get("mol_id")
        if rid and mid:
            existing_pairs.add((rid, mid))
    print(f"Existing pairs: {len(existing_pairs)}")
    
    temp_loader = DataLoader(Subset(dataset, [0]), batch_size=1, collate_fn=collate_rmpred_batch)
    batch0 = next(iter(temp_loader))
    dims = {
        "dim_rna_llm": batch0["rna_llm"].shape[-1],
        "dim_mole_llm": batch0["mole_llm"].shape[-1],
        "c_onehot_rna": batch0["rna_onehot"].shape[-1],
        "c_onehot_mole": batch0["mole_onehot"].shape[-1],
        "d_pssm": batch0["rna_pssm"].shape[-1],
    }
    
    entry_ids = getattr(dataset, "keys", list(stores.entry_binding.keys()))
    folds = get_blind_mole_folds(stores, entry_ids, args.folds, args.seed)
    
    if args.run_folds > 0:
        folds = folds[:args.run_folds]
    
    teacher_results, student_results = [], []
    
    for fold_id, (train_idx, val_idx, train_moles, val_moles) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_id}/{len(folds)-1} | train={len(train_idx)} val={len(val_idx)}")
        print(f"{'='*60}")
        
        fold_dir = os.path.join(args.out_dir, f"fold_{fold_id:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        
        val_loader = make_loader(dataset, val_idx.tolist(), args.batch_size, False, args.seed + 999)
        
        teacher_fold_dir = os.path.join(args.teacher_dir, f"fold_{fold_id:02d}")
        print(f"\nLoading teacher from {teacher_fold_dir}...")
        teacher_models = load_teacher_ensemble(teacher_fold_dir, device, dims)
        
        print("Evaluating teacher...")
        t_rmse, t_pearson, t_n = evaluate_ensemble(teacher_models, val_loader, device)
        print(f"\n[TEACHER FOLD {fold_id:02d}] RMSE={t_rmse:.4f} P={t_pearson:.4f}")
        teacher_results.append({"rmse": t_rmse, "pearson": t_pearson, "n": t_n})
        
        print(f"\n=== Generating Conservative Pseudo-Labels ===")
        
        pseudo_entries = []
        
        print(f"\n[Strategy] High-confidence ensemble on novel training pairs (max={args.max_pseudo_total})")
        
        rng = np.random.RandomState(args.seed + fold_id * 100)
        
        pssm_dir = stores.pssm_dir
        rna_ids = list(stores.rna_map.keys())
        valid_rna_ids = [rid for rid in rna_ids if os.path.exists(os.path.join(pssm_dir, f"{rid}.npy"))]
        train_mol_list = list(train_moles)
        
        print(f"  Valid RNAs: {len(valid_rna_ids)}, Training mols: {len(train_mol_list)}")
        
        candidate_pairs = []
        seen_pairs = set(existing_pairs)
        max_attempts = args.max_pseudo_total * 50
        
        for _ in range(max_attempts):
            if len(candidate_pairs) >= args.max_pseudo_total * 10:
                break
            rid = rng.choice(valid_rna_ids)
            mid = rng.choice(train_mol_list)
            key = (rid, mid)
            if key not in seen_pairs:
                seen_pairs.add(key)
                candidate_pairs.append({'rna_id': rid, 'mol_id': mid})
        
        print(f"  Sampled {len(candidate_pairs)} candidate pairs")
        
        if candidate_pairs:
            pseudo_dataset_temp = PseudoLabelDataset(stores, 
                [{'rna_id': p['rna_id'], 'mol_id': p['mol_id'], 'pseudo_label': 0.0, 'weight': 1.0} 
                 for p in candidate_pairs], 1024, 2048)
            pseudo_loader_temp = DataLoader(pseudo_dataset_temp, batch_size=16, 
                                             shuffle=False, num_workers=0, 
                                             collate_fn=collate_pseudo_batch)
            
            all_preds = []
            all_rids = []
            all_mids = []
            
            for model in teacher_models:
                model.eval()
            
            for batch in pseudo_loader_temp:
                if batch is None:
                    continue
                batch = move_batch_to_device(batch, device)
                batch_preds = []
                
                for model in teacher_models:
                    with torch.no_grad():
                        with autocast():
                            mu = model(
                                rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"],
                                rna_edges=batch["rna_edges"], rna_pssm=batch["rna_pssm"],
                                rna_mask=batch["rna_mask"], mole_llm=batch["mole_llm"],
                                mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                                mole_mask=batch["mole_mask"],
                            )
                        batch_preds.append(mu.squeeze(-1).cpu().numpy())
                
                batch_preds = np.stack(batch_preds, axis=0)
                ens_mean = np.mean(batch_preds, axis=0)
                ens_std = np.std(batch_preds, axis=0)
                
                for i in range(len(ens_mean)):
                    all_preds.append((ens_mean[i], ens_std[i]))
                    all_rids.append(batch["rna_ids"][i])
                    all_mids.append(batch["mol_ids"][i])
            
            print(f"  Got {len(all_preds)} predictions")
            if all_preds:
                mean_pkds = [p[0] for p in all_preds]
                mean_stds = [p[1] for p in all_preds]
                print(f"  pKd: mean={np.mean(mean_pkds):.2f}, min={np.min(mean_pkds):.2f}, max={np.max(mean_pkds):.2f}")
                print(f"  std: mean={np.mean(mean_stds):.3f}, min={np.min(mean_stds):.3f}, max={np.max(mean_stds):.3f}")
            
            for i, (mean_pkd, std_pkd) in enumerate(all_preds):
                if std_pkd < args.confidence_threshold and mean_pkd >= args.min_pkd:
                    weight = np.exp(-std_pkd)
                    pseudo_entries.append({
                        'rna_id': all_rids[i],
                        'mol_id': all_mids[i],
                        'pseudo_label': float(mean_pkd),
                        'weight': float(np.clip(weight, 0.1, 1.0)),
                        'source': 'ensemble',
                    })
            
            pseudo_entries = sorted(pseudo_entries, key=lambda x: -x['weight'])[:args.max_pseudo_total]
            print(f"  High-confidence pseudolabels: {len(pseudo_entries)}")
        
        seen = set()
        unique_entries = []
        for e in pseudo_entries:
            key = (e['rna_id'], e['mol_id'])
            if key not in seen:
                seen.add(key)
                unique_entries.append(e)
        
        if len(unique_entries) > args.max_pseudo_total:
            unique_entries = sorted(unique_entries, key=lambda x: -x['weight'])[:args.max_pseudo_total]
        
        pseudo_entries = unique_entries
        
        print(f"\n[Total] {len(pseudo_entries)} pseudolabels")
        if pseudo_entries:
            weights = [e['weight'] for e in pseudo_entries]
            pkds = [e['pseudo_label'] for e in pseudo_entries]
            print(f"  Weight: {min(weights):.3f} - {max(weights):.3f}")
            print(f"  pKd: {min(pkds):.2f} - {max(pkds):.2f}")
        
        pseudo_loader = None
        if pseudo_entries:
            pseudo_dataset = PseudoLabelDataset(stores, pseudo_entries, 1024, 2048)
            pseudo_loader = DataLoader(pseudo_dataset, batch_size=8, shuffle=True,
                                       num_workers=0, collate_fn=collate_pseudo_batch)
            print(f"Pseudo-label loader: {len(pseudo_entries)} samples")
        
        print(f"\n--- Training Student Ensemble ---")
        student_ckpts = []
        
        for m in range(args.k):
            member_seed = args.seed + 2000 * (fold_id * args.k + m)
            set_seed(member_seed)
            
            train_loader = make_loader(dataset, train_idx.tolist(), args.batch_size, True, member_seed)
            
            model = RMPred(
                d_llm_rna=dims["dim_rna_llm"], c_onehot_rna=dims["c_onehot_rna"],
                d_pssm_rna=dims["d_pssm"], d_llm_mole=dims["dim_mole_llm"],
                c_onehot_mole=dims["c_onehot_mole"],
                d_model_inner=256, d_model_fusion=512, dropout=0.2,
                fusion_layers=2, fusion_heads=4, rna_gnn_layers=4, rna_gnn_heads=4,
                mole_gnn_layers=4, mole_gnn_heads=4, mole_num_edge_types=8, rna_max_len=1024,
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scaler = GradScaler()
            
            best_pearson = float("-inf")
            best_path = os.path.join(fold_dir, f"student_{m:02d}_best.pt")
            patience = 0
            
            print(f"\n--- Student {m+1}/{args.k} ---")
            
            for epoch in range(1, args.epochs + 1):
                tr = train_epoch_semi(model, train_loader, pseudo_loader, optimizer,
                                      device, scaler, pseudo_weight=args.pseudo_weight)
                
                model.eval()
                _, val_p, _ = evaluate_ensemble([model], val_loader, device)
                
                if val_p > best_pearson:
                    best_pearson = val_p
                    torch.save({"model_state": model.state_dict(), "epoch": epoch}, best_path)
                    patience = 0
                else:
                    patience += 1
                
                if epoch % 5 == 0 or epoch == 1:
                    print(f"[Ep {epoch:03d}] Loss={tr['loss']:.4f} PseudoL={tr['pseudo_loss']:.4f} | P={val_p:.4f}")
                
                if patience >= args.early_patience:
                    print(f"  -> Early stop. Best={best_pearson:.4f}")
                    break
            
            print(f"Student {m:02d} done. Best={best_pearson:.4f}")
            student_ckpts.append(best_path)
        
        print("\nEvaluating Student Ensemble...")
        student_models = []
        for ckpt_path in student_ckpts:
            sd = torch.load(ckpt_path, map_location=device)
            model = RMPred(
                d_llm_rna=dims["dim_rna_llm"], c_onehot_rna=dims["c_onehot_rna"],
                d_pssm_rna=dims["d_pssm"], d_llm_mole=dims["dim_mole_llm"],
                c_onehot_mole=dims["c_onehot_mole"],
                d_model_inner=256, d_model_fusion=512, dropout=0.2,
                fusion_layers=2, fusion_heads=4, rna_gnn_layers=4, rna_gnn_heads=4,
                mole_gnn_layers=4, mole_gnn_heads=4, mole_num_edge_types=8, rna_max_len=1024,
            ).to(device)
            model.load_state_dict(sd["model_state"])
            model.eval()
            student_models.append(model)
        
        s_rmse, s_pearson, s_n = evaluate_ensemble(student_models, val_loader, device)
        print(f"\n[STUDENT FOLD {fold_id:02d}] RMSE={s_rmse:.4f} P={s_pearson:.4f}")
        
        delta = s_pearson - t_pearson
        if delta > 0:
            print(f">>> Student IMPROVED by {delta:.4f}! <<<")
        else:
            print(f">>> Student degraded by {delta:.4f} <<<")
        
        student_results.append({"rmse": s_rmse, "pearson": s_pearson, "n": s_n})
        
        del teacher_models, student_models
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    t_rmses = [r["rmse"] for r in teacher_results]
    t_ps = [r["pearson"] for r in teacher_results]
    s_rmses = [r["rmse"] for r in student_results]
    s_ps = [r["pearson"] for r in student_results]
    
    print(f"\nTeacher: RMSE={np.mean(t_rmses):.4f}±{np.std(t_rmses):.4f} | Pearson={np.mean(t_ps):.4f}±{np.std(t_ps):.4f}")
    print(f"Student: RMSE={np.mean(s_rmses):.4f}±{np.std(s_rmses):.4f} | Pearson={np.mean(s_ps):.4f}±{np.std(s_ps):.4f}")
    print(f"\nDelta Pearson: {np.mean(s_ps) - np.mean(t_ps):+.4f}")
    
    sys.stdout.flush()


if __name__ == "__main__":
    main()
