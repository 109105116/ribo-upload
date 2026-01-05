import os
import sys
import argparse
import json
import glob
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from config import BASE_DIR, PSSM_DIR, DICT_PATH, CODE_DIR

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torch.cuda.amp import autocast, GradScaler
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RMPred import RMPred
from RNAdataset import (
    GlobalStores, RMPredDataset, collate_rmpred_batch,
    seq_to_onehot_rna, SmartRDKitEncoder,
    _as_float_tensor, _as_long_edges, _filter_edges, _auto_to_0_based
)
from utils import *


MODEL_CONFIG = {
    "d_model_inner": 256,
    "d_model_fusion": 512,
    "dropout": 0.2,
    "fusion_layers": 2,
    "fusion_heads": 4,
    "rna_gnn_layers": 4,
    "rna_gnn_heads": 4,
    "mole_gnn_layers": 4,
    "mole_gnn_heads": 4,
    "mole_num_edge_types": 8,
    "rna_max_len": 1024,
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "val_batch_size": 32,
    "epochs": 60,
    "lr": 2e-4,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
    "ccc_weight": 1.0,
    "early_patience": 20,
    "k": 5,
}

SEMI_CONFIG_V2 = {
    "confidence_threshold": 0.18,
    "pseudo_samples": 30,
    "pseudo_weight": 0.03,
    "curriculum_stages": 3,
    "curriculum_warmup_epochs": 5,
    "use_distillation": True,
    "distill_weight": 0.1,
    "distill_temperature": 2.0,
    "min_agreement_ratio": 0.8,
    "uncertainty_weighting": "inverse_variance",
    "n_epochs": 20,
    "n_members": 5,
    "batch_size": 16,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "patience": 8,
    "seed": 42,
    "use_ccc_loss": False,
}

MODEL_CFG = {
    "in_rna_dim": 2565,
    "in_mole_dim": 800,
    "hidden": 256,
    "heads": 4,
    "gnn_layers": 2,
    "cross_layers": 2,
    "dropout": 0.2,
}


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


def find_optimal_weights(preds_matrix: np.ndarray, targets: np.ndarray, metric: str = 'pearson') -> np.ndarray:
    num_models = preds_matrix.shape[1]
    init_weights = np.ones(num_models) / num_models
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(num_models)]
    
    def objective(weights):
        final_pred = np.average(preds_matrix, axis=1, weights=weights)
        if metric == 'rmse':
            return numpy_rmse(final_pred, targets)
        elif metric == 'pearson':
            return -numpy_pearson(final_pred, targets)
        return 0.0
    
    try:
        result = minimize(objective, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 100, 'ftol': 1e-6})
        return result.x
    except Exception:
        return init_weights


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


def load_rna_type_map(json_path: str) -> Dict[str, str]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    id_to_type = {}
    for k, v in data.items():
        if isinstance(v, dict):
            id_to_type[str(k)] = v.get("RNA_type", "Unknown")
        else:
            id_to_type[str(k)] = str(v)
    return id_to_type


def evaluate_ensemble_with_weights(
    models: List[RMPred],
    weights: List[float],
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    all_preds = []
    all_labels = []
    
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            
            batch_model_preds = []
            for model in models:
                rna_x = torch.cat([batch["rna_llm"], batch["rna_onehot"], batch["rna_pssm"]], dim=-1).to(device)
                mole_x = torch.cat([batch["mole_llm"], batch["mole_onehot"]], dim=-1).to(device)
                
                out = model(
                    rna_x, batch["rna_edges"], batch["rna_mask"].to(device),
                    mole_x, batch["mole_edges"], batch["mole_mask"].to(device)
                )
                batch_model_preds.append(out.cpu().numpy())
            
            batch_preds = np.stack(batch_model_preds, axis=0)
            weights_arr = np.array(weights)[:, None]
            weighted_preds = np.sum(batch_preds * weights_arr, axis=0)
            
            all_preds.extend(weighted_preds.tolist())
            all_labels.extend(batch["pkd"].numpy().tolist())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    return numpy_pearson(preds, labels), numpy_rmse(preds, labels)


def print_fold_results(fold_id: int, results: Dict, prefix: str = ""):
    print(f"\n[{prefix}FOLD {fold_id:02d} RESULTS]")
    print(f"  Simple Avg:   RMSE={results['simple_avg']['rmse']:.4f} P={results['simple_avg']['pearson']:.4f}")
    print(f"  Weighted Avg: RMSE={results['weighted_avg']['rmse']:.4f} P={results['weighted_avg']['pearson']:.4f}")


def print_comparison(fold_id: int, teacher_pcc: float, student_pcc: float):
    delta = student_pcc - teacher_pcc
    status = "✓ IMPROVED" if delta > 0.005 else ("≈ Maintained" if delta > -0.005 else "✗ DEGRADED")
    print(f"\n[FOLD {fold_id:02d}] Teacher={teacher_pcc:.4f} Student={student_pcc:.4f} Delta={delta:+.4f} {status}")
    return delta


class PseudoLabelDataset(Dataset):
    
    def __init__(self, stores: GlobalStores, pseudo_entries: List[Dict],
                 max_rna_len: int = 1024, max_mole_len: int = 2048):
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
                "rna_id": rna_id, "mol_id": mol_id,
                "rna_llm": r_llm, "rna_onehot": r_oh, "rna_edges": r_edges, "rna_pssm": r_pssm,
                "mole_llm": m_llm, "mole_onehot": m_oh, "mole_edges": m_edges,
                "label": pseudo_label, "sample_weight": weight,
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
    
    entry_ids, rna_ids, mol_ids, labels, weights = [], [], [], [], []
    
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
        "entry_ids": entry_ids, "rna_ids": rna_ids, "mol_ids": mol_ids,
        "rna_llm": rna_llm, "rna_onehot": rna_oh, "rna_edges": rna_edges,
        "rna_pssm": rna_pssm, "rna_mask": rna_mask,
        "mole_llm": mole_llm, "mole_onehot": mole_oh, "mole_edges": mole_edges, "mole_mask": mole_mask,
        "pkd": torch.tensor(labels, dtype=torch.float32),
        "sample_weights": torch.tensor(weights, dtype=torch.float32),
    }


def make_subset_loader(dataset, indices, batch_size, shuffle, num_workers, seed):
    subset = Subset(dataset, indices)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_rmpred_batch, generator=g)


def load_teacher_ensemble(ckpt_dir: str, fold: int, device: torch.device,
                          model_kwargs: dict = None) -> Tuple[List[RMPred], List[float]]:
    if model_kwargs is None:
        model_kwargs = MODEL_CFG.copy()
    
    fold_dir = os.path.join(ckpt_dir, f"fold_{fold:02d}")
    pattern = os.path.join(fold_dir, "member_*_best.pt")
    ckpt_files = sorted(glob.glob(pattern))
    
    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No checkpoints found in {fold_dir}")
    
    models, weights = [], []
    for ckpt_path in ckpt_files:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = RMPred(**model_kwargs).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models.append(model)
        weights.append(ckpt.get('weight', 1.0))
    
    total = sum(weights)
    weights = [w / total for w in weights]
    return models, weights


def ensemble_predict(models: List[RMPred], weights: List[float], batch: dict,
                     device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    rna_x = torch.cat([batch["rna_llm"], batch["rna_onehot"], batch["rna_pssm"]], dim=-1).to(device)
    mole_x = torch.cat([batch["mole_llm"], batch["mole_onehot"]], dim=-1).to(device)
    
    preds = []
    for model in models:
        with torch.no_grad():
            out = model(rna_x, batch["rna_edges"], batch["rna_mask"].to(device),
                       mole_x, batch["mole_edges"], batch["mole_mask"].to(device))
        preds.append(out.cpu().numpy())
    
    preds = np.stack(preds, axis=0)
    weights_arr = np.array(weights)[:, None]
    mean_preds = np.sum(preds * weights_arr, axis=0)
    std_preds = np.std(preds, axis=0)
    return mean_preds, std_preds


def create_student_model(model_kwargs: dict = None, device: torch.device = None) -> RMPred:
    if model_kwargs is None:
        model_kwargs = MODEL_CFG.copy()
    model = RMPred(**model_kwargs)
    if device is not None:
        model = model.to(device)
    return model


def get_teacher_soft_labels(models: List[RMPred], weights: List[float], batch: dict,
                            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rna_x = torch.cat([batch["rna_llm"], batch["rna_onehot"], batch["rna_pssm"]], dim=-1).to(device)
    mole_x = torch.cat([batch["mole_llm"], batch["mole_onehot"]], dim=-1).to(device)
    
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(rna_x, batch["rna_edges"], batch["rna_mask"].to(device),
                       mole_x, batch["mole_edges"], batch["mole_mask"].to(device))
        preds.append(out)
    
    preds = torch.stack(preds, dim=0)
    weights_tensor = torch.tensor(weights, device=device)[:, None]
    soft_labels = (preds * weights_tensor).sum(dim=0)
    uncertainty = preds.std(dim=0)
    return soft_labels, uncertainty


def generate_pseudo_labels_v2(
    models: List[RMPred], stores, labeled_pairs: set, device: torch.device,
    all_rna_ids: List[str], all_mol_ids: List[str],
    max_samples: int = 30, confidence_threshold: float = 0.18,
    min_agreement_ratio: float = 0.8, curriculum_stage: int = 0,
    total_stages: int = 3, verbose: bool = True,
) -> List[Dict]:
    for m in models:
        m.eval()
    
    novel_pairs = []
    for rna_id in all_rna_ids:
        for mol_id in all_mol_ids:
            if (rna_id, mol_id) not in labeled_pairs:
                novel_pairs.append({'rna_id': rna_id, 'mol_id': mol_id, 'pseudo_label': 0.0})
    
    if verbose:
        print(f"[PseudoLabel V2] Found {len(novel_pairs)} novel pairs")
    
    if len(novel_pairs) == 0:
        return []
    
    np.random.shuffle(novel_pairs)
    sample_size = min(len(novel_pairs), max_samples * 50)
    candidates = novel_pairs[:sample_size]
    
    temp_ds = PseudoLabelDataset(stores, candidates)
    temp_loader = DataLoader(temp_ds, batch_size=32, shuffle=False, 
                             num_workers=0, collate_fn=collate_pseudo_batch)
    
    all_predictions, all_rna_ids_out, all_mol_ids_out = [], [], []
    
    with torch.no_grad():
        for batch in temp_loader:
            if batch is None:
                continue
            
            batch_preds = []
            for model in models:
                rna_x = torch.cat([batch["rna_llm"], batch["rna_onehot"], batch["rna_pssm"]], dim=-1).to(device)
                mole_x = torch.cat([batch["mole_llm"], batch["mole_onehot"]], dim=-1).to(device)
                out = model(rna_x, batch["rna_edges"], batch["rna_mask"].to(device),
                           mole_x, batch["mole_edges"], batch["mole_mask"].to(device))
                batch_preds.append(out.cpu().numpy())
            
            batch_preds = np.stack(batch_preds, axis=0)
            for i in range(batch_preds.shape[1]):
                all_predictions.append(batch_preds[:, i])
                all_rna_ids_out.append(batch["rna_ids"][i])
                all_mol_ids_out.append(batch["mol_ids"][i])
    
    if len(all_predictions) == 0:
        return []
    
    candidates_scored = []
    n_models = len(models)
    
    for i, preds in enumerate(all_predictions):
        mean_pred = np.mean(preds)
        std_pred = np.std(preds)
        agreement_margin = 0.5
        n_agree = np.sum(np.abs(preds - mean_pred) < agreement_margin)
        agreement_ratio = n_agree / n_models
        in_range = 2.0 <= mean_pred <= 12.0
        
        candidates_scored.append({
            'rna_id': all_rna_ids_out[i], 'mol_id': all_mol_ids_out[i],
            'pseudo_label': float(mean_pred), 'std': float(std_pred),
            'agreement_ratio': float(agreement_ratio), 'in_range': in_range,
        })
    
    stage_multiplier = 1.0 + (curriculum_stage / total_stages) * 0.5
    effective_threshold = confidence_threshold * stage_multiplier
    effective_agreement = min_agreement_ratio - (curriculum_stage / total_stages) * 0.1
    
    if verbose:
        print(f"[PseudoLabel V2] Stage {curriculum_stage}/{total_stages}: "
              f"threshold={effective_threshold:.3f}, agreement={effective_agreement:.2f}")
    
    filtered = [c for c in candidates_scored 
                if c['std'] < effective_threshold and c['agreement_ratio'] >= effective_agreement and c['in_range']]
    
    if verbose:
        print(f"[PseudoLabel V2] After filtering: {len(filtered)} / {len(candidates_scored)} candidates")
    
    if len(filtered) == 0:
        filtered = sorted(candidates_scored, key=lambda x: x['std'])[:max_samples]
        if verbose:
            print(f"[PseudoLabel V2] Fallback: using top-{len(filtered)} by std")
    
    filtered = sorted(filtered, key=lambda x: x['std'])[:max_samples]
    
    for entry in filtered:
        confidence_weight = 1.0 - (entry['std'] / effective_threshold)
        entry['weight'] = max(0.5, min(1.0, confidence_weight))
        entry['confidence'] = 1.0 - entry['std']
    
    if verbose and filtered:
        stds = [e['std'] for e in filtered]
        print(f"[PseudoLabel V2] Selected {len(filtered)} samples, std range: [{min(stds):.3f}, {max(stds):.3f}]")
    
    return filtered


def get_curriculum_schedule(total_epochs: int, n_stages: int = 3) -> List[int]:
    epochs_per_stage = total_epochs // n_stages
    schedule = []
    for epoch in range(total_epochs):
        stage = min(epoch // epochs_per_stage, n_stages - 1)
        schedule.append(stage)
    return schedule


def compute_pseudo_label_diversity(pseudo_entries: List[Dict]) -> Dict[str, float]:
    if len(pseudo_entries) == 0:
        return {'n_samples': 0, 'n_rnas': 0, 'n_moles': 0, 'label_std': 0.0}
    rnas = set(e['rna_id'] for e in pseudo_entries)
    moles = set(e['mol_id'] for e in pseudo_entries)
    labels = [e['pseudo_label'] for e in pseudo_entries]
    return {
        'n_samples': len(pseudo_entries), 'n_rnas': len(rnas), 'n_moles': len(moles),
        'label_mean': float(np.mean(labels)), 'label_std': float(np.std(labels)),
        'label_range': (float(min(labels)), float(max(labels))),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised training V2')
    parser.add_argument('--fold', type=int, default=None, help='Fold to train (0-9)')
    parser.add_argument('--all', action='store_true', help='Train all folds')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--epochs', type=int, default=SEMI_CONFIG_V2['n_epochs'], help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--teacher_ckpt', type=str, default='ckpt_weights_rnatype_0101',
                        help='Teacher checkpoint directory')
    parser.add_argument('--output_dir', type=str, default='ckpt_semi_v2_rnatype_0101',
                        help='Output directory for student checkpoints')
    return parser.parse_args()


def setup_device(gpu_idx: int):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_idx}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_idx)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def train_one_epoch_semi_v2(student, teachers, teacher_weights, labeled_loader, pseudo_loader,
                            optimizer, scaler, device, epoch, config) -> dict:
    student.train()
    
    criterion = ccc_loss if config.get('use_ccc_loss', False) else nn.MSELoss(reduction='none')
    use_distill = config.get('use_distillation', True)
    distill_weight = config.get('distill_weight', 0.1)
    distill_temp = config.get('distill_temperature', 2.0)
    pseudo_weight = config.get('pseudo_weight', 0.03)
    
    labeled_iter = iter(labeled_loader)
    pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None
    
    total_loss, labeled_loss_sum, pseudo_loss_sum, distill_loss_sum = 0.0, 0.0, 0.0, 0.0
    n_batches = 0
    
    for batch_idx in range(len(labeled_loader)):
        optimizer.zero_grad()
        
        try:
            batch = next(labeled_iter)
        except StopIteration:
            break
        
        if batch is None:
            continue
        
        with autocast():
            rna_x = torch.cat([batch["rna_llm"], batch["rna_onehot"], batch["rna_pssm"]], dim=-1).to(device)
            mole_x = torch.cat([batch["mole_llm"], batch["mole_onehot"]], dim=-1).to(device)
            labels = batch["pkd"].to(device)
            
            student_out = student(rna_x, batch["rna_edges"], batch["rna_mask"].to(device),
                                  mole_x, batch["mole_edges"], batch["mole_mask"].to(device))
            
            labeled_losses = criterion(student_out, labels)
            labeled_loss = labeled_losses.mean() if hasattr(labeled_losses, 'mean') else labeled_losses
            total_batch_loss = labeled_loss
            labeled_loss_sum += labeled_loss.item()
            
            if use_distill and len(teachers) > 0:
                with torch.no_grad():
                    soft_labels, _ = get_teacher_soft_labels(teachers, teacher_weights, batch, device)
                distill_loss = nn.MSELoss()(student_out, soft_labels) / (distill_temp ** 2)
                total_batch_loss = total_batch_loss + distill_weight * distill_loss
                distill_loss_sum += distill_loss.item()
            
            if pseudo_iter is not None:
                try:
                    pseudo_batch = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)
                    pseudo_batch = next(pseudo_iter)
                
                if pseudo_batch is not None:
                    p_rna_x = torch.cat([pseudo_batch["rna_llm"], pseudo_batch["rna_onehot"], 
                                         pseudo_batch["rna_pssm"]], dim=-1).to(device)
                    p_mole_x = torch.cat([pseudo_batch["mole_llm"], pseudo_batch["mole_onehot"]], dim=-1).to(device)
                    p_labels = pseudo_batch["pkd"].to(device)
                    p_weights = pseudo_batch["sample_weights"].to(device)
                    
                    p_student_out = student(p_rna_x, pseudo_batch["rna_edges"], pseudo_batch["rna_mask"].to(device),
                                            p_mole_x, pseudo_batch["mole_edges"], pseudo_batch["mole_mask"].to(device))
                    
                    p_losses = criterion(p_student_out, p_labels)
                    p_loss = (p_losses * p_weights).mean() if hasattr(p_losses, 'mean') else p_losses
                    total_batch_loss = total_batch_loss + pseudo_weight * p_loss
                    pseudo_loss_sum += p_loss.item()
        
        scaler.scale(total_batch_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += total_batch_loss.item()
        n_batches += 1
    
    return {
        'total_loss': total_loss / max(n_batches, 1),
        'labeled_loss': labeled_loss_sum / max(n_batches, 1),
        'pseudo_loss': pseudo_loss_sum / max(n_batches, 1),
        'distill_loss': distill_loss_sum / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_model(model: RMPred, loader: DataLoader, device: torch.device) -> tuple:
    model.eval()
    all_preds, all_labels = [], []
    
    for batch in loader:
        if batch is None:
            continue
        rna_x = torch.cat([batch["rna_llm"], batch["rna_onehot"], batch["rna_pssm"]], dim=-1).to(device)
        mole_x = torch.cat([batch["mole_llm"], batch["mole_onehot"]], dim=-1).to(device)
        out = model(rna_x, batch["rna_edges"], batch["rna_mask"].to(device),
                   mole_x, batch["mole_edges"], batch["mole_mask"].to(device))
        all_preds.extend(out.cpu().numpy().tolist())
        all_labels.extend(batch["pkd"].numpy().tolist())
    
    preds, labels = np.array(all_preds), np.array(all_labels)
    return numpy_pearson(preds, labels), numpy_rmse(preds, labels), preds, labels


def train_fold_v2(fold, stores, full_dataset, df, rna_type_map, device,
                  teacher_ckpt_dir, output_dir, config, verbose=True):
    print(f"\n{'='*60}")
    print(f"FOLD {fold} - Semi-Supervised Training V2")
    print(f"{'='*60}")
    
    fold_out_dir = os.path.join(output_dir, f"fold_{fold:02d}")
    os.makedirs(fold_out_dir, exist_ok=True)
    
    print(f"\n[1] Loading teacher ensemble from {teacher_ckpt_dir}/fold_{fold:02d}")
    teachers, teacher_weights = load_teacher_ensemble(teacher_ckpt_dir, fold, device)
    print(f"    Loaded {len(teachers)} teachers with weights: {[f'{w:.3f}' for w in teacher_weights]}")
    
    from sklearn.model_selection import StratifiedKFold
    rna_ids_list = df['RNA id'].values
    rna_types = [rna_type_map.get(rid, 'unknown') for rid in rna_ids_list]
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.get('seed', 42))
    splits = list(skf.split(range(len(df)), rna_types))
    train_idx, val_idx = splits[fold]
    
    print(f"\n[2] Data split: Train={len(train_idx)}, Val={len(val_idx)}")
    
    train_loader = make_subset_loader(full_dataset, train_idx, batch_size=config.get('batch_size', 16),
                                       shuffle=True, num_workers=4, seed=config.get('seed', 42))
    val_loader = make_subset_loader(full_dataset, val_idx, batch_size=config.get('batch_size', 16),
                                     shuffle=False, num_workers=4, seed=config.get('seed', 42))
    
    print(f"\n[3] Evaluating teacher ensemble on validation set...")
    teacher_pcc, teacher_rmse = evaluate_ensemble_with_weights(teachers, teacher_weights, val_loader, device)
    print(f"    Teacher: PCC={teacher_pcc:.4f}, RMSE={teacher_rmse:.4f}")
    
    labeled_pairs = set()
    for _, row in df.iloc[train_idx].iterrows():
        labeled_pairs.add((row['RNA id'], row['SM id']))
    
    all_rna_ids = df['RNA id'].unique().tolist()
    all_mol_ids = df['SM id'].unique().tolist()
    
    n_members = config.get('n_members', 5)
    n_epochs = config.get('n_epochs', 20)
    curriculum_stages = config.get('curriculum_stages', 3)
    curriculum_schedule = get_curriculum_schedule(n_epochs, curriculum_stages)
    
    student_results, all_val_preds = [], []
    
    for member_idx in range(n_members):
        print(f"\n[4.{member_idx+1}] Training student member {member_idx+1}/{n_members}")
        
        student = create_student_model(device=device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=config.get('lr', 5e-5),
                                       weight_decay=config.get('weight_decay', 0.01))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        scaler = GradScaler()
        
        best_pcc, best_state = -1.0, None
        patience = config.get('patience', 8)
        no_improve = 0
        pseudo_loader = None
        current_stage = -1
        
        for epoch in range(n_epochs):
            epoch_stage = curriculum_schedule[epoch]
            
            if epoch_stage != current_stage or epoch == 0:
                current_stage = epoch_stage
                print(f"\n    Epoch {epoch+1}: Curriculum stage {current_stage} - regenerating pseudo-labels...")
                
                pseudo_entries = generate_pseudo_labels_v2(
                    teachers, stores, labeled_pairs, device, all_rna_ids, all_mol_ids,
                    max_samples=config.get('pseudo_samples', 30),
                    confidence_threshold=config.get('confidence_threshold', 0.18),
                    min_agreement_ratio=config.get('min_agreement_ratio', 0.8),
                    curriculum_stage=current_stage, total_stages=curriculum_stages,
                    verbose=verbose and member_idx == 0,
                )
                
                if len(pseudo_entries) > 0:
                    pseudo_ds = PseudoLabelDataset(stores, pseudo_entries)
                    pseudo_loader = DataLoader(pseudo_ds, batch_size=config.get('batch_size', 16),
                                               shuffle=True, num_workers=0, collate_fn=collate_pseudo_batch)
                    if verbose and member_idx == 0:
                        diversity = compute_pseudo_label_diversity(pseudo_entries)
                        print(f"    Pseudo-label diversity: {diversity}")
                else:
                    pseudo_loader = None
            
            losses = train_one_epoch_semi_v2(student, teachers, teacher_weights, train_loader,
                                              pseudo_loader, optimizer, scaler, device, epoch, config)
            scheduler.step()
            
            val_pcc, val_rmse, _, _ = evaluate_model(student, val_loader, device)
            
            if verbose:
                print(f"    Epoch {epoch+1:2d}: Loss={losses['total_loss']:.4f} "
                      f"(L={losses['labeled_loss']:.4f}, P={losses['pseudo_loss']:.4f}, D={losses['distill_loss']:.4f}) "
                      f"Val PCC={val_pcc:.4f}, RMSE={val_rmse:.4f}")
            
            if val_pcc > best_pcc:
                best_pcc = val_pcc
                best_state = {'model_state_dict': student.state_dict(), 'epoch': epoch,
                              'pcc': val_pcc, 'rmse': val_rmse}
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            save_path = os.path.join(fold_out_dir, f"student_{member_idx:02d}_best.pt")
            torch.save(best_state, save_path)
            print(f"    Saved: {save_path} (PCC={best_pcc:.4f})")
            
            student.load_state_dict(best_state['model_state_dict'])
            _, _, val_preds, val_labels = evaluate_model(student, val_loader, device)
            all_val_preds.append(val_preds)
            student_results.append({'member': member_idx, 'pcc': best_pcc})
    
    if len(all_val_preds) > 0:
        ensemble_preds = np.mean(all_val_preds, axis=0)
        _, _, _, val_labels = evaluate_model(teachers[0], val_loader, device)
        
        student_ensemble_pcc = numpy_pearson(ensemble_preds, val_labels)
        student_ensemble_rmse = numpy_rmse(ensemble_preds, val_labels)
        
        print(f"\n    Student Ensemble: PCC={student_ensemble_pcc:.4f}, RMSE={student_ensemble_rmse:.4f}")
        print(f"    Teacher Ensemble: PCC={teacher_pcc:.4f}, RMSE={teacher_rmse:.4f}")
        
        delta_pcc = student_ensemble_pcc - teacher_pcc
        status = "✓ IMPROVED" if delta_pcc > 0.005 else ("≈ Maintained" if delta_pcc > -0.005 else "✗ DEGRADED")
        print(f"    Delta: {delta_pcc:+.4f} {status}")
        
        fold_results = {
            'fold': fold, 'teacher_pcc': teacher_pcc, 'teacher_rmse': teacher_rmse,
            'student_pcc': student_ensemble_pcc, 'student_rmse': student_ensemble_rmse,
            'delta_pcc': delta_pcc, 'members': student_results,
        }
        
        with open(os.path.join(fold_out_dir, 'results.json'), 'w') as f:
            json.dump(fold_results, f, indent=2)
        
        return fold_results
    return None


def main():
    args = parse_args()
    
    print("="*60)
    print("Semi-Supervised Training V2 - Consolidated")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    device = setup_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    config = dict(SEMI_CONFIG_V2)
    config['n_epochs'] = args.epochs
    config['seed'] = args.seed
    
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    print(f"\nLoading data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(base_dir, 'data')
    
    df = pd.read_csv(os.path.join(data_dir, 'All_sf_dataset_v1.csv'))
    print(f"  Dataset: {len(df)} entries")
    
    rna_type_map = load_rna_type_map(os.path.join(data_dir, 'type', 'rna_types.json'))
    
    stores = GlobalStores(
        msa_h5=os.path.join(data_dir, 'processed', 'msa_embed.h5'),
        mole_h5=os.path.join(data_dir, 'processed', 'mole_embed.h5'),
        rna_pkl=os.path.join(data_dir, 'processed', 'rna_graphs.pkl'),
        mole_pkl=os.path.join(data_dir, 'processed', 'mole_graphs.pkl'),
        seq_json=os.path.join(data_dir, 'processed', 'rna_seqs.json'),
        smiles_json=os.path.join(data_dir, 'processed', 'mole_smiles.json'),
        pssm_dir=os.path.join(data_dir, 'pssm_npy_5d'),
    )
    
    full_dataset = RMPredDataset(df=df, stores=stores, label_col='pKd', rna_col='RNA id', mole_col='SM id')
    
    if args.all:
        folds = list(range(10))
    elif args.fold is not None:
        folds = [args.fold]
    else:
        folds = [0]
    
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    teacher_ckpt_dir = os.path.join(script_dir, args.teacher_ckpt)
    
    all_results = []
    for fold in folds:
        result = train_fold_v2(fold=fold, stores=stores, full_dataset=full_dataset, df=df,
                               rna_type_map=rna_type_map, device=device,
                               teacher_ckpt_dir=teacher_ckpt_dir, output_dir=output_dir,
                               config=config, verbose=args.verbose)
        if result:
            all_results.append(result)
    
    if len(all_results) > 0:
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        
        teacher_pccs = [r['teacher_pcc'] for r in all_results]
        student_pccs = [r['student_pcc'] for r in all_results]
        deltas = [r['delta_pcc'] for r in all_results]
        
        print(f"\nTeacher: PCC={np.mean(teacher_pccs):.4f} ± {np.std(teacher_pccs):.4f}")
        print(f"Student: PCC={np.mean(student_pccs):.4f} ± {np.std(student_pccs):.4f}")
        print(f"Delta:   {np.mean(deltas):+.4f}")
        
        n_improved = sum(1 for d in deltas if d > 0.005)
        n_maintained = sum(1 for d in deltas if -0.005 <= d <= 0.005)
        n_degraded = sum(1 for d in deltas if d < -0.005)
        print(f"\nImproved: {n_improved}, Maintained: {n_maintained}, Degraded: {n_degraded}")
        
        overall = {
            'timestamp': datetime.now().isoformat(), 'config': config, 'folds': all_results,
            'summary': {
                'teacher_pcc_mean': float(np.mean(teacher_pccs)),
                'teacher_pcc_std': float(np.std(teacher_pccs)),
                'student_pcc_mean': float(np.mean(student_pccs)),
                'student_pcc_std': float(np.std(student_pccs)),
                'delta_mean': float(np.mean(deltas)),
                'n_improved': n_improved, 'n_maintained': n_maintained, 'n_degraded': n_degraded,
            }
        }
        
        with open(os.path.join(output_dir, 'overall_results.json'), 'w') as f:
            json.dump(overall, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
