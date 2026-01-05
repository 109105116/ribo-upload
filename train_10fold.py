import os
import math
import argparse
import random
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import shuffle as sklearn_shuffle 
from scipy.optimize import minimize

from RNAdataset import *
from RMPred import RMPred
from RNAdataset import compute_metrics
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


def find_optimal_weights(preds_matrix: np.ndarray, targets: np.ndarray, metric: str = 'rmse') -> np.ndarray:
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
        result = minimize(
            objective, 
            init_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        return result.x
    except Exception as e:
        print(f"  Warning: Weight optimization failed ({e}), using equal weights")
        return init_weights


@torch.no_grad()
def evaluate_mu(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device, 
    type_map: Dict[str, str] = None
) -> Dict[str, Any]:
    model.eval()
    all_preds, all_ys, all_types = [], [], []

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

        with autocast():
            mu = model(
                rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                mole_mask=batch["mole_mask"],
            )
        
        all_preds.append(mu.squeeze(-1)[keep].detach().cpu())
        all_ys.append(y[keep].detach().cpu())

        if type_map is not None and batch_ids is not None:
            keep_cpu = keep.cpu().tolist()
            valid_ids = [bid for bid, k in zip(batch_ids, keep_cpu) if k]
            all_types.extend([type_map.get(str(eid), "Unknown") for eid in valid_ids])

    if not all_preds:
        return {"global": {"pearson": float("nan"), "rmse": float("nan")}}

    mu_all = torch.cat(all_preds, dim=0)
    y_all = torch.cat(all_ys, dim=0)
    results = {"global": compute_metrics(mu_all, y_all)}

    if type_map is not None and len(all_types) == len(mu_all):
        type_buckets = defaultdict(lambda: {"preds": [], "ys": []})
        for p, t, r_type in zip(mu_all, y_all, all_types):
            type_buckets[r_type]["preds"].append(p)
            type_buckets[r_type]["ys"].append(t)
        for r_type, data in type_buckets.items():
            results[r_type] = compute_metrics(torch.tensor(data["preds"]), torch.tensor(data["ys"]))

    return results


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


def evaluate_ensemble_with_weights(
    preds_matrix: np.ndarray, 
    y_all: np.ndarray, 
    all_types: List[str],
    optimize_metric: str = 'pearson'
) -> Dict[str, Any]:
    results = {}
    
    simple_preds = np.mean(preds_matrix, axis=1)
    results["simple_avg"] = {
        "rmse": numpy_rmse(simple_preds, y_all),
        "pearson": numpy_pearson(simple_preds, y_all),
        "count": len(y_all)
    }
    
    if num_models >= 3:
        sorted_preds = np.sort(preds_matrix, axis=1)
        trimmed_preds = np.mean(sorted_preds[:, 1:-1], axis=1)
        results["trimmed_avg"] = {
            "rmse": numpy_rmse(trimmed_preds, y_all),
            "pearson": numpy_pearson(trimmed_preds, y_all),
            "count": len(y_all)
        }
    else:
        results["trimmed_avg"] = results["simple_avg"].copy()
    
    best_weights = find_optimal_weights(preds_matrix, y_all, metric=optimize_metric)
    weighted_preds = np.average(preds_matrix, axis=1, weights=best_weights)
    results["weighted_avg"] = {
        "rmse": numpy_rmse(weighted_preds, y_all),
        "pearson": numpy_pearson(weighted_preds, y_all),
        "count": len(y_all)
    }
    results["best_weights"] = best_weights.tolist()
    
    if all_types and len(all_types) == len(weighted_preds):
        type_buckets = defaultdict(lambda: {"preds": [], "ys": []})
        for pred, target, rna_type in zip(weighted_preds, y_all, all_types):
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


def train_one_epoch_mu(model, loader, optimizer, device, scaler, grad_clip=1.0, ccc_weight=0.2):
    model.train()
    total_loss, total_mse, total_ccc, total_n = 0.0, 0.0, 0.0, 0
    
    for batch in loader:
        if batch is None: 
            continue
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None: 
            raise KeyError("Batch has no 'pkd' or 'labels'.")
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
        
    denom = max(1, total_n)
    return {"loss": total_loss/denom, "mse": total_mse/denom, "ccc_loss": total_ccc/denom}


def make_subset_loader(dataset, indices, batch_size, shuffle, num_workers, seed):
    subset = Subset(dataset, indices)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                      collate_fn=collate_rmpred_batch, generator=g)


def _get_entry_label(entry, label_key):
    for k in (label_key, label_key.lower(), label_key.upper(), "pKd", "pKD", "affinity", "label"):
        if k in entry and entry[k] is not None:
            try: 
                return float(entry[k])
            except: 
                pass
    return None


class regressor_stratified_cv:
    def __init__(self, n_splits=10, n_repeats=2, group_count=10, random_state=0, strategy='quantile'):
        self.group_count = group_count
        self.strategy = strategy
        self.cvkwargs = dict(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.cv = RepeatedStratifiedKFold(**self.cvkwargs)
        self.discretizer = KBinsDiscretizer(n_bins=self.group_count, encode='ordinal', strategy=self.strategy)
        
    def split(self, X, y, groups=None):
        kgroups = self.discretizer.fit_transform(y[:, None])[:, 0]
        return self.cv.split(X, kgroups, groups)
    
    def get_n_splits(self, X, y, groups=None):
        return self.cv.get_n_splits(X, y, groups)


def repeated_stratified_kfold_indices(*, stores, entry_ids, label_key, n_splits, n_repeats, group_count, strategy, seed):
    y_raw = np.array([_get_entry_label(stores.entry_binding[eid], label_key) for eid in entry_ids], dtype=np.float32)
    idx_raw = np.arange(len(entry_ids))
    keep_mask = np.isfinite(y_raw)
    y_clean = y_raw[keep_mask]
    idx_clean = idx_raw[keep_mask]
    
    if len(y_clean) < n_splits: 
        raise ValueError("Not enough finite labels.")
    idx_shuffled, y_shuffled = sklearn_shuffle(idx_clean, y_clean, random_state=seed)

    cv = regressor_stratified_cv(
        n_splits=n_splits, n_repeats=n_repeats, group_count=group_count,
        random_state=seed, strategy=strategy,
    )
    X_dummy = np.zeros((len(y_shuffled), 1), dtype=np.float32)
    folds = []
    for tr_rel, va_rel in cv.split(X_dummy, y_shuffled):
        folds.append((idx_shuffled[tr_rel], idx_shuffled[va_rel]))

    print(f"[Split] bins={group_count} | strategy={strategy} | splits={n_splits} | repeats={n_repeats} | seed={seed}")
    return folds, y_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="ensemble size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--bootstrap", action="store_true", help="bagging")
    ap.add_argument("--out_dir", type=str, default="ckpt_final")
    ap.add_argument("--max_rna_len", type=int, default=1024)
    ap.add_argument("--max_mole_len", type=int, default=2048)
    ap.add_argument("--early_patience", type=int, default=20)
    ap.add_argument("--label_key", type=str, default="pkd")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--strat_bins", type=int, default=5)
    ap.add_argument("--strat_strategy", type=str, default="uniform")
    ap.add_argument("--metric", type=str, default="rmse", choices=["rmse", "pearson"])
    
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Best Metric: {args.metric.upper()}")
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
    else:
        print(f"Warning: Dictionary not found at {DICT_PATH}")

    dataset = RMPredDataset(
        stores, strict=True, max_rna_len=args.max_rna_len,
        max_mole_len=(None if args.max_mole_len == 0 else args.max_mole_len),
        truncate_if_exceed=False, label_key=args.label_key,
    )

    n_total = len(dataset)
    print(f"Total={n_total} | Stratified {args.folds}-fold")

    set_seed(args.seed)

    temp_loader = DataLoader(Subset(dataset, [0]), batch_size=1, collate_fn=collate_rmpred_batch)
    batch0 = next(iter(temp_loader))
    dim_rna_llm = batch0["rna_llm"].shape[-1]
    dim_mole_llm = batch0["mole_llm"].shape[-1]
    c_onehot_rna = batch0["rna_onehot"].shape[-1]
    c_onehot_mole = batch0["mole_onehot"].shape[-1]
    d_pssm = batch0["rna_pssm"].shape[-1]

    entry_ids = getattr(dataset, "keys", None)
    if entry_ids is None: 
        entry_ids = list(stores.entry_binding.keys())

    folds, y_all = repeated_stratified_kfold_indices(
        stores=stores, entry_ids=entry_ids, label_key=args.label_key,
        n_splits=args.folds, n_repeats=args.repeats, group_count=args.strat_bins, 
        strategy=args.strat_strategy, seed=args.seed,
    )

    fold_simple = []
    fold_trimmed = []
    fold_weighted = []
    all_val_preds = [None] * n_total
    all_val_targets = [None] * n_total

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        print(f"\n===== FOLD {fold_id+1}/{len(folds)} | train={len(train_idx)} val={len(val_idx)} =====")
        fold_dir = os.path.join(args.out_dir, f"fold_{fold_id:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        val_loader = make_subset_loader(dataset, val_idx.tolist(), batch_size=args.val_batch_size, 
                                        shuffle=False, num_workers=0, seed=args.seed + 999 + fold_id)

        best_ckpts = []
        
        for m in range(args.k):
            member_seed = args.seed + 1000 * (fold_id * args.k + m)
            set_seed(member_seed)

            if args.bootstrap:
                tr_indices = [random.choice(train_idx.tolist()) for _ in range(len(train_idx))]
            else:
                tr_indices = train_idx.tolist()

            train_loader = make_subset_loader(dataset, tr_indices, batch_size=args.batch_size, 
                                              shuffle=True, num_workers=0, seed=member_seed)

            model = RMPred(
                d_llm_rna=dim_rna_llm, c_onehot_rna=c_onehot_rna, d_pssm_rna=d_pssm,
                d_llm_mole=dim_mole_llm, c_onehot_mole=c_onehot_mole,
                d_model_inner=256, d_model_fusion=512, dropout=0.2,
                fusion_layers=2, fusion_heads=4, rna_gnn_layers=4, rna_gnn_heads=4,
                mole_gnn_layers=4, mole_gnn_heads=4, mole_num_edge_types=8, rna_max_len=args.max_rna_len,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scaler = GradScaler()

            best_score = float("-inf") if args.metric == "pearson" else float("inf")
            best_path = os.path.join(fold_dir, f"member_{m:02d}_best.pt")
            patience = 0

            print(f"\n--- Train member {m+1}/{args.k} ---")
            
            for epoch in range(1, args.epochs + 1):
                tr = train_one_epoch_mu(model, train_loader, optimizer, device, scaler, grad_clip=1.0, ccc_weight=1.0)
                val_results = evaluate_mu(model, val_loader, device, type_map=type_map)

                current_rmse = val_results["global"]["rmse"]
                current_pearson = val_results["global"]["pearson"]
                current_score = current_pearson if args.metric == "pearson" else current_rmse

                print(f"[Ep {epoch:03d}] Loss={tr['loss']:.4f} | Global: RMSE={current_rmse:.4f} P={current_pearson:.4f}")
                for rna_type, metrics in val_results.items():
                    if rna_type != "global":
                        print(f"    - {rna_type}: RMSE={metrics['rmse']:.4f} P={metrics['pearson']:.4f}")

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
                        "dims": {
                            "dim_rna_llm": dim_rna_llm, "dim_mole_llm": dim_mole_llm,
                            "c_onehot_rna": c_onehot_rna, "c_onehot_mole": c_onehot_mole, "d_pssm": d_pssm,
                        },
                    }, best_path)
                else:
                    patience += 1
                    if patience >= args.early_patience:
                        print(f"  -> Early stop. Best {args.metric}={best_score:.4f}")
                        break

            print(f"Member {m:02d} finished. Best {args.metric}={best_score:.4f}")
            best_ckpts.append(best_path)

        print(f"\nEvaluating Ensemble for Fold {fold_id}...")
        models = []
        for ckpt in best_ckpts:
            sd = torch.load(ckpt, map_location=device)
            model = RMPred(
                d_llm_rna=dim_rna_llm, c_onehot_rna=c_onehot_rna, d_pssm_rna=d_pssm,
                d_llm_mole=dim_mole_llm, c_onehot_mole=c_onehot_mole,
                d_model_inner=256, d_model_fusion=512, dropout=0.2,
                fusion_layers=2, fusion_heads=4, rna_gnn_layers=4, rna_gnn_heads=4,
                mole_gnn_layers=4, mole_gnn_heads=4, mole_num_edge_types=8, rna_max_len=args.max_rna_len,
            ).to(device)
            model.load_state_dict(sd["model_state"], strict=True)
            models.append(model)

        preds_matrix, y_val, all_types = collect_ensemble_predictions(models, val_loader, device, type_map)
        
        if len(y_val) == 0:
            print("  Warning: No validation samples!")
            continue
            
        ens_results = evaluate_ensemble_with_weights(preds_matrix, y_val, all_types, optimize_metric=args.metric)
        
        print(f"\n[FOLD {fold_id:02d} ENSEMBLE RESULTS]")
        print(f"  Simple Avg:   RMSE={ens_results['simple_avg']['rmse']:.4f} P={ens_results['simple_avg']['pearson']:.4f}")
        print(f"  Trimmed Avg:  RMSE={ens_results['trimmed_avg']['rmse']:.4f} P={ens_results['trimmed_avg']['pearson']:.4f}")
        print(f"  Weighted Avg: RMSE={ens_results['weighted_avg']['rmse']:.4f} P={ens_results['weighted_avg']['pearson']:.4f}")
        print(f"  -> Weights: {[f'{w:.3f}' for w in ens_results['best_weights']]}")
        
        for key, metrics in ens_results.items():
            if key not in ["simple_avg", "weighted_avg", "trimmed_avg", "best_weights"]:
                print(f"    Type {key}: RMSE={metrics['rmse']:.4f} P={metrics['pearson']:.4f} N={metrics['count']}")

        fold_simple.append(ens_results["simple_avg"])
        fold_trimmed.append(ens_results["trimmed_avg"])
        fold_weighted.append(ens_results["weighted_avg"])

        best_w = np.array(ens_results["best_weights"])
        weighted_preds = np.average(preds_matrix, axis=1, weights=best_w)
        
        eid_to_idx = {eid: i for i, eid in enumerate(entry_ids)}
        val_idx_list = val_idx.tolist()
        
        for i, (pred, target) in enumerate(zip(weighted_preds, y_val)):
            if i < len(val_idx_list):
                di = val_idx_list[i]
                all_val_preds[di] = float(pred)
                all_val_targets[di] = float(target)

    print(f"\n===== {args.folds}-FOLD SUMMARY =====")
    
    simple_rmses = [m["rmse"] for m in fold_simple]
    simple_pears = [m["pearson"] for m in fold_simple]
    print(f"\nSimple Average:")
    print(f"  RMSE:    mean={np.nanmean(simple_rmses):.4f} std={np.nanstd(simple_rmses):.4f}")
    print(f"  Pearson: mean={np.nanmean(simple_pears):.4f} std={np.nanstd(simple_pears):.4f}")
    
    trimmed_rmses = [m["rmse"] for m in fold_trimmed]
    trimmed_pears = [m["pearson"] for m in fold_trimmed]
    print(f"\nTrimmed Average:")
    print(f"  RMSE:    mean={np.nanmean(trimmed_rmses):.4f} std={np.nanstd(trimmed_rmses):.4f}")
    print(f"  Pearson: mean={np.nanmean(trimmed_pears):.4f} std={np.nanstd(trimmed_pears):.4f}")
    
    weighted_rmses = [m["rmse"] for m in fold_weighted]
    weighted_pears = [m["pearson"] for m in fold_weighted]
    print(f"\nWeighted Average (Optimized):")
    print(f"  RMSE:    mean={np.nanmean(weighted_rmses):.4f} std={np.nanstd(weighted_rmses):.4f}")
    print(f"  Pearson: mean={np.nanmean(weighted_pears):.4f} std={np.nanstd(weighted_pears):.4f}")

    valid_oof = [(p, t) for p, t in zip(all_val_preds, all_val_targets) if p is not None and t is not None]
    if valid_oof:
        p_arr = np.array([x[0] for x in valid_oof])
        t_arr = np.array([x[1] for x in valid_oof])
        print(f"\n===== OOF (all folds concatenated) =====")
        print(f"OOF RMSE={numpy_rmse(p_arr, t_arr):.4f} | OOF Pearson={numpy_pearson(p_arr, t_arr):.4f}")


if __name__ == "__main__":
    main()
