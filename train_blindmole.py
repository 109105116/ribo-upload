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

from RNAdataset import *

from utils import * 
from config import BASE_DIR, PSSM_DIR, DICT_PATH

from RMPred import RMPred
def get_blind_mole_kfold_indices(stores, entry_ids, n_splits, seed):
    all_sample_mole_ids = []
    
    sample_entry = stores.entry_binding[entry_ids[0]]
    if 'ligand_name' in sample_entry:
        mole_key = 'ligand_name'
    elif 'mole_id' in sample_entry:
        mole_key = 'mole_id'
    else:
        mole_key = 'ligand_name' 
    
    print(f"[Info] Using key '{mole_key}' for Molecule Blind Split.")

    for eid in entry_ids:
        if eid in stores.entry_binding:
            val = stores.entry_binding[eid].get(mole_key, "UNKNOWN")
            all_sample_mole_ids.append(val)
        else:
            print(f"Warning: Entry ID {eid} not found in binding store.")
            all_sample_mole_ids.append("UNKNOWN")

    unique_moles = np.array(sorted(list(set(all_sample_mole_ids))))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    mole_to_sample_indices = defaultdict(list)
    for idx, m_id in enumerate(all_sample_mole_ids):
        mole_to_sample_indices[m_id].append(idx)
        
    folds = []
    print(f"\n[Split Strategy] Blind Molecule (Scaffold Split) | Unique Molecules: {len(unique_moles)} | Folds: {n_splits}")
    
    for i, (train_mole_indices, val_mole_indices) in enumerate(kf.split(unique_moles)):
        train_moles = set(unique_moles[train_mole_indices])
        val_moles = set(unique_moles[val_mole_indices])
        
        train_idx = []
        val_idx = []
        
        for m_id in train_moles:
            train_idx.extend(mole_to_sample_indices[m_id])
        for m_id in val_moles:
            val_idx.extend(mole_to_sample_indices[m_id])
            
        random.Random(seed + i).shuffle(train_idx)
        random.Random(seed + i).shuffle(val_idx)
        
        folds.append((np.array(train_idx), np.array(val_idx)))
        print(f"  Fold {i+1}: Train Samples={len(train_idx)} (Moles={len(train_moles)}) | Val Samples={len(val_idx)} (Moles={len(val_moles)})")
    
    return folds


@torch.no_grad()
def evaluate_mu(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device, 
    type_map: Dict[str, str] = None
) -> Dict[str, Any]:
    
    model.eval()
    all_preds = []
    all_ys = []
    all_types = [] 

    for batch in loader:
        if batch is None: continue
        batch_ids = batch.get("entry_ids") or batch.get("entry_id")
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None: continue
        y = y.to(device)
        
        keep = torch.isfinite(y)
        if keep.sum().item() == 0: continue

        with autocast(): 
            mu = model(
                rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                mole_mask=batch["mole_mask"],
            )
        
        valid_preds = mu.squeeze(-1)[keep].detach().cpu()
        valid_ys = y[keep].detach().cpu()
        all_preds.append(valid_preds)
        all_ys.append(valid_ys)

        if type_map is not None and batch_ids is not None:
            keep_cpu = keep.cpu().tolist()
            valid_ids = [bid for bid, k in zip(batch_ids, keep_cpu) if k]
            batch_types = [type_map.get(str(eid), "Unknown") for eid in valid_ids]
            all_types.extend(batch_types)

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

def ccc_loss(mu: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mu = mu.float().view(-1); y = y.float().view(-1)
    if mu.numel() < 2: return torch.mean((mu - y) ** 2)
    mu_mean = mu.mean(); y_mean = y.mean()
    mu_var = mu.var(unbiased=False); y_var = y.var(unbiased=False)
    cov = ((mu - mu_mean) * (y - y_mean)).mean()
    ccc = (2.0 * cov) / (mu_var + y_var + (mu_mean - y_mean).pow(2) + eps)
    return 1.0 - torch.clamp(ccc, -1.0, 1.0)

@torch.no_grad()
def evaluate_ensemble_mu(models, loader, device, type_map=None):
    for m in models: m.eval()
    all_preds = []; all_ys = []; all_types = []

    for batch in loader:
        if batch is None: continue
        batch_ids = batch.get("entry_ids") or batch.get("entry_id")
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None: continue
        y = y.to(device)
        keep = torch.isfinite(y)
        if keep.sum().item() == 0: continue

        member_preds = []
        for m in models:
            with autocast():
                mu, _ = m(
                    rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                    rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                    mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                    mole_mask=batch["mole_mask"],
                )
            member_preds.append(mu.squeeze(-1)[keep].detach().cpu())
        
        mu_mean = torch.stack(member_preds, dim=0).mean(dim=0)
        valid_ys = y[keep].detach().cpu()
        all_preds.append(mu_mean)
        all_ys.append(valid_ys)

        if type_map is not None and batch_ids is not None:
            keep_cpu = keep.cpu().tolist()
            valid_ids = [bid for bid, k in zip(batch_ids, keep_cpu) if k]
            batch_types = [type_map.get(str(eid), "Unknown") for eid in valid_ids]
            all_types.extend(batch_types)

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

def train_one_epoch_mu(model, loader, optimizer, device, scaler, grad_clip=1.0, ccc_weight=1.0):
    model.train()
    total_loss, total_mse, total_ccc, total_n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        if batch is None: continue
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None: raise KeyError("Batch has no 'pkd' or 'labels'.")
        y = y.to(device)
        keep = torch.isfinite(y)
        if keep.sum().item() == 0: continue
        yy = y[keep]

        with autocast():
            mu = model(
                rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                mole_mask=batch["mole_mask"],
            )
            mu = mu.squeeze(-1)[keep]
            mse, ccc_l = mse_loss(mu, yy), ccc_loss(mu, yy)
            loss = 0.5 * (mse + ccc_weight * ccc_l)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()
        
        bs = int(yy.numel())
        total_loss += loss.item() * bs; total_mse += mse.item() * bs; total_ccc += ccc_l.item() * bs; total_n += bs
    denom = max(1, total_n)
    return {"loss": total_loss/denom, "mse": total_mse/denom, "ccc_loss": total_ccc/denom}

def make_subset_loader(dataset, indices, batch_size, shuffle, num_workers, seed):
    subset = Subset(dataset, indices)
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_rmpred_batch, generator=g)

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
    ap.add_argument("--out_dir", type=str, default="ckpt_mole_blind")
    ap.add_argument("--max_rna_len", type=int, default=1024)
    ap.add_argument("--max_mole_len", type=int, default=2048)
    ap.add_argument("--early_patience", type=int, default=20)
    ap.add_argument("--label_key", type=str, default="pkd")
    ap.add_argument("--folds", type=int, default=5, help="Number of folds for Molecule scaffold split")
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
        try: type_map = load_rna_type_map(DICT_PATH)
        except: print(f"Warning: Could not load dictionary from {DICT_PATH}")

    dataset = RMPredDataset(
        stores, strict=True, max_rna_len=args.max_rna_len,
        max_mole_len=(None if args.max_mole_len == 0 else args.max_mole_len),
        truncate_if_exceed=False, label_key=args.label_key,
    )

    n_total = len(dataset)
    print(f"Total Samples={n_total}")
    set_seed(args.seed)

    temp_loader = DataLoader(Subset(dataset, [0]), batch_size=1, collate_fn=collate_rmpred_batch)
    batch0 = next(iter(temp_loader))
    dim_rna_llm = batch0["rna_llm"].shape[-1]
    dim_mole_llm = batch0["mole_llm"].shape[-1]
    c_onehot_rna = batch0["rna_onehot"].shape[-1]
    c_onehot_mole = batch0["mole_onehot"].shape[-1]
    d_pssm = batch0["rna_pssm"].shape[-1]

    entry_ids = getattr(dataset, "keys", None)
    if entry_ids is None: entry_ids = list(stores.entry_binding.keys())

    folds = get_blind_mole_kfold_indices(
        stores=stores, 
        entry_ids=entry_ids, 
        n_splits=args.folds, 
        seed=args.seed
    )

    fold_metrics = []
    all_val_preds = [None] * n_total
    all_val_targets = [None] * n_total

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        print(f"\n===== FOLD {fold_id+1}/{len(folds)} | train={len(train_idx)} val={len(val_idx)} =====")
        fold_dir = os.path.join(args.out_dir, f"fold_{fold_id:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        val_loader = make_subset_loader(dataset, val_idx.tolist(), batch_size=args.val_batch_size, shuffle=False, num_workers=0, seed=args.seed + 999 + fold_id)
        best_ckpts = []

        for m in range(args.k):
            member_seed = args.seed + 1000 * (fold_id * args.k + m)
            set_seed(member_seed)

            if args.bootstrap:
                tr_indices = [random.choice(train_idx.tolist()) for _ in range(len(train_idx))]
            else:
                tr_indices = train_idx.tolist()

            train_loader = make_subset_loader(dataset, tr_indices, batch_size=args.batch_size, shuffle=True, num_workers=0, seed=member_seed)

            model = RMPred(
                d_llm_rna=dim_rna_llm, c_onehot_rna=c_onehot_rna, d_pssm_rna=d_pssm,
                d_llm_mole=dim_mole_llm, c_onehot_mole=c_onehot_mole,
                d_model_inner=256, d_model_fusion=512, dropout=0.2,
                fusion_layers=2, fusion_heads=4, rna_gnn_layers=4, rna_gnn_heads=4,
                mole_gnn_layers=4, mole_gnn_heads=4, mole_num_edge_types=8, rna_max_len=args.max_rna_len,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scaler = GradScaler()

            if args.metric == "pearson": best_score = float("-inf")
            else: best_score = float("inf")
            
            best_path = os.path.join(fold_dir, f"member_{m:02d}_best.pt")
            patience = 0

            print(f"\n--- Train member {m+1}/{args.k} ---")
            for epoch in range(1, args.epochs + 1):
                tr = train_one_epoch_mu(model, train_loader, optimizer, device, scaler, grad_clip=1.0, ccc_weight=1.0)
                val_results = evaluate_mu(model, val_loader, device, type_map=type_map)

                current_rmse = val_results["global"]["rmse"]
                current_pearson = val_results["global"]["pearson"]
                if args.metric == "pearson": current_score = current_pearson
                else: current_score = current_rmse

                print(f"[Ep {epoch:03d}] Loss={tr['loss']:.4f} | Global: RMSE={current_rmse:.4f} P={current_pearson:.4f}")
                
                is_best = False
                if math.isfinite(current_score):
                    if args.metric == "pearson":
                        if current_score > best_score: is_best = True
                    else: 
                        if current_score < best_score: is_best = True

                if is_best:
                    best_score = current_score
                    patience = 0
                    torch.save(
                        {
                            "fold": fold_id, "member": m, "epoch": epoch,
                            "model_state": model.state_dict(),
                            "best_score": best_score, "metric": args.metric,
                            "dims": {
                                "dim_rna_llm": dim_rna_llm, "dim_mole_llm": dim_mole_llm,
                                "c_onehot_rna": c_onehot_rna, "c_onehot_mole": c_onehot_mole, "d_pssm": d_pssm,
                            },
                        }, best_path,
                    )
                else:
                    patience += 1
                    if patience >= args.early_patience:
                        print(f"  -> Early stop. Best {args.metric}={best_score:.4f}")
                        break

            print(f"Member {m:02d} finished. Best {args.metric}={best_score:.4f}")
            best_ckpts.append(best_path)

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

        ens_results = evaluate_ensemble_mu(models, val_loader, device, type_map=type_map)
        fold_metrics.append(ens_results["global"])

        print(f"\n[FOLD {fold_id:02d} ENSEMBLE RESULTS]")
        print(f"Global: RMSE={ens_results['global']['rmse']:.4f} P={ens_results['global']['pearson']:.4f}")
        for rna_type, metrics in ens_results.items():
            if rna_type != "global":
                print(f"Type {rna_type}: RMSE={metrics['rmse']:.4f} P={metrics['pearson']:.4f} N={metrics['count']}")

        for batch in val_loader:
            if batch is None: continue
            batch_entry_ids = batch.get("entry_ids") or batch.get("entry_id")
            if batch_entry_ids is None: break
            batch = move_batch_to_device(batch, device)
            yb = batch.get("pkd", batch.get("labels"))
            if yb is None: continue
            keep = torch.isfinite(yb)
            if keep.sum().item() == 0: continue

            member_preds = []
            for m in models:
                with autocast():
                    mu, _ = m(
                        rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                        rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                        mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                        mole_mask=batch["mole_mask"],
                    )
                member_preds.append(mu.squeeze(-1)[keep].detach().cpu())
            mu_mean = torch.stack(member_preds, dim=0).mean(dim=0)
            y_cpu = yb.detach().cpu()
            
            eid_to_idx = {eid: i for i, eid in enumerate(entry_ids)}
            for j, eid in enumerate(batch_entry_ids):
                di = eid_to_idx.get(eid)
                if di is not None and torch.isfinite(y_cpu[j]):
                    all_val_preds[di] = float(mu_mean[j].item())
                    all_val_targets[di] = float(y_cpu[j].item())

    rmses = [m["rmse"] for m in fold_metrics]
    pears = [m["pearson"] for m in fold_metrics]
    print(f"\n===== {args.folds}-FOLD BLIND MOLECULE SUMMARY (Metric: {args.metric}) =====")
    print(f"RMSE: mean={np.nanmean(rmses):.4f} std={np.nanstd(rmses):.4f} | {rmses}")
    print(f"Pearson: mean={np.nanmean(pears):.4f} std={np.nanstd(pears):.4f} | {pears}")

    valid_mask = [i for i, v in enumerate(all_val_preds) if v is not None and all_val_targets[i] is not None]
    if len(valid_mask) > 0:
        p = torch.tensor([all_val_preds[i] for i in valid_mask])
        t = torch.tensor([all_val_targets[i] for i in valid_mask])
        print("\n===== OOF (concatenated) =====")
        print(f"Count={len(p)} | OOF RMSE={rmse(p, t):.4f} | OOF Pearson={pearson_corr(p, t):.4f}")

if __name__ == "__main__":
    main()