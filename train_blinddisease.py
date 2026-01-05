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
from RMPred import RMPred
from utils import *
from config import BASE_DIR, PSSM_DIR, DICT_PATH, DISEASE_DICT_PATH

def get_cancer_holdout_indices(entry_ids, disease_map, target_disease):
    train_idx = []
    val_idx = []
    
    print(f"\n[Split Strategy] Specific Hold-out: Test on '{target_disease}'")
    
    missing_count = 0
    
    for i, eid in enumerate(entry_ids):
        eid_str = str(eid)
        
        disease = disease_map.get(eid_str)
        if disease == target_disease:
            val_idx.append(i)
        else:
            train_idx.append(i)
    
    print(f"Train samples: {len(train_idx)}, Test samples: {len(val_idx)}")
    return [(torch.tensor(train_idx), torch.tensor(val_idx))]

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
        
        valid_preds = mu.view(-1)[keep].detach().cpu()
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
    mu = mu.float().view(-1)
    y = y.float().view(-1)
    if mu.numel() < 2: return torch.mean((mu - y) ** 2)
    mu_mean = mu.mean()
    y_mean = y.mean()
    mu_var = mu.var(unbiased=False)
    y_var = y.var(unbiased=False)
    cov = ((mu - mu_mean) * (y - y_mean)).mean()
    ccc = (2.0 * cov) / (mu_var + y_var + (mu_mean - y_mean).pow(2) + eps)
    ccc = torch.clamp(ccc, -1.0, 1.0)
    return 1.0 - ccc

@torch.no_grad()
def evaluate_ensemble_mu(
    models: List[nn.Module], 
    loader: DataLoader, 
    device: torch.device,
    type_map: Dict[str, str] = None
) -> Dict[str, Any]:
    
    for m in models: m.eval()
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

        member_preds = []
        for m in models:
            with autocast():
                mu, _ = m(
                    rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                    rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                    mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                    mole_mask=batch["mole_mask"],
                )
            member_preds.append(mu.view(-1)[keep].detach().cpu())
        
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
            mu = mu.view(-1)[keep]
            mse, ccc_l = mse_loss(mu, yy), ccc_loss(mu, yy)
            loss = 0.5 * (mse + ccc_weight * ccc_l)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        scaler.step(optimizer)
        scaler.update()
        
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
    ap.add_argument("--out_dir", type=str, default="ckpt_cancer_holdout")
    ap.add_argument("--max_rna_len", type=int, default=1024)
    ap.add_argument("--max_mole_len", type=int, default=2048)
    ap.add_argument("--early_patience", type=int, default=20)
    ap.add_argument("--label_key", type=str, default="pkd")
    
    ap.add_argument("--disease", type=str, default="Acquired immunodeficiency syndrome (AIDS)", help="Target disease for hold-out")
    
    ap.add_argument("--metric", type=str, default="pearson", choices=["rmse", "pearson"], 
                    help="Metric to determine best model")
    
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
        try:
            type_map = load_rna_type_map(DICT_PATH)
        except:
            print(f"Warning: Could not load rna_type_map from {DICT_PATH}")
    else:
        print(f"Warning: rna_type_map not found at {DICT_PATH}")

    disease_map = None
    if os.path.exists(DISEASE_DICT_PATH):
        print(f"Loading Disease Dictionary from: {DISEASE_DICT_PATH}")
        with open(DISEASE_DICT_PATH, 'r') as f:
            disease_map = json.load(f)
    else:
        raise FileNotFoundError(f"Error: Disease dictionary not found at {DISEASE_DICT_PATH}.")

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

    folds = get_cancer_holdout_indices(entry_ids, disease_map, args.disease)

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        print(f"\n===== {args.disease.upper()} HOLDOUT EXPERIMENT | train={len(train_idx)} test({args.disease})={len(val_idx)} =====")
        fold_dir = os.path.join(args.out_dir, f"{args.disease.lower().replace(' ', '_')}_holdout")
        os.makedirs(fold_dir, exist_ok=True)

        val_loader = make_subset_loader(dataset, val_idx.tolist(), batch_size=args.val_batch_size, shuffle=False, num_workers=0, seed=args.seed + 999)

        best_ckpts = []

        for m in range(args.k):
            member_seed = args.seed + 1000 * m
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

            if args.metric == "pearson":
                best_score = float("-inf")
            else:
                best_score = float("inf")
                
            best_path = os.path.join(fold_dir, f"member_{m:02d}_best.pt")
            patience = 0

            print(f"\n--- Train member {m+1}/{args.k} ---")
            
            for epoch in range(1, args.epochs + 1):
                tr = train_one_epoch_mu(model, train_loader, optimizer, device, scaler, grad_clip=1.0, ccc_weight=1.0)
                
                val_results = evaluate_mu(model, val_loader, device, type_map=type_map)

                current_rmse = val_results["global"]["rmse"]
                current_pearson = val_results["global"]["pearson"]
                
                if args.metric == "pearson":
                    current_score = current_pearson
                else:
                    current_score = current_rmse

                print(f"[Ep {epoch:03d}] Loss={tr['loss']:.4f} | {args.disease} Test: RMSE={current_rmse:.4f} P={current_pearson:.4f}")
                
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
                            "member": m, "epoch": epoch,
                            "model_state": model.state_dict(),
                            "best_score": best_score, "metric": args.metric,
                        },
                        best_path,
                    )
                else:
                    patience += 1
                    if patience >= args.early_patience:
                        print(f"  -> Early stop. Best {args.disease} {args.metric}={best_score:.4f}")
                        break

            print(f"Member {m:02d} finished. Best {args.disease} {args.metric}={best_score:.4f}")
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
        
        print(f"\n[FINAL {args.disease.upper()} ENSEMBLE RESULTS]")
        print(f"Global ({args.disease}): RMSE={ens_results['global']['rmse']:.4f} P={ens_results['global']['pearson']:.4f}")
        
        for rna_type, metrics in ens_results.items():
            if rna_type != "global":
                print(f"Type {rna_type}: RMSE={metrics['rmse']:.4f} P={metrics['pearson']:.4f} N={metrics['count']}")

if __name__ == "__main__":
    main()