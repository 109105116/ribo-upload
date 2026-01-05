# all 2*2*2 of mse_ccc, ensemble_single, bootstrap_not

import os
import math
import argparse
import random
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import KFold

from RNAdataset import load_global_stores, RMPredDataset, collate_rmpred_batch
from utils import *
from RMPred import RMPred


from config import BASE_DIR, PSSM_DIR


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_ccc: bool = True,
    ccc_weight: float = 1.0,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_ccc = 0.0
    total_n = 0

    for batch in loader:
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

        mu = model(
            rna_llm=batch["rna_llm"],
            rna_onehot=batch["rna_onehot"],
            rna_edges=batch["rna_edges"],
            rna_pssm=batch["rna_pssm"],
            rna_mask=batch["rna_mask"],
            mole_llm=batch["mole_llm"],
            mole_onehot=batch["mole_onehot"],
            mole_edges=batch["mole_edges"],
            mole_mask=batch["mole_mask"],
        )

        mu = mu.squeeze(-1)[keep]
        yy = y[keep]

        mse = mse_loss(mu, yy)
        
        if use_ccc:
            ccc_l = ccc_loss(mu, yy)
            loss = 0.5 * (mse + ccc_weight * ccc_l)
        else:
            ccc_l = torch.tensor(0.0)
            loss = mse

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = int(yy.numel())
        total_loss += loss.item() * bs
        total_mse += mse.item() * bs
        total_ccc += ccc_l.item() * bs
        total_n += bs

    denom = max(1, total_n)
    return {
        "loss": total_loss / denom,
        "mse": total_mse / denom,
        "ccc_loss": total_ccc / denom,
    }


@torch.no_grad()
def evaluate_single(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds, ys = [], []
    
    for batch in loader:
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

        mu = model(
            rna_llm=batch["rna_llm"],
            rna_onehot=batch["rna_onehot"],
            rna_edges=batch["rna_edges"],
            rna_pssm=batch["rna_pssm"],
            rna_mask=batch["rna_mask"],
            mole_llm=batch["mole_llm"],
            mole_onehot=batch["mole_onehot"],
            mole_edges=batch["mole_edges"],
            mole_mask=batch["mole_mask"],
        )
        preds.append(mu.squeeze(-1)[keep].detach().cpu())
        ys.append(y[keep].detach().cpu())

    if len(preds) == 0:
        return {"pearson": float("nan"), "rmse": float("nan")}
    mu_all = torch.cat(preds, dim=0)
    y_all = torch.cat(ys, dim=0)
    return {"pearson": pearson_corr(mu_all, y_all), "rmse": rmse(mu_all, y_all)}


@torch.no_grad()
def evaluate_ensemble(models: List[nn.Module], loader: DataLoader, device: torch.device) -> Dict[str, float]:
    for m in models:
        m.eval()

    ens_preds, ys = [], []
    
    for batch in loader:
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

        member_preds = []
        for m in models:
            mu, _ = m(
                rna_llm=batch["rna_llm"],
                rna_onehot=batch["rna_onehot"],
                rna_edges=batch["rna_edges"],
                rna_pssm=batch["rna_pssm"],
                rna_mask=batch["rna_mask"],
                mole_llm=batch["mole_llm"],
                mole_onehot=batch["mole_onehot"],
                mole_edges=batch["mole_edges"],
                mole_mask=batch["mole_mask"],
            )
            member_preds.append(mu.squeeze(-1)[keep].detach().cpu())
        
        mu_mean = torch.stack(member_preds, dim=0).mean(dim=0)
        ens_preds.append(mu_mean)
        ys.append(y[keep].detach().cpu())

    if len(ens_preds) == 0:
        return {"pearson": float("nan"), "rmse": float("nan")}
    mu_all = torch.cat(ens_preds, dim=0)
    y_all = torch.cat(ys, dim=0)
    return {"pearson": pearson_corr(mu_all, y_all), "rmse": rmse(mu_all, y_all)}


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
    for i, (train_rna_indices, val_rna_indices) in enumerate(kf.split(unique_rnas)):
        train_rnas = set(unique_rnas[train_rna_indices])
        val_rnas = set(unique_rnas[val_rna_indices])
        
        train_idx = []
        val_idx = []
        
        for r_id in train_rnas:
            train_idx.extend(rna_to_sample_indices[r_id])
        for r_id in val_rnas:
            val_idx.extend(rna_to_sample_indices[r_id])
            
        random.Random(seed + i).shuffle(train_idx)
        random.Random(seed + i).shuffle(val_idx)
        
        folds.append((np.array(train_idx), np.array(val_idx)))
    
    return folds


def make_subset_loader(dataset, indices, batch_size, shuffle, num_workers, seed):
    subset = Subset(dataset, indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_rmpred_batch,
        generator=generator,
        drop_last=False,
    )


class AblationConfig:
    def __init__(self, name: str, use_ccc: bool, use_ensemble: bool, use_bootstrap: bool):
        self.name = name
        self.use_ccc = use_ccc
        self.use_ensemble = use_ensemble
        self.use_bootstrap = use_bootstrap
    
    def __repr__(self):
        return f"AblationConfig({self.name}: CCC={self.use_ccc}, Ens={self.use_ensemble}, Boot={self.use_bootstrap})"


def get_ablation_configs() -> List[AblationConfig]:
    configs = []
    
    for use_ccc in [False, True]:
        for use_ensemble in [False, True]:
            for use_bootstrap in [False, True]:
                ccc_str = "CCC" if use_ccc else "MSE"
                ens_str = "Ens" if use_ensemble else "Single"
                boot_str = "Boot" if use_bootstrap else "NoBoot"
                name = f"{ccc_str}_{ens_str}_{boot_str}"
                configs.append(AblationConfig(name, use_ccc, use_ensemble, use_bootstrap))
    
    return configs


class TrainingLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.curves = defaultdict(list)
    
    def log(self, config_name: str, fold: int, member: int, epoch: int, 
            train_loss: float, train_mse: float, train_ccc: float,
            val_rmse: float, val_pearson: float):
        self.curves[config_name].append({
            "fold": fold,
            "member": member,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "train_ccc": train_ccc,
            "val_rmse": val_rmse,
            "val_pearson": val_pearson,
        })
    
    def save(self):
        for config_name, data in self.curves.items():
            path = os.path.join(self.log_dir, f"curve_{config_name}.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        print(f"[Logger] Saved training curves to {self.log_dir}")


def run_ablation(
    config: AblationConfig,
    dataset,
    stores,
    entry_ids: List,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    args,
    device: torch.device,
    dims: Dict[str, int],
    logger: TrainingLogger,
) -> List[Dict[str, float]]:
    
    print(f"\n{'='*60}", flush=True)
    print(f"Running: {config.name}", flush=True)
    print(f"  CCC Loss: {config.use_ccc}", flush=True)
    print(f"  Ensemble: {config.use_ensemble} (K={args.k if config.use_ensemble else 1})", flush=True)
    print(f"  Bootstrap: {config.use_bootstrap}", flush=True)
    print(f"{'='*60}", flush=True)
    
    fold_metrics = []
    k = args.k if config.use_ensemble else 1
    
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_id+1}/{len(folds)} | train={len(train_idx)} val={len(val_idx)} ---", flush=True)
        
        val_loader = make_subset_loader(
            dataset, val_idx.tolist(), 
            batch_size=args.val_batch_size, 
            shuffle=False, num_workers=0, 
            seed=args.seed + 999 + fold_id
        )
        
        trained_models = []
        
        for m in range(k):
            member_seed = args.seed + 1000 * (fold_id * k + m)
            set_seed(member_seed)
            
            if config.use_bootstrap:
                tr_indices = [random.choice(train_idx.tolist()) for _ in range(len(train_idx))]
            else:
                tr_indices = train_idx.tolist()
            
            train_loader = make_subset_loader(
                dataset, tr_indices, 
                batch_size=args.batch_size, 
                shuffle=True, num_workers=0, 
                seed=member_seed
            )
            
            model = RMPred(
                d_llm_rna=dims["dim_rna_llm"],
                c_onehot_rna=dims["c_onehot_rna"],
                d_pssm_rna=dims["d_pssm"],
                d_llm_mole=dims["dim_mole_llm"],
                c_onehot_mole=dims["c_onehot_mole"],
                d_model_inner=256,
                d_model_fusion=512,
                dropout=0.2,
                fusion_layers=2,
                fusion_heads=4,
                rna_gnn_layers=4,
                rna_gnn_heads=4,
                mole_gnn_layers=4,
                mole_gnn_heads=4,
                mole_num_edge_types=8,
                rna_max_len=args.max_rna_len,
            ).to(device)
            
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=args.lr, 
                weight_decay=args.weight_decay
            )
            
            best_rmse = float("inf")
            best_state = None
            patience = 0
            
            for epoch in range(1, args.epochs + 1):
                tr = train_one_epoch(
                    model, train_loader, optimizer, device,
                    use_ccc=config.use_ccc,
                    ccc_weight=args.ccc_weight,
                    grad_clip=1.0,
                )
                val_metrics = evaluate_single(model, val_loader, device)
                
                logger.log(
                    config.name, fold_id, m, epoch,
                    tr["loss"], tr["mse"], tr["ccc_loss"],
                    val_metrics["rmse"], val_metrics["pearson"]
                )
                
                print(f"  [F{fold_id} M{m} E{epoch:03d}] Loss={tr['loss']:.4f} | "
                      f"Val RMSE={val_metrics['rmse']:.4f} Pearson={val_metrics['pearson']:.4f}", flush=True)
                
                if math.isfinite(val_metrics["rmse"]) and val_metrics["rmse"] < best_rmse:
                    best_rmse = val_metrics["rmse"]
                    best_state = model.state_dict().copy()
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.early_patience:
                        print(f"  -> Early stop at epoch {epoch}. Best RMSE={best_rmse:.4f}", flush=True)
                        break
            
            if best_state is not None:
                model.load_state_dict(best_state)
            trained_models.append(model)
        
        if config.use_ensemble:
            metrics = evaluate_ensemble(trained_models, val_loader, device)
        else:
            metrics = evaluate_single(trained_models[0], val_loader, device)
        
        print(f"[Fold {fold_id}] RMSE={metrics['rmse']:.4f} | Pearson={metrics['pearson']:.4f}", flush=True)
        fold_metrics.append(metrics)
        
        del trained_models
        torch.cuda.empty_cache()
    
    return fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Training Method Components")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--k", type=int, default=5, help="Ensemble size")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early_patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--ccc_weight", type=float, default=1.0, help="Weight for CCC loss term")
    parser.add_argument("--max_rna_len", type=int, default=1024)
    parser.add_argument("--max_mole_len", type=int, default=2048)
    
    parser.add_argument("--out_dir", type=str, default="ablation_training_method")
    
    parser.add_argument("--configs", type=str, nargs="*", default=None,
                        help="Specific configs to run, e.g., 'CCC_Ens_Boot MSE_Single_NoBoot'")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{args.out_dir}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n[Loading data...]")
    stores = load_global_stores(
        ids_path=os.path.join(BASE_DIR, "all_processed_v4_ids.pkl"),
        rna_embed_path=os.path.join(BASE_DIR, "rna_embed.pkl"),
        rna_graph_path=os.path.join(BASE_DIR, "rna_graph_edges.pkl"),
        mole_embed_path=os.path.join(BASE_DIR, "mole_embeddings_v2.pkl"),
        mole_edge_path=os.path.join(BASE_DIR, "mole_edges.pkl"),
        pssm_dir=PSSM_DIR,
    )
    
    dataset = RMPredDataset(
        stores, strict=True, 
        max_rna_len=args.max_rna_len,
        max_mole_len=(None if args.max_mole_len == 0 else args.max_mole_len),
        truncate_if_exceed=False, 
        label_key="pkd",
    )
    
    n_total = len(dataset)
    print(f"Total samples: {n_total}")
    
    set_seed(args.seed)
    temp_loader = DataLoader(
        Subset(dataset, list(range(min(n_total, 64)))), 
        batch_size=min(args.batch_size, 8), 
        shuffle=False, num_workers=0, 
        collate_fn=collate_rmpred_batch, 
        drop_last=False
    )
    batch0 = next(iter(temp_loader))
    dims = {
        "dim_rna_llm": batch0["rna_llm"].shape[-1],
        "dim_mole_llm": batch0["mole_llm"].shape[-1],
        "c_onehot_rna": batch0["rna_onehot"].shape[-1],
        "c_onehot_mole": batch0["mole_onehot"].shape[-1],
        "d_pssm": batch0["rna_pssm"].shape[-1],
    }
    print(f"Feature dims: {dims}")
    
    entry_ids = getattr(dataset, "keys", None)
    if entry_ids is None:
        entry_ids = list(stores.entry_binding.keys())
    
    folds = get_blind_rna_kfold_indices(stores, entry_ids, args.folds, args.seed)
    
    logger = TrainingLogger(os.path.join(out_dir, "curves"))
    
    all_configs = get_ablation_configs()
    if args.configs:
        all_configs = [c for c in all_configs if c.name in args.configs]
    
    print(f"\nRunning {len(all_configs)} ablation configurations:")
    for c in all_configs:
        print(f"  - {c.name}")
    
    results = {}
    for config in all_configs:
        fold_metrics = run_ablation(
            config, dataset, stores, entry_ids, folds,
            args, device, dims, logger
        )
        results[config.name] = fold_metrics
    
    logger.save()
    
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    summary_data = []
    for config_name, fold_metrics in results.items():
        rmses = [m["rmse"] for m in fold_metrics]
        pears = [m["pearson"] for m in fold_metrics]
        
        row = {
            "config": config_name,
            "rmse_mean": np.nanmean(rmses),
            "rmse_std": np.nanstd(rmses),
            "pearson_mean": np.nanmean(pears),
            "pearson_std": np.nanstd(pears),
        }
        summary_data.append(row)
        
        print(f"{config_name:25s} | RMSE: {row['rmse_mean']:.4f}±{row['rmse_std']:.4f} | "
              f"Pearson: {row['pearson_mean']:.4f}±{row['pearson_std']:.4f}")
    
    csv_path = os.path.join(out_dir, "ablation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "rmse_mean", "rmse_std", "pearson_mean", "pearson_std"])
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"\nSummary saved to: {csv_path}")
    
    details_path = os.path.join(out_dir, "ablation_details.json")
    with open(details_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {details_path}")
    
if __name__ == "__main__":
    main()
