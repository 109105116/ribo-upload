
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem

from utils import *
from config import BASE_DIR, PSSM_DIR, DICT_PATH

def _as_float_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float32)


def _as_long_edges(x) -> torch.Tensor:
    if x is None:
        return torch.empty((0, 2), dtype=torch.long)
    if isinstance(x, torch.Tensor):
        t = x.long()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x).long()
    else:
        t = torch.tensor(x, dtype=torch.long)

    if t.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)

    if t.dim() == 2 and t.shape[0] == 2 and t.shape[1] >= 1:
        t = t.t().contiguous()

    if t.dim() == 1:
        if t.numel() % 2 != 0:
            raise ValueError(f"Edge tensor error: numel={t.numel()}")
        t = t.view(-1, 2)

    return t

def _auto_to_0_based(edges: torch.Tensor, L: int) -> torch.Tensor:
    if edges.numel() == 0: return edges
    uv = edges[:, :2]
    mn = int(uv.min().item())
    mx = int(uv.max().item())
    if mn >= 1 and mx <= L:
        edges = edges.clone()
        edges[:, :2] = edges[:, :2] - 1
    return edges

def _filter_edges(edges: torch.Tensor, L: int) -> torch.Tensor:
    if edges.numel() == 0: return edges
    u = edges[:, 0]
    v = edges[:, 1]
    keep = (u >= 0) & (v >= 0) & (u < L) & (v < L) & (u != v)
    return edges[keep]

def seq_to_onehot_rna(seq: str) -> torch.Tensor:
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4, '-': 4, '.': 4}
    idx = [mapping.get(s.upper(), 4) for s in seq]
    L = len(idx)
    t = torch.zeros(L, 5, dtype=torch.float32)
    t[torch.arange(L), torch.tensor(idx)] = 1.0
    return t


class SmartRDKitEncoder:
    def __init__(self):
        self.atom_vocab = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H', 'B', 'Si', 'Se']
        self.atom_map = {a: i for i, a in enumerate(self.atom_vocab)}
        self.vocab_size = len(self.atom_vocab) + 1 

    def get_onehot(self, smiles: str, target_len: Optional[int] = None):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None, "Invalid_SMILES"

        if target_len is None or mol.GetNumAtoms() == target_len:
            mol_final, status = mol, "Implicit_H"
        else:
            mol_h = Chem.AddHs(mol)
            if mol_h.GetNumAtoms() == target_len:
                mol_final, status = mol_h, "Explicit_H"
            else:
                return None, f"Mismatch"

        num_atoms = mol_final.GetNumAtoms()
        oh = torch.zeros(num_atoms, self.vocab_size, dtype=torch.float32)
        for i, atom in enumerate(mol_final.GetAtoms()):
            sym = atom.GetSymbol()
            idx = self.atom_map.get(sym, self.vocab_size - 1)
            oh[i, idx] = 1.0
        return oh, status

    @staticmethod
    def bonds_as_edges(mol: Chem.Mol) -> torch.Tensor:
        edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        if len(edges) == 0: return torch.empty((0, 2), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long)



@dataclass
class GlobalStores:
    entry_binding: Dict
    rna_map: Dict
    mol_map: Dict
    rna_seqs: Dict
    mole_smiles: Dict
    rna_embed: Dict
    rna_graph: Dict
    mole_embed: Dict
    mole_graph: Dict
    pssm_dir: str
    pkd_norm: Dict[str, float]


def load_global_stores(
    ids_path: str,
    rna_embed_path: str,
    rna_graph_path: str,
    mole_embed_path: str,
    mole_edge_path: str,
    pssm_dir: str,
) -> GlobalStores:
    with open(ids_path, "rb") as f:
        data = pickle.load(f)
    with open(rna_embed_path, "rb") as f:
        rna_embed = pickle.load(f)
    with open(rna_graph_path, "rb") as f:
        rna_graph = pickle.load(f)
    with open(mole_embed_path, "rb") as f:
        mole_embed = pickle.load(f)
    with open(mole_edge_path, "rb") as f:
        mole_graph = pickle.load(f)

    pkd_norm = data.get("pkd_normalizer", {'mean': 0.0, 'std': 1.0})

    return GlobalStores(
        entry_binding=data["entry_binding_dict"],
        rna_map=data["rna_id_to_name_dict"],
        mol_map=data["mol_id_to_name_dict"],
        rna_seqs=data["rna_seq_dict"],
        mole_smiles=data["mole_smiles_dict"],
        rna_embed=rna_embed,
        rna_graph=rna_graph,
        mole_embed=mole_embed,
        mole_graph=mole_graph,
        pssm_dir=pssm_dir,
        pkd_norm=pkd_norm
    )


class RMPredDataset(Dataset):
    def __init__(
        self,
        stores: GlobalStores,
        keys: Optional[List[Any]] = None,
        *,
        strict: bool = True,
        cache_mole_onehot: bool = True,
        max_rna_len: Optional[int] = None,
        max_mole_len: Optional[int] = None,
        truncate_if_exceed: bool = False,
        label_key: str = "pkd"
    ):
        self.s = stores
        self.strict = strict
        self.encoder = SmartRDKitEncoder()
        self.cache_mole_onehot = cache_mole_onehot
        self.max_rna_len = max_rna_len
        self.max_mole_len = max_mole_len
        self.truncate_if_exceed = truncate_if_exceed
        self._mole_onehot_cache: Dict[Any, tuple] = {}
        self.label_key = label_key

        self.keys = keys if keys is not None else list(self.s.entry_binding.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        entry_id = self.keys[idx]
        entry = self.s.entry_binding[entry_id]

        rna_id = entry.get("rna_id")
        mol_id = entry.get("mole_id") or entry.get("mol_id")
        if rna_id is None or mol_id is None:
            if self.strict: raise KeyError(f"Missing ID in {entry_id}")
            return None

        rna_name = self.s.rna_map.get(rna_id)
        if not rna_name: return None
        r_llm = smart_fetch(self.s.rna_embed, [rna_name, rna_id])
        if r_llm is None: return None
        r_llm = _as_float_tensor(r_llm)
        r_seq = smart_fetch(self.s.rna_seqs, [rna_id, rna_name])
        r_oh = seq_to_onehot_rna(r_seq)
        r_edges_raw = smart_fetch(self.s.rna_graph, [rna_id, rna_name])
        r_edges = _as_long_edges(r_edges_raw)
        
        pssm_path = os.path.join(self.s.pssm_dir, f"{rna_id}.npy")
        if not os.path.exists(pssm_path):
            if self.strict: raise FileNotFoundError(f"PSSM missing {rna_id}")
            return None
        p = np.load(pssm_path)
        if p.ndim != 2: p = p.reshape(p.shape[0], -1)
        r_pssm = torch.from_numpy(p).float()

        Lr = min(r_llm.shape[0], r_oh.shape[0], r_pssm.shape[0])
        if self.max_rna_len and Lr > self.max_rna_len:
             if self.truncate_if_exceed: Lr = int(self.max_rna_len)
             else: return None
        r_llm, r_oh, r_pssm = r_llm[:Lr], r_oh[:Lr], r_pssm[:Lr]
        r_edges = _filter_edges(_auto_to_0_based(r_edges, Lr), Lr)

        mol_name = self.s.mol_map.get(mol_id)
        if not mol_name: return None
        lookup = [mol_name, mol_id]
        m_llm = smart_fetch(self.s.mole_embed, lookup)
        if m_llm is None: return None
        m_llm = _as_float_tensor(m_llm)
        Lm = int(m_llm.shape[0])
        
        smiles = smart_fetch(self.s.mole_smiles, [mol_id, mol_name])
        if not smiles: return None
        
        if self.cache_mole_onehot and mol_id in self._mole_onehot_cache:
            m_oh, status = self._mole_onehot_cache[mol_id]
            if m_oh.shape[0] != Lm: m_oh, status = self.encoder.get_onehot(smiles, target_len=Lm)
        else:
            m_oh, status = self.encoder.get_onehot(smiles, target_len=Lm)
        if m_oh is None: return None
        if self.cache_mole_onehot: self._mole_onehot_cache[mol_id] = (m_oh, status)

        if status == "Explicit_H":
            mol_h = Chem.AddHs(Chem.MolFromSmiles(smiles))
            m_edges = self.encoder.bonds_as_edges(mol_h)
        else:
            m_edges_raw = smart_fetch(self.s.mole_graph, lookup)
            m_edges = _as_long_edges(m_edges_raw) if m_edges_raw is not None else torch.empty((0,2), dtype=torch.long)
        
        m_edges = _filter_edges(_auto_to_0_based(m_edges, Lm), Lm)

        y = entry.get(self.label_key) 
        if y is None: y = entry.get("affinity") or entry.get("label")
        
        y = float(y) if y is not None else None

        return {
            "entry_id": entry_id,
            "rna_id": rna_id,
            "mol_id": mol_id,
            "rna_llm": r_llm, "rna_onehot": r_oh, "rna_edges": r_edges, "rna_pssm": r_pssm,
            "mole_llm": m_llm, "mole_onehot": m_oh, "mole_edges": m_edges,
            "label": y,
        }


def collate_rmpred_batch(batch_list: List[Optional[dict]]):
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
    rna_edges: List[torch.Tensor] = []

    mole_llm = torch.zeros(B, max_mole_len, d_mole_llm)
    mole_oh = torch.zeros(B, max_mole_len, c_mole_oh)
    mole_mask = torch.zeros(B, max_mole_len)
    mole_edges: List[torch.Tensor] = []

    entry_ids, rna_ids, mol_ids = [], [], []
    labels = []

    for i, item in enumerate(batch_list):
        entry_ids.append(item["entry_id"])
        rna_ids.append(item["rna_id"])
        mol_ids.append(item["mol_id"])
        labels.append(item["label"])

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

    out = {
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
    }
    
    if any(v is not None for v in labels):
        valid_labels = [v if v is not None else float("nan") for v in labels]
        out["pkd"] = torch.tensor(valid_labels, dtype=torch.float32) 
        out["labels"] = out["pkd"]
        
    return out


def make_dataloader(
    stores: GlobalStores,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    strict: bool = True,
    keys: Optional[List[Any]] = None,
    max_rna_len: Optional[int] = None,
    max_mole_len: Optional[int] = None,
    truncate_if_exceed: bool = False,
    label_key: str = "pkd"
) -> DataLoader:
    ds = RMPredDataset(
        stores,
        keys=keys,
        strict=strict,
        max_rna_len=max_rna_len,
        max_mole_len=max_mole_len,
        truncate_if_exceed=truncate_if_exceed,
        label_key=label_key
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_rmpred_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


__all__ = [
    "GlobalStores",
    "load_global_stores",
    "RMPredDataset",
    "collate_rmpred_batch",
    "make_dataloader",
    "SmartRDKitEncoder",
]
