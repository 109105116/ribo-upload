"""
The data directory has the following structure by default.
The 'processed/' dir contains all final precomputed features used 
by the training scripts. All other dirs contain intermediate/raw data
from preprocessing.

data/
├── processed/                         # Pre-processed embeddings and metadata
│   ├── all_processed_v4_ids.pkl       # Entry IDs and binding affinity labels
│   ├── rna_embed.pkl                  # RinalMo embeddings (1280D per residue)
│   ├── rna_graph_edges.pkl            # RNA graph adjacency matrices
│   ├── mole_embeddings_v2.pkl         # Uni-Mol molecule embeddings (512D)
│   └── mole_edges.pkl                 # Molecule graph adjacency matrices
├── pssm_npy_5d/                       # Position-Specific Scoring Matrices
│   └── Target_*.npy                   # PSSM features (5D: A,C,G,U,-)
├── fasta/                             # Raw RNA sequences
│   └── Target_*.fasta                 # Individual RNA FASTA files
├── msa_a3m/                           # Multiple Sequence Alignments
│   ├── Target_*__*.a3m                # MSA files from MMseqs2
│   ├── Target_*__*.a3m.dbtype         # MSA database type info
│   └── Target_*__*.a3m.index          # MSA index files
├── type/                              # RNA type-specific datasets
│   ├── Aptamers_dataset_v1.csv        # Aptamer RNA-ligand pairs
│   ├── Ribosomal_dataset_v1.csv       # Ribosomal RNA-ligand pairs
│   ├── Riboswitch_dataset_v1.csv      # Riboswitch RNA-ligand pairs
│   ├── Viral_RNA_dataset_v1.csv       # Viral RNA-ligand pairs
│   ├── miRNA_dataset_v1.csv           # miRNA-ligand pairs
│   └── Repeats_dataset_v1.csv         # Repeat RNA-ligand pairs
├── rmsa_npy/                          # RNA MSA representations
│   └── representations_cv/            # Cross-validation MSA features
├── rna_ss/                            # RNA secondary structure files
│   └── Target_*.bpseq                 # Base pair secondary structure
├── All_sf_dataset_v1.csv              # Complete dataset (all RNA types)
├── data_dictionary_type.json          # RNA type mappings (Entry_ID -> type)
├── data_dictionary_tissue.json        # Tissue mappings for tissue-specific training
└── data_dictionary_disease.json       # Disease mappings for disease-specific training
"""

import os

DATA_ROOT = "/path/to/data"
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(DATA_ROOT, "processed")
PSSM_DIR = os.path.join(DATA_ROOT, "pssm_npy_5d")

DICT_PATH = os.path.join(DATA_ROOT, "data_dictionary_type.json")
TISSUE_DICT_PATH = os.path.join(DATA_ROOT, "data_dictionary_tissue.json")
DISEASE_DICT_PATH = os.path.join(DATA_ROOT, "data_dictionary_disease.json")

IDS_FILE = "all_processed_v4_ids.pkl"
RNA_EMBED_FILE = "rna_embed.pkl"
RNA_GRAPH_FILE = "rna_graph_edges.pkl"
MOLE_EMBED_FILE = "mole_embeddings_v2.pkl"
MOLE_EDGE_FILE = "mole_edges.pkl"

# for load_global_stores()
def get_data_paths():
    return {
        "ids_path": os.path.join(BASE_DIR, IDS_FILE),
        "rna_embed_path": os.path.join(BASE_DIR, RNA_EMBED_FILE),
        "rna_graph_path": os.path.join(BASE_DIR, RNA_GRAPH_FILE),
        "mole_embed_path": os.path.join(BASE_DIR, MOLE_EMBED_FILE),
        "mole_edge_path": os.path.join(BASE_DIR, MOLE_EDGE_FILE),
        "pssm_dir": PSSM_DIR,
    }

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
}

def validate_paths(verbose=True):
    paths = get_data_paths()
    missing = []
    for name, path in paths.items():
        exists = os.path.exists(path)
        if verbose:
            status = "✓" if exists else "✗ MISSING"
            print(f"  {status} {name}: {path}")
        if not exists:
            missing.append(path)
    if missing:
        print(f"\n {len(missing)} required file(s) missing!")
        return False
    elif verbose:
        print("\n✓ All data files found.")
    return True


if __name__ == "__main__":
    print("Configuration paths:")
    print(f"  DATA_ROOT: {DATA_ROOT}")
    print(f"  CODE_DIR:  {CODE_DIR}")
    print(f"  BASE_DIR:  {BASE_DIR}")
    print(f"  PSSM_DIR:  {PSSM_DIR}")
    print(f"  DICT_PATH: {DICT_PATH}")
    print()
    print("Validating data files...")
    validate_paths()