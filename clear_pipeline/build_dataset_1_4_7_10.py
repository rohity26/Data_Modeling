"""build_dataset_1_4_7_10.py — Standalone dataset builder for the NEW
1/4/7/10-DU topology set (train-only, LOO training set).

This is a SEPARATE file from build_dataset.py — nothing in the original file
is touched. Differences from build_dataset.py:
  - TOPOLOGY_REGISTRY has exactly 4 entries: cu1_du2 (1 DU), cu3_du6du7du8du9
    (4 DU) — both reused unchanged from the original 50h collection — plus two
    NEW topologies, cu0_7du (7 DU) and cu0_10du (10 DU), each backed by its own
    freshly-collected 50h CSV (train_normal_1cu7du_50h.csv /
    train_normal_1cu10du_50h.csv).
  - Each topology now carries its own "train_csv" source (previously all
    topologies shared one global TRAIN_CSV); main() loads each distinct CSV
    exactly once and reuses it for every topology that points at it.
  - Train-only build for every topology (no TEST_CSV exists for the new
    topologies yet) — same train-only fallback behavior as build_dataset.py.
  - Output goes to a NEW base dir, CU_1_4_7_10_STRESS, so it can never
    collide with the original CU_DL_UDP_STRESS output.

USAGE
-----
    cd project_root/step3_topoar/clear_pipeline
    python build_dataset_1_4_7_10.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# USER INPUTS
# =============================================================================

STRESS_TYPE = 3           # 1=CPU | 2=MEM | 3=NET — kept for parity with build_dataset.py;
                           # irrelevant here since every build below is train-only.

BASE_DIR = Path("CU_1_4_7_10_STRESS")

STRESS_NAMES = {0: "normal", 1: "CPU", 2: "MEM", 3: "NET"}

ORIGINAL_50H_CSV = Path("data_input/train_normal_dl_udp_50h.csv")
CSV_7DU = Path("data_input/train_normal_1cu7du_50h.csv")
CSV_10DU = Path("data_input/train_normal_1cu10du_50h.csv")

# cu_id     : container name of the CU in that topology's source CSV
# du_ids    : container names of all DUs under this CU
# pci_ids   : PCI number for each DU (PCI-{n}_... columns in CSV)
# train_csv : which CSV this topology's normal data comes from
TOPOLOGY_REGISTRY = {
    "cu1_du2": {                                   # 1 DU — unchanged, reused as-is
        "cu_id":  "srscu1",
        "du_ids": ["srsdu2"],
        "pci_ids": [3],
        "train_csv": ORIGINAL_50H_CSV,
    },
    "cu3_du6du7du8du9": {                          # 4 DU — unchanged, reused as-is
        "cu_id":  "srscu3",
        "du_ids": ["srsdu6", "srsdu7", "srsdu8", "srsdu9"],
        "pci_ids": [7, 8, 9, 10],
        "train_csv": ORIGINAL_50H_CSV,
    },
    "cu0_7du": {                                   # 7 DU — NEW topology
        "cu_id":  "srscu0",
        "du_ids": [f"srsdu{i}" for i in range(7)],
        "pci_ids": list(range(1, 8)),
        "train_csv": CSV_7DU,
    },
    "cu0_10du": {                                  # 10 DU — NEW topology
        "cu_id":  "srscu0",
        "du_ids": [f"srsdu{i}" for i in range(10)],
        "pci_ids": list(range(1, 11)),
        "train_csv": CSV_10DU,
    },
}

# =============================================================================
# COLUMN DETECTION — identical logic to build_dataset.py
# =============================================================================

CADVISOR_PATTERNS = [
    ("cpu",       "container_cpu_user_seconds_total",       "machine_cpu_cores",     None),
    ("mem_pct",   "machine_memory_bytes",                   None,                    None),
    ("mem_bytes", "container_memory_cache",                 None,                    "machine_memory_bytes"),
    ("fs_reads",  "container_fs_reads_bytes_total",         None,                    None),
    ("fs_writes", "container_fs_writes_bytes_total",        None,                    None),
    ("net_tx",    "container_network_transmit_bytes_total", 'interface="eth0"',      None),
    ("net_rx",    "container_network_receive_bytes_total",  'interface="eth0"',      None),
]


def _find_cadvisor_col(
    entity: str, all_cols: list[str],
    required: str, extra: str | None, forbidden: str | None,
) -> str:
    matches = [
        c for c in all_cols
        if f'name="{entity}"' in c
        and 'instance="cadvisor:8080"' in c
        and "by (instance)" in c
        and required in c
        and (extra is None or extra in c)
        and (forbidden is None or forbidden not in c)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{entity}: expected 1 col for '{required}', got {len(matches)}: {matches}"
        )
    return matches[0]


def _find_cu_eth2_rx(cu_id: str, all_cols: list[str]) -> str:
    """CU downlink ingress from the core (eth2, no 'by (instance)' form)."""
    matches = [
        c for c in all_cols
        if f'name="{cu_id}"' in c
        and 'instance="cadvisor:8080"' in c
        and "container_network_receive_bytes_total" in c
        and 'interface="eth2"' in c
    ]
    if len(matches) != 1:
        raise ValueError(f"{cu_id}: expected 1 eth2 rx col, got {len(matches)}: {matches}")
    return matches[0]


def get_cu_columns(cu_id: str, all_cols: list[str]) -> list[str]:
    """7 cadvisor columns for the CU; net_rx redirected to eth2 (2-interface CU)."""
    cols = []
    for name, req, extra, forb in CADVISOR_PATTERNS:
        if name == "net_rx":
            cols.append(_find_cu_eth2_rx(cu_id, all_cols))
        else:
            cols.append(_find_cadvisor_col(cu_id, all_cols, req, extra, forb))
    return cols


def get_du_columns(du_id: str, pci_id: int, all_cols: list[str]) -> list[str]:
    """37 columns for one DU: 7 cadvisor + 30 PCI, in fixed order."""
    cad = [_find_cadvisor_col(du_id, all_cols, req, extra, forb)
           for _, req, extra, forb in CADVISOR_PATTERNS]
    pci = sorted([c for c in all_cols if c.startswith(f"PCI-{pci_id}_")])
    if len(pci) == 0:
        raise ValueError(f"No PCI-{pci_id}_* columns found")
    return cad + pci


# =============================================================================
# BUILD ONE TOPOLOGY (train-only)
# =============================================================================

def build_one(topology: str, train: pd.DataFrame):
    assert topology in TOPOLOGY_REGISTRY, \
        f"Unknown topology '{topology}'. Add it to TOPOLOGY_REGISTRY."

    topo_cfg = TOPOLOGY_REGISTRY[topology]
    cu_id    = topo_cfg["cu_id"]
    du_ids   = topo_cfg["du_ids"]
    pci_ids  = topo_cfg["pci_ids"]
    n_du     = len(du_ids)
    train_csv = topo_cfg["train_csv"]

    out_dir = BASE_DIR / f"{topology}_stress{STRESS_TYPE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Topology   : {topology}  ({cu_id} + {du_ids})")
    print(f"Train CSV  : {train_csv}")
    print(f"Output dir : {out_dir}")
    print(f"{'='*60}")

    all_cols = train.columns.tolist()

    print("\nDetecting KPI columns ...")
    cu_cols = get_cu_columns(cu_id, all_cols)
    du_cols_per_instance = [
        get_du_columns(du_id, pci_id, all_cols)
        for du_id, pci_id in zip(du_ids, pci_ids)
    ]
    cu_dim = len(cu_cols)
    du_dim = len(du_cols_per_instance[0])
    print(f"  CU dim: {cu_dim}  |  DU dim: {du_dim}  |  N_DU: {n_du}")

    print("\nBuilding train arrays (normal only, no filtering needed) ...")
    train_cu = train[cu_cols].fillna(0.0).astype(np.float32).values
    train_du = np.stack(
        [train[cols].fillna(0.0).astype(np.float32).values for cols in du_cols_per_instance],
        axis=1,
    )
    train_block_id = np.zeros(len(train_cu), dtype=np.int64)
    print(f"  train_cu: {train_cu.shape}  train_du: {train_du.shape}")

    np.savez(
        out_dir / "train.npz",
        cu=train_cu,
        du=train_du,
        block_id=train_block_id,
    )
    print(f"\nSaved {out_dir}/train.npz — cu{train_cu.shape} du{train_du.shape}")

    (out_dir / "summary.txt").write_text(
        f"Topology   : {topology}\n"
        f"Train CSV  : {train_csv}\n"
        f"Test CSV   : (none — train-only build)\n"
        f"\nTRAIN ROW COUNT: {len(train_cu):,}\n"
    )
    print("Train-only build (no test.npz/blocks.csv for this topology set).")


def main():
    loaded_csvs: dict[Path, pd.DataFrame] = {}

    def get_train(path: Path) -> pd.DataFrame:
        if path not in loaded_csvs:
            print(f"\nLoading {path.name} ...")
            df = pd.read_csv(path)
            print(f"  shape: {df.shape}")
            loaded_csvs[path] = df
        return loaded_csvs[path]

    for topology, cfg in TOPOLOGY_REGISTRY.items():
        train = get_train(cfg["train_csv"])
        build_one(topology, train)

    print(f"\nAll done. {len(TOPOLOGY_REGISTRY)} topologies built under {BASE_DIR}/")


if __name__ == "__main__":
    main()
