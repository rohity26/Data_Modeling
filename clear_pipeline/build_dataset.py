"""build_dataset.py — Standalone dataset builder for topology + stress-type experiments.

USAGE
-----
Edit the USER INPUTS block below, then run:
    cd project_root/step3_topoar/clear_pipeline
    python build_dataset.py

WHAT IT DOES
------------
Given a raw train CSV (normal only) and a raw test CSV (all stress types mixed):

  Step 1 — Keep only the stress type you care about on the selected topology.
            Drop rows where the selected topology's CU or DU has a stress type
            that is neither 0 (normal) nor STRESS_TYPE (your target).
            Example: if STRESS_TYPE=1 (CPU), drop rows where cu1 has stress=2 or 3.

  Step 2 — Drop rows where any entity from a DIFFERENT topology is stressed.
            Example: for cu1_du2, drop rows where srscu0, srsdu0, or srsdu1 are stressed.

Result rows in test output:
    - Truly normal:  all entities at stress=0
    - Target stress: selected topology entity at STRESS_TYPE, every other entity at 0

OUTPUT
------
    <OUT_DIR>/
        train.npz    keys: cu(T,7)  du(T,N,37)  block_id(T,)
        test.npz     keys: cu(T,7)  du(T,N,37)  block_id(T,)
                           cu_stress(T,)  du_stress(T,N)
        blocks.csv   per-block summary: block_id, n_rows, cu_stress, du_stress
        summary.txt  row counts by category
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# USER INPUTS — change these only
# =============================================================================

#TRAIN_CSV   = Path("/home/somya/workspace/thesis3/topoar_gpu_run/clear_pipeline/data/train_normal_random_traffic_72h_exp.csv")
#TEST_CSV    = Path("/home/somya/workspace/thesis3/topoar_gpu_run/clear_pipeline/data/anomaly_random_cu_net_stress_exp.csv")
# Look for the input CSV paths at the top or inside the main block of build_dataset.py
# New downlink-UDP, 2-interface-CU dataset (see data_input/DATASET_CONTEXT.md).
# No matching stress test CSV exists yet — "Test sets with stress are collected
# separately" — so TEST_CSV is left pointing at a not-yet-existing path and
# main() falls back to a train-only build (train.npz only) when it's missing.
TRAIN_CSV = Path("data_input/train_normal_dl_udp_50h.csv")
TEST_CSV = Path("data_input/test_anomaly_dl_udp_net.csv")

TOPOLOGY    = "cu0_du0du1"   # options: "cu0_du0du1" | "cu1_du2" | "cu2_du3du4du5" | "cu3_du6du7du8du9"
STRESS_TYPE = 3           # 1=CPU | 2=MEM | 3=NET

BASE_DIR    = Path("CU_DL_UDP_STRESS")
OUT_DIR     = BASE_DIR / f"{TOPOLOGY}_stress{STRESS_TYPE}"
BUILD_ALL_TOPOS = True   # True: main() loops over every topology in TOPOLOGY_REGISTRY (ignores TOPOLOGY above)

# =============================================================================
# TOPOLOGY REGISTRY — add new topologies here if needed
# =============================================================================

STRESS_NAMES = {0: "normal", 1: "CPU", 2: "MEM", 3: "NET"}

# cu_id     : container name of the CU
# du_ids    : container names of all DUs under this CU
# pci_ids   : PCI number for each DU (PCI-{n}_... columns in CSV)
TOPOLOGY_REGISTRY = {
    "cu0_du0du1": {
        "cu_id":  "srscu0",
        "du_ids": ["srsdu0", "srsdu1"],
        "pci_ids": [1, 2],
    },
    "cu1_du2": {
        "cu_id":  "srscu1",
        "du_ids": ["srsdu2"],
        "pci_ids": [3],
    },
    "cu2_du3du4du5": {
        "cu_id":  "srscu2",
        "du_ids": ["srsdu3", "srsdu4", "srsdu5"],
        "pci_ids": [4, 5, 6],
    },
    # ─── ADD YOUR NEW 4-DU TOPOLOGY HERE ───
    "cu3_du6du7du8du9": {
        "cu_id":  "srscu3",                               # The exact container name of the 4th CU
        "du_ids": ["srsdu6", "srsdu7", "srsdu8", "srsdu9"], # List of all 4 DUs under this CU
        "pci_ids": [7, 8, 9, 10],                          # Their matching PCI Prefix IDs in the CSV
    },
}

# =============================================================================
# COLUMN DETECTION
# Each cadvisor KPI is identified by substrings in the Prometheus column name.
# Order here is the fixed output feature order (indices 0-6 in CU/DU arrays).
#
# The raw CSV has duplicate columns per metric (two Prometheus scrape forms):
#   "sum(..."     — the normalized/ratio form  → we WANT this one
#   "sum  (..."   — the raw irate form         → we skip this
# Universal discriminator: all correct columns contain "by (instance)".
# Network columns also have eth0 and eth1 variants — we always pick eth0.
# =============================================================================

# (metric_name, required_substring, extra_required_or_None, forbidden_or_None)
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
        and "by (instance)" in c          # skips the "sum  (...)" duplicate form
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
    """CU downlink ingress from the core.

    The CU now has two data interfaces: eth0 faces the DU (F1) and eth2 faces the
    5G core. Downlink enters on eth2 and leaves on eth0, so the conservation pair
    is eth0_tx / eth2_rx. The eth2 queries were emitted in the raw
    `sum  (irate(...))` form, which has no "by (instance)", so that discriminator
    cannot be applied here. interface="eth2" is itself unique enough.
    """
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
    """Return 7 cadvisor columns for the CU in fixed order.

    net_tx stays on eth0 (bytes out to the DU); net_rx moves to eth2 (bytes in
    from the core). Feature order and array shape are unchanged.
    """
    cols = []
    for name, req, extra, forb in CADVISOR_PATTERNS:
        if name == "net_rx":
            cols.append(_find_cu_eth2_rx(cu_id, all_cols))
        else:
            cols.append(_find_cadvisor_col(cu_id, all_cols, req, extra, forb))
    return cols


def get_du_columns(du_id: str, pci_id: int, all_cols: list[str]) -> list[str]:
    """Return 37 columns for one DU: 7 cadvisor + 30 PCI, in fixed order."""
    cad = [_find_cadvisor_col(du_id, all_cols, req, extra, forb)
           for _, req, extra, forb in CADVISOR_PATTERNS]
    pci = sorted([c for c in all_cols if c.startswith(f"PCI-{pci_id}_")])
    if len(pci) == 0:
        raise ValueError(f"No PCI-{pci_id}_* columns found")
    return cad + pci


# =============================================================================
# BLOCK ID — increments each time the (cu_stress, du_stress) pattern changes
# =============================================================================

def compute_block_id(cu_stress: np.ndarray, du_stress: np.ndarray) -> np.ndarray:
    state = np.concatenate([cu_stress[:, None], du_stress], axis=1)  # (T, 1+N)
    same_as_prev = np.all(state[1:] == state[:-1], axis=1)
    return np.concatenate([[0], np.cumsum(~same_as_prev)]).astype(np.int64)


# =============================================================================
# FILTERING
# =============================================================================

def filter_test(
    test: pd.DataFrame,
    topo_cfg: dict,
    stress_type: int,
    all_topo_cfgs: dict,
) -> pd.DataFrame:
    """Apply two-step filter to raw test CSV.

    Step 1 — Selected topology's entities: keep stress ∈ {0, stress_type} only.
    Step 2 — All other topology entities:  keep stress == 0 only.
    """
    keep = np.ones(len(test), dtype=bool)

    cu_id  = topo_cfg["cu_id"]
    du_ids = topo_cfg["du_ids"]
    selected_entities = {cu_id} | set(du_ids)

    # ── Step 1: selected topology must have stress in {0, stress_type} ────────
    cu_stress_col = f"{cu_id}_stressType"
    cu_s = test[cu_stress_col].fillna(0).astype(int).values
    keep &= np.isin(cu_s, [0, stress_type])

    for du_id in du_ids:
        du_s = test[f"{du_id}_stressType"].fillna(0).astype(int).values
        keep &= np.isin(du_s, [0, stress_type])

    # ── Step 2: all other entities must be at stress=0 ────────────────────────
    all_stress_cols = [c for c in test.columns if c.endswith("_stressType")]
    for col in all_stress_cols:
        entity = col.replace("_stressType", "")
        if entity in selected_entities:
            continue
        keep &= (test[col].fillna(0).astype(int).values == 0)

    n_dropped = (~keep).sum()
    print(f"  Filter dropped {n_dropped:,} rows, kept {keep.sum():,}")
    # Keep the ORIGINAL row index: dropped rows create hidden time-splices in
    # the output stream; the caller derives seg_id from gaps in this index so
    # downstream consumers can reset stateful processing (rolling stats, LSTM
    # hidden state, CUSUM) at each splice instead of treating the stream as
    # continuous time.
    return test[keep]


# =============================================================================
# SUMMARY PRINTER
# =============================================================================

def print_summary(cu_stress: np.ndarray, du_stress: np.ndarray, label: str) -> str:
    truly_normal = (cu_stress == 0) & np.all(du_stress == 0, axis=1)
    cu_target    = (cu_stress == STRESS_TYPE)
    du_target    = (cu_stress == 0) & np.any(du_stress == STRESS_TYPE, axis=1)

    lines = [
        f"\n{label}",
        f"  Total rows         : {len(cu_stress):>7,}",
        f"  Truly normal       : {truly_normal.sum():>7,}  (all stress=0)",
        f"  CU {STRESS_NAMES[STRESS_TYPE]} stress     : {cu_target.sum():>7,}",
        f"  DU {STRESS_NAMES[STRESS_TYPE]} stress     : {du_target.sum():>7,}",
        f"  Total anomaly rows : {(cu_target | du_target).sum():>7,}",
    ]
    for line in lines:
        print(line)
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def build_one(topology: str, train: pd.DataFrame, test_raw: pd.DataFrame | None):
    assert topology in TOPOLOGY_REGISTRY, \
        f"Unknown topology '{topology}'. Add it to TOPOLOGY_REGISTRY."
    assert STRESS_TYPE in STRESS_NAMES and STRESS_TYPE != 0, \
        "STRESS_TYPE must be 1 (CPU), 2 (MEM), or 3 (NET)."

    topo_cfg = TOPOLOGY_REGISTRY[topology]
    cu_id    = topo_cfg["cu_id"]
    du_ids   = topo_cfg["du_ids"]
    pci_ids  = topo_cfg["pci_ids"]
    n_du     = len(du_ids)

    out_dir = BASE_DIR / f"{topology}_stress{STRESS_TYPE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Topology   : {topology}  ({cu_id} + {du_ids})")
    print(f"Stress type: {STRESS_TYPE} = {STRESS_NAMES[STRESS_TYPE]}")
    print(f"Output dir : {out_dir}")
    print(f"{'='*60}")

    all_cols = train.columns.tolist()

    # ── Detect KPI columns ────────────────────────────────────────────────────
    print("\nDetecting KPI columns ...")
    cu_cols = get_cu_columns(cu_id, all_cols)
    du_cols_per_instance = [
        get_du_columns(du_id, pci_id, all_cols)
        for du_id, pci_id in zip(du_ids, pci_ids)
    ]
    cu_dim = len(cu_cols)
    du_dim = len(du_cols_per_instance[0])
    print(f"  CU dim: {cu_dim}  |  DU dim: {du_dim}  |  N_DU: {n_du}")

    # ── TRAINING: train CSV is all normal — no filtering needed ───────────────
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

    if test_raw is None:
        (out_dir / "summary.txt").write_text(
            f"Topology   : {topology}\n"
            f"Stress type: {STRESS_TYPE} = {STRESS_NAMES[STRESS_TYPE]}\n"
            f"Train CSV  : {TRAIN_CSV}\n"
            f"Test CSV   : (none yet — train-only build)\n"
            f"\nTRAIN ROW COUNT: {len(train_cu):,}\n"
            f"No test.npz written — no matching stress test CSV was found at {TEST_CSV}.\n"
        )
        print("No TEST_CSV found — skipping test.npz/blocks.csv (train-only build).")
        return

    # ── TESTING: apply two-step filter ────────────────────────────────────────
    print(f"\nFiltering test rows (Step 1: keep stress ∈ {{0,{STRESS_TYPE}}} on {topology}")
    print(f"                     Step 2: keep other-topology entities at stress=0) ...")
    test = filter_test(test_raw, topo_cfg, STRESS_TYPE, TOPOLOGY_REGISTRY)

    # seg_id: contiguous-time segments of the filtered stream. A new segment
    # starts wherever filtering removed rows (gap in the raw CSV index) —
    # adjacent output rows there are NOT adjacent in real time.
    raw_idx = test.index.values
    seg_id = np.concatenate([[0], np.cumsum(np.diff(raw_idx) > 1)]).astype(np.int64)
    n_seg = int(seg_id[-1]) + 1
    print(f"  Time segments after filtering: {n_seg} "
          f"(stateful consumers should reset at each boundary)")
    test = test.reset_index(drop=True)

    test_cu = test[cu_cols].fillna(0.0).astype(np.float32).values
    test_du = np.stack(
        [test[cols].fillna(0.0).astype(np.float32).values for cols in du_cols_per_instance],
        axis=1,
    )
    cu_stress = test[f"{cu_id}_stressType"].fillna(0).astype(np.int8).values
    du_stress = np.stack(
        [test[f"{du_id}_stressType"].fillna(0).astype(np.int8).values for du_id in du_ids],
        axis=1,
    )
    test_block_id = compute_block_id(cu_stress.astype(np.int64), du_stress.astype(np.int64))
    print(f"  test_cu:  {test_cu.shape}  test_du:  {test_du.shape}")

    np.savez(
        out_dir / "test.npz",
        cu=test_cu,
        du=test_du,
        block_id=test_block_id,
        cu_stress=cu_stress,
        du_stress=du_stress,
        seg_id=seg_id,
    )

    # ── Save blocks.csv ───────────────────────────────────────────────────────
    block_rows = []
    for bid in np.unique(test_block_id):
        mask = test_block_id == bid
        block_rows.append({
            "block_id":  int(bid),
            "n_rows":    int(mask.sum()),
            "cu_stress": int(cu_stress[mask][0]),
            "du_stress": ",".join(str(int(x)) for x in du_stress[mask][0]),
        })
    pd.DataFrame(block_rows).to_csv(out_dir / "blocks.csv", index=False)

    # ── Print and save summary ────────────────────────────────────────────────
    summary = print_summary(cu_stress.astype(np.int64), du_stress.astype(np.int64), "TEST ROW BREAKDOWN")
    (out_dir / "summary.txt").write_text(
        f"Topology   : {topology}\n"
        f"Stress type: {STRESS_TYPE} = {STRESS_NAMES[STRESS_TYPE]}\n"
        f"Train CSV  : {TRAIN_CSV}\n"
        f"Test CSV   : {TEST_CSV}\n"
        + summary + "\n"
    )

    print(f"\nDone. Files written:")
    print(f"  {out_dir}/train.npz  — cu{train_cu.shape} du{train_du.shape}")
    print(f"  {out_dir}/test.npz   — cu{test_cu.shape} du{test_du.shape}")
    print(f"  {out_dir}/blocks.csv")
    print(f"  {out_dir}/summary.txt")


def main():
    print(f"\nLoading {TRAIN_CSV.name} ...")
    train = pd.read_csv(TRAIN_CSV)
    print(f"  shape: {train.shape}")

    test_raw = None
    if TEST_CSV.exists():
        print(f"Loading {TEST_CSV.name} ...")
        test_raw = pd.read_csv(TEST_CSV)
        print(f"  shape: {test_raw.shape}")
    else:
        print(f"\nTEST_CSV not found at {TEST_CSV} — building train.npz only for every topology.")

    topologies = list(TOPOLOGY_REGISTRY) if BUILD_ALL_TOPOS else [TOPOLOGY]
    for topology in topologies:
        build_one(topology, train, test_raw)


if __name__ == "__main__":
    main()
