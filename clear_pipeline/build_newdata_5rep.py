"""Build test_newdata.npz for the new test_csv_tcp_net_5rep collection.

Unlike the other test_csv_* folders, this one only has CU-side files (one per
CU per rep, 5 reps), each with the FULL 865-column schema (all entities'
columns are present in every rep file; only that rep's own CU actually went
into NET stress -- everything else, including its own topology's DUs, is
just normal background readings). So each rep file is already a complete,
self-contained topology-window sample; no per-entity file concatenation is
needed the way build_newdata_test.py does it.

Each of the 5 reps gets its own seg_id (via the GAP-offset trick), so the
segmented, own-file evaluation gives 5 independent per-rep results per CU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import build_dataset as bd

NEW_CSV_DIR = Path("data_input/test_csv_tcp_net_5rep")
BASE_DIR = Path("CU_NET_5REP_STRESS")
GAP = 1000
STRESS_TYPE = 3  # NET


def load_cu_reps(cu_id: str) -> pd.DataFrame:
    rep_paths = sorted(NEW_CSV_DIR.glob(f"test_{cu_id}_NET_rep*.csv"))
    assert rep_paths, f"No rep files found for {cu_id}"
    frames = []
    offset = 0
    for p in rep_paths:
        df = pd.read_csv(p)
        n = len(df)
        df.index = np.arange(offset, offset + n)
        frames.append(df)
        offset += n + GAP
    return pd.concat(frames, axis=0), [p.name for p in rep_paths]


def build_one(topology: str):
    bd.STRESS_TYPE = STRESS_TYPE
    topo_cfg = bd.TOPOLOGY_REGISTRY[topology]
    cu_id, du_ids = topo_cfg["cu_id"], topo_cfg["du_ids"]

    print(f"\n{'='*60}")
    print(f"Topology: {topology}  Stress: NET (5-rep)")
    raw, rep_names = load_cu_reps(cu_id)
    print(f"  Concatenated raw rows: {len(raw)}  (from {len(rep_names)} reps: {rep_names})")

    all_cols = raw.columns.tolist()
    cu_cols = bd.get_cu_columns(cu_id, all_cols)
    du_cols_per_instance = [
        bd.get_du_columns(du_id, pci_id, all_cols)
        for du_id, pci_id in zip(du_ids, topo_cfg["pci_ids"])
    ]

    test = bd.filter_test(raw, topo_cfg, STRESS_TYPE, bd.TOPOLOGY_REGISTRY)

    raw_idx = test.index.values
    seg_id = np.concatenate([[0], np.cumsum(np.diff(raw_idx) > 1)]).astype(np.int64)
    n_seg = int(seg_id[-1]) + 1
    print(f"  Time segments after filtering: {n_seg} (expect {len(rep_names)})")
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
    test_block_id = bd.compute_block_id(cu_stress.astype(np.int64), du_stress.astype(np.int64))
    print(f"  test_cu: {test_cu.shape}  test_du: {test_du.shape}")
    print(f"  CU NET-stress rows: {(cu_stress==STRESS_TYPE).sum()}")

    out_dir = BASE_DIR / f"{topology}_stress{STRESS_TYPE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    assert (out_dir / "train.npz").exists(), f"Expected existing train.npz in {out_dir}"
    out_path = out_dir / "test_newdata.npz"
    np.savez(
        out_path,
        cu=test_cu,
        du=test_du,
        block_id=test_block_id,
        cu_stress=cu_stress,
        du_stress=du_stress,
        seg_id=seg_id,
    )
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    for topology in bd.TOPOLOGY_REGISTRY:
        build_one(topology)
