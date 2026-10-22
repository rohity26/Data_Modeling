"""build_newdata_test.py — Build test_newdata.npz for each topology x stress type
from the new per-entity CSVs in data_input/test_csv/, reusing the existing
train.npz files (built by build_dataset.py, unchanged) so the already-trained
checkpoints (random_model_ckpt_test_*.pt) can be evaluated on this new data
without retraining.

Each file in data_input/test_csv/ is named test_<entity>_<STRESS>_rep1.csv and
contains ONLY that entity stressed (everything else at stressType=0), e.g.
test_srscu0_CPU_rep1.csv has srscu0_stressType in {0,1} and every other
*_stressType column == 0 for all rows.

For a given topology + stress type, we concatenate just that topology's own
CU + DU files for that stress type (e.g. cu0_du0du1 + NET -> srscu0, srsdu0,
srsdu1 NET files). Because each source file is an independent experiment
repetition, we assign each file a synthetic row-index block separated by a
gap, so build_dataset.py's existing gap-based seg_id logic correctly marks a
new time-segment at every file boundary instead of treating unrelated reps as
one continuous stream.

Output: CU_NET_random_STRESS/<topo>_stress<N>/test_newdata.npz
        (train.npz in that folder is left untouched)

USAGE
-----
    cd project_root/step3_topoar/clear_pipeline
    python build_newdata_test.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import build_dataset as bd

NEW_CSV_DIR = Path("data_input/test_csv")
BASE_DIR = Path("CU_NET_random_STRESS")
GAP = 1000  # synthetic index gap inserted between concatenated reps

STRESS_TYPES = [1, 2, 3]  # CPU, MEM, NET


def load_topology_frame(topo_cfg: dict, stress_name: str) -> pd.DataFrame:
    """Concatenate this topology's own CU + DU rep files for one stress type,
    assigning a gapped synthetic index so file boundaries show up as seg_id
    boundaries downstream."""
    entities = [topo_cfg["cu_id"]] + list(topo_cfg["du_ids"])
    frames = []
    offset = 0
    for ent in entities:
        rep_paths = sorted(NEW_CSV_DIR.glob(f"test_{ent}_{stress_name}_rep*.csv"))
        assert rep_paths, f"No files found for entity={ent} stress={stress_name}"
        for p in rep_paths:
            df = pd.read_csv(p)
            n = len(df)
            df.index = np.arange(offset, offset + n)
            frames.append(df)
            offset += n + GAP
    return pd.concat(frames, axis=0)


def build_one(topology: str, stress_type: int):
    bd.STRESS_TYPE = stress_type  # print_summary() reads this module global
    stress_name = bd.STRESS_NAMES[stress_type]
    topo_cfg = bd.TOPOLOGY_REGISTRY[topology]
    cu_id, du_ids = topo_cfg["cu_id"], topo_cfg["du_ids"]

    print(f"\n{'='*60}")
    print(f"Topology: {topology}  Stress: {stress_name}")
    raw = load_topology_frame(topo_cfg, stress_name)
    print(f"  Concatenated raw rows: {len(raw)}  (from {1+len(du_ids)} files)")

    all_cols = raw.columns.tolist()
    cu_cols = bd.get_cu_columns(cu_id, all_cols)
    du_cols_per_instance = [
        bd.get_du_columns(du_id, pci_id, all_cols)
        for du_id, pci_id in zip(du_ids, topo_cfg["pci_ids"])
    ]

    # Defensive: apply the same two-step filter as build_dataset.py, even
    # though by construction every row already satisfies it (only this
    # topology's own entity is ever non-zero in these files).
    test = bd.filter_test(raw, topo_cfg, stress_type, bd.TOPOLOGY_REGISTRY)

    raw_idx = test.index.values
    seg_id = np.concatenate([[0], np.cumsum(np.diff(raw_idx) > 1)]).astype(np.int64)
    n_seg = int(seg_id[-1]) + 1
    print(f"  Time segments after filtering: {n_seg}")
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

    out_dir = BASE_DIR / f"{topology}_stress{stress_type}"
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
    bd.print_summary(cu_stress.astype(np.int64), du_stress.astype(np.int64),
                      f"{topology} / {stress_name} (new test data)")
    print(f"  Saved -> {out_path}")


def main():
    for stress_type in STRESS_TYPES:
        for topology in bd.TOPOLOGY_REGISTRY:
            build_one(topology, stress_type)


if __name__ == "__main__":
    main()
