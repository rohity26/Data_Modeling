import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path("CU_NET_random_STRESS")
TOPOLOGIES = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
STRESS_TYPES = {1: "CPU", 2: "MEM", 3: "NET"}

CU_METRICS = [
    "cpu",
    "mem_pct",
    "mem_bytes",
    "fs_reads",
    "fs_writes",
    "net_tx",
    "net_rx",
]

rows = []

for topo in TOPOLOGIES:
    for st_id, st_name in STRESS_TYPES.items():
        npz_path = BASE_DIR / f"{topo}_stress{st_id}" / "test.npz"

        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        cu = data["cu"]  # (T, 7)
        cu_stress = data["cu_stress"]  # (T,)

        norm_mask = cu_stress == 0
        strm_mask = cu_stress == st_id

        if norm_mask.sum() == 0 or strm_mask.sum() == 0:
            continue

        row_data = {"Topology": topo, "Stress": st_name}

        for idx, feat in enumerate(CU_METRICS):
            norm_mean = cu[norm_mask, idx].mean()
            norm_std = cu[norm_mask, idx].std() + 1e-8
            strm_mean = cu[strm_mask, idx].mean()

            shift = abs(strm_mean - norm_mean) / norm_std
            row_data[feat] = round(shift, 2)

        # Record max shift across all 7 CU metrics
        max_shift = max([row_data[f] for f in CU_METRICS])
        row_data["MAX_SHIFT"] = max_shift
        row_data["CU_DETECTABLE?"] = "YES" if max_shift >= 1.0 else "NO"

        rows.append(row_data)

df = pd.DataFrame(rows)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("========================================================================================================================")
print("                                  FULL CU FEATURE SHIFT MATRIX (ALL 7 METRICS)")
print("========================================================================================================================")
print(df.to_string(index=False))
print("========================================================================================================================")


