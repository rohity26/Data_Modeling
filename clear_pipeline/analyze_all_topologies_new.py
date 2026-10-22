import os
from pathlib import Path
import numpy as np

BASE_DIR = Path("CU_NET_random_STRESS")
TOPOLOGIES = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
STRESS_TYPES = {1: "CPU", 2: "MEM", 3: "NET"}

CU_METRICS = ["cpu", "mem_pct", "mem_bytes", "net_tx", "net_rx"]
METRIC_INDICES = [0, 1, 2, 5, 6]  # Exclude fs_reads (3) and fs_writes (4)

print(
    f"{'TOPOLOGY':<18} | {'STRESS':<6} | {'cpu':<7} | {'mem_pct':<7} | {'mem_bytes':<10} | {'net_tx':<7} | {'net_rx':<7} | {'NET TARGET SIGNAL'}"
)
print("=" * 95)

for topo in TOPOLOGIES:
    for st_id, st_name in STRESS_TYPES.items():
        npz_path = BASE_DIR / f"{topo}_stress{st_id}" / "test.npz"

        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        cu = data["cu"]
        cu_stress = data["cu_stress"]

        norm_mask = cu_stress == 0
        strm_mask = cu_stress == st_id

        if norm_mask.sum() == 0 or strm_mask.sum() == 0:
            continue

        shifts = {}
        for idx, feat_name in zip(METRIC_INDICES, CU_METRICS):
            norm_mean = cu[norm_mask, idx].mean()
            norm_std = cu[norm_mask, idx].std()

            # Handle constant/zero variance metrics
            if norm_std < 1e-5:
                norm_std = 1.0

            strm_mean = cu[strm_mask, idx].mean()
            shift = abs(strm_mean - norm_mean) / norm_std
            shifts[feat_name] = shift

        net_signal = max(shifts["net_tx"], shifts["net_rx"])
        net_status = "STRONG" if net_signal >= 1.0 else "NO SIGNAL"

        print(
            f"{topo:<18} | {st_name:<6} | {shifts['cpu']:<7.2f} | {shifts['mem_pct']:<7.2f} | {shifts['mem_bytes']:<10.2f} | {shifts['net_tx']:<7.2f} | {shifts['net_rx']:<7.2f} | net_tx/rx max: {net_signal:.2f} ({net_status})"
        )
    print("-" * 95)


