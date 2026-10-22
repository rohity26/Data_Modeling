import os
from pathlib import Path
import numpy as np

BASE_DIR = Path("CU_NET_random_STRESS")
TOPOLOGIES = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
STRESS_TYPES = {
    1: "CPU STRESS (Stress Type 1)",
    2: "MEMORY STRESS (Stress Type 2)",
    3: "NETWORK STRESS (Stress Type 3)",
}

CU_METRICS = ["cpu", "mem_pct", "mem_bytes", "net_tx", "net_rx"]
METRIC_INDICES = [0, 1, 2, 5, 6]  # Excludes fs_reads (3) and fs_writes (4)


def analyze_stress_type(st_id, st_title):
    print("\n" + "=" * 105)
    print(f"                               {st_title}")
    print("=" * 105)

    for topo in TOPOLOGIES:
        npz_path = BASE_DIR / f"{topo}_stress{st_id}" / "test.npz"

        if not npz_path.exists():
            print(f"\n[!] Dataset not found for Topology: {topo}")
            continue

        data = np.load(npz_path)
        cu = data["cu"]  # (T, 7)
        cu_stress = data["cu_stress"]  # (T,)

        norm_mask = cu_stress == 0
        strm_mask = cu_stress == st_id

        norm_count = norm_mask.sum()
        strm_count = strm_mask.sum()

        print(
            f"\nTOPOLOGY: {topo:<18} | Normal Rows: {norm_count:<6} | Stress Rows: {strm_count:<5}"
        )
        print("-" * 105)
        print(
            f"{'Feature':<12} | {'Normal Mean':<18} | {'Stress Mean':<18} | {'Normal Std':<18} | {'Signal Shift (Δμ/σ)'}"
        )
        print("-" * 105)

        for idx, feat_name in zip(METRIC_INDICES, CU_METRICS):
            n_mean = cu[norm_mask, idx].mean()
            s_mean = cu[strm_mask, idx].mean() if strm_count > 0 else n_mean
            n_std = cu[norm_mask, idx].std()

            # Prevent divide-by-zero for constant features
            std_for_calc = n_std if n_std > 1e-6 else 1.0
            shift = abs(s_mean - n_mean) / std_for_calc

            # Format large memory byte numbers properly, small floats normally
            if "bytes" in feat_name:
                print(
                    f"{feat_name:<12} | {n_mean:<18.2f} | {s_mean:<18.2f} | {n_std:<18.2f} | {shift:.2f}"
                )
            else:
                print(
                    f"{feat_name:<12} | {n_mean:<18.4f} | {s_mean:<18.4f} | {n_std:<18.4f} | {shift:.2f}"
                )


def main():
    for st_id, st_title in STRESS_TYPES.items():
        analyze_stress_type(st_id, st_title)
    print("\n" + "=" * 105 + "\n")


if __name__ == "__main__":
    main()

