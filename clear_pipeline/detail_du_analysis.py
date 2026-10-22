import os
from pathlib import Path
import numpy as np

BASE_DIR = Path("CU_NET_random_STRESS")
TOPOLOGIES = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
STRESS_TYPES = {
    1: "CPU STRESS (Stress Type 1)",
#    2: "MEMORY STRESS (Stress Type 2)",
#    3: "NETWORK STRESS (Stress Type 3)",
}

_CU_FEAT_NAMES = ["cpu", "mem_pct", "mem_bytes", "net_tx", "net_rx", "net_diff", "net_ratio"]
_DU_FEAT_NAMES = (
    ["cpu", "mem_pct", "mem_bytes", "fs_writes", "net_tx", "net_rx"]
    + [f"pci_{i}" for i in range(22)]
    + ["net_diff", "net_ratio"]
)


def analyze_stress_type(st_id, st_title):
    print("\n" + "=" * 105)
    print(f"                                {st_title}")
    print("=" * 105)

    for topo in TOPOLOGIES:
        npz_path = BASE_DIR / f"{topo}_stress{st_id}" / "test.npz"

        if not npz_path.exists():
            print(f"\n[!] Dataset not found for Topology: {topo}")
            continue

        data = np.load(npz_path)
        du = data["du"]               # Shape: (T, N_DU, 30)
        du_stress = data["du_stress"] # Shape: (T, N_DU) or (T,)

        if du_stress.ndim == 1:
            du_stress = np.repeat(du_stress[:, np.newaxis], du.shape[1], axis=1)

        du_flat = du.reshape(-1, du.shape[-1])
        stress_flat = du_stress.reshape(-1)

        norm_mask = stress_flat == 0
        strm_mask = stress_flat == st_id

        norm_count = norm_mask.sum()
        strm_count = strm_mask.sum()

        print(
            f"\nTOPOLOGY: {topo:<18} | Normal DU Rows: {norm_count:<8} | Stress DU Rows: {strm_count:<6}"
        )
        print("-" * 105)
        print(
            f"{'Feature':<12} | {'Normal Mean':<18} | {'Stress Mean':<18} | {'Normal Std':<18} | {'Signal Shift (Δμ/σ)'}"
        )
        print("-" * 105)

        for idx, feat_name in enumerate(_DU_FEAT_NAMES):
            if idx >= du_flat.shape[1]:
                break

            n_vals = du_flat[norm_mask, idx]
            s_vals = du_flat[strm_mask, idx] if strm_count > 0 else n_vals

            n_mean = n_vals.mean() if len(n_vals) > 0 else 0.0
            s_mean = s_vals.mean() if len(s_vals) > 0 else n_mean
            n_std = n_vals.std() if len(n_vals) > 0 else 0.0

            std_for_calc = n_std if n_std > 1e-6 else 1.0
            shift = abs(s_mean - n_mean) / std_for_calc

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


