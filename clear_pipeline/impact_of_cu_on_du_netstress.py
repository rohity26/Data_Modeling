import os
from pathlib import Path
import numpy as np

BASE_DIR = Path("CU_NET_random_STRESS")
TOPOLOGIES = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
NET_STRESS_ID = 3  # Network Stress

_DU_FEAT_NAMES = (
    ["cpu", "mem_pct", "mem_bytes", "fs_writes", "net_tx", "net_rx"]
    + [f"pci_{i}" for i in range(22)]
    + ["net_diff", "net_ratio"]
)


def analyze_du_deviation_from_normal():
    print("=" * 115)
    print("         DU METRIC DEVIATION FROM NORMAL BASELINE DURING CU NETWORK STRESS")
    print("=" * 115)

    for topo in TOPOLOGIES:
        npz_path = BASE_DIR / f"{topo}_stress{NET_STRESS_ID}" / "test.npz"

        if not npz_path.exists():
            print(f"\n[!] Dataset not found: {npz_path}")
            continue

        data = np.load(npz_path)
        du = data["du"]                 # Shape: (T, N_DU, 30)
        cu_stress = data["cu_stress"]   # Shape: (T,) or (T, N_CU)

        if cu_stress.ndim > 1:
            cu_stress = cu_stress.max(axis=1)

        # Masks for normal baseline vs network stress
        cu_norm_mask = cu_stress == 0
        cu_strm_mask = cu_stress == NET_STRESS_ID

        du_norm = du[cu_norm_mask].reshape(-1, du.shape[-1])
        du_strm = du[cu_strm_mask].reshape(-1, du.shape[-1])

        print(
            f"\nTOPOLOGY: {topo:<18} | Connected DUs: {du.shape[1]} | Normal Samples: {len(du_norm)} | Stressed Samples: {len(du_strm)}"
        )
        print("-" * 115)
        print(
            f"{'DU Feature':<12} | {'Normal Mean':<16} | {'Stressed Mean':<16} | {'Delta (Δ)':<14} | {'% Deviation':<12} | {'Z-Shift (Δ/σ)'}"
        )
        print("-" * 115)

        for idx, feat_name in enumerate(_DU_FEAT_NAMES):
            if idx >= du.shape[-1]:
                break

            n_vals = du_norm[:, idx]
            s_vals = du_strm[:, idx]

            n_mean = n_vals.mean() if len(n_vals) > 0 else 0.0
            s_mean = s_vals.mean() if len(s_vals) > 0 else 0.0
            n_std = n_vals.std() if len(n_vals) > 0 else 0.0

            delta = s_mean - n_mean
            
            # Percentage deviation from normal baseline
            pct_dev = (delta / n_mean * 100) if abs(n_mean) > 1e-6 else 0.0
            
            # Z-Score shift (deviation relative to normal variance)
            z_shift = abs(delta) / (n_std if n_std > 1e-6 else 1.0)

            if "bytes" in feat_name:
                print(
                    f"{feat_name:<12} | {n_mean:<16.2f} | {s_mean:<16.2f} | {delta:<14.2f} | {pct_dev:<11.2f}% | {z_shift:.2f}"
                )
            else:
                print(
                    f"{feat_name:<12} | {n_mean:<16.4f} | {s_mean:<16.4f} | {delta:<14.4f} | {pct_dev:<11.2f}% | {z_shift:.2f}"
                )


if __name__ == "__main__":
    analyze_du_deviation_from_normal()


