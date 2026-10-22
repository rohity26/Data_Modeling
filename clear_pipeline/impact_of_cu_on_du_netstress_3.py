import os
from pathlib import Path
import numpy as np

BASE_DIR = Path("CU_NET_random_STRESS")
TOPOLOGIES = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
NET_STRESS_ID = 3  # Network Stress

# Exactly 25 features matching your NPZ shape (6 basic + 19 PCI)
_DU_FEAT_NAMES = (
    ["cpu", "mem_pct", "mem_bytes", "fs_writes", "net_tx", "net_rx"]
    + [f"pci_{i}" for i in range(19)]
)


def analyze_per_du_deviation_from_normal():
    print("=" * 125)
    print("        INDIVIDUAL DU METRIC DEVIATION FROM NORMAL BASELINE DURING CU NETWORK STRESS")
    print("=" * 125)

    for topo in TOPOLOGIES:
        npz_path = BASE_DIR / f"{topo}_stress{NET_STRESS_ID}" / "test.npz"

        if not npz_path.exists():
            print(f"\n[!] Dataset not found: {npz_path}")
            continue

        data = np.load(npz_path)
        du = data["du"]                # Shape: (T, N_DU, 25)
        cu_stress = data["cu_stress"]   # Shape: (T,) or (T, N_CU)
        du_stress = data["du_stress"]   # Shape: (T,) or (T, N_DU)

        if cu_stress.ndim > 1:
            cu_stress = cu_stress.max(axis=1)
        if du_stress.ndim > 1:
            du_stress = du_stress.max(axis=1)

        # 1. Clean Baseline Mask: Ensure NO stress on CU or DU
        cu_norm_mask = (cu_stress == 0) & (du_stress == 0)

        # 2. Network Stress Window Mask [t1, t2]
        cu_strm_mask = (cu_stress == NET_STRESS_ID)

        num_dus = du.shape[1]
        print(f"\n" + "#" * 125)
        print(f" TOPOLOGY: {topo:<18} | Connected DUs Count: {num_dus}")
        print("#" * 125)

        # 3. Iterate over each connected DU separately
        for du_idx in range(num_dus):
            # Extract data specifically for DU #du_idx
            # Shape of du_norm_i and du_strm_i: (T_samples, 25)
            du_norm_i = du[cu_norm_mask, du_idx, :]
            du_strm_i = du[cu_strm_mask, du_idx, :]

            print(f"\n  >>> Target: DU Index {du_idx} (DU_{du_idx}) | Normal Samples: {len(du_norm_i)} | Stressed Samples: {len(du_strm_i)}")
            print("  " + "-" * 121)
            print(
                f"  {'DU Feature':<12} | {'Normal Mean':<16} | {'Stressed Mean':<16} | {'Delta (Δ)':<14} | {'% Deviation':<12} | {'Z-Shift (Δ/σ)':<13} | {'Significant?'}"
            )
            print("  " + "-" * 121)

            for idx, feat_name in enumerate(_DU_FEAT_NAMES):
                if idx >= du.shape[-1]:
                    break

                n_vals = du_norm_i[:, idx]
                s_vals = du_strm_i[:, idx]

                n_mean = n_vals.mean() if len(n_vals) > 0 else 0.0
                s_mean = s_vals.mean() if len(s_vals) > 0 else 0.0
                n_std = n_vals.std() if len(n_vals) > 0 else 0.0

                delta = s_mean - n_mean

                # Percentage deviation from normal baseline
                pct_dev = (delta / n_mean * 100) if abs(n_mean) > 1e-6 else 0.0

                # Z-Score shift (deviation relative to normal baseline standard deviation)
                z_shift = abs(delta) / (n_std if n_std > 1e-6 else 1.0)

                # Robust rule: Significant if shift exceeds 0.50 standard deviations AND baseline > 0.0001
                is_sig = "YES" if (z_shift >= 0.50 and abs(n_mean) > 1e-4) else "NO"

                if "bytes" in feat_name:
                    print(
                        f"  {feat_name:<12} | {n_mean:<16.2f} | {s_mean:<16.2f} | {delta:<14.2f} | {pct_dev:<11.2f}% | {z_shift:<13.2f} | {is_sig}"
                    )
                else:
                    print(
                        f"  {feat_name:<12} | {n_mean:<16.4f} | {s_mean:<16.4f} | {delta:<14.4f} | {pct_dev:<11.2f}% | {z_shift:<13.2f} | {is_sig}"
                    )


if __name__ == "__main__":
    analyze_per_du_deviation_from_normal()


