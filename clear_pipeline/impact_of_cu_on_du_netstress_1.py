import os
from pathlib import Path
import numpy as np
from scipy import stats

BASE_DIR = Path("CU_NET_random_STRESS")
TOPOLOGIES = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
NET_STRESS_ID = 3  # Network Stress

# Exactly 25 features: 6 basic + 19 PCI (pci_0 to pci_18)
_DU_FEAT_NAMES = (
    ["cpu", "mem_pct", "mem_bytes", "fs_writes", "net_tx", "net_rx"]
    + [f"pci_{i}" for i in range(19)]
)


def analyze_du_deviation_from_normal():
    print("=" * 135)
    print("          DU METRIC DEVIATION FROM NORMAL BASELINE DURING CU NETWORK STRESS")
    print("=" * 135)

    for topo in TOPOLOGIES:
        npz_path = BASE_DIR / f"{topo}_stress{NET_STRESS_ID}" / "test.npz"

        if not npz_path.exists():
            print(f"\n[!] Dataset not found: {npz_path}")
            continue

        data = np.load(npz_path)
        du = data["du"]               # Shape: (T, N_DU, 25)
        cu_stress = data["cu_stress"]   # Shape: (T,) or (T, N_CU)
        du_stress = data["du_stress"]   # Shape: (T,) or (T, N_DU)

        if cu_stress.ndim > 1:
            cu_stress = cu_stress.max(axis=1)

        if du_stress.ndim > 1:
            du_stress = du_stress.max(axis=1)

        # 1. Clean Baseline Filtering: Ensure NO stress exists on CU or DU
        cu_norm_mask = (cu_stress == 0) & (du_stress == 0)

        # 2. Network Stress Window Mask [t1, t2]
        cu_strm_mask = (cu_stress == NET_STRESS_ID)

        # 3. Pooled Aggregation across all connected DUs in the topology
        du_norm = du[cu_norm_mask].reshape(-1, du.shape[-1])
        du_strm = du[cu_strm_mask].reshape(-1, du.shape[-1])

        print(
            f"\nTOPOLOGY: {topo:<18} | Connected DUs: {du.shape[1]} | Normal Samples: {len(du_norm)} | Stressed Samples: {len(du_strm)}"
        )
        print("-" * 135)
        print(
            f"{'DU Feature':<12} | {'Normal Mean':<16} | {'Stressed Mean':<16} | {'Delta (Δ)':<14} | {'% Deviation':<12} | {'Z-Shift (Δ/σ)':<13} | {'p-value':<10} | {'Significant?'}"
        )
        print("-" * 135)

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

            # 4. Statistical Significance Testing via Welch's t-test
            if len(n_vals) > 1 and len(s_vals) > 1 and (n_vals.std() > 0 or s_vals.std() > 0):
                _, p_val = stats.ttest_ind(s_vals, n_vals, equal_var=False)
            else:
                p_val = 1.0

            # Flag as significant if p < 0.05, Z >= 0.30, and non-zero baseline
            is_sig = "YES" if (p_val < 0.05 and z_shift >= 0.30 and abs(n_mean) > 1e-4) else "NO"
            p_str = f"{p_val:.2e}" if p_val < 0.01 else f"{p_val:.4f}"

            if "bytes" in feat_name:
                print(
                    f"{feat_name:<12} | {n_mean:<16.2f} | {s_mean:<16.2f} | {delta:<14.2f} | {pct_dev:<11.2f}% | {z_shift:<13.2f} | {p_str:<10} | {is_sig}"
                )
            else:
                print(
                    f"{feat_name:<12} | {n_mean:<16.4f} | {s_mean:<16.4f} | {delta:<14.4f} | {pct_dev:<11.2f}% | {z_shift:<13.2f} | {p_str:<10} | {is_sig}"
                )


if __name__ == "__main__":
    analyze_du_deviation_from_normal()


