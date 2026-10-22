import os
from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION — Matches run_experiment.py for CU NET STRESS
# =============================================================================
BASE_DIR = Path("CU_NET_random_STRESS")
TOPOLOGIES = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]
STRESS_TYPE = 3  # NET Stress

CU_FEAT_NAMES = ["cpu", "mem_pct", "mem_bytes", "net_tx", "net_rx", "net_diff", "net_ratio"]
DU_FEAT_NAMES = (
    ["cpu", "mem_pct", "mem_bytes", "fs_reads", "fs_writes", "net_tx", "net_rx"]
    + [f"pci_{i}" for i in range(18)]
    + ["net_diff", "net_ratio"]
)

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = list(range(25))


def process_features(z):
    """Applies exact feature slicing & derivations from run_experiment.py."""
    cu = z["cu"].astype(np.float32)
    du = z["du"].astype(np.float32)
    N_DU = du.shape[1]

    # Normalize CU traffic by DU count
    cu[:, 5] = cu[:, 5] / N_DU
    cu[:, 6] = cu[:, 6] / N_DU

    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]

    # Derived CU features
    _tx, _rx = cu[:, 3:4], cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)

    # Derived DU features
    _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)

    return cu, du


def analyze_cu_net_cascade():
    print(f"=" * 85)
    print(f" VALIDATING CASCADING IMPACT OF CU NET STRESS (STRESS_TYPE = {STRESS_TYPE})")
    print(f" Dataset Directory: {BASE_DIR}")
    print(f"=" * 85)

    all_topo_results = []

    for topo in TOPOLOGIES:
        npz_path = BASE_DIR / f"{topo}_stress{STRESS_TYPE}" / "test.npz"
        if not npz_path.exists():
            print(f"\n[-] File not found: {npz_path}, skipping...")
            continue

        z = np.load(npz_path)
        cu, du = process_features(z)
        cu_stress = z["cu_stress"].astype(np.int32)
        du_stress = z["du_stress"].astype(np.int32)

        N_DU = du.shape[1]

        # 1. Condition A: Baseline Normal (Both CU and all DUs unstressed)
        norm_mask = (cu_stress == 0) & np.all(du_stress == 0, axis=1)

        # 2. Condition B: CU NET Stress ONLY (CU stressed with NET, DUs explicitly marked unstressed)
        cu_net_stress_mask = (cu_stress == STRESS_TYPE) & np.all(du_stress == 0, axis=1)

        n_norm = norm_mask.sum()
        n_cascade = cu_net_stress_mask.sum()

        print(f"\nTopology: {topo:<16s} | Pure Normal Timesteps: {n_norm:>5d} | CU-NET Stress Timesteps: {n_cascade:>5d}")

        if n_norm < 10 or n_cascade < 10:
            print("  [-] Not enough timesteps to calculate stats reliably.")
            continue

        for d_idx in range(N_DU):
            du_norm = du[norm_mask, d_idx, :]       # (T_norm, 27)
            du_casc = du[cu_net_stress_mask, d_idx, :]  # (T_casc, 27)

            norm_means = du_norm.mean(axis=0)
            norm_stds = du_norm.std(axis=0)
            casc_means = du_casc.mean(axis=0)

            # Avoid division by zero on flat/constant features
            stds_clean = np.where(norm_stds < 1e-6, 1.0, norm_stds)

            # Signal Shift (Z-score shift = |mean_cascade - mean_normal| / std_normal)
            shifts = np.abs(casc_means - norm_means) / stds_clean

            # Identify features shifted by >= 1.5 standard deviations
            shifted_indices = np.where(shifts >= 1.5)[0]

            if len(shifted_indices) > 0:
                print(f"  [!] DU_{d_idx} SHOWS PHYSICAL FEATURE SHIFT DURING CU NET STRESS:")
                records = []
                for idx in sorted(shifted_indices, key=lambda i: shifts[i], reverse=True):
                    feat_name = DU_FEAT_NAMES[idx] if idx < len(DU_FEAT_NAMES) else f"feat_{idx}"
                    records.append({
                        "DU_Feature": feat_name,
                        "Normal_Mean": round(float(norm_means[idx]), 4),
                        "CU_NET_Stress_DU_Mean": round(float(casc_means[idx]), 4),
                        "Normal_Std": round(float(norm_stds[idx]), 4),
                        "Signal_Shift (Δμ/σ)": round(float(shifts[idx]), 2)
                    })
                    all_topo_results.append({
                        "Topology": topo,
                        "DU_Idx": f"DU_{d_idx}",
                        "Feature": feat_name,
                        "Shift_ZScore": round(float(shifts[idx]), 2)
                    })
                df = pd.DataFrame(records)
                print(df.to_string(index=False))
            else:
                max_shift_idx = np.argmax(shifts)
                max_feat_name = DU_FEAT_NAMES[max_shift_idx]
                print(f"  [-] DU_{d_idx}: No significant shift. Max shift was in '{max_feat_name}' (Δμ/σ = {shifts[max_shift_idx]:.2f})")

    # Overall Summary Table
    print("\n" + "=" * 85)
    print(" SUMMARY OF DU FEATURES IMPACTED BY CU NET STRESS")
    print("=" * 85)
    if all_topo_results:
        df_summary = pd.DataFrame(all_topo_results)
        print(df_summary.to_string(index=False))
        print("\nCONCLUSION: Mentor's hypothesis CONFIRMED.")
        print("CU NET stress propagates downstream to DU physical metrics.")
        print("DU False Positives (FPs) are valid downstream cascade effects.")
    else:
        print("No DU features shifted significantly (Δμ/σ < 1.5).")
        print("\nCONCLUSION: CU DU hypothesis REJECTED.")
        print("DU metrics remain stable under CU NET stress. DU FPs are model noise.")


if __name__ == "__main__":
    analyze_cu_net_cascade()
