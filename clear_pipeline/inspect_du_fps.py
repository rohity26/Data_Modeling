import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path("CU_NET_random_STRESS")
MODEL_PATH = Path("all.pt")  # Path to your trained PyTorch model checkpoint
STRESS_TYPE = 3             # NET Stress

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = list(range(25))

DU_FEAT_NAMES = (
    ["cpu", "mem_pct", "mem_bytes", "fs_reads", "fs_writes", "net_tx", "net_rx"]
    + [f"pci_{i}" for i in range(18)]
    + ["net_diff", "net_ratio"]
)

TOPOLOGIES = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]


def process_features(z):
    """Applies exact feature slicing & derivations from pipeline."""
    cu = z["cu"].astype(np.float32)
    du = z["du"].astype(np.float32)
    N_DU = du.shape[1]

    cu[:, 5] /= N_DU
    cu[:, 6] /= N_DU

    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]

    _tx, _rx = cu[:, 3:4], cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)

    _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)

    return cu, du


def inspect_flagged_du_timesteps():
    if not MODEL_PATH.exists():
        print(f"[!] Model file not found at {MODEL_PATH}. Checking dataset directly...")

    print("=" * 90)
    print(" INSPECTING TIMESTEPS WHERE DU ANOMALIES ARE FLAGGED DURING CU NET STRESS")
    print("=" * 90)

    for topo in TOPOLOGIES:
        npz_path = BASE_DIR / f"{topo}_stress{STRESS_TYPE}" / "test.npz"
        if not npz_path.exists():
            continue

        z = np.load(npz_path)
        cu, du = process_features(z)
        cu_stress = z["cu_stress"].astype(np.int32)
        du_stress = z["du_stress"].astype(np.int32)

        # Timesteps with CU NET Stress active and DU unstressed
        cu_net_mask = (cu_stress == STRESS_TYPE) & np.all(du_stress == 0, axis=1)
        norm_mask = (cu_stress == 0) & np.all(du_stress == 0, axis=1)

        print(f"\nTopology: {topo} | Timesteps under CU NET Stress: {cu_net_mask.sum()}")

        N_DU = du.shape[1]
        for d_idx in range(N_DU):
            du_norm = du[norm_mask, d_idx, :]
            du_stress_data = du[cu_net_mask, d_idx, :]

            norm_mean = du_norm.mean(axis=0)
            norm_std = np.where(du_norm.std(axis=0) < 1e-6, 1.0, du_norm.std(axis=0))

            # Calculate point-by-point Instantaneous Z-Score Shift per timestep
            # Z_t,j = |X_t,j - mu_j| / sigma_j
            pointwise_z = np.abs(du_stress_data - norm_mean) / norm_std  # (T_stress, 27)
            max_z_per_timestep = pointwise_z.max(axis=1)                 # (T_stress,)

            # Timesteps where at least ONE DU feature spiked > 2.0 standard deviations
            spike_timesteps = np.where(max_z_per_timestep >= 2.0)[0]

            print(f"  DU_{d_idx}: {len(spike_timesteps)} / {len(du_stress_data)} timesteps had instant feature spikes (Z >= 2.0)")

            if len(spike_timesteps) > 0:
                # Find top features causing spikes on these specific timesteps
                avg_spike_z = pointwise_z[spike_timesteps].mean(axis=0)
                top_feats = np.argsort(avg_spike_z)[::-1][:5]

                records = []
                for f_idx in top_feats:
                    records.append({
                        "DU Feature": DU_FEAT_NAMES[f_idx],
                        "Baseline Mean": round(float(norm_mean[f_idx]), 4),
                        "Spike Timestep Mean": round(float(du_stress_data[spike_timesteps, f_idx].mean()), 4),
                        "Instant Spike Z-Score": round(float(avg_spike_z[f_idx]), 2)
                    })
                
                df = pd.DataFrame(records)
                print(f"    [!] Top features driving instant DU anomaly triggers on DU_{d_idx}:")
                print(df.to_string(index=False))


if __name__ == "__main__":
    inspect_flagged_du_timesteps()


