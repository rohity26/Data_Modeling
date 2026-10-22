import os
from pathlib import Path
import numpy as np
import pandas as pd

base_dir = Path("CU_NET_random_STRESS")
output_excel = "network_stress_net_only.xlsx"

# Locate network stress folders (*_stress3)
stress_folders = sorted(base_dir.glob("*_stress3"))

if not stress_folders:
    print("No network stress folders (*_stress3) found.")
else:
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for folder in stress_folders:
            npz_path = folder / "test.npz"
            if not npz_path.exists():
                continue

            # Load binary NPZ dataset
            data = np.load(npz_path)
            cu = data["cu"]                 # Shape: (T, 7)
            du = data["du"]                 # Shape: (T, N_du, 37)
            cu_stress = data["cu_stress"]   # Shape: (T,)
            du_stress = data["du_stress"]   # Shape: (T, N_du)

            T, n_du, _ = du.shape

            # Build dataset containing ONLY Timestamp, Stress, net_tx, and net_rx
            df_dict = {
                "time_step": np.arange(T),
                "cu_stress": cu_stress,
                "cu_net_tx": cu[:, 5],  # net_tx index
                "cu_net_rx": cu[:, 6],  # net_rx index
            }

            # Extract net_tx and net_rx for each DU instance
            for d in range(n_du):
                df_dict[f"du{d}_stress"] = du_stress[:, d]
                df_dict[f"du{d}_net_tx"] = du[:, d, 5]
                df_dict[f"du{d}_net_rx"] = du[:, d, 6]

            df = pd.DataFrame(df_dict)

            # Optional: Uncomment the line below if you want ONLY rows where stress is active (>0)
            # df = df[(df["cu_stress"] > 0) | (df[[f"du{d}_stress" for d in range(n_du)]] > 0).any(axis=1)]

            sheet_name = folder.name.replace("_stress3", "")[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Exported {folder.name} -> Sheet: '{sheet_name}' ({df.shape[0]} rows, {df.shape[1]} columns)")

    print(f"\nSuccessfully saved targeted metrics to '{output_excel}'!")


