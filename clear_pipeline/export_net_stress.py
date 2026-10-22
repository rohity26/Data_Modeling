import os
from pathlib import Path
import numpy as np
import pandas as pd

# Core cAdvisor telemetry feature names
CADVISOR_METRICS = ["cpu", "mem_pct", "mem_bytes", "fs_reads", "fs_writes", "net_tx", "net_rx"]

base_dir = Path("CU_NET_random_STRESS")
output_excel = "network_stress_telemetry_full.xlsx"

# Locate all Network Stress folders (*_stress3)
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
            block_id = data["block_id"]     # Shape: (T,)
            seg_id = data["seg_id"]         # Shape: (T,)

            T, n_du, du_dim = du.shape

            # Build metadata columns
            df_dict = {
                "time_step": np.arange(T),
                "block_id": block_id,
                "seg_id": seg_id,
                "cu_stress": cu_stress,
            }

            # Extract 7 CU metrics
            for i, name in enumerate(CADVISOR_METRICS):
                df_dict[f"cu_{name}"] = cu[:, i]

            # Extract DU metrics per DU instance (7 cAdvisor + 30 PCI features)
            for d in range(n_du):
                df_dict[f"du{d}_stress"] = du_stress[:, d]
                
                # 7 cAdvisor metrics
                for i, name in enumerate(CADVISOR_METRICS):
                    df_dict[f"du{d}_{name}"] = du[:, d, i]
                    
                # 30 PCI metrics
                for p in range(du_dim - 7):
                    df_dict[f"du{d}_pci_{p}"] = du[:, d, 7 + p]

            # Convert to DataFrame and write to Excel tab
            df = pd.DataFrame(df_dict)
            sheet_name = folder.name.replace("_stress3", "")[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Exported {folder.name} -> Sheet: '{sheet_name}' ({df.shape[0]} rows, {df.shape[1]} columns)")

    print(f"\nSuccessfully created '{output_excel}' containing all network stress values!")


