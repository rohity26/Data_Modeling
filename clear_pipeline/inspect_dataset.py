import numpy as np
from pathlib import Path

# Path to dataset directory
BASE_DIR = Path("CU_NET_random_STRESS")

def inspect_topology_data():
    if not BASE_DIR.exists():
        print(f"Directory {BASE_DIR} does not exist.")
        return

    npz_files = sorted(list(BASE_DIR.glob("**/*.npz")))
    if not npz_files:
        print(f"No .npz files found under {BASE_DIR}")
        return

    print("=" * 80)
    print(f"{'Folder / File':<35} | {'CU Shape (T, Feats)':<20} | {'DU Shape (T, N_DU, Feats)':<22}")
    print("=" * 80)

    for p in npz_files:
        rel_path = p.relative_to(BASE_DIR)
        data = np.load(p)
        
        cu_shape = data["cu"].shape if "cu" in data else "Missing"
        du_shape = data["du"].shape if "du" in data else "Missing"
        
        print(f"{str(rel_path):<35} | {str(cu_shape):<20} | {str(du_shape):<22}")

    print("=" * 80)

    # Detailed channel summary from the first found train.npz
    sample_file = npz_files[0]
    sample_data = np.load(sample_file)
    cu_raw = sample_data["cu"]
    du_raw = sample_data["du"]

    print(f"\nDetailed Breakdown (from {sample_file.relative_to(BASE_DIR)}):")
    print(f"  * Raw CU Total Features  : {cu_raw.shape[-1]}")
    print(f"  * Raw DU Total Features  : {du_raw.shape[-1]}")
    print(f"  * Number of DUs (N_DU)  : {du_raw.shape[1]}")

if __name__ == "__main__":
    inspect_topology_data()
