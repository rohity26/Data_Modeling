import os
import numpy as np

FILES = ["train.npz", "test.npz"]

BASE_NAMES = [
    "cpu",
    "mem_pct",
    "mem_bytes",
    "fs_reads",
    "fs_writes",
    "net_tx",
    "net_rx",
]


def inspect_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}\n")
        return

    print("=" * 75)
    print(f" FILE: {file_path}")
    print("=" * 75)

    data = np.load(file_path)

    # 1. Inspect CU Array
    if "cu" in data:
        cu = data["cu"]
        print(f"\n[CU Array Shape: {cu.shape}]")
        print("-" * 70)
        # Collapse all axes except the last feature dimension
        axes_to_reduce = tuple(range(cu.ndim - 1))
        cu_min = np.min(cu, axis=axes_to_reduce)
        cu_max = np.max(cu, axis=axes_to_reduce)

        for idx, (min_val, max_val) in enumerate(zip(cu_min, cu_max)):
            name = BASE_NAMES[idx] if idx < len(BASE_NAMES) else f"feature_{idx}"
            # Active if non-zero anywhere or if values actually vary
            has_signal = not (min_val == 0 and max_val == 0)
            print(
                f"CU  | Idx {idx:2d} | {name:<10} | Active: {str(has_signal):<5} | Min: {min_val:10.4f} | Max: {max_val:10.4f}"
            )
    else:
        print("\n[CU Array not found in file]")

    # 2. Inspect DU Array
    if "du" in data:
        du = data["du"]
        print(f"\n[DU Array Shape: {du.shape}]")
        print("-" * 70)
        axes_to_reduce = tuple(range(du.ndim - 1))
        du_min = np.min(du, axis=axes_to_reduce)
        du_max = np.max(du, axis=axes_to_reduce)

        for idx, (min_val, max_val) in enumerate(zip(du_min, du_max)):
            if idx < 7:
                name = BASE_NAMES[idx]
            else:
                name = f"PCI Ch {idx - 7}"

            has_signal = not (min_val == 0 and max_val == 0)
            print(
                f"DU  | Idx {idx:2d} | {name:<12} | Active: {str(has_signal):<5} | Min: {min_val:10.4f} | Max: {max_val:10.4f}"
            )
    else:
        print("\n[DU Array not found in file]")

    print("\n")


if __name__ == "__main__":
    for f in FILES:
        inspect_dataset(f)


