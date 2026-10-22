import numpy as np

# Load your dataset
data = np.load("train.npz")  # Replace with your actual npz file path
du_data = data["du"]  # Expected shape: (samples, time_steps, 37)

print(f"DU Array Shape: {du_data.shape}")

# Calculate variance or max values per metric channel (axis 0 & 1)
metric_activity = np.max(np.abs(du_data), axis=(0, 1))

print("\n--- PCI Channel Status ---")
# Metrics 0 to 6 are base CADVISOR metrics (including fs_reads at index 3)
# Metrics 7 to 36 correspond to PCI channels
for idx in range(7, du_data.shape[-1]):
    max_val = metric_activity[idx]
    has_signal = max_val > 0  # True if channel contains data, False if constant zero
    pci_num = idx - 7
    print(
        f"Raw Index {idx:2d} | PCI Channel {pci_num:2d} | Active: {has_signal} (Max Val: {max_val:.4f})"
    )


