import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Extract data directly from terminal run
features = [
    "cpu", "mem_pct", "mem_bytes", "fs_writes", "net_tx", "net_rx",
    "pci_0", "pci_1", "pci_2", "pci_3", "pci_4", "pci_5", "pci_6",
    "pci_7", "pci_8", "pci_9", "pci_10", "pci_11", "pci_12", "pci_13",
    "pci_14", "pci_15", "pci_16", "pci_17", "pci_18"
]

topologies = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5", "cu3_du6du7du8du9"]

# Z-Shift matrix (Rows: Features, Columns: Topologies)
z_shift_data = np.array([
    [0.28, 0.32, 0.34, 0.03], # cpu
    [0.07, 0.01, 0.08, 0.02], # mem_pct
    [0.08, 0.02, 0.15, 0.23], # mem_bytes
    [0.00, 0.00, 0.00, 0.00], # fs_writes
    [0.10, 0.17, 0.16, 0.03], # net_tx
    [0.58, 0.87, 0.61, 0.17], # net_rx
    [0.58, 0.87, 0.61, 0.17], # pci_0
    [0.27, 0.74, 0.45, 0.08], # pci_1
    [0.00, 0.00, 0.00, 0.00], # pci_2
    [0.64, 1.31, 0.88, 0.16], # pci_3
    [0.06, 0.18, 0.03, 0.03], # pci_4
    [0.17, 0.17, 0.17, 0.17], # pci_5
    [0.00, 0.00, 0.00, 0.00], # pci_6
    [0.63, 1.31, 0.88, 0.16], # pci_7
    [0.35, 0.95, 0.48, 0.05], # pci_8
    [0.00, 0.00, 0.00, 0.00], # pci_9
    [0.17, 0.17, 0.17, 0.17], # pci_10
    [0.10, 0.18, 0.16, 0.03], # pci_11
    [0.00, 0.00, 0.00, 0.00], # pci_12
    [0.00, 0.00, 0.00, 0.00], # pci_13
    [0.11, 0.19, 0.16, 0.03], # pci_14
    [0.57, 1.27, 0.68, 0.15], # pci_15
    [0.17, 0.17, 0.17, 0.17], # pci_16
    [0.00, 0.00, 0.00, 0.00], # pci_17
    [0.87, 1.12, 1.01, 0.02]  # pci_18
])

# % Deviation matrix
pct_dev_data = np.array([
    [-1.76,  -1.89,  -2.07,   0.19],
    [-0.36,  -0.07,  -0.37,   0.11],
    [-0.02,   0.00,  -0.03,  -0.05],
    [ 0.00,   0.00,   0.00,   0.00],
    [-36.59, -62.95, -54.68, -11.83],
    [ 2.85,   4.06,   2.90,   0.80],
    [ 2.84,   4.06,   2.90,   0.80],
    [-22.07, -57.99, -36.07,  -6.02],
    [ 0.00,   0.00,   0.00,   0.00],
    [-23.21, -47.30, -32.07,   5.90],
    [-27.46, -86.88, -16.92,  15.15],
    [ 2.82,   2.80,   2.83,   2.84],
    [ 0.00,   0.00,   0.00,   0.00],
    [-22.99, -47.26, -32.04,   5.85],
    [ 1.02,   1.41,   1.27,   0.06],
    [ 0.00,   0.00,   0.00,   0.00],
    [ 2.84,   2.81,   2.84,   2.82],
    [-65.87,-100.00,-100.00, -16.02],
    [ 0.00,   0.00,   0.00,   0.00],
    [ 0.00,   0.00,   0.00,   0.00],
    [-71.56,-100.00,-100.00, -16.52],
    [-22.29, -47.19, -26.00,  -5.67],
    [ 2.83,   2.80,   2.83,   2.83],
    [ 0.00,   0.00,   0.00,   0.00],
    [-35.35, -44.70, -40.54,   0.93]
])

df_z = pd.DataFrame(z_shift_data, index=features, columns=topologies)
df_pct = pd.DataFrame(pct_dev_data, index=features, columns=topologies)

# --- FIGURE 1: Z-SHIFT HEATMAP WITH SIGNIFICANCE THRESHOLD ---
plt.figure(figsize=(10, 10))
sns.heatmap(df_z, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Z-Shift (Δ/σ)'}, linewidths=0.5)
plt.axhline(y=0, color='black', linewidth=1)
plt.title("DU Feature Z-Shift Across Topologies\n(Values >= 0.50 flagged as Significant)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Topology", fontsize=12, labelpad=10)
plt.ylabel("DU Telemetry Feature", fontsize=12)
plt.tight_layout()
plt.savefig("du_zshift_heatmap.png", dpi=300)
plt.show()

# --- FIGURE 2: % DEVIATION BAR CHART FOR IMPACTED METRICS ---
sig_features = ["net_tx", "net_rx", "pci_0", "pci_1", "pci_3", "pci_7", "pci_15", "pci_18"]
df_pct_sig = df_pct.loc[sig_features].reset_index().melt(id_vars="index", var_name="Topology", value_name="Pct_Dev")
df_pct_sig.rename(columns={"index": "Feature"}, inplace=True)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_pct_sig, x="Feature", y="Pct_Dev", hue="Topology", palette="tab10")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Percentage Deviation (% Δ) for Impacted DU Features", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("DU Metric", fontsize=12, labelpad=10)
plt.ylabel("Percentage Deviation (%)", fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.legend(title="Topology", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("du_impacted_metrics_dev.png", dpi=300)
plt.show()



