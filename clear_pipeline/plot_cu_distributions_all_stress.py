import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================
TEST_CSV_PATH = "data_input/test_anomaly_th3.csv"
ALL_CU_IDS = ["srscu0", "srscu1", "srscu2", "srscu3"]
OUTPUT_DIR = os.path.join("cu-plot", "stress_comparison")

STRESS_LABELS = {0: "Normal", 1: "CPU Stress", 2: "Memory Stress", 3: "Network Stress"}

# 5 CORE / DERIVED METRICS FOR CUs
CADVISOR_PATTERNS = [
    ("CPU Usage", "container_cpu_user_seconds_total", "machine_cpu_cores", None),
    ("Mem Pct", "machine_memory_bytes", None, None),
    ("Mem Bytes", "container_memory_cache", None, "machine_memory_bytes"),
    ("Net TX", "container_network_transmit_bytes_total", 'interface="eth0"', None),
    ("Net RX", "container_network_receive_bytes_total", 'interface="eth0"', None),
]


def find_cu_columns(cu_id: str, all_cols: list[str]):
    """Extracts the 5 CU metric column names based on CADVISOR_PATTERNS."""
    cu_cols = []
    metric_names = []

    for name, required, extra, forbidden in CADVISOR_PATTERNS:
        matches = [
            c
            for c in all_cols
            if f'name="{cu_id}"' in c
            and 'instance="cadvisor:8080"' in c
            and "by (instance)" in c
            and required in c
            and (extra is None or extra in c)
            and (forbidden is None or forbidden not in c)
        ]
        if len(matches) == 1:
            cu_cols.append(matches[0])
            metric_names.append(name)
        else:
            print(f"[{cu_id}] Warning: Expected 1 match for {name}, found {len(matches)}")

    return cu_cols, metric_names


def plot_combined_stress_type(df: pd.DataFrame, target_stress_label: str, stress_color: str):
    """
    Generates a single combined grid figure comparing 'Normal' vs target_stress_label
    across ALL 4 CUs and ALL 5 metrics.
    """
    sns.set_theme(style="whitegrid")
    target_labels = ["Normal", target_stress_label]

    custom_palette = {
        "Normal": "#1f77b4",            # Blue
        target_stress_label: stress_color # Specific Stress Color
    }

    # Grid layout: 4 rows (one per CU) x 5 columns (one per metric)
    fig, axes = plt.subplots(len(ALL_CU_IDS), 5, figsize=(22, 14))

    for row_idx, cu_id in enumerate(ALL_CU_IDS):
        stress_col = f"{cu_id}_stressType"
        if stress_col not in df.columns:
            print(f"Skipping {cu_id}: '{stress_col}' not found.")
            continue

        # Prepare and filter CU data
        df_cu = df.copy()
        df_cu[stress_col] = df_cu[stress_col].fillna(0).astype(int)
        df_cu["Stress_Type_Label"] = df_cu[stress_col].map(STRESS_LABELS)
        filtered_df = df_cu[df_cu["Stress_Type_Label"].isin(target_labels)].copy()

        cu_cols, metric_names = find_cu_columns(cu_id, filtered_df.columns.tolist())

        for col_idx, (col_name, metric_title) in enumerate(zip(cu_cols, metric_names)):
            ax = axes[row_idx, col_idx]

            sns.kdeplot(
                data=filtered_df,
                x=col_name,
                hue="Stress_Type_Label",
                hue_order=target_labels,
                palette=custom_palette,
                common_norm=False,
                fill=True,
                alpha=0.35,
                ax=ax,
            )

            # Titles & Formatting
            if row_idx == 0:
                ax.set_title(metric_title, fontsize=13, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{cu_id.upper()}\nDensity", fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("Density")

            ax.set_xlabel(metric_title, fontsize=10)

    plt.suptitle(
        f"CU Metrics Density Comparison: Normal vs {target_stress_label} (Across All CUs)",
        fontsize=16,
        fontweight="bold",
        y=0.99
    )
    plt.tight_layout()

    filename = f"combined_all_cus_{target_stress_label.lower().replace(' ', '_')}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory initialized at: {OUTPUT_DIR}")

    print(f"Loading {TEST_CSV_PATH}...")
    df = pd.read_csv(TEST_CSV_PATH)

    # 1. CPU Stress Combined Plot (Orange)
    print("\nGenerating combined figure for CPU Stress...")
    plot_combined_stress_type(df, target_stress_label="CPU Stress", stress_color="#ff7f0e")

    # 2. Memory Stress Combined Plot (Green)
    print("Generating combined figure for Memory Stress...")
    plot_combined_stress_type(df, target_stress_label="Memory Stress", stress_color="#2ca02c")

    # 3. Network Stress Combined Plot (Red)
    print("Generating combined figure for Network Stress...")
    plot_combined_stress_type(df, target_stress_label="Network Stress", stress_color="#d62728")

    print(f"\nDone! All 3 combined figures saved in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
