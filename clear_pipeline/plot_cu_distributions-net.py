import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================
TEST_CSV_PATH = "data_input/test_anomaly_th3.csv"
ALL_CU_IDS = ["srscu0", "srscu1", "srscu2", "srscu3"]  # All CUs to process
OUTPUT_DIR = os.path.join("cu-plot", "net-stress")      # Target folder path

STRESS_LABELS = {0: "Normal", 1: "CPU Stress", 2: "Memory Stress", 3: "Network Stress"}

# 5 CORE/DERIVED CU METRIC PATTERNS
CADVISOR_PATTERNS = [
    (
        "CPU Usage",
        "container_cpu_user_seconds_total",
        "machine_cpu_cores",
        None,
    ),
    ("Mem Pct", "machine_memory_bytes", None, None),
    (
        "Mem Bytes",
        "container_memory_cache",
        None,
        "machine_memory_bytes",
    ),
    (
        "Net TX",
        "container_network_transmit_bytes_total",
        'interface="eth0"',
        None,
    ),
    (
        "Net RX",
        "container_network_receive_bytes_total",
        'interface="eth0"',
        None,
    ),
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


def generate_plots_for_cu(df: pd.DataFrame, cu_id: str, output_dir: str):
    """Generates and saves KDE and Boxplot figures for a specific CU."""
    stress_col = f"{cu_id}_stressType"
    if stress_col not in df.columns:
        print(f"Skipping {cu_id}: Column '{stress_col}' not found in CSV.")
        return

    # Clean stress column and map labels
    df_copy = df.copy()
    df_copy[stress_col] = df_copy[stress_col].fillna(0).astype(int)
    df_copy["Stress_Type_Label"] = df_copy[stress_col].map(STRESS_LABELS)

    # Filter for only 'Normal' and 'Network Stress'
    target_labels = ["Normal", "Network Stress"]
    filtered_df = df_copy[df_copy["Stress_Type_Label"].isin(target_labels)].copy()

    # Match metric columns for this CU
    cu_cols, metric_names = find_cu_columns(cu_id, filtered_df.columns.tolist())

    if len(cu_cols) == 0:
        print(f"Skipping {cu_id}: No metric columns matched.")
        return

    sns.set_theme(style="whitegrid")

    # Custom color palette: Blue for Normal, Red for Network Stress
    custom_palette = {
        "Normal": "#1f77b4",
        "Network Stress": "#d62728",
    }

    # =========================================================================
    # 1. KDE / DENSITY DISTRIBUTIONS
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (col, name) in enumerate(zip(cu_cols, metric_names)):
        sns.kdeplot(
            data=filtered_df,
            x=col,
            hue="Stress_Type_Label",
            hue_order=target_labels,
            palette=custom_palette,
            common_norm=False,
            fill=True,
            alpha=0.3,
            ax=axes[idx],
        )
        axes[idx].set_title(f"Distribution: {name}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel(name)
        axes[idx].set_ylabel("Density")

    # Remove the 6th unused subplot slot cleanly
    for i in range(len(cu_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(
        f"CU Metrics Density Distribution: Normal vs Network Stress ({cu_id})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    kde_save_path = os.path.join(output_dir, f"{cu_id}_net_kde_5metrics.png")
    plt.savefig(kde_save_path, dpi=300)
    plt.close(fig)  # Close memory buffer
    print(f"Saved: {kde_save_path}")

    # =========================================================================
    # 2. BOXPLOTS
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (col, name) in enumerate(zip(cu_cols, metric_names)):
        sns.boxplot(
            data=filtered_df,
            x="Stress_Type_Label",
            y=col,
            order=target_labels,
            ax=axes[idx],
            palette=custom_palette,
        )
        axes[idx].set_title(f"Boxplot: {name}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Condition")
        axes[idx].set_ylabel(name)

    # Remove the 6th unused subplot slot cleanly
    for i in range(len(cu_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(
        f"CU Metrics Boxplots: Normal vs Network Stress ({cu_id})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    box_save_path = os.path.join(output_dir, f"{cu_id}_net_boxplots_5metrics.png")
    plt.savefig(box_save_path, dpi=300)
    plt.close(fig)  # Close memory buffer
    print(f"Saved: {box_save_path}")


def main():
    # Ensure destination directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory initialized at: {OUTPUT_DIR}")

    print(f"Loading {TEST_CSV_PATH}...")
    df = pd.read_csv(TEST_CSV_PATH)

    # Process each CU in a single run
    for cu_id in ALL_CU_IDS:
        print(f"\nProcessing {cu_id}...")
        generate_plots_for_cu(df, cu_id, OUTPUT_DIR)

    print(f"\nDone! All plots saved in '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()
