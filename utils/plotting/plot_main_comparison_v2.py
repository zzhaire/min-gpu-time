import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.plotting.colors import get_scheduler_color, get_scheduler_display_name

# Set style for publication quality
try:
    plt.style.use("seaborn-v0_8-paper")
except:
    plt.style.use("ggplot")


def plot_main_comparison():
    results_dir = "results"
    csv_file = os.path.join(results_dir, "comparison-main.csv")
    fallback_csv = os.path.join(results_dir, "comparison.csv")
    scalability_csv = os.path.join(results_dir, "scalability_completion.csv")
    output_file = os.path.join(results_dir, "main_comparison_metrics.png")

    df = None
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    elif os.path.exists(scalability_csv):
        # Prefer Experiment 1 results for the main comparison summary.
        # Default to the high-load setting (N=1000).
        sc = pd.read_csv(scalability_csv)
        sc = sc[sc["Num Tasks"] == 1000].copy()
        if sc.empty:
            print(f"Error: no rows with Num Tasks == 1000 in {scalability_csv}")
            return

        # Normalize scheduler naming to match the rest of the plotting scripts.
        sc["Scheduler"] = sc["Scheduler"].astype(str).str.replace("-", "_", regex=False)

        df = pd.DataFrame(
            {
                "Scheduler": sc["Scheduler"],
                "Total GPU Time (s)": sc["Total GPU Time"],
                "Avg JCT (s)": sc["Avg JCT"],
                "Avg Wait (s)": sc["Avg Wait"],
                "Cost Per Task (GPU-s)": sc["Total GPU Time"] / sc["Num Tasks"],
            }
        )
        n_for_title = 1000
    elif os.path.exists(fallback_csv):
        df = pd.read_csv(fallback_csv)
        n_for_title = None
    else:
        print(f"Error: {csv_file} not found.")
        return

    # Filter schedulers if needed, or keep all
    # df = df[df['Scheduler'].isin(['pollux_patient', 'pollux', 'rack_aware', 'min_gpu_time'])]

    # Sort by Scheduler for consistent ordering, or by GPU Time to show trend
    # Let's sort to put Ours first or last.
    # Sorting by Total GPU Time makes sense to show the "ranking".
    df = df.sort_values("Total GPU Time (s)", ascending=True)

    # Clean up names - 使用全局配置
    df["Display Name"] = df["Scheduler"].apply(get_scheduler_display_name)

    # Metrics configuration
    # (Metric Column, Title, Y-Label, Color Palette Base)
    metrics_config = [
        (
            "Total GPU Time (s)",
            "Total GPU Time (Cost)",
            "GPU-Seconds (Lower is Better)",
            "Greens",
        ),
        (
            "Avg JCT (s)",
            "Avg Job Completion Time",
            "Seconds (Lower is Better)",
            "Blues",
        ),
        ("Avg Wait (s)", "Avg Wait Time", "Seconds (Lower is Better)", "Reds"),
        (
            "Cost Per Task (GPU-s)",
            "Cost Per Task",
            "GPU-Seconds / Task (Lower is Better)",
            "Purples",
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    for idx, (col, title, ylabel, cmap_name) in enumerate(metrics_config):
        ax = axes[idx]

        # 使用全局颜色配置
        bar_colors = [get_scheduler_color(sched) for sched in df["Scheduler"]]

        bars = ax.bar(
            df["Display Name"],
            df[col],
            color=bar_colors,
            alpha=0.9,
            width=0.7,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add labels on top
        for bar in bars:
            height = bar.get_height()
            val_str = f"{height:,.0f}" if height > 100 else f"{height:.1f}"

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                val_str,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                rotation=0,
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=45)

        # Special handling for GPU Time to highlight the "Min GPU Time" achievement
        if col == "Total GPU Time (s)":
            # Add an arrow or annotation to Ours
            # Assuming df is sorted, Ours is index 0 (lowest)
            ours_row = df[df["Scheduler"] == "pollux_patient"]
            if not ours_row.empty:
                idx_ours = df.index.get_loc(ours_row.index[0])
                val_ours = ours_row[col].values[0]

                # Compare with Rack Aware
                ra_row = df[df["Scheduler"] == "rack_aware"]
                if not ra_row.empty:
                    val_ra = ra_row[col].values[0]
                    reduction = (val_ra - val_ours) / val_ra * 100

                    ax.annotate(
                        f"-{reduction:.1f}% vs Baseline",
                        xy=(
                            bars[idx_ours].get_x() + bars[idx_ours].get_width() / 2,
                            val_ours,
                        ),
                        xytext=(20, 30),
                        textcoords="offset points",
                        arrowprops=dict(
                            arrowstyle="->",
                            color="black",
                            connectionstyle="arc3,rad=.2",
                        ),
                        fontsize=10,
                        color="#F47F72",
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            fc="white",
                            ec="#F47F72",
                            alpha=0.9,
                        ),
                    )

    title_suffix = f" (N={int(n_for_title)})" if n_for_title is not None else ""
    plt.suptitle(
        f"Main Experiment Results: Efficiency & Performance{title_suffix}",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Main comparison plot saved to {output_file}")


if __name__ == "__main__":
    plot_main_comparison()
