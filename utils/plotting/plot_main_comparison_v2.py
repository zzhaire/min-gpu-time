import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style for publication quality
try:
    plt.style.use("seaborn-v0_8-paper")
except:
    plt.style.use("ggplot")


def plot_main_comparison():
    results_dir = "results"
    csv_file = os.path.join(results_dir, "comparison-main.csv")
    output_file = os.path.join(results_dir, "main_comparison_metrics.png")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # Filter schedulers if needed, or keep all
    # df = df[df['Scheduler'].isin(['pollux_patient', 'pollux', 'rack_aware', 'min_gpu_time'])]

    # Sort by Scheduler for consistent ordering, or by GPU Time to show trend
    # Let's sort to put Ours first or last.
    # Sorting by Total GPU Time makes sense to show the "ranking".
    df = df.sort_values("Total GPU Time (s)", ascending=True)

    # Clean up names
    name_map = {
        "pollux_patient": "Pollux Patient\n(Ours)",
        "pollux": "Pollux",
        "rack_aware": "Rack Aware",
        "min_gpu_time": "Min GPU Time",
        "first_fit": "First Fit",
        "best_fit": "Best Fit",
    }
    df["Display Name"] = df["Scheduler"].map(name_map)

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
        ("Completed", "Completion Rate", "Tasks Completed (Max 100)", "Greys"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for idx, (col, title, ylabel, cmap_name) in enumerate(metrics_config):
        ax = axes[idx]

        # Prepare colors
        # We want to highlight 'Pollux Patient (Ours)'
        # Let's use a specific color for Ours and a neutral one for others,
        # OR use the cmap but force Ours to be distinct.

        bar_colors = []
        for sched in df["Scheduler"]:
            if sched == "pollux_patient":
                bar_colors.append("#2ca02c")  # Strong Green
            elif sched == "pollux":
                bar_colors.append("#ff7f0e")  # Orange
            elif sched == "rack_aware":
                bar_colors.append("#1f77b4")  # Blue
            else:
                bar_colors.append("#bdc3c7")  # Gray

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
            if col == "Completed":
                val_str = f"{int(height)}"

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
                        color="#2ca02c",
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            fc="white",
                            ec="#2ca02c",
                            alpha=0.9,
                        ),
                    )

    plt.suptitle(
        "Main Experiment Results: Efficiency & Performance (N=100)",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Main comparison plot saved to {output_file}")


if __name__ == "__main__":
    plot_main_comparison()
