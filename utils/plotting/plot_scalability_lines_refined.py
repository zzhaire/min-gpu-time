import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.ticker import FuncFormatter

# Set professional style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def plot_scalability_lines_refined():
    results_dir = "results"
    csv_file = os.path.join(results_dir, "scalability.csv")
    output_file = os.path.join(results_dir, "scalability_lines_refined.png")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # --- Data Preprocessing ---
    # Calculate actual completed count
    df["Completed Count"] = df["Num Tasks"] * (df["Completion Rate"] / 100.0)

    # Metric 1: Avg GPU Seconds per Job (Cost Efficiency)
    # This is the most important metric to show "Min GPU Time"
    df["GPU Seconds Per Job"] = df.apply(
        lambda row: (
            row["Total GPU Time"] / row["Completed Count"]
            if row["Completed Count"] > 0
            else np.nan
        ),
        axis=1,
    )

    # --- Metrics Configuration ---
    # We will plot 4 key metrics
    metrics_config = [
        {
            "col": "GPU Seconds Per Job",
            "title": "Resource Cost (GPU-Seconds per Job)",
            "ylabel": "GPU-Sec / Job",
            "xlabel": "System Load (Number of Tasks)",
            "lower_better": True,
        },
        {
            "col": "Avg JCT",
            "title": "Job Completion Time",
            "ylabel": "Seconds",
            "xlabel": "System Load (Number of Tasks)",
            "lower_better": True,
        },
        {
            "col": "Avg Wait",
            "title": "Job Wait Time (Queueing)",
            "ylabel": "Seconds",
            "xlabel": "System Load (Number of Tasks)",
            "lower_better": True,
        },
        {
            "col": "Completion Rate",
            "title": "System Throughput (Completion Rate)",
            "ylabel": "Percentage (%)",
            "xlabel": "System Load (Number of Tasks)",
            "lower_better": False,
        },
    ]

    # --- Scheduler Styling (High Contrast) ---
    schedulers = [
        "pollux_patient",  # Ours
        "pollux",  # SOTA
        "rack_aware",  # Baseline 1
        "min_gpu_time",  # Baseline 2
        "first_fit",  # Baseline 3
    ]

    style_map = {
        "pollux_patient": {
            "color": "#00A300",
            "marker": "o",
            "ls": "-",
            "lw": 3.5,
            "label": "Eco-Pollux (Ours)",
            "zorder": 10,
        },  # Strong Green
        "pollux": {
            "color": "#FF8C00",
            "marker": "s",
            "ls": "--",
            "lw": 2.0,
            "label": "Pollux",
            "zorder": 5,
        },  # Orange
        "rack_aware": {
            "color": "#1E90FF",
            "marker": "^",
            "ls": "-.",
            "lw": 2.0,
            "label": "Rack Aware",
            "zorder": 4,
        },  # Blue
        "min_gpu_time": {
            "color": "#9370DB",
            "marker": "D",
            "ls": ":",
            "lw": 2.0,
            "label": "Min GPU Time",
            "zorder": 3,
        },  # Purple
        "first_fit": {
            "color": "#808080",
            "marker": "x",
            "ls": ":",
            "lw": 1.5,
            "label": "First Fit",
            "zorder": 2,
        },  # Grey
    }

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    for idx, config in enumerate(metrics_config):
        ax = axes[idx]
        col = config["col"]

        for sched in schedulers:
            # Get data for this scheduler
            data = df[df["Scheduler"] == sched].sort_values("Num Tasks")
            if data.empty:
                continue

            # Filter NaNs
            valid_data = data.dropna(subset=[col])

            # Get style
            s = style_map.get(
                sched, {"color": "black", "marker": ".", "ls": "-", "lw": 1}
            )

            ax.plot(
                valid_data["Num Tasks"],
                valid_data[col],
                color=s["color"],
                marker=s["marker"],
                linestyle=s["ls"],
                linewidth=s["lw"],
                label=s["label"],
                markersize=8,
                zorder=s["zorder"],
                alpha=0.9,
            )

        # Aesthetics
        ax.set_title(config["title"], fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel(config["xlabel"], fontsize=12)
        ax.set_ylabel(config["ylabel"], fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4, color="gray")

        # Format Y-axis with commas for large numbers
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ",")))

        # Highlight "Lower is Better" or "Higher is Better"
        if config["lower_better"]:
            # Add arrow pointing down
            # ax.annotate('Lower is Better', xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10, color='green', fontweight='bold')
            pass

    # --- Global Legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=len(schedulers),
        fontsize=13,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Refined scalability plot saved to {output_file}")


if __name__ == "__main__":
    plot_scalability_lines_refined()
