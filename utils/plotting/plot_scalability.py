import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

try:
    import seaborn as sns

    _HAVE_SEABORN = True
except Exception:
    sns = None
    _HAVE_SEABORN = False

# Set style for publication quality
try:
    plt.style.use("seaborn-v0_8-paper")
except:
    plt.style.use("ggplot")


def plot_scalability():
    results_dir = "results"
    csv_file = os.path.join(results_dir, "scalability.csv")
    output_file = os.path.join(results_dir, "scalability_analysis.png")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # Rename scheduler for paper
    df["Scheduler"] = df["Scheduler"].replace({"pollux-patient": "Eco-Pollux (Ours)"})

    # --- Data Preprocessing & New Metrics ---
    # Calculate actual completed count to derive per-task metrics
    df["Completed Count"] = df["Num Tasks"] * (df["Completion Rate"] / 100.0)

    # Cost Per Task: How much GPU time did it cost to finish one task?
    # This exposes the "bloat" of RackAware.
    df["Cost Per Task"] = df.apply(
        lambda row: (
            row["Total GPU Time"] / row["Completed Count"]
            if row["Completed Count"] > 0
            else np.nan
        ),
        axis=1,
    )

    # Define metrics layout (2 Rows x 3 Cols)
    # Row 1: System Performance
    # Row 2: User Experience & Cost
    metrics_config = [
        # (Metric Column, Y-Label, Title, Invert Axis?)
        (
            "Completion Rate",
            "Completion Rate (%)",
            "Completion Rate (Higher is Better)",
            False,
        ),
        (
            "Avg Wait",
            "Avg Wait Time (s)",
            "Wait Time (Lower is Better)",
            False,
        ),
        (
            "Total GPU Time",
            "Total GPU-Seconds",
            "Cluster Saturation (Resource Usage)",
            False,
        ),
        (
            "Avg Slowdown",
            "Slowdown (Ratio)",
            "User Experience (Lower is Better)",
            True,
        ),
        ("Avg JCT", "Seconds", "Avg Job Completion Time (Lower is Better)", False),
        (
            "Cost Per Task",
            "GPU-Sec / Task",
            "Cost Efficiency (Lower is Better)",
            False,
        ),
    ]

    # Scheduler styling
    schedulers = df["Scheduler"].unique()
    # Use a high-contrast palette
    if _HAVE_SEABORN:
        colors = sns.color_palette("deep", len(schedulers))
    else:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(schedulers))]
    sched_map = {sched: colors[i] for i, sched in enumerate(schedulers)}

    markers = ["o", "s", "^", "D", "v", "P"]
    marker_map = {
        sched: markers[i % len(markers)] for i, sched in enumerate(schedulers)
    }

    linestyles = ["-", "--", "-.", ":", "-", "--"]
    style_map = {
        sched: linestyles[i % len(linestyles)] for i, sched in enumerate(schedulers)
    }

    # Create Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric, unit, title, _) in enumerate(metrics_config):
        ax = axes[idx]

        for sched in schedulers:
            data = df[df["Scheduler"] == sched].sort_values("Num Tasks")
            if data.empty:
                continue

            # Filter out NaNs for plotting
            valid_data = data.dropna(subset=[metric])

            ax.plot(
                valid_data["Num Tasks"],
                valid_data[metric],
                marker=marker_map[sched],
                color=sched_map[sched],
                linestyle=style_map[sched],
                label=sched,
                linewidth=2.5,
                markersize=9,
                alpha=0.85,
            )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Load (Number of Tasks)", fontsize=12)
        ax.set_ylabel(unit, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4, color="gray")
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Log scale for Cost Per Task if differences are huge
        if metric == "Cost Per Task" or metric == "Avg Slowdown":
            # Check range
            ymin, ymax = ax.get_ylim()
            if ymax / (ymin + 1e-9) > 10:  # If dynamic range > 10x
                ax.set_yscale("log")
                ax.set_ylabel(unit + " (Log Scale)", fontsize=12)

    # Global Legend
    # Place it at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(schedulers),
        fontsize=14,
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )

    plt.suptitle(
        "Scalability & Stress Test: Pollux Patient (Ours) vs Baselines",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Improved scalability plot saved to {output_file}")


if __name__ == "__main__":
    plot_scalability()
