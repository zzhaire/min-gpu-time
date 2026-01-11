import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import sys

# Set style for publication quality
try:
    plt.style.use("seaborn-v0_8-paper")
except:
    plt.style.use("ggplot")

# Improve PDF export quality (vector text)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def plot_scalability_completion_detailed():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(project_root, "results")
    csv_file = os.path.join(results_dir, "scalability_completion.csv")
    output_file = os.path.join(results_dir, "scalability_completion_detailed.png")
    output_pdf = os.path.join(results_dir, "scalability_completion_detailed.pdf")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Keep only the main experiment points: 50, 100, 200, 300, ..., 1000
    main_points = [50, 100] + list(range(200, 1001, 100))
    df = df[df["Num Tasks"].isin(main_points)].copy()

    # Rename scheduler for paper
    df["Scheduler"] = df["Scheduler"].replace(
        {"pollux-patient": "Eco-Pollux (ours)"}
    )

    # --- Data Preprocessing & New Metrics ---
    # Since this is run-to-completion, Completed Count is Num Tasks
    df["Completed Count"] = df["Num Tasks"]

    # Cost Per Task: Total GPU Time / Num Tasks
    df["Cost Per Task"] = df["Total GPU Time"] / df["Num Tasks"]

    # Print Table for User Verification (Important)
    print("\n--- Detailed Results Table (Check for bugs) ---")
    # Pivot for easier reading of GPU Time
    gpu_time_pivot = df.pivot(
        index="Num Tasks", columns="Scheduler", values="Total GPU Time"
    )
    print("\nTotal GPU Time (Lower is Better):")
    print(gpu_time_pivot)

    print("\nAvg JCT (Lower is Better):")
    jct_pivot = df.pivot(index="Num Tasks", columns="Scheduler", values="Avg JCT")
    print(jct_pivot)

    print("\nCost Per Task (Lower is Better):")
    cost_pivot = df.pivot(
        index="Num Tasks", columns="Scheduler", values="Cost Per Task"
    )
    print(cost_pivot)
    print("-----------------------------------------------")

    # Define metrics layout (2 Rows x 3 Cols)
    metrics_config = [
        # (Metric Column, Y-Label, Title, Invert Axis?)
        (
            "Makespan",
            "Makespan (s)",
            "Makespan (Time to Finish All)",
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
            "Total GPU Cost (Resource Usage)",
            False,
        ),
        (
            "Avg Slowdown",
            "Slowdown (Ratio)",
            "User Experience (Lower is Better)",
            True,  # Log scale candidate
        ),
        ("Avg JCT", "Seconds", "Avg Job Completion Time (Lower is Better)", False),
        (
            "Cost Per Task",
            "GPU-Sec / Task",
            "Cost Efficiency (Lower is Better)",
            False,
        ),
    ]

    # Scheduler styling (consistent with previous plots)
    schedulers = df["Scheduler"].unique()
    # Use a high-contrast palette
    # Custom palette to ensure consistency if possible, or use seaborn deep
    colors = sns.color_palette("deep", len(schedulers))

    # Try to match previous specific colors if possible for consistency
    custom_palette = {
        "Eco-Pollux (ours)": "#D62728",  # Red
        "pollux": "#FF7F0E",  # Orange
        "min-gpu-time": "#2CA02C",  # Green
        "rack-aware": "#1F77B4",  # Blue
        "first-fit": "#9467BD",  # Purple
        "best-fit": "#8C564B",  # Brown
    }

    sched_map = {}
    for i, sched in enumerate(schedulers):
        if sched in custom_palette:
            sched_map[sched] = custom_palette[sched]
        else:
            sched_map[sched] = colors[i % len(colors)]

    markers = ["o", "s", "^", "D", "v", "P"]
    marker_map = {
        sched: markers[i % len(markers)] for i, sched in enumerate(schedulers)
    }

    linestyles = ["-", "--", "-.", ":", "-", "--"]
    style_map = {
        sched: linestyles[i % len(linestyles)] for i, sched in enumerate(schedulers)
    }

    # Create Figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()

    for idx, (metric, unit, title, use_log) in enumerate(metrics_config):
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

        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.set_xlabel("Load (Number of Tasks)", fontsize=14)
        ax.set_ylabel(unit, fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.4, color="gray")
        ax.tick_params(axis="both", which="major", labelsize=12)

        # Force x-ticks to main experiment points
        ax.set_xticks(main_points)

        # Log scale logic
        if use_log:
            # Check range
            ymin, ymax = ax.get_ylim()
            if ymin > 0 and ymax / ymin > 20:  # If dynamic range is large
                ax.set_yscale("log")
                ax.set_ylabel(unit + " (Log Scale)", fontsize=14)

        # Special formatting for large numbers
        if metric in ["Total GPU Time", "Makespan", "Avg JCT", "Cost Per Task"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Global Legend
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

    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"\nPlot saved to {output_file}")
    print(f"Plot saved to {output_pdf}")


if __name__ == "__main__":
    plot_scalability_completion_detailed()
