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


def plot_scalability_lines():
    results_dir = "results"
    csv_file = os.path.join(results_dir, "scalability.csv")
    output_file = os.path.join(results_dir, "scalability_lines_efficiency.png")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # --- Data Preprocessing ---
    # Calculate actual completed count to derive per-task metrics
    # Note: In the CSV, 'Completion Rate' is percentage (0-100)
    df["Completed Count"] = df["Num Tasks"] * (df["Completion Rate"] / 100.0)

    # Critical Metric: Avg GPU Time Consumed per Completed Task
    # This normalizes the "Total GPU Time" (which is saturated in stress tests) by the actual work done.
    # This reflects "How many GPU-seconds does it take to finish ONE task on average?"
    df["GPU Time Per Task"] = df.apply(
        lambda row: (
            row["Total GPU Time"] / row["Completed Count"]
            if row["Completed Count"] > 0
            else np.nan
        ),
        axis=1,
    )

    # Filter out Min GPU Time if it makes the plot unreadable (optional, but keeping it for completeness is usually better)
    # df = df[df['Scheduler'] != 'min-gpu-time']

    # --- Metrics Configuration ---
    metrics_config = [
        # (Metric Column, Y-Label, Title, Invert Axis?, Log Scale?)
        (
            "GPU Time Per Task",
            "GPU-Seconds / Task",
            "Resource Efficiency (Lower Cost is Better)",
            False,
            True,
        ),  # The Hero Metric
        ("Avg JCT", "Seconds", "Avg Job Completion Time (Faster)", False, False),
        (
            "Avg Wait",
            "Seconds",
            "Avg Wait Time (Less Queueing)",
            False,
            False,
        ),  # Changed from Throughput to Wait
        (
            "Completion Rate",
            "Rate (%)",
            "System Stability (Higher is Better)",
            False,
            False,
        ),
    ]

    # --- Styling ---
    schedulers = df["Scheduler"].unique()

    # Custom Palette: Highlight Pollux Patient
    colors = {}
    markers = {}
    linestyles = {}
    linewidths = {}
    zorders = {}

    for sched in schedulers:
        if sched == "pollux_patient":
            colors[sched] = "#2ca02c"  # Green
            markers[sched] = "o"
            linestyles[sched] = "-"
            linewidths[sched] = 3.0
            zorders[sched] = 10  # Top
        elif sched == "pollux":
            colors[sched] = "#ff7f0e"  # Orange
            markers[sched] = "s"
            linestyles[sched] = "--"
            linewidths[sched] = 2.0
            zorders[sched] = 5
        elif sched == "rack_aware":
            colors[sched] = "#1f77b4"  # Blue
            markers[sched] = "^"
            linestyles[sched] = "-."
            linewidths[sched] = 2.0
            zorders[sched] = 4
        else:
            # Gray out others
            colors[sched] = "#7f7f7f"
            markers[sched] = "x"
            linestyles[sched] = ":"
            linewidths[sched] = 1.5
            zorders[sched] = 3

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for idx, (metric, ylabel, title, invert, use_log) in enumerate(metrics_config):
        ax = axes[idx]

        for sched in schedulers:
            data = df[df["Scheduler"] == sched].sort_values("Num Tasks")
            if data.empty:
                continue

            # Filter NaNs
            valid_data = data.dropna(subset=[metric])

            ax.plot(
                valid_data["Num Tasks"],
                valid_data[metric],
                marker=markers.get(sched, "o"),
                color=colors.get(sched, "gray"),
                linestyle=linestyles.get(sched, "-"),
                linewidth=linewidths.get(sched, 1.5),
                label=sched if idx == 0 else "",  # Legend only on first temporarily
                markersize=8,
                zorder=zorders.get(sched, 1),
                alpha=0.9,
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Task Load (Number of Tasks)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

        if use_log:
            ax.set_yscale("log")
            # Add annotation for log scale
            # ax.text(0.05, 0.95, 'Log Scale', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

        if invert:
            ax.invert_yaxis()

    # --- Global Legend ---
    # Create custom legend handles to ensure order
    # Order: Pollux Patient, Pollux, Rack Aware, others
    ordered_scheds = [
        "pollux_patient",
        "pollux",
        "rack_aware",
        "min-gpu-time",
        "first-fit",
    ]
    handles = []
    labels = []

    # Dummy plot to get handles is risky, let's manually build from the last ax plot loop
    # actually, axes[0] has labels for everyone if we iterated all scheds.
    # Let's rebuild handles based on the dicts
    from matplotlib.lines import Line2D

    for sched in ordered_scheds:
        if sched in schedulers:
            h = Line2D(
                [0],
                [0],
                color=colors[sched],
                marker=markers[sched],
                linestyle=linestyles[sched],
                linewidth=linewidths[sched],
                markersize=8,
            )
            handles.append(h)

            # Pretty print label
            lbl = sched
            if sched == "pollux_patient":
                lbl = "Pollux Patient (Ours)"
            elif sched == "min-gpu-time":
                lbl = "Min GPU Time"
            elif sched == "first-fit":
                lbl = "First Fit"
            elif sched == "rack_aware":
                lbl = "Rack Aware"
            elif sched == "pollux":
                lbl = "Pollux"
            labels.append(lbl)

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
    print(f"Efficiency line charts saved to {output_file}")


if __name__ == "__main__":
    plot_scalability_lines()
