import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("ggplot")


def main():
    results_dir = "results"
    csv_file = os.path.join(results_dir, "pareto_frontier.csv")
    output_file = os.path.join(results_dir, "pareto_frontier.png")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # Calculate Cost ($)
    # Assumption: $3 per GPU-Hour
    gpu_price_per_hour = 3.0
    df["Cost"] = (df["Total GPU Time"] / 3600.0) * gpu_price_per_hour

    plt.figure(figsize=(10, 8))

    # Plot Baselines
    baselines = df[df["Type"] == "Baseline"]
    # Define markers/colors for baselines
    baseline_markers = {
        "rack-aware": ("s", "blue"),
        "min-gpu-time": ("^", "purple"),
        "first-fit": ("v", "gray"),
        "pollux": ("D", "orange"),
    }

    for _, row in baselines.iterrows():
        sched = row["Scheduler"]
        marker, color = baseline_markers.get(sched, ("o", "black"))
        plt.scatter(
            row["Cost"],
            row["Avg JCT"],
            label=f"{sched} (Baseline)",
            marker=marker,
            color=color,
            s=100,
            zorder=10,
            edgecolors="black",
        )

        # Annotate
        plt.text(
            row["Cost"],
            row["Avg JCT"] + 150,
            sched,
            fontsize=9,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot Ours (Pollux Patient)
    ours = df[df["Type"] == "Ours"].sort_values("Cost")
    plt.plot(
        ours["Cost"],
        ours["Avg JCT"],
        color="green",
        linestyle="--",
        alpha=0.5,
        zorder=5,
        label="Pareto Frontier (Ours)",
    )

    plt.scatter(
        ours["Cost"],
        ours["Avg JCT"],
        label="Pollux Patient (Ours)",
        marker="o",
        color="green",
        s=120,
        zorder=10,
        edgecolors="black",
    )

    # Annotate Our Points
    for _, row in ours.iterrows():
        param = row["Param"]
        # Simplified label: P=1.5
        label = param.replace("P=", "Patience=")
        plt.text(
            row["Cost"],
            row["Avg JCT"] - 250,
            label,
            fontsize=9,
            ha="center",
            va="top",
            color="green",
            fontweight="bold",
        )

    # Formatting
    plt.xlabel("Total Cloud Cost ($) [@ $3/GPU-hr]", fontsize=12)
    plt.ylabel("Average Job Completion Time (s) [Lower is Better]", fontsize=12)
    plt.title('Cost-Speed Pareto Frontier: The "Sweet Spot"', fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", frameon=True, framealpha=0.9)

    # Invert X axis? No, lower cost is better (left). Lower JCT is better (bottom).
    # Ideal point is Bottom-Left.

    # Add an arrow pointing to the "Sweet Spot" (P=1.5 usually)
    # Find point with min (Cost * JCT) or just the "knee"
    # P=1.5 in our data: Cost ~464, JCT ~5219.
    # Wait, in the run output P=1.5 was Cost 557k, JCT 5219. P=1.0 was Cost 471k, JCT 7823.
    # The knee is likely P=1.5.

    # Optional: Highlight the optimal region
    # plt.axvspan(min(ours['Cost']), max(ours['Cost']), color='green', alpha=0.05)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Pareto plot saved to {output_file}")


if __name__ == "__main__":
    main()
