import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.plotting.colors import get_scheduler_color, get_scheduler_display_name

# Set style
try:
    plt.style.use("seaborn-v0_8-paper")
except:
    plt.style.use("ggplot")


def plot_main_comparison():
    results_dir = "results"
    csv_file = os.path.join(results_dir, "comparison-main.csv")
    fallback_csv = os.path.join(results_dir, "comparison.csv")
    scalability_csv = os.path.join(results_dir, "scalability_completion.csv")
    output_file = os.path.join(results_dir, "main_gpu_time_comparison.png")

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
                "Makespan (s)": sc["Makespan"],
                "Cost Per Task (GPU-s)": sc["Total GPU Time"] / sc["Num Tasks"],
            }
        )
    elif os.path.exists(fallback_csv):
        df = pd.read_csv(fallback_csv)
    else:
        print(f"Error: {csv_file} not found.")
        return

    # Sort to make the comparison clear (e.g., descending order of GPU Time)
    # But we want 'pollux_patient' to be distinct, maybe first or last.
    # Let's sort by Total GPU Time descending so the smallest bar (Ours) is at the end or prominent.
    df = df.sort_values("Total GPU Time (s)", ascending=False)

    # Clean up names for display - 使用全局配置
    df["Display Name"] = df["Scheduler"].apply(get_scheduler_display_name)

    # Calculate savings relative to Rack Aware (Baseline)
    baseline_val = df[df["Scheduler"] == "rack_aware"]["Total GPU Time (s)"].values[0]
    ours_val = df[df["Scheduler"] == "pollux_patient"]["Total GPU Time (s)"].values[0]
    savings_pct = ((baseline_val - ours_val) / baseline_val) * 100

    plt.figure(figsize=(10, 6))

    # 使用全局颜色配置
    colors = [get_scheduler_color(sched) for sched in df["Scheduler"]]

    bars = plt.bar(
        df["Display Name"], df["Total GPU Time (s)"], color=colors, alpha=0.8, width=0.6
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Highlight the savings
    # Find the 'Pollux Patient' bar index
    # We can add a text box or arrow
    plt.title(
        "Total GPU Time Consumption (Lower is Better)", fontsize=14, fontweight="bold"
    )
    plt.ylabel("Total GPU Seconds", fontsize=12)
    plt.xlabel("Scheduler Strategy", fontsize=12)

    # Add a text box about the reduction
    text_str = f"Ours reduces GPU Time by {savings_pct:.1f}%\nvs Rack Aware Baseline"
    plt.text(
        0.95,
        0.95,
        text_str,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#F47F72"),
    )

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Comparison plot saved to {output_file}")


if __name__ == "__main__":
    plot_main_comparison()
