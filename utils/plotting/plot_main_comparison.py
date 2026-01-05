import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
try:
    plt.style.use("seaborn-v0_8-paper")
except:
    plt.style.use("ggplot")


def plot_main_comparison():
    results_dir = "results"
    csv_file = os.path.join(results_dir, "comparison-main.csv")
    output_file = os.path.join(results_dir, "main_gpu_time_comparison.png")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # Sort to make the comparison clear (e.g., descending order of GPU Time)
    # But we want 'pollux_patient' to be distinct, maybe first or last.
    # Let's sort by Total GPU Time descending so the smallest bar (Ours) is at the end or prominent.
    df = df.sort_values("Total GPU Time (s)", ascending=False)

    # Clean up names for display
    name_map = {
        "pollux_patient": "Pollux Patient (Ours)",
        "pollux": "Pollux",
        "rack_aware": "Rack Aware",
        "min_gpu_time": "Min GPU Time (Scheduler)",
        "first_fit": "First Fit",
        "best_fit": "Best Fit",
    }
    df["Display Name"] = df["Scheduler"].map(name_map)

    # Calculate savings relative to Rack Aware (Baseline)
    baseline_val = df[df["Scheduler"] == "rack_aware"]["Total GPU Time (s)"].values[0]
    ours_val = df[df["Scheduler"] == "pollux_patient"]["Total GPU Time (s)"].values[0]
    savings_pct = ((baseline_val - ours_val) / baseline_val) * 100

    plt.figure(figsize=(10, 6))

    # Create color list: Highlight ours with Green, others Gray/Blue
    colors = []
    for sched in df["Scheduler"]:
        if sched == "pollux_patient":
            colors.append("#2ca02c")  # Green
        elif sched == "pollux":
            colors.append("#ff7f0e")  # Orange
        else:
            colors.append("#1f77b4")  # Blue (or gray '#7f7f7f' for less emphasis)

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
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#2ca02c"),
    )

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Comparison plot saved to {output_file}")


if __name__ == "__main__":
    plot_main_comparison()
