import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# Set style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["lines.linewidth"] = 2.5
plt.rcParams["lines.markersize"] = 9


def plot_scalability_completion(csv_path, output_path):
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter schedulers if needed (optional, keeping all for now)
    schedulers = df["Scheduler"].unique()

    # Define colors
    palette = {
        "pollux-patient": "#D62728",  # Red
        "pollux": "#FF7F0E",  # Orange
        "min-gpu-time": "#2CA02C",  # Green
        "rack-aware": "#1F77B4",  # Blue
        "first-fit": "#9467BD",  # Purple
        "best-fit": "#8C564B",  # Brown
    }

    # Markers
    markers = {
        "pollux-patient": "o",
        "pollux": "s",
        "min-gpu-time": "^",
        "rack-aware": "D",
        "first-fit": "v",
        "best-fit": "P",
    }

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Metrics to plot
    metrics = [
        ("Total GPU Time", "Total GPU Time (seconds)", "Total GPU Cost"),
        ("Avg JCT", "Average JCT (seconds)", "Average Job Completion Time"),
        ("Avg Wait", "Average Wait Time (seconds)", "Average Wait Time"),
    ]

    for i, (col, ylabel, title) in enumerate(metrics):
        ax = axes[i]

        sns.lineplot(
            data=df,
            x="Num Tasks",
            y=col,
            hue="Scheduler",
            style="Scheduler",
            palette=palette,
            markers=markers,
            dashes=False,
            ax=ax,
            legend=(i == 2),  # Only show legend on last plot
        )

        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_xlabel("Number of Tasks", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Format y-axis for Total GPU Time (e.g., 1M, 2M)
        if col == "Total GPU Time":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{x/1e6:.1f}M")
            )

        # Ensure all x-ticks are shown
        ax.set_xticks(df["Num Tasks"].unique())

    # Adjust legend
    if len(axes) > 0:
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend(
            handles=handles,
            labels=labels,
            title="Scheduler",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=True,
        )

    plt.tight_layout()

    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("Done!")


if __name__ == "__main__":
    csv_file = "results/scalability_completion.csv"
    output_file = "results/scalability_completion.png"

    if os.path.exists(csv_file):
        plot_scalability_completion(csv_file, output_file)
    else:
        print(f"Error: {csv_file} not found.")
