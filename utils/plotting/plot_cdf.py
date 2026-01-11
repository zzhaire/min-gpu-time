import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from utils.plotting.colors import get_scheduler_color, get_scheduler_display_name

# Set style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("ggplot")


def get_cdf(data):
    """Calculate CDF for a given dataset."""
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, yvals


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    results_dir = os.path.join(project_root, "results")
    output_png = os.path.join(results_dir, "jct_cdf.png")
    output_pdf = os.path.join(results_dir, "jct_cdf.pdf")

    # Schedulers to plot and their display names - 使用全局配置
    # 直接使用全局颜色和标签配置
    scheduler_keys = [
        "pollux_patient",
        "pollux",
        "rack_aware",
        "min_gpu_time",
        "first_fit",
    ]

    plt.figure(figsize=(8, 5))

    # 使用全局颜色配置
    scheduler_display_names = {k: get_scheduler_display_name(k) for k in scheduler_keys}

    print(
        f"{'Scheduler':<20} {'Avg JCT':<10} {'Median':<10} {'P99 JCT':<10} {'Max JCT':<10}"
    )
    print("-" * 65)

    for idx, scheduler_key in enumerate(scheduler_keys):
        label = get_scheduler_display_name(scheduler_key)
        file_path = os.path.join(results_dir, f"tasks_{scheduler_key}.csv")

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        try:
            df_task = pd.read_csv(file_path)
            if "jct" not in df_task.columns:
                print(f"Warning: 'jct' column not found in {file_path}. Skipping.")
                continue

            jct_data = df_task["jct"].dropna()

            if len(jct_data) == 0:
                print(f"Warning: No JCT data in {file_path}. Skipping.")
                continue

            # Stats
            avg_jct = np.mean(jct_data)
            median_jct = np.median(jct_data)
            p99_jct = np.percentile(jct_data, 99)
            max_jct = np.max(jct_data)
            print(
                f"{label:<20} {avg_jct:<10.1f} {median_jct:<10.1f} {p99_jct:<10.1f} {max_jct:<10.1f}"
            )

            x, y = get_cdf(jct_data)

            # Highlight PACE with a thicker line
            linewidth = 3.0 if scheduler_key == "pollux_patient" else 1.5
            alpha = 1.0 if scheduler_key == "pollux_patient" else 0.7

            # 使用全局颜色配置
            color = get_scheduler_color(scheduler_key)

            plt.plot(x, y, label=label, linewidth=linewidth, alpha=alpha, color=color)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plt.xscale("log")
    plt.xlabel("Job Completion Time (s) [Log Scale]", fontsize=17, fontweight="bold")
    plt.ylabel("CDF", fontsize=17, fontweight="bold")
    plt.title("JCT Distribution & Fairness Analysis", fontsize=19, fontweight="bold")
    plt.legend(fontsize=15, frameon=True, facecolor="white", edgecolor="gray")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tick_params(axis="both", labelsize=15)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    print(f"\nCDF plot saved to {output_png}")
    print(f"CDF plot saved to {output_pdf}")


if __name__ == "__main__":
    main()
