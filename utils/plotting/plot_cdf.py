import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

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
    results_dir = "results"
    output_file = os.path.join(results_dir, "jct_cdf.png")

    # Schedulers to plot and their display names
    schedulers = {
        "pollux_patient": "Pollux Patient",
        "pollux": "Pollux",
        "rack_aware": "Rack Aware",
        "min_gpu_time": "Min GPU Time",
        "first_fit": "First Fit",
        "best_fit": "Best Fit",
    }

    plt.figure(figsize=(10, 6))

    # Use a colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(schedulers)))

    print(
        f"{'Scheduler':<20} {'Avg JCT':<10} {'Median':<10} {'P99 JCT':<10} {'Max JCT':<10}"
    )
    print("-" * 65)

    for idx, (scheduler_key, label) in enumerate(schedulers.items()):
        file_path = os.path.join(results_dir, f"tasks_{scheduler_key}.csv")

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        try:
            df = pd.read_csv(file_path)
            if "jct" not in df.columns:
                print(f"Warning: 'jct' column not found in {file_path}. Skipping.")
                continue

            jct_data = df["jct"].dropna()

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

            # Highlight Pollux Patient with a thicker line
            linewidth = 3.0 if scheduler_key == "pollux_patient" else 1.5
            alpha = 1.0 if scheduler_key == "pollux_patient" else 0.7

            # Ensure color consistency
            color = colors[idx]

            plt.plot(x, y, label=label, linewidth=linewidth, alpha=alpha, color=color)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plt.xscale("log")
    plt.xlabel("Job Completion Time (s) [Log Scale]", fontsize=12)
    plt.ylabel("CDF (Cumulative Distribution Function)", fontsize=12)
    plt.title("JCT Distribution & Fairness Analysis", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nCDF plot saved to {output_file}")


if __name__ == "__main__":
    main()
