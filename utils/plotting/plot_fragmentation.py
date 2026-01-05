import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os

# Set style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("ggplot")


def get_active_gpu_timeline(csv_file):
    """
    Reconstructs the number of active GPUs over time from task logs.
    Active GPU = A GPU running at least one task.
    """
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found.")
        return [], []

    df = pd.read_csv(csv_file)
    events = []

    for _, row in df.iterrows():
        try:
            start = float(row["start_time"])
            end = float(row["completion_time"])

            # Parse allocated_gpus list string
            gpus_str = str(row["allocated_gpus"])
            if pd.isna(gpus_str) or gpus_str == "" or gpus_str == "nan":
                gpus = []
            else:
                # Handle comma-separated format from metrics.py
                gpus = [g.strip() for g in gpus_str.split(",") if g.strip()]

            if pd.isna(start) or pd.isna(end) or not gpus:
                continue

            # +1 for task start, -1 for task end
            events.append((start, 1, gpus))
            events.append((end, -1, gpus))
        except Exception as e:
            continue

    # Sort by time
    events.sort(key=lambda x: x[0])

    times = [0]
    active_counts = [0]

    # Track how many tasks are running on each GPU
    gpu_ref_counts = {}

    # Process events
    for t, type, gpus in events:
        if t > times[-1]:
            # Record the state just before this new time point (step function)
            times.append(t)
            # Calculate active GPUs (count > 0)
            active_count = sum(1 for c in gpu_ref_counts.values() if c > 0)
            active_counts.append(active_count)

        # Update ref counts
        for gpu in gpus:
            gpu_ref_counts[gpu] = gpu_ref_counts.get(gpu, 0) + type

    # Final cleanup (extend to end)
    if events:
        final_time = events[-1][0]
        times.append(final_time)
        active_counts.append(0)

    print(
        f"[{csv_file}] Processed {len(events)} events, Time range: 0 -> {times[-1]:.1f}s"
    )

    return times, active_counts


def plot_fragmentation():
    results_dir = "results"
    output_file = os.path.join(results_dir, "cluster_fragmentation.png")

    schedulers = {
        "pollux_patient": "Pollux Patient (Ours)",
        "pollux": "Pollux",
        "rack_aware": "Rack Aware",
        "min_gpu_time": "Min GPU Time",
        "first_fit": "First Fit",
        "best_fit": "Best Fit",
    }

    total_gpus = 64  # Hardcoded from config (8 racks * 8 gpus)

    fig, axes = plt.subplots(
        len(schedulers), 1, figsize=(12, 12), sharex=True, sharey=True
    )
    if len(schedulers) == 1:
        axes = [axes]

    # Use a colormap for distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(schedulers)))

    for idx, (key, label) in enumerate(schedulers.items()):
        ax = axes[idx]
        csv_file = os.path.join(results_dir, f"tasks_{key}.csv")

        times, active_counts = get_active_gpu_timeline(csv_file)

        color = colors[idx]

        if not times:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            continue

        # Convert to numpy for easier manipulation
        times = np.array(times)
        active_counts = np.array(active_counts)

        # Plot "Active" (Goodput/Used) Area
        ax.fill_between(
            times,
            0,
            active_counts,
            color=color,
            alpha=0.6,
            label="Active GPUs (Occupied)",
        )

        # Plot "Fragmentation" (Idle) Area
        ax.fill_between(
            times, active_counts, total_gpus, color="lightgray", alpha=0.2, hatch="//"
        )

        # Formatting
        ax.set_ylim(0, total_gpus)
        ax.set_ylabel("Active GPUs", fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Add stats text
        avg_active = np.mean(active_counts) if len(active_counts) > 0 else 0
        occupancy = (avg_active / total_gpus) * 100
        # Renamed to "Avg Occupancy" to imply resource footprint
        ax.text(
            0.02,
            0.85,
            f"Avg Occupancy: {occupancy:.1f}%",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        # Add Max Active text
        max_active = np.max(active_counts) if len(active_counts) > 0 else 0
        ax.text(
            0.25,
            0.85,
            f"Peak: {max_active} GPUs",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    axes[-1].set_xlabel("Time (s)", fontsize=12)

    # Legend
    # axes[0].legend(loc='upper right')

    plt.suptitle(
        "Cluster Resource Footprint: Lower is Better (for same workload)", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Fragmentation plot saved to {output_file}")


if __name__ == "__main__":
    plot_fragmentation()
