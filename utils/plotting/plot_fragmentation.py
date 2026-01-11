import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os
from utils.plotting.colors import get_scheduler_color, get_scheduler_display_name

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

    def _parse_allocated_gpus(raw_val):
        s = str(raw_val)
        if pd.isna(raw_val) or s == "" or s == "nan":
            return []
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        parts = [p.strip() for p in s.split(",") if p.strip()]
        cleaned = []
        for p in parts:
            p2 = p.strip().strip("[").strip("]").strip('"').strip("'")
            if p2:
                cleaned.append(p2)
        return cleaned

    for _, row in df.iterrows():
        try:
            start = float(row["start_time"])
            end = float(row["completion_time"])

            gpus = _parse_allocated_gpus(row.get("allocated_gpus", ""))

            if pd.isna(start) or pd.isna(end) or not gpus:
                continue

            # +1 for task start, -1 for task end
            events.append((start, 1, gpus))
            events.append((end, -1, gpus))
        except Exception as e:
            continue

    # Sort by time
    events.sort(key=lambda x: x[0])

    times = [0.0]
    active_counts = [0]

    gpu_ref_counts = {}

    i = 0
    while i < len(events):
        t = float(events[i][0])
        while i < len(events) and float(events[i][0]) == t:
            _, typ, gpus = events[i]
            for gpu in gpus:
                gpu_ref_counts[gpu] = gpu_ref_counts.get(gpu, 0) + typ
                if gpu_ref_counts[gpu] == 0:
                    del gpu_ref_counts[gpu]
            i += 1

        active_count = len(gpu_ref_counts)
        if t >= times[-1]:
            times.append(t)
            active_counts.append(active_count)

    print(
        f"[{csv_file}] Processed {len(events)} events, Time range: 0 -> {times[-1]:.1f}s"
    )

    return times, active_counts


def plot_fragmentation():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    results_dir = os.path.join(project_root, "results")
    output_png = os.path.join(results_dir, "cluster_fragmentation.png")
    output_pdf = os.path.join(results_dir, "cluster_fragmentation.pdf")

    # 统一使用全局配置的 key -> display name
    schedulers = [
        "pollux_patient",
        "pollux",
        "rack_aware",
        "min_gpu_time",
        "first_fit",
    ]

    total_gpus = 64  # Hardcoded from config (8 racks * 8 gpus)

    fig, axes = plt.subplots(len(schedulers), 1, figsize=(10, 8), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0)  # 去掉子图间距
    if len(schedulers) == 1:
        axes = [axes]

    # Use global colors from colors.py
    scheduler_colors = {key: get_scheduler_color(key) for key in schedulers}

    for idx, key in enumerate(schedulers):
        ax = axes[idx]
        csv_file = os.path.join(results_dir, f"tasks_{key}.csv")
        sched_name = get_scheduler_display_name(key)

        times, active_counts = get_active_gpu_timeline(csv_file)

        color = scheduler_colors[key]

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
            step="post",
        )

        # Plot "Fragmentation" (Idle) Area
        ax.fill_between(
            times,
            active_counts,
            total_gpus,
            color="lightgray",
            alpha=0.2,
            hatch="//",
            step="post",
        )

        # Formatting
        ax.set_ylim(0, total_gpus)
        ax.set_ylabel("")  # 去掉y轴标签
        ax.set_title("")  # 去掉顶部标题
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=16, width=1.5)

        # y轴刻度加粗
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontweight("bold")

        # 调度策略名字放到图内左边
        ax.text(
            0.02,
            0.5,
            sched_name,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )

        # Add stats text
        avg_active = 0.0
        if len(times) >= 2:
            dt = np.diff(times)
            if np.sum(dt) > 0:
                avg_active = float(np.sum(active_counts[:-1] * dt) / np.sum(dt))
        occupancy = (avg_active / total_gpus) * 100
        # Renamed to "Avg Occupancy" to imply resource footprint
        max_active = np.max(active_counts) if len(active_counts) > 0 else 0
        ax.text(
            0.98,
            0.85,
            f"Avg Occupancy: {occupancy:.1f}%  |  Peak: {max_active} GPUs",
            transform=ax.transAxes,
            ha="right",
            fontsize=15,
            fontweight="bold",
            bbox=dict(
                facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round,pad=0.3"
            ),
        )

    axes[-1].set_xlabel("Time (s)", fontsize=18, fontweight="bold")

    # x轴刻度格式化为 k
    def format_k(x, pos):
        if x >= 1000:
            return f"{int(x/1000)}k"
        return f"{int(x)}"

    from matplotlib.ticker import FuncFormatter

    axes[-1].xaxis.set_major_formatter(FuncFormatter(format_k))
    # x轴刻度加粗加大
    for label in axes[-1].get_xticklabels():
        label.set_fontsize(16)
        label.set_fontweight("bold")

    # y轴统一标记（放在中间图的左边）
    fig.text(
        0.02,
        0.5,
        "Active GPUs",
        va="center",
        rotation="vertical",
        fontsize=18,
        fontweight="bold",
    )

    # Legend
    # axes[0].legend(loc='upper right')

    plt.suptitle(
        "Cluster Resource Footprint: Lower is Better (for same workload)",
        fontsize=20,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0)  # tight_layout后再设置间距为0
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    print(f"Fragmentation plot saved to {output_png}")
    print(f"Fragmentation plot saved to {output_pdf}")


if __name__ == "__main__":
    plot_fragmentation()
