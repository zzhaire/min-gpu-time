import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import argparse
import time
from utils.plotting.colors import get_scheduler_color, get_scheduler_display_name

# ============== 可调参数 (Parameters) ==============
# ---------- 图片尺寸 ----------
FIG_WIDTH = 12  # 图片宽度（英寸），2x3布局建议12（瘦高）
FIG_HEIGHT = 14  # 图片高度（英寸），2x3布局建议14（瘦高）

# ---------- 子图间距 ----------
HSPACE = 0.35  # 上下子图间距
WSPACE = 0.30  # 左右子图间距

# ---------- 折线和数据点 ----------
LINE_WIDTH = 3  # 折线粗细
MARKER_SIZE = 7 # 数据点大小
MARKER_EDGE_WIDTH = 1.0  # 数据点边框粗细

# ---------- 标题 ----------
TITLE_FONTSIZE = 14  # 子图标题字体大小
TITLE_PAD = 16  # 标题与图的距离

# ---------- 坐标轴标签 ----------
YLABEL_FONTSIZE = 11  # 纵轴单位标签字体大小（放在纵轴顶部）
XLABEL_FONTSIZE = 14  # 底部统一的 "Number of Tasks" 字体大小

# ---------- 刻度 ----------
TICK_LABELSIZE = 13  # 刻度数字字体大小（增大）
X_TICKS = [100, 300, 500, 700, 900, 1000]  # x轴显示的刻度值
X_TICK_ROTATION = 45  # x轴刻度旋转角度
Y_TICK_COUNT = 6  # 纵轴刻度数量

# ---------- 图例 ----------
LEGEND_FONTSIZE = 11  # 图例文字大小
LEGEND_HANDLELENGTH = 3.0  # 图例中线段长度
LEGEND_HANDLEHEIGHT = 1.2  # 图例中线段高度
LEGEND_COLUMNSPACING = 0.8  # 图例各项水平间距
LEGEND_LINE_WIDTH = 4  # 图例中线条粗细
LEGEND_MARKER_SIZE = 8  # 图例中数据点大小
LEGEND_BBOX_Y = -0.02  # 图例纵向位置

# ---------- 布局 (tight_layout) ----------
LAYOUT_PAD = 0  # 图与边框的整体间距
LAYOUT_H_PAD = 0  # 子图之间额外垂直间距
LAYOUT_W_PAD = 0  # 子图之间额外水平间距
LAYOUT_RECT_BOTTOM = 0.10  # 子图区域底部起点（留空给图例）
LAYOUT_RECT_TOP = 1.0  # 子图区域顶部终点

# ---------- 底部x轴标签 ----------
XLABEL_Y_POS = 0.05  # "Number of Tasks" 的纵向位置
# ===================================================

# Set style for publication quality
try:
    plt.style.use("seaborn-v0_8-paper")
except:
    plt.style.use("ggplot")

# Improve PDF export quality (vector text)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def plot_scalability_completion_detailed(
    csv_file=None, output_file=None, output_pdf=None
):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    results_dir = os.path.join(project_root, "results")
    csv_file = csv_file or os.path.join(results_dir, "scalability_completion.csv")
    output_file = output_file or os.path.join(
        results_dir, "scalability_completion_detailed.png"
    )
    output_pdf = output_pdf or os.path.join(
        results_dir, "scalability_completion_detailed.pdf"
    )

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Keep only the main experiment points: 50, 100, 200, 300, ..., 1000
    main_points = [50, 100] + list(range(200, 1001, 100))
    df = df[df["Num Tasks"].isin(main_points)].copy()

    # Rename scheduler - 使用全局配置
    df["Scheduler Display"] = df["Scheduler"].apply(get_scheduler_display_name)

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

    # Define metrics layout (3 Rows x 2 Cols) - shorter titles
    metrics_config = [
        ("Makespan", "Seconds", "Makespan", False),
        ("Avg Wait", "Seconds", "Avg Wait Time", False),
        ("Total GPU Time", "GPU-Seconds", "Total GPU Cost", False),
        ("Avg Slowdown", "Ratio", "Slowdown", True),
        ("Avg JCT", "Seconds", "Avg JCT", False),
        ("Cost Per Task", "GPU-Sec/Task", "Cost Per Task", False),
    ]

    # Scheduler styling - 使用全局颜色配置
    schedulers = df["Scheduler"].unique()

    # 使用全局颜色配置构建映射
    sched_map = {}
    for sched in schedulers:
        sched_map[sched] = get_scheduler_color(sched)

    markers = ["o", "s", "^", "D", "v", "P"]
    marker_map = {
        sched: markers[i % len(markers)] for i, sched in enumerate(schedulers)
    }

    linestyles = ["-", "--", "-.", ":", "-", "--"]
    style_map = {
        sched: linestyles[i % len(linestyles)] for i, sched in enumerate(schedulers)
    }

    # Create Figure (3x2 layout for two-column paper)
    fig, axes = plt.subplots(3, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.subplots_adjust(hspace=HSPACE, wspace=WSPACE)

    for idx, (metric, unit, title, use_log) in enumerate(metrics_config):
        # 3x2布局: row = idx // 2, col = idx % 2
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        for sched in schedulers:
            data = df[df["Scheduler"] == sched].sort_values("Num Tasks")
            if data.empty:
                continue

            # Filter out NaNs for plotting
            valid_data = data.dropna(subset=[metric])

            display_name = get_scheduler_display_name(sched)

            ax.plot(
                valid_data["Num Tasks"],
                valid_data[metric],
                marker=marker_map[sched],
                color=sched_map[sched],
                linestyle=style_map[sched],
                label=display_name,
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                alpha=0.9,
                markeredgewidth=MARKER_EDGE_WIDTH,
            )

        # 标题放在上方
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=TITLE_PAD)
        ax.set_xlabel("")  # 去掉每个图的x轴标签
        ax.set_ylabel("")  # 纵轴标签用text放在顶部
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontweight("bold")

        # x轴刻度
        ax.set_xticks(X_TICKS)
        ax.set_xticklabels(
            X_TICKS, rotation=X_TICK_ROTATION, ha="right", fontweight="bold"
        )

        # y轴刻度
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=Y_TICK_COUNT))

        # Log scale logic
        if use_log:
            ymin, ymax = ax.get_ylim()
            if ymin > 0 and ymax / ymin > 20:
                ax.set_yscale("log")

        # Compact number formatting (K/M suffix, no decimals)
        def compact_formatter(x, p):
            if x >= 1e6:
                return f"{int(x/1e6)}M"
            elif x >= 1e3:
                return f"{int(x/1e3)}K"
            else:
                return f"{int(x)}"

        ax.yaxis.set_major_formatter(plt.FuncFormatter(compact_formatter))

        # 纵轴单位放在纵轴最上方，靠左对齐
        ax.text(
            0,
            1.02,
            f"({unit})",
            transform=ax.transAxes,
            fontsize=YLABEL_FONTSIZE,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    # 1. 先调用 tight_layout，再用 subplots_adjust 强制设置间距
    plt.tight_layout(
        rect=[0, LAYOUT_RECT_BOTTOM, 1, LAYOUT_RECT_TOP],
        pad=LAYOUT_PAD,
        h_pad=LAYOUT_H_PAD,
        w_pad=LAYOUT_W_PAD,
    )
    # 强制覆盖间距（tight_layout 之后）
    plt.subplots_adjust(hspace=HSPACE, wspace=WSPACE)

    # 2. 再添加图例（在 tight_layout 之后，字体设置才不会被覆盖）
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
        ncol=len(schedulers),
        frameon=True,
        edgecolor="black",
        facecolor="white",
        fancybox=False,
        handlelength=LEGEND_HANDLELENGTH,
        handleheight=LEGEND_HANDLEHEIGHT,
        columnspacing=LEGEND_COLUMNSPACING,
        prop={"weight": "bold", "size": LEGEND_FONTSIZE},
    )
    for line in legend.get_lines():
        line.set_linewidth(LEGEND_LINE_WIDTH)
        line.set_markersize(LEGEND_MARKER_SIZE)

    # 3. 最后添加统一x轴标签
    fig.text(
        0.5,
        XLABEL_Y_POS,
        "Number of Tasks",
        ha="center",
        fontsize=XLABEL_FONTSIZE,
        fontweight="bold",
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"\nPlot saved to {output_file}")
    print(f"Plot saved to {output_pdf}")
    plt.close(fig)


def _file_signature(path: str):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return None


def _with_timestamp(path: str, ts: int):
    root, ext = os.path.splitext(path)
    return f"{root}_{ts}{ext}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--out-png", type=str, default=None)
    parser.add_argument("--out-pdf", type=str, default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--keep-history", action="store_true")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        csv_path = os.path.join(project_root, "results", "scalability_completion.csv")

    out_png = args.out_png
    out_pdf = args.out_pdf
    if out_png is None or out_pdf is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        results_dir = os.path.join(project_root, "results")
        out_png = out_png or os.path.join(
            results_dir, "scalability_completion_detailed.png"
        )
        out_pdf = out_pdf or os.path.join(
            results_dir, "scalability_completion_detailed.pdf"
        )

    def _render_once():
        if args.keep_history:
            ts = int(time.time())
            plot_scalability_completion_detailed(
                csv_file=csv_path,
                output_file=_with_timestamp(out_png, ts),
                output_pdf=_with_timestamp(out_pdf, ts),
            )
        else:
            plot_scalability_completion_detailed(
                csv_file=csv_path,
                output_file=out_png,
                output_pdf=out_pdf,
            )

    if not args.watch:
        _render_once()
        raise SystemExit(0)

    last_sig = None
    try:
        while True:
            sig = _file_signature(csv_path)
            if sig is not None and sig != last_sig:
                last_sig = sig
                _render_once()
            time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        raise SystemExit(0)
