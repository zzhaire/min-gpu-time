#!/usr/bin/env python3
"""
Plot N=1000 summary: 6 metrics in a 2x3 grid layout for two-column paper.
Metrics: Makespan, Wait Time, Total GPU Cost, Slowdown, Avg JCT, Cost Per Task
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from utils.plotting.colors import get_scheduler_color, get_scheduler_display_name

# ============== 可调参数 (Parameters) ==============
# ---------- 图片尺寸 ----------
FIG_WIDTH = 14  # 图片宽度（英寸），2x3布局建议14
FIG_HEIGHT = 12  # 图片高度（英寸），2x3布局建议12（瘦高）

# ---------- 子图间距 ----------
HSPACE = 0.35  # 上下子图间距
WSPACE = 0.30  # 左右子图间距

# ---------- 柱状图 ----------
BAR_WIDTH = 0.7  # 柱子宽度，0-1之间
BAR_EDGE_WIDTH = 1.0  # 柱子边框粗细
BAR_HIGHLIGHT_WIDTH = 3  # PACE 柱子边框粗细（突出显示）

# ---------- 标题 ----------
TITLE_FONTSIZE = 14  # 子图标题字体大小
TITLE_PAD = 16  # 标题与图的距离

# ---------- 坐标轴标签 ----------
YLABEL_FONTSIZE = 11  # 纵轴单位标签字体大小（放在纵轴顶部）

# ---------- 刻度 ----------
TICK_LABELSIZE = 13  # 刻度数字字体大小（增大）
Y_TICK_COUNT = 6  # 纵轴刻度数量

# ---------- 柱顶数字标签 ----------
BAR_LABEL_FONTSIZE = 10  # 柱顶数字字体大小
BAR_LABEL_OFFSET = 3  # 数字距离柱顶的距离（像素）

# ---------- 图例 ----------
LEGEND_FONTSIZE = 12  # 图例文字大小
LEGEND_NCOL = 5  # 图例列数
LEGEND_BBOX_Y = -0.02  # 图例纵向位置

# ---------- 布局 (tight_layout) ----------
LAYOUT_RECT_BOTTOM = 0.10  # 子图区域底部起点（留空给图例）
LAYOUT_RECT_TOP = 1.0  # 子图区域顶部终点

# ---------- Y轴范围 ----------
YLIM_MULTIPLIER = 1.18  # Y轴范围倍率（给柱顶数字留空间）
# ===================================================

# Scheduler display order
SCHEDULER_ORDER = [
    "pollux-patient",
    "pollux",
    "rack-aware",
    "min-gpu-time",
    "first-fit",
]


def plot_n1000_summary(csv_path: str, out_png: str, out_pdf: str, n_tasks: int = 1000):
    """Generate 2x3 grid of bar charts for N=n_tasks."""

    df = pd.read_csv(csv_path)
    df_n = df[df["Num Tasks"] == n_tasks].copy()

    if df_n.empty:
        raise ValueError(f"No data found for N={n_tasks}")

    # Compute Cost Per Task
    df_n["Cost Per Task"] = df_n["Total GPU Time"] / df_n["Num Tasks"]

    # Define metrics (6 metrics for 2x3 grid)
    metrics = [
        ("Makespan", "Makespan (s)", "Time to Finish All Jobs"),
        ("Avg Wait", "Avg Wait Time (s)", "Avg Wait Time"),
        ("Total GPU Time", "GPU-Seconds", "Total GPU Cost"),
        ("Avg Slowdown", "Slowdown Ratio", "User Experience (Slowdown)"),
        ("Avg JCT", "Avg JCT (s)", "Avg Job Completion Time"),
        ("Cost Per Task", "GPU-Sec / Task", "Cost Per Task"),
    ]

    # 2x3 布局适合双栏论文 - 2列3行
    fig, axes = plt.subplots(3, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.subplots_adjust(hspace=HSPACE, wspace=WSPACE)

    # y轴格式化函数
    def format_k(x, pos):
        if x >= 1e6:
            return f"{x/1e6:.0f}M"
        elif x >= 1e3:
            return f"{x/1e3:.0f}k"
        return f"{x:.0f}"

    for idx, (metric_col, ylabel, title) in enumerate(metrics):
        # 2x3布局: row = idx // 2, col = idx % 2
        row = idx // 2
        col_idx = idx % 2
        ax = axes[row, col_idx]

        values = []
        colors = []
        labels = []

        for sched in SCHEDULER_ORDER:
            row_data = df_n[df_n["Scheduler"] == sched]
            if not row_data.empty:
                # 获取该调度器的指标值
                metric_value = row_data[metric_col].values[0]
                values.append(metric_value)
                colors.append(get_scheduler_color(sched))
                labels.append(get_scheduler_display_name(sched))

        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            values,
            color=colors,
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            width=BAR_WIDTH,
        )

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if val >= 1e6:
                label = f"{val/1e6:.2f}M"
            elif val >= 1e3:
                label = f"{val/1e3:.1f}K"
            else:
                label = f"{val:.1f}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, BAR_LABEL_OFFSET),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_FONTSIZE,
                fontweight="bold",
            )

        ax.set_ylabel("")  # 纵轴标签用text放在顶部
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=TITLE_PAD)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
        ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        ax.yaxis.set_major_formatter(FuncFormatter(format_k))
        ax.yaxis.set_major_locator(
            plt.MaxNLocator(nbins=Y_TICK_COUNT)
        )
        ax.set_ylim(0, max(values) * YLIM_MULTIPLIER)

        # 纵轴单位放在纵轴最上方，靠左对齐
        ax.text(
            0,
            1.02,
            f"({ylabel})",
            transform=ax.transAxes,
            fontsize=YLABEL_FONTSIZE,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        # Highlight PACE (first bar)
        if len(bars) > 0:
            bars[0].set_edgecolor(get_scheduler_color("pollux-patient"))
            bars[0].set_linewidth(BAR_HIGHLIGHT_WIDTH)

    # 统一图例放在最下面
    legend_handles = [
        Patch(
            facecolor=get_scheduler_color(s), edgecolor="black", label=get_scheduler_display_name(s)
        )
        for s in SCHEDULER_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=LEGEND_NCOL,
        frameon=True,
        facecolor="white",
        edgecolor="gray",
        bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
        prop={"weight": "bold", "size": LEGEND_FONTSIZE},
    )

    plt.tight_layout(rect=[0, LAYOUT_RECT_BOTTOM, 1, LAYOUT_RECT_TOP])

    # Save outputs
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot N=1000 summary (2x3 grid)")
    parser.add_argument(
        "--csv",
        type=str,
        default="results/scalability_completion.csv",
        help="Path to scalability_completion.csv",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="results/n1000_summary.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default="results/n1000_summary.pdf",
        help="Output PDF path",
    )
    parser.add_argument(
        "--n", type=int, default=1000, help="Number of tasks to plot (default: 1000)"
    )
    args = parser.parse_args()

    plot_n1000_summary(args.csv, args.out_png, args.out_pdf, args.n)


if __name__ == "__main__":
    main()
