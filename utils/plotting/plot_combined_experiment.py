#!/usr/bin/env python3
"""
合并图：左边是scalability折线图，右边是n1000柱状图
3行4列布局，每行2个指标，每个指标有折线图+柱状图

所有可调参数都在下面的 "============== 可调参数 ==============" 区域内
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from utils.plotting.colors import get_scheduler_color, get_scheduler_display_name

# ============== 可调参数 (Parameters) ==============
# 修改下面的参数来调整图表样式

# ---------- 图片尺寸 ----------
FIG_WIDTH = 22        # 图片宽度（英寸），4列布局
FIG_HEIGHT = 18        # 图片高度（英寸），3行布局

# ---------- 子图间距 ----------
HSPACE = 0.10         # 上下子图间距 (0-1之间)
WSPACE = 0.18         # 左右子图间距 (0-1之间)

# ---------- 折线图参数 ----------
LINE_WIDTH = 2.5       # 折线粗细
MARKER_SIZE = 10       # 数据点大小
MARKER_EDGE_WIDTH = 1.5  # 数据点边框粗细

# ---------- 柱状图参数 ----------
BAR_WIDTH = 0.55      # 柱子宽度 (0-1之间)
BAR_EDGE_WIDTH = 0.8  # 柱子边框粗细
BAR_HIGHLIGHT_WIDTH = 1.5  # PACE 柱子边框粗细（突出显示）
BAR_LABEL_FONTSIZE = 14  # 柱顶数字字体大小
BAR_LABEL_OFFSET = 4   # 柱顶数字距离柱顶的距离（像素）

# 柱状图 X 轴标签 (PACE, Pollux, rack-aware 等) 的字体大小
EBAR_XTICK_FONTSIZE = 20

# ---------- 标题 ----------
TITLE_FONTSIZE = 20   # 子图标题字体大小
TITLE_PAD = 20         # 标题与图的距离

# ---------- 坐标轴标签 (Y轴单位/纵轴单位) ----------
# 设为 True 表示把单位放在左边(传统的 set_ylabel 方式)，设为 False 表示放在上方(用 text)
YLABEL_ON_LEFT = True   # 纵轴单位放在左边还是上方
YLABEL_FONTSIZE = 18    # 纵轴单位字体大小
YLABEL_LABELPAD = 6     # 纵轴单位离y轴的距离（仅当 YLABEL_ON_LEFT=True 时有效）

# ---------- 底部横轴标签 (Number of Tasks) ----------
# 设为 True 表示每个折线图自己显示 xlabel，设为 False 表示用全局的 fig.text
XLABEL_PER_SUBPLOT = True   # 是否在每个折线图画 xlabel
XLABEL_FONTSIZE = 18        # 底部 "Number of Tasks" 字体大小
XLABEL_LABELPAD = 6         # xlabel 离轴的距离（仅当 XLABEL_PER_SUBPLOT=True 时有效）
XLABEL_GLOBAL_Y_POS = 0.05  # 全局 xlabel 的位置（仅当 XLABEL_PER_SUBPLOT=False 时有效，0-1之间）
XLABEL_GLOBAL_X1 = 0.12     # 第一个 xlabel 的横向位置（仅当 XLABEL_PER_SUBPLOT=False 时有效）
XLABEL_GLOBAL_X2 = 0.62     # 第二个 xlabel 的横向位置（仅当 XLABEL_PER_SUBPLOT=False 时有效）

# ---------- 刻度 ----------
TICK_LABELSIZE = 15     # 刻度数字字体大小
X_TICKS = [100, 300, 500, 600, 800, 1000]  # x轴显示的刻度值
X_TICK_ROTATION = 45    # x轴刻度旋转角度
Y_TICK_COUNT = 5        # 纵轴刻度数量

# ---------- 图例 ----------
LEGEND_FONTSIZE = 20    # 图例文字大小
LEGEND_NCOL = 5         # 图例列数
LEGEND_BBOX_Y = 0.05   # 图例纵向位置 (负值向下移动，越小越往下)
LEGEND_MARKER_SIZE = 15 # 图例中 marker 的大小

# ---------- 布局 (tight_layout) ----------
LAYOUT_RECT_BOTTOM = 0.10  # 子图区域底部起点
LAYOUT_RECT_TOP = 1.0      # 子图区域顶部终点
LAYOUT_RECT_LEFT = 0.04    # 左边留白

# ---------- Y轴范围 ----------
YLIM_MULTIPLIER = 1.15     # Y轴范围倍率（给柱顶数字留空间）

# ==================================================

# Scheduler display order
SCHEDULER_ORDER = [
    "pollux-patient",
    "pollux",
    "rack-aware",
    "min-gpu-time",
    "first-fit",
]

# Set style for publication quality
try:
    plt.style.use("seaborn-v0_8-paper")
except:
    plt.style.use("ggplot")

# Improve PDF export quality (vector text)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def format_k(x, pos):
    """Y轴格式化函数"""
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    elif x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def plot_combined(csv_path: str, out_png: str, out_pdf: str, n_tasks: int = 1000):
    """Generate combined figure: 3 rows x 4 columns"""
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 提取 N=1000 的数据用于柱状图
    df_n = df[df["Num Tasks"] == n_tasks].copy()
    df_n["Cost Per Task"] = df_n["Total GPU Time"] / df_n["Num Tasks"]
    
    # 筛选主实验点用于折线图
    main_points = [50, 100] + list(range(200, 1001, 100))
    df_line = df[df["Num Tasks"].isin(main_points)].copy()
    df_line["Cost Per Task"] = df_line["Total GPU Time"] / df_line["Num Tasks"]
    
    # 定义6个指标配置 (3行 x 4列)
    # 每行2个指标，每个指标有折线图和柱状图
    metrics_config = [
        # Row 1: Makespan, Avg Wait
        ("Makespan", "Seconds", "Makespan", False, False),   # col 0: line
        ("Makespan", "Makespan (s)", "N=1000", False, True),   # col 1: bar
        ("Avg Wait", "Seconds", "Avg Wait Time", False, False),  # col 2: line
        ("Avg Wait", "Avg Wait Time (s)", "N=1000", False, True), # col 3: bar
        # Row 2: Total GPU Time, Avg Slowdown
        ("Total GPU Time", "GPU-Seconds", "Total GPU Cost", False, False),  # col 0
        ("Total GPU Time", "GPU-Sec", "N=1000", False, True),  # col 1
        ("Avg Slowdown", "Ratio", "Slowdown", False, False),  # col 2
        ("Avg Slowdown", "Ratio", "N=1000", False, True),      # col 3
        # Row 3: Avg JCT, Cost Per Task
        ("Avg JCT", "Seconds", "Avg JCT", False, False),       # col 0
        ("Avg JCT", "Avg JCT (s)", "N=1000", False, True),     # col 1
        ("Cost Per Task", "GPU-Sec/Task", "Cost Per Task", False, False),  # col 2
        ("Cost Per Task", "GPU-Sec/Task", "N=1000", False, True),  # col 3
    ]
    
    # 创建3行4列的图
    fig, axes = plt.subplots(3, 4, figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.subplots_adjust(hspace=HSPACE, wspace=WSPACE)
    
    # 定义样式
    markers = ["o", "s", "^", "D", "v"]
    linestyles = ["-", "--", "-.", ":", "-"]
    
    # 折线图scheduler排序
    line_schedulers = [s for s in SCHEDULER_ORDER if s in df_line["Scheduler"].unique()]
    
    for idx, (metric_col, ylabel, title, use_log, is_bar) in enumerate(metrics_config):
        row = idx // 4  # 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2
        col = idx % 4   # 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
        ax = axes[row, col]
        
        if is_bar:
            # ========== 柱状图 ==========
            values = []
            colors = []
            labels = []
            
            for sched in SCHEDULER_ORDER:
                row_data = df_n[df_n["Scheduler"] == sched]
                if not row_data.empty:
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
            
            # 先只对 y 轴应用通用刻度字体，然后单独设置 X 轴标签字体
            ax.set_xticks(x)
            ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
            ax.set_xticklabels(
                labels,
                rotation=45,
                ha="right",
                fontsize=EBAR_XTICK_FONTSIZE,
            )
            
            # Highlight PACE (first bar)
            if len(bars) > 0:
                bars[0].set_edgecolor(get_scheduler_color("pollux-patient"))
                bars[0].set_linewidth(BAR_HIGHLIGHT_WIDTH)
            
            ax.set_ylim(0, max(values) * YLIM_MULTIPLIER)
            
        else:
            # ========== 折线图 ==========
            marker_idx = 0
            for sched in line_schedulers:
                data = df_line[df_line["Scheduler"] == sched].sort_values("Num Tasks")
                if data.empty:
                    continue
                
                valid_data = data.dropna(subset=[metric_col])
                display_name = get_scheduler_display_name(sched)
                
                ax.plot(
                    valid_data["Num Tasks"],
                    valid_data[metric_col],
                    marker=markers[marker_idx % len(markers)],
                    color=get_scheduler_color(sched),
                    linestyle=linestyles[marker_idx % len(linestyles)],
                    label=display_name,
                    linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE,
                    alpha=0.9,
                    markeredgewidth=MARKER_EDGE_WIDTH,
                )
                marker_idx += 1
            
            # x轴刻度
            ax.set_xticks(X_TICKS)
            ax.set_xticklabels(
                X_TICKS, rotation=X_TICK_ROTATION, ha="right", fontweight="bold"
            )
            
            # Log scale logic
            if use_log:
                ymin, ymax = ax.get_ylim()
                if ymin > 0 and ymax / ymin > 20:
                    ax.set_yscale("log")
        
        # 通用设置
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=TITLE_PAD)
        
        # ---------- 纵轴单位设置 ----------
        if YLABEL_ON_LEFT:
            # 方式1: 放在左边 (set_ylabel)
            ax.set_ylabel(
                ylabel,
                fontsize=YLABEL_FONTSIZE,
                fontweight="bold",
                labelpad=YLABEL_LABELPAD,
            )
        else:
            # 方式2: 放在上方 (text)
            ax.set_ylabel("")
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
        
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
        # 折线图对 x/y 同时应用刻度字体；柱状图的 y 轴刻度在上面单独设置
        if not is_bar:
            ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=Y_TICK_COUNT))
        ax.yaxis.set_major_formatter(FuncFormatter(format_k))
        
        # ---------- 底部 x 轴标签设置 ----------
        # 仅对最底下一行的折线图添加 xlabel
        if XLABEL_PER_SUBPLOT and not is_bar and row == 2:
            ax.set_xlabel(
                "Number of Tasks",
                fontsize=XLABEL_FONTSIZE,
                fontweight="bold",
                labelpad=XLABEL_LABELPAD,
            )
    
    # 统一图例放在最下面
    from matplotlib.lines import Line2D
    legend_handles = []
    for i, s in enumerate(SCHEDULER_ORDER):
        if s in df_line["Scheduler"].unique():
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker=markers[i % len(markers)],
                    color="w",
                    markerfacecolor=get_scheduler_color(s),
                    markersize=LEGEND_MARKER_SIZE,
                    label=get_scheduler_display_name(s),
                )
            )
    
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
    
    # 添加底部的 "Number of Tasks" 标签（仅当不使用 per-subplot xlabel 时）
    if not XLABEL_PER_SUBPLOT:
        fig.text(
            XLABEL_GLOBAL_X1,
            XLABEL_GLOBAL_Y_POS,
            "Number of Tasks",
            ha="center",
            fontsize=XLABEL_FONTSIZE,
            fontweight="bold",
        )
        fig.text(
            XLABEL_GLOBAL_X2,
            XLABEL_GLOBAL_Y_POS,
            "Number of Tasks",
            ha="center",
            fontsize=XLABEL_FONTSIZE,
            fontweight="bold",
        )
    
    plt.tight_layout(rect=[LAYOUT_RECT_LEFT, LAYOUT_RECT_BOTTOM, 1, LAYOUT_RECT_TOP])
    
    # Save outputs
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot combined figure")
    parser.add_argument(
        "--csv",
        type=str,
        default="results/scalability_completion.csv",
        help="Path to scalability_completion.csv",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="results/combined_experiment.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default="results/combined_experiment.pdf",
        help="Output PDF path",
    )
    parser.add_argument(
        "--n", type=int, default=1000, help="Number of tasks for bar chart (default: 1000)"
    )
    args = parser.parse_args()
    
    plot_combined(args.csv, args.out_png, args.out_pdf, args.n)


if __name__ == "__main__":
    main()
