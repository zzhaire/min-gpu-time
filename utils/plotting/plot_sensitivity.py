from utils.plotting.colors import get_scheduler_color, get_scheduler_display_name

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

# ============== 可调参数 (Parameters) ==============
FIG_WIDTH = 10
FIG_HEIGHT = 6
TITLE = "Sensitivity to Interference"
TITLE_FONTSIZE = 18
TITLE_PAD = 30
YLABEL = "Normalized Total GPU Time"
YLABEL_FONTSIZE = 16
XLABEL = ""
YLIM_MIN = 0
YLIM_MAX = 1.18
BAR_LABEL_FONTSIZE = 13
BAR_LABEL_PADDING = 3

# Ticks
TICK_LABELSIZE = 18

# Legend
LEGEND_NCOL = 4
LEGEND_FONTSIZE = 14
# 把图例移动到图的下方，避免挡住标题
LEGEND_BBOX = (0.5, -0.15)  # (x, y) position (below axes)
LEGEND_LOC = "upper center"
LEGEND_COLUMNSPACING = 1.2

# Baseline（用 First Fit 的全局颜色）
BASELINE_COLOR = get_scheduler_color("first-fit")
BASELINE_LINESTYLE = "--"
BASELINE_LINEWIDTH = 2.5
BASELINE_ALPHA = 0.7
BASELINE_LABEL = f"Baseline ({get_scheduler_display_name('first-fit')})"
# ===================================================

try:
    import seaborn as sns

    _HAVE_SEABORN = True
except Exception:
    sns = None
    _HAVE_SEABORN = False

if _HAVE_SEABORN:
    sns.set_theme(style="whitegrid")

# Read data (自动定位项目根目录)
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
csv_path = os.path.join(project_root, "results/sensitivity_interference.csv")
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Define order for plotting
interference_order = ["Low", "Medium", "High"]
scheduler_order = [
    "first-fit",
    "rack-aware",
    "min-gpu-time",
    "pollux",
    "pollux-patient",
]

# Create the plot
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
ax = plt.gca()
g = None

if _HAVE_SEABORN:
    # 使用全局颜色配置：键是调度器 key，值是颜色
    custom_palette = {sched: get_scheduler_color(sched) for sched in scheduler_order}
    g = sns.barplot(
        data=df,
        x="Interference Level",
        y="Normalized Time",
        hue="Scheduler",
        order=interference_order,
        hue_order=scheduler_order,
        palette=custom_palette,
    )
else:
    df_plot = df.copy()
    df_plot["Interference Level"] = pd.Categorical(
        df_plot["Interference Level"], categories=interference_order, ordered=True
    )
    df_plot["Scheduler"] = pd.Categorical(
        df_plot["Scheduler"], categories=scheduler_order, ordered=True
    )
    df_plot = df_plot.sort_values(["Interference Level", "Scheduler"])

    pivot = (
        df_plot.pivot(
            index="Interference Level", columns="Scheduler", values="Normalized Time"
        )
        .reindex(interference_order)
        .reindex(columns=scheduler_order)
    )

    x = np.arange(len(interference_order))
    width = 0.8 / max(1, len(scheduler_order))

    cmap = plt.get_cmap("viridis")
    # 使用全局颜色配置
    colors = [get_scheduler_color(sched) for sched in scheduler_order]

    containers = []
    for i, sched in enumerate(scheduler_order):
        vals = pivot[sched].values
        offsets = x - 0.4 + width / 2 + i * width
        bars = ax.bar(
            offsets, vals, width=width, label=sched, color=colors[i], alpha=0.9
        )
        containers.append(bars)

    ax.set_xticks(x)
    ax.set_xticklabels(interference_order, fontsize=TICK_LABELSIZE, fontweight="bold")
    ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)

    for bars in containers:
        for bar in bars:
            height = bar.get_height()
            if height is None or pd.isna(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_FONTSIZE,
                rotation=45,
            )

# Add a horizontal line at y=1.0 (Baseline)
plt.axhline(
    y=1.0,
    color=BASELINE_COLOR,
    linestyle=BASELINE_LINESTYLE,
    linewidth=BASELINE_LINEWIDTH,
    alpha=BASELINE_ALPHA,
)

# Customize
plt.ylabel(YLABEL, fontsize=YLABEL_FONTSIZE, fontweight="bold")
plt.xlabel(XLABEL)
plt.title(TITLE, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=TITLE_PAD)

# 统一刻度字号（seaborn 分支也适用）
ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)

# Get handles and labels, then add baseline at the end
handles, labels = ax.get_legend_handles_labels()

# 把图例里的调度器 key 映射成统一的显示名
pretty_labels = []
for lab in labels:
    pretty_labels.append(get_scheduler_display_name(lab))

baseline_handle = Line2D(
    [0],
    [0],
    color=BASELINE_COLOR,
    linestyle=BASELINE_LINESTYLE,
    linewidth=BASELINE_LINEWIDTH,
    alpha=BASELINE_ALPHA,
)
handles.append(baseline_handle)
pretty_labels.append(BASELINE_LABEL)

# Legend（脚注）：字体加粗，并放在图下方
legend = plt.legend(
    handles,
    pretty_labels,
    bbox_to_anchor=LEGEND_BBOX,
    loc=LEGEND_LOC,
    ncol=LEGEND_NCOL,
    fontsize=LEGEND_FONTSIZE,
    frameon=True,
    edgecolor="black",
    facecolor="white",
    columnspacing=LEGEND_COLUMNSPACING,
)
for text in legend.get_texts():
    text.set_fontweight("bold")
plt.ylim(YLIM_MIN, YLIM_MAX)

# Add value labels on top of bars
if _HAVE_SEABORN and g is not None:
    for i in g.containers:
        g.bar_label(
            i,
            fmt="%.2f",
            fontsize=BAR_LABEL_FONTSIZE,
            padding=BAR_LABEL_PADDING,
            rotation=45,
        )

plt.tight_layout()

# Save PNG and PDF
output_png = os.path.join(project_root, "results/sensitivity_plot.png")
output_pdf = os.path.join(project_root, "results/sensitivity_plot.pdf")
plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.savefig(output_pdf, bbox_inches="tight")
print(f"Plot saved to {output_png}")
print(f"Plot saved to {output_pdf}")
