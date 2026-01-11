import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.plotting.colors import get_scheduler_color, get_scheduler_display_name

# ============== 可调参数 (Parameters) ==============
# Patience 标注位置 (dx, dy) - offset points
PATIENCE_POS_1 = (10, 10)  # Patience < δ_intra
PATIENCE_POS_2 = (0, -20)  # δ_intra ≤ Patience < δ_inter
PATIENCE_POS_3 = (18, 0)  # Patience ≥ δ_inter
# ===================================================

try:
    import seaborn as sns

    _HAVE_SEABORN = True
except Exception:
    sns = None
    _HAVE_SEABORN = False

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("ggplot")


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    results_dir = os.path.join(project_root, "results")
    csv_file = os.path.join(results_dir, "pareto_frontier.csv")
    output_file = os.path.join(results_dir, "pareto_frontier.png")
    output_pdf = os.path.join(results_dir, "pareto_frontier.pdf")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # Calculate Cost ($)
    # Assumption: $3 per GPU-Hour
    gpu_price_per_hour = 3.0

    # Topology penalty thresholds (from default config)
    delta_intra = 1.4
    delta_inter = 2.1
    df["Cost"] = (df["Total GPU Time"] / 3600.0) * gpu_price_per_hour

    # Improve PDF export quality (vector text)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # Plot Baselines
    baselines = df[df["Type"] == "Baseline"]
    # Define markers for baselines，颜色统一用全局配置的 get_scheduler_color
    baseline_markers = {
        "rack_aware": "s",
        "min_gpu_time": "^",
        "first_fit": "v",
        "pollux": "D",
    }

    for _, row in baselines.iterrows():
        sched = row["Scheduler"]  # 例如 'rack-aware', 'min-gpu-time'
        norm_key = sched.lower().replace("-", "_")
        marker = baseline_markers.get(norm_key, "o")
        color = get_scheduler_color(sched)
        display_name = get_scheduler_display_name(sched)
        ax.scatter(
            row["Cost"],
            row["Avg JCT"],
            label=display_name,
            marker=marker,
            color=color,
            s=250,
            zorder=10,
            edgecolors="black",
            linewidths=1.5,
        )

        # Annotate
        ax.annotate(
            display_name,
            xy=(row["Cost"], row["Avg JCT"]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=12,
            ha="left",
            va="bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
            zorder=20,
        )

    # Plot Ours (PACE)
    ours = df[df["Type"] == "Ours"].copy()
    if not ours.empty:
        ours["Patience"] = ours["Param"].astype(str).str.replace("P=", "", regex=False)
        ours["Patience"] = pd.to_numeric(ours["Patience"], errors="coerce")

    # Group identical points so dense sweeps remain readable.
    # (Many Patience values can yield the same outcome due to topology thresholds.)
    grouped_rows = []
    if not ours.empty:
        for (tgt, jct), g in ours.groupby(["Total GPU Time", "Avg JCT"], dropna=False):
            pats = [p for p in g["Patience"].tolist() if pd.notna(p)]
            pats_sorted = sorted(pats)

            if pats_sorted:
                pmin = pats_sorted[0]
                pmax = pats_sorted[-1]

                # Label by regime instead of concrete values.
                if pmax < delta_intra:
                    label = "Patience < δ_intra"
                elif pmin >= delta_inter:
                    label = "Patience ≥ δ_inter"
                else:
                    label = "δ_intra ≤ Patience < δ_inter"
            else:
                label = "Patience=N/A"

            cost_dollars = (float(tgt) / 3600.0) * gpu_price_per_hour
            grouped_rows.append(
                {
                    "Total GPU Time": float(tgt),
                    "Avg JCT": float(jct),
                    "Cost": cost_dollars,
                    "Label": label,
                    "PatienceMin": float(pmin) if pats_sorted else float("nan"),
                    "PatienceMax": float(pmax) if pats_sorted else float("nan"),
                }
            )

    ours_u = (
        pd.DataFrame(grouped_rows).sort_values("Cost")
        if grouped_rows
        else pd.DataFrame()
    )

    # PACE color
    pace_color = get_scheduler_color("pollux_patient")

    if not ours_u.empty:
        ax.plot(
            ours_u["Cost"],
            ours_u["Avg JCT"],
            color=pace_color,
            linestyle="--",
            alpha=0.6,
            zorder=5,
            label="Pareto Frontier (PACE)",
        )

        ax.scatter(
            ours_u["Cost"],
            ours_u["Avg JCT"],
            label="PACE (Ours)",
            marker="o",
            color=pace_color,
            s=280,
            zorder=10,
            edgecolors="black",
            linewidths=1.5,
        )

    # Annotate Our (Unique) Points
    if not ours_u.empty:
        y_min = float(np.nanmin(df["Avg JCT"])) if len(df) else 0.0
        y_max = float(np.nanmax(df["Avg JCT"])) if len(df) else 0.0
        y_span = max(1.0, y_max - y_min)

        # 使用顶部参数
        offsets = [PATIENCE_POS_1, PATIENCE_POS_2, PATIENCE_POS_3]

        for i, (_, row) in enumerate(ours_u.iterrows()):
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(
                row["Label"],
                xy=(row["Cost"], row["Avg JCT"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=12,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                color=pace_color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                zorder=30,
            )

    # Formatting
    ax.set_xlabel("Total Cloud Cost ($) [@ $3/GPU-hr]", fontsize=15)
    ax.set_ylabel("Average Job Completion Time (s) [Lower is Better]", fontsize=15)
    ax.set_title('Cost-Speed Pareto Frontier: The "Sweet Spot"', fontsize=17)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Give the plot more breathing room
    ax.margins(x=0.08, y=0.12)

    # Legend inside (upper-left) with better spacing
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="lightgray",
        fontsize=11,
        labelspacing=0.8,  # 行距
        handletextpad=0.8,
    )
    # 加粗图例文字
    for text in legend.get_texts():
        text.set_fontweight("bold")

    # Invert X axis? No, lower cost is better (left). Lower JCT is better (bottom).
    # Ideal point is Bottom-Left.

    # Add an arrow pointing to the "Sweet Spot" (P=1.5 usually)
    # Find point with min (Cost * JCT) or just the "knee"
    # P=1.5 in our data: Cost ~464, JCT ~5219.
    # Wait, in the run output P=1.5 was Cost 557k, JCT 5219. P=1.0 was Cost 471k, JCT 7823.
    # The knee is likely P=1.5.

    # Optional: Highlight the optimal region
    # plt.axvspan(min(ours['Cost']), max(ours['Cost']), color='green', alpha=0.05)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"Pareto plot saved to {output_file}")
    print(f"Pareto plot saved to {output_pdf}")


if __name__ == "__main__":
    main()
