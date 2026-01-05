import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style
sns.set_theme(style="whitegrid")

# Read data
csv_path = "results/sensitivity_interference.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Define order for plotting
interference_order = ["Low", "Medium", "High"]
scheduler_order = [
    "first-fit",
    "best-fit",
    "rack-aware",
    "min-gpu-time",
    "pollux",
    "pollux-patient",
]

# Create the plot
plt.figure(figsize=(12, 8))
g = sns.barplot(
    data=df,
    x="Interference Level",
    y="Normalized Time",
    hue="Scheduler",
    order=interference_order,
    hue_order=scheduler_order,
    palette="viridis",
)

# Add a horizontal line at y=1.0 (Baseline)
plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Baseline")

# Customize
plt.title("Sensitivity to Interference: Scheduler Robustness", fontsize=16)
plt.ylabel("Normalized Total GPU Time (Lower is Better)", fontsize=12)
plt.xlabel("Interference Level", fontsize=12)
plt.legend(title="Scheduler", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.ylim(0, 1.2)  # Adjust Y-axis limit for better visibility

# Add value labels on top of bars
for i in g.containers:
    g.bar_label(i, fmt="%.2f", fontsize=9, padding=3)

plt.tight_layout()

# Save
output_path = "results/sensitivity_plot.png"
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
