"""
Scalability Experiment: Varying number of tasks to test scheduler performance at scale.
We scale the submission window linearly with the number of tasks to maintain a constant Average Load (Arrival Rate).
This isolates the effect of "Scale" (Queue Depth management, Fragmentation over time) from "Load" (Saturation).
"""

import sys
import os
import csv
import time
import numpy as np
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    default_cluster_config,
    default_task_config,
    default_simulator_config,
)
from utils.task_generator import TaskGenerator
from simulator import Simulator
from schedulers import (
    FirstFitScheduler,
    BestFitScheduler,
    RackAwareScheduler,
    MinGPUTimeScheduler,
    PolluxScheduler,
    PolluxPatientScheduler,
)
from core.cluster import Cluster


def run_experiment(scheduler_name, num_tasks, submission_window, max_time):
    """Run a single experiment with a specific number of tasks"""

    # 1. Configure
    cluster_config = default_cluster_config
    simulator_config = deepcopy(default_simulator_config)

    # Strict deadline for Stress Test
    simulator_config.max_time = max_time

    # 2. Create Cluster
    cluster = Cluster(
        num_racks=cluster_config.num_racks,
        gpus_per_rack=cluster_config.gpus_per_rack,
        gpu_memory=cluster_config.gpu_memory,
        intra_rack_penalty=cluster_config.intra_rack_penalty,
        inter_rack_penalty=cluster_config.inter_rack_penalty,
    )

    # 3. Create Scheduler
    if scheduler_name == "first-fit":
        scheduler = FirstFitScheduler(cluster)
    elif scheduler_name == "best-fit":
        scheduler = BestFitScheduler(cluster)
    elif scheduler_name == "rack-aware":
        scheduler = RackAwareScheduler(cluster)
    elif scheduler_name == "min-gpu-time":
        scheduler = MinGPUTimeScheduler(
            cluster, patience_threshold=1.1, starvation_limit=2000.0
        )
    elif scheduler_name == "pollux":
        scheduler = PolluxScheduler(cluster, alpha=0.5)
    elif scheduler_name == "pollux-patient":
        # Ours: P=1.5 (Pareto Optimal)
        scheduler = PolluxPatientScheduler(
            cluster, alpha=0.5, patience_threshold=1.5, starvation_limit=2000.0
        )
        scheduler.sharing_penalty_map = simulator_config.sharing_penalty_map
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # 4. Generate Tasks
    gen = TaskGenerator(seed=42 + num_tasks)
    tasks = gen.generate_tasks(
        num_tasks=num_tasks,
        min_gpus=default_task_config.min_gpus,
        max_gpus=default_task_config.max_gpus,
        min_memory=default_task_config.min_memory,
        max_memory=default_task_config.max_memory,
        min_duration=default_task_config.min_duration,
        max_duration=default_task_config.max_duration,
        submission_window=submission_window,
    )

    # 5. Run
    simulator = Simulator(cluster, scheduler, simulator_config)
    metrics = simulator.run(tasks)

    # 6. Collect Metrics
    summary = metrics.get_summary()
    completed_count = summary["completed_tasks"]

    # Completion Rate
    completion_rate = (completed_count / num_tasks) * 100.0

    # Avg JCT (only for completed)
    avg_jct = summary["average_jct"] if summary["average_jct"] else 0
    avg_wait = summary["average_wait_time"] if summary["average_wait_time"] else 0

    # Total GPU Time (Cost)
    total_gpu_time = metrics.total_gpu_time

    # Slowdown
    completed_tasks = [m for m in metrics.task_metrics if m["status"] == "completed"]
    if completed_tasks:
        slowdowns = [m["jct"] / m["estimated_duration"] for m in completed_tasks]
        avg_slowdown = sum(slowdowns) / len(slowdowns)
    else:
        avg_slowdown = 0

    # Throughput (Tasks / Hour) relative to Simulation Time (max_time)
    throughput = (completed_count / max_time) * 3600.0

    return {
        "Completion Rate": completion_rate,
        "Avg JCT": avg_jct,
        "Avg Wait": avg_wait,
        "Avg Slowdown": avg_slowdown,
        "Throughput": throughput,
        "Total GPU Time": total_gpu_time,
    }


def main():
    output_file = "results/scalability.csv"
    os.makedirs("results", exist_ok=True)

    # Stress Test Scales
    task_counts = [100, 300, 500, 800, 1000]

    # Fixed Constraints
    submission_window = 1800.0  # 30 mins
    max_time = 3600.0  # 1 hour hard limit

    schedulers = [
        "pollux-patient",
        "pollux",
        "rack-aware",
        "min-gpu-time",
        "first-fit",
    ]

    print("=" * 60)
    print("Starting Stress Test (Scalability) Experiment")
    print(f"Window: {submission_window}s, Max Time: {max_time}s")
    print("=" * 60)

    # Initialize CSV
    fieldnames = [
        "Scheduler",
        "Num Tasks",
        "Completion Rate",
        "Avg JCT",
        "Avg Wait",
        "Avg Slowdown",
        "Throughput",
        "Total GPU Time",
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for n in task_counts:
        print(f"\n[Load N={n} tasks / {submission_window}s]")

        for sched in schedulers:
            try:
                t0 = time.time()
                metrics = run_experiment(sched, n, submission_window, max_time)
                duration = time.time() - t0

                print(
                    f"  > {sched:<15}: Rate={metrics['Completion Rate']:5.1f}%, GPU_Time={metrics['Total GPU Time']:.0f} (Sim: {duration:.1f}s)"
                )

                res = {"Scheduler": sched, "Num Tasks": n}
                res.update(metrics)

                with open(output_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(res)

            except Exception as e:
                print(f"  > {sched:<15}: FAILED ({e})")

    print("\nScalability experiment completed. Results saved.")


if __name__ == "__main__":
    main()
