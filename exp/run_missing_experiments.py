"""
补充运行缺失的实验数据
"""

import sys
import os
import csv
import time
import numpy as np
from copy import deepcopy

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

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


def run_experiment(scheduler_name, num_tasks, submission_window):
    """Run a single experiment with a specific number of tasks until completion"""

    # 1. Configure
    cluster_config = default_cluster_config
    simulator_config = deepcopy(default_simulator_config)

    # Run to completion (Large max_time)
    simulator_config.max_time = 200000.0

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
        # Pure patience, Rigid
        scheduler = MinGPUTimeScheduler(
            cluster, patience_threshold=1.1, starvation_limit=5000.0
        )
    elif scheduler_name == "pollux":
        # Elastic, Impatient
        scheduler = PolluxScheduler(cluster, alpha=0.5)
    elif scheduler_name == "pollux-patient":
        # Elastic + Patient (Ours)
        scheduler = PolluxPatientScheduler(
            cluster, alpha=0.5, patience_threshold=1.5, starvation_limit=5000.0
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

    # Total GPU Time (Cost) - This is the key metric for Run-to-Completion
    total_gpu_time = metrics.total_gpu_time

    # Slowdown
    completed_tasks = [m for m in metrics.task_metrics if m["status"] == "completed"]
    if completed_tasks:
        slowdowns = [m["jct"] / m["estimated_duration"] for m in completed_tasks]
        avg_slowdown = sum(slowdowns) / len(slowdowns)
    else:
        avg_slowdown = 0

    makespan = metrics.timeline[-1]["time"] if metrics.timeline else 0

    return {
        "Completion Rate": completion_rate,
        "Avg JCT": avg_jct,
        "Avg Wait": avg_wait,
        "Avg Slowdown": avg_slowdown,
        "Total GPU Time": total_gpu_time,
        "Makespan": makespan,
    }


def main():
    output_file = "results/scalability_completion.csv"

    # 需要补充的实验
    missing_experiments = [
        (950, "min-gpu-time"),
        (950, "first-fit"),
        (1000, "pollux-patient"),
        (1000, "pollux"),
        (1000, "rack-aware"),
        (1000, "min-gpu-time"),
        (1000, "first-fit"),
    ]

    fieldnames = [
        "Scheduler",
        "Num Tasks",
        "Completion Rate",
        "Avg JCT",
        "Avg Wait",
        "Avg Slowdown",
        "Total GPU Time",
        "Makespan",
    ]

    print("=" * 60)
    print("补充运行缺失的实验数据")
    print("=" * 60)

    for n, sched in missing_experiments:
        window = n * 18.0
        print(f"\n[补充实验: {sched} - {n} tasks, Window={window:.0f}s]")

        try:
            t0 = time.time()
            metrics = run_experiment(sched, n, window)
            duration = time.time() - t0

            print(
                f"  > {sched:<15}: GPU_Time={metrics['Total GPU Time']:,.0f}, JCT={metrics['Avg JCT']:.0f}s (Sim: {duration:.1f}s)"
            )

            res = {"Scheduler": sched, "Num Tasks": n}
            res.update(metrics)

            with open(output_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(res)

        except Exception as e:
            print(f"  > {sched:<15}: FAILED ({e})")

    print("\n补充实验完成。")


if __name__ == "__main__":
    main()
