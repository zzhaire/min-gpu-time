"""
Scalability Experiment (Run-to-Completion):
Varying number of tasks to test scheduler performance.
Unlike the Stress Test, this runs until ALL tasks are completed (or hopelessly starved).
This allows comparing the TRUE "Total GPU Time" required to process a fixed workload.
"""

import sys
import os
import csv
import time
import argparse
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


def run_experiment(scheduler_name, num_tasks, submission_window, seed_anchor_n=None):
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
    seed_n = num_tasks
    if seed_anchor_n is not None and num_tasks == 1000:
        seed_n = seed_anchor_n
    gen = TaskGenerator(seed=42 + seed_n)
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


def _load_existing_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/scalability_completion.csv")
    parser.add_argument("--only-n", type=int, default=None)
    parser.add_argument("--update-existing", action="store_true")
    parser.add_argument("--seed-anchor-n", type=int, default=None)
    args = parser.parse_args()

    output_file = args.output
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Run-to-completion Scales
    # Smaller scales than stress test because we wait for everything to finish
    task_counts = [
        50,
        100,
        150,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    ]

    if args.only_n is not None:
        task_counts = [args.only_n]

    # Fixed Density: 100 tasks per 1800s
    # So window = N * 18.0

    schedulers = ["pollux-patient", "pollux", "rack-aware", "min-gpu-time", "first-fit"]

    print("=" * 60)
    print("Starting Run-to-Completion Scalability Experiment")
    print("Metric: Total Cost to process N tasks")
    print("=" * 60)

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

    existing_rows = []
    if args.update_existing:
        existing_rows = _load_existing_rows(output_file)

    for n in task_counts:
        window = n * 18.0
        print(f"\n[Workload N={n} tasks, Window={window:.0f}s]")

        for sched in schedulers:
            try:
                t0 = time.time()
                metrics = run_experiment(
                    sched, n, window, seed_anchor_n=args.seed_anchor_n
                )
                duration = time.time() - t0

                print(
                    f"  > {sched:<15}: GPU_Time={metrics['Total GPU Time']:,.0f}, JCT={metrics['Avg JCT']:.0f}s (Sim: {duration:.1f}s)"
                )

                res = {"Scheduler": sched, "Num Tasks": n}
                res.update(metrics)

                if args.update_existing:
                    existing_rows = [
                        r
                        for r in existing_rows
                        if not (
                            str(r.get("Scheduler")) == str(sched)
                            and int(float(r.get("Num Tasks", 0))) == int(n)
                        )
                    ]
                    existing_rows.append({k: res.get(k, "") for k in fieldnames})
                else:
                    if os.path.exists(output_file):
                        rows = _load_existing_rows(output_file)
                    else:
                        rows = []
                    rows.append({k: res.get(k, "") for k in fieldnames})
                    _write_rows(output_file, fieldnames, rows)

            except Exception as e:
                print(f"  > {sched:<15}: FAILED ({e})")

    if args.update_existing:
        def _sort_key(r):
            try:
                return (str(r.get("Scheduler")), int(float(r.get("Num Tasks", 0))))
            except Exception:
                return (str(r.get("Scheduler")), 0)

        existing_rows.sort(key=_sort_key)
        _write_rows(output_file, fieldnames, existing_rows)

    print("\nRun-to-Completion experiment completed. Results saved.")


if __name__ == "__main__":
    main()
