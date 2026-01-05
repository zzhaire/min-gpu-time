"""
Pareto Frontier 实验脚本：权衡成本与速度
运行不同 alpha 值的 Pollux Patient 以及各个 Baseline，收集 (Cost, JCT) 数据点。
"""

import sys
import os
import csv
from copy import deepcopy

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    default_cluster_config,
    default_task_config,
    default_simulator_config,
    default_experiment_config,
    default_scheduler_config,
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


def run_experiment(scheduler_name, tasks, alpha=None, patience=None):
    """运行单次实验"""
    print(f"Running {scheduler_name} (alpha={alpha}, patience={patience})...")

    # 1. 配置环境
    cluster_config = default_cluster_config
    simulator_config = deepcopy(default_simulator_config)

    # 2. 创建集群
    cluster = Cluster(
        num_racks=cluster_config.num_racks,
        gpus_per_rack=cluster_config.gpus_per_rack,
        gpu_memory=cluster_config.gpu_memory,
        intra_rack_penalty=cluster_config.intra_rack_penalty,
        inter_rack_penalty=cluster_config.inter_rack_penalty,
    )

    # 3. 创建调度器
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
        # Pollux standard (alpha affects trade-off)
        a = alpha if alpha is not None else 0.5
        scheduler = PolluxScheduler(cluster, alpha=a)
    elif scheduler_name == "pollux-patient":
        # Pollux Patient
        # Alpha 0.5 (Balanced)
        a = alpha if alpha is not None else 0.5
        p = patience if patience is not None else 1.1
        scheduler = PolluxPatientScheduler(
            cluster,
            alpha=a,
            patience_threshold=p,
            starvation_limit=2000.0,
        )
        scheduler.sharing_penalty_map = simulator_config.sharing_penalty_map
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # 4. 运行模拟
    tasks_copy = deepcopy(tasks)
    simulator = Simulator(cluster, scheduler, simulator_config)
    metrics = simulator.run(tasks_copy)

    # 收集关键指标
    # Total GPU Time (Cost)
    total_gpu_time = metrics.total_gpu_time
    # Avg JCT (Speed)
    summary = metrics.get_summary()
    avg_jct = summary["average_jct"]

    return total_gpu_time, avg_jct


def main():
    output_file = "results/pareto_frontier.csv"

    # 生成任务 (固定种子)
    task_config = default_task_config
    generator = TaskGenerator(seed=42)
    tasks = generator.generate_tasks(
        num_tasks=task_config.num_tasks,
        min_gpus=task_config.min_gpus,
        max_gpus=task_config.max_gpus,
        min_memory=task_config.min_memory,
        max_memory=task_config.max_memory,
        min_duration=task_config.min_duration,
        max_duration=task_config.max_duration,
        submission_window=task_config.submission_window,
    )

    results = []

    print("=" * 60)
    print("开始 Pareto Frontier 实验：成本 vs 速度")
    print("=" * 60)

    # 1. Baselines
    baselines = ["rack-aware", "min-gpu-time", "first-fit", "pollux"]
    for sched in baselines:
        cost, speed = run_experiment(sched, tasks)
        results.append(
            {
                "Scheduler": sched,
                "Param": "N/A",
                "Total GPU Time": cost,
                "Avg JCT": speed,
                "Type": "Baseline",
            }
        )

    # 2. Pollux Patient (Varying Patience)
    # Vary Patience Threshold
    patience_levels = [1.0, 1.2, 1.5, 2.2, 100.0]
    for p in patience_levels:
        cost, speed = run_experiment("pollux-patient", tasks, alpha=0.5, patience=p)
        results.append(
            {
                "Scheduler": "pollux-patient",
                "Param": f"P={p}",
                "Total GPU Time": cost,
                "Avg JCT": speed,
                "Type": "Ours",
            }
        )

    # 保存结果
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Scheduler", "Param", "Total GPU Time", "Avg JCT", "Type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 60)
    print(f"实验完成！结果已保存至 {output_file}")
    print("=" * 60)

    # 打印简表
    print(
        f"{'Scheduler':<20} {'Param':<10} {'Cost (GPU-s)':<15} {'Speed (Avg JCT)':<15}"
    )
    print("-" * 65)
    for r in results:
        param_str = str(r["Param"])
        print(
            f"{r['Scheduler']:<20} {param_str:<10} {r['Total GPU Time']:<15.2f} {r['Avg JCT']:<15.2f}"
        )


if __name__ == "__main__":
    main()
