"""
敏感性分析脚本：干扰强度 (Sensitivity to Interference)
运行 First-Fit (Baseline) 和 Pollux-Patient 在不同干扰强度下的表现。
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


def get_sharing_map(level: str):
    """
    根据干扰强度返回共享惩罚表 (Efficiency)
    """
    if level == "Low":
        # 计算密集型: 干扰很小
        return {1: 1.0, 2: 0.95, 3: 0.90, 4: 0.85, 5: 0.80}
    elif level == "Medium":
        # 默认场景: 适中干扰
        return {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.40, 5: 0.20}
    elif level == "High":
        # 带宽密集型: 干扰巨大
        return {1: 1.0, 2: 0.60, 3: 0.40, 4: 0.25, 5: 0.10}
    else:
        raise ValueError(f"Unknown level: {level}")


def run_experiment(interference_level, scheduler_name, tasks):
    """运行单次实验"""
    print(f"Running {scheduler_name} under {interference_level} Interference...")

    # 1. 配置环境
    cluster_config = default_cluster_config
    simulator_config = deepcopy(default_simulator_config)

    # 注入干扰配置
    simulator_config.sharing_penalty_map = get_sharing_map(interference_level)

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
        # 标准 Pollux (alpha=0.5, 追求平衡)
        scheduler = PolluxScheduler(cluster, alpha=0.5)
    elif scheduler_name == "pollux-patient":
        # 优化版 Pollux (alpha=0.0, 追求 Min-GPU-Time)
        scheduler = PolluxPatientScheduler(
            cluster, alpha=0.0, patience_threshold=1.1, starvation_limit=2000.0
        )
        # 关键：手动更新调度器内部的 penalty_map，使其与 simulator 一致
        scheduler.sharing_penalty_map = simulator_config.sharing_penalty_map
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # 4. 运行模拟
    # 深拷贝任务列表，防止状态污染
    tasks_copy = deepcopy(tasks)
    simulator = Simulator(cluster, scheduler, simulator_config)
    metrics = simulator.run(tasks_copy)

    return metrics.total_gpu_time


def main():
    # 实验设置
    levels = ["Low", "Medium", "High"]
    schedulers = [
        "first-fit",
        "best-fit",
        "rack-aware",
        "min-gpu-time",
        "pollux",
        "pollux-patient",
    ]
    output_file = "results/sensitivity_interference.csv"

    # 生成任务 (固定种子，保证公平对比)
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
    print("开始敏感性分析：干扰强度 (Sensitivity to Interference)")
    print("=" * 60)

    for level in levels:
        baseline_time = 0
        for scheduler in schedulers:
            gpu_time = run_experiment(level, scheduler, tasks)

            # 记录结果
            res = {
                "Interference Level": level,
                "Scheduler": scheduler,
                "Total GPU Time": gpu_time,
                "Normalized Time": 1.0,  # 稍后计算
            }

            if scheduler == "first-fit":
                baseline_time = gpu_time
            else:
                # 计算归一化 (相对于 Baseline)
                res["Normalized Time"] = (
                    gpu_time / baseline_time if baseline_time > 0 else 0
                )

            results.append(res)
            print(
                f"  -> Result: {gpu_time:.2f}s (Normalized: {res['Normalized Time']:.4f})\n"
            )

    # 保存结果 (使用 csv 模块)
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Interference Level",
            "Scheduler",
            "Total GPU Time",
            "Normalized Time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 60)
    print(f"实验完成！结果已保存至 {output_file}")
    print("=" * 60)

    # 打印表格
    print(
        f"{'Interference':<15} {'Scheduler':<20} {'Total GPU Time':<15} {'Normalized':<10}"
    )
    print("-" * 65)
    for r in results:
        print(
            f"{r['Interference Level']:<15} {r['Scheduler']:<20} {r['Total GPU Time']:<15.2f} {r['Normalized Time']:<10.4f}"
        )


if __name__ == "__main__":
    main()
