"""
主运行脚本：运行调度实验
"""
from config.config import (
    ClusterConfig, TaskConfig, SimulatorConfig, ExperimentConfig,
    default_cluster_config, default_task_config,
    default_simulator_config, default_experiment_config,
    default_scheduler_config
)
from utils.task_generator import TaskGenerator
from simulator import Simulator
from schedulers import (
    FirstFitScheduler, BestFitScheduler,
    RackAwareScheduler, MinGPUTimeScheduler, PolluxScheduler,
    PolluxPatientScheduler
)
from core.cluster import Cluster
import argparse
import sys
import os
import csv
import glob
from utils.plotter import Plotter

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_scheduler(scheduler_name: str, cluster: Cluster, scheduler_config=None):
    """根据名称获取调度器"""
    schedulers = {
        'first-fit': FirstFitScheduler,
        'best-fit': BestFitScheduler,
        'rack-aware': RackAwareScheduler,
        'min-gpu-time': MinGPUTimeScheduler,
        'pollux': PolluxScheduler,
        'pollux-patient': PolluxPatientScheduler
    }

    if scheduler_name not in schedulers:
        raise ValueError(
            f"未知的调度器: {scheduler_name}. 可选: {list(schedulers.keys())}")

    if scheduler_name == 'min-gpu-time':
        cfg = getattr(scheduler_config, 'min_gpu_time', None) if scheduler_config else None
        patience = cfg.patience_threshold if cfg else 1.1
        starvation_limit = cfg.starvation_limit if cfg else 2000.0
        return MinGPUTimeScheduler(cluster,
                                   patience_threshold=patience,
                                   starvation_limit=starvation_limit)
    
    if scheduler_name == 'pollux':
        cfg = getattr(scheduler_config, 'pollux', None) if scheduler_config else None
        alpha = cfg.alpha if cfg else 0.5
        # 兼容性：pollux 原生版不需要 patience
        return PolluxScheduler(cluster, alpha=alpha)
    
    if scheduler_name == 'pollux-patient':
        cfg = getattr(scheduler_config, 'pollux', None) if scheduler_config else None
        alpha = cfg.alpha if cfg else 0.5
        patience = cfg.patience_threshold if cfg else 1.1
        starvation_limit = cfg.starvation_limit if cfg else 2000.0
        return PolluxPatientScheduler(cluster, alpha=alpha,
                                      patience_threshold=patience,
                                      starvation_limit=starvation_limit)

    return schedulers[scheduler_name](cluster)


def summarize_results(output_dir: str):
    """
    汇总所有调度算法的实验结果
    """
    summary_files = glob.glob(os.path.join(output_dir, "summary_*.csv"))
    results = []

    for summary_file in summary_files:
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = {row[0]: row[1] for row in reader if len(row) >= 2}
                
                results.append({
                    "Scheduler": rows.get("调度器", "Unknown"),
                    "Total Tasks": rows.get("总任务数", "0"),
                    "Completed": rows.get("完成任务数", "0"),
                    "Starved": rows.get("饿死任务数", "0"),
                    "Total GPU Time (s)": rows.get("总GPU时间(秒)", "0"),
                    "Avg JCT (s)": rows.get("平均JCT(秒)", "N/A"),
                    "Avg Wait (s)": rows.get("平均等待时间(秒)", "N/A")
                })
        except Exception as e:
            print(f"读取文件 {summary_file} 失败: {e}")

    # 排序（为了表格美观）
    results.sort(key=lambda x: x["Scheduler"])

    # 打印控制台表格
    print("\n" + "="*100)
    print(f"{'调度算法对比汇总':^100}")
    print("="*100)
    
    header = f"{'Scheduler':<20} {'Tasks':<10} {'Done':<10} {'Starved':<10} {'GPU Time':<15} {'Avg JCT':<15} {'Avg Wait':<15}"
    print(header)
    print("-" * 100)
    
    for res in results:
        print(f"{res['Scheduler']:<20} {res['Total Tasks']:<10} {res['Completed']:<10} "
              f"{res['Starved']:<10} {res['Total GPU Time (s)']:<15} "
              f"{res['Avg JCT (s)']:<15} {res['Avg Wait (s)']:<15}")
    
    print("="*100)

    # 保存对比 CSV
    comparison_file = os.path.join(output_dir, "comparison.csv")
    if results:
        fieldnames = ["Scheduler", "Total Tasks", "Completed", "Starved", 
                      "Total GPU Time (s)", "Avg JCT (s)", "Avg Wait (s)"]
        with open(comparison_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n对比结果已保存到: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description='最小化GPU时间调度算法实验平台')

    # 调度器选择
    parser.add_argument('--scheduler', type=str, default='min-gpu-time',
                        choices=['first-fit', 'best-fit',
                                 'rack-aware', 'min-gpu-time', 'pollux', 'pollux-patient'],
                        help='调度算法')

    # 运行模式
    parser.add_argument('--run-all', action='store_true',
                        help='运行所有调度算法并对比')
    
    # 绘图
    parser.add_argument('--plot', action='store_true',
                        help='实验结束后绘制时空图')

    args = parser.parse_args()

    # 从配置文件读取参数
    cluster_config = default_cluster_config
    task_config = default_task_config
    simulator_config = default_simulator_config
    experiment_config = default_experiment_config
    scheduler_config = default_scheduler_config

    print("="*60)
    print("实验配置")
    print("="*60)
    print(f"集群: {cluster_config.num_racks}机架 x {cluster_config.gpus_per_rack}GPU, "
          f"每GPU {cluster_config.gpu_memory}GB")
    print(f"任务: {task_config.num_tasks}个, "
          f"GPU范围[{task_config.min_gpus}, {task_config.max_gpus}], "
          f"内存范围[{task_config.min_memory}, {task_config.max_memory}]GB")
    print("="*60)

    # 初始化绘图工具（如果需要）
    plotter = Plotter(experiment_config.output_dir) if args.plot else None

    if args.run_all:
        # 运行所有调度算法
        schedulers_to_test = ['first-fit',
                              'best-fit', 'rack-aware', 'min-gpu-time', 'pollux', 'pollux-patient']
        print(f"\n将运行所有调度算法: {schedulers_to_test}\n")

        for scheduler_name in schedulers_to_test:
            print(f"\n{'='*60}")
            print(f"运行调度器: {scheduler_name}")
            print(f"{'='*60}\n")

            # 创建集群
            cluster = Cluster(
                num_racks=cluster_config.num_racks,
                gpus_per_rack=cluster_config.gpus_per_rack,
                gpu_memory=cluster_config.gpu_memory,
                intra_rack_penalty=cluster_config.intra_rack_penalty,
                inter_rack_penalty=cluster_config.inter_rack_penalty
            )

            # 创建调度器
            scheduler = get_scheduler(scheduler_name, cluster, scheduler_config)

            # 生成任务（使用相同种子确保一致性）
            generator = TaskGenerator(seed=experiment_config.seed)
            tasks = generator.generate_tasks(
                num_tasks=task_config.num_tasks,
                min_gpus=task_config.min_gpus,
                max_gpus=task_config.max_gpus,
                min_memory=task_config.min_memory,
                max_memory=task_config.max_memory,
                min_duration=task_config.min_duration,
                max_duration=task_config.max_duration,
                submission_window=task_config.submission_window
            )

            # 创建模拟器并运行
            simulator = Simulator(cluster, scheduler, simulator_config)
            metrics = simulator.run(tasks)

            # 打印结果
            metrics.print_summary()

            # 保存到表格
            scheduler_tag = scheduler_name.replace('-', '_')
            metrics.save_to_tables(
                experiment_config.output_dir,
                scheduler_tag
            )
            
            # 绘图
            if plotter:
                tasks_file = os.path.join(experiment_config.output_dir, f"tasks_{scheduler_tag}.csv")
                plotter.plot_gantt_chart(tasks_file, f"gantt_{scheduler_tag}.png", scheduler_name)
        
        # 汇总所有结果
        summarize_results(experiment_config.output_dir)
                
    else:
        # 运行单个调度算法
        print(f"\n运行调度器: {args.scheduler}\n")

        # 创建集群
        cluster = Cluster(
            num_racks=cluster_config.num_racks,
            gpus_per_rack=cluster_config.gpus_per_rack,
            gpu_memory=cluster_config.gpu_memory,
            intra_rack_penalty=cluster_config.intra_rack_penalty,
            inter_rack_penalty=cluster_config.inter_rack_penalty
        )

        # 创建调度器
        scheduler = get_scheduler(args.scheduler, cluster, scheduler_config)

        # 生成任务
        generator = TaskGenerator(seed=experiment_config.seed)
        tasks = generator.generate_tasks(
            num_tasks=task_config.num_tasks,
            min_gpus=task_config.min_gpus,
            max_gpus=task_config.max_gpus,
            min_memory=task_config.min_memory,
            max_memory=task_config.max_memory,
            min_duration=task_config.min_duration,
            max_duration=task_config.max_duration,
            submission_window=task_config.submission_window
        )

        # 创建模拟器并运行
        simulator = Simulator(cluster, scheduler, simulator_config)
        metrics = simulator.run(tasks)

        # 打印结果
        metrics.print_summary()
        metrics.print_task_table()

        # 保存到表格
        scheduler_tag = args.scheduler.replace('-', '_')
        metrics.save_to_tables(
            experiment_config.output_dir,
            scheduler_tag
        )
        
        # 绘图
        if plotter:
            tasks_file = os.path.join(experiment_config.output_dir, f"tasks_{scheduler_tag}.csv")
            plotter.plot_gantt_chart(tasks_file, f"gantt_{scheduler_tag}.png", args.scheduler)

    print("\n实验完成！")


if __name__ == '__main__':
    main()
