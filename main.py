"""
主运行脚本：运行调度实验
"""
import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cluster import Cluster
from schedulers import (
    FirstFitScheduler, BestFitScheduler, 
    RackAwareScheduler, MinGPUTimeScheduler
)
from simulator import Simulator
from utils.task_generator import TaskGenerator
from config.config import (
    ClusterConfig, TaskConfig, SimulatorConfig, ExperimentConfig,
    default_cluster_config, default_task_config, 
    default_simulator_config, default_experiment_config
)


def get_scheduler(scheduler_name: str, cluster: Cluster):
    """根据名称获取调度器"""
    schedulers = {
        'first-fit': FirstFitScheduler,
        'best-fit': BestFitScheduler,
        'rack-aware': RackAwareScheduler,
        'min-gpu-time': MinGPUTimeScheduler
    }
    
    if scheduler_name not in schedulers:
        raise ValueError(f"未知的调度器: {scheduler_name}. 可选: {list(schedulers.keys())}")
    
    return schedulers[scheduler_name](cluster)


def main():
    parser = argparse.ArgumentParser(description='最小化GPU时间调度算法实验平台')
    
    # 调度器选择
    parser.add_argument('--scheduler', type=str, default='min-gpu-time',
                       choices=['first-fit', 'best-fit', 'rack-aware', 'min-gpu-time'],
                       help='调度算法')
    
    # 运行模式
    parser.add_argument('--run-all', action='store_true',
                       help='运行所有调度算法并对比')
    
    args = parser.parse_args()
    
    # 从配置文件读取参数
    cluster_config = default_cluster_config
    task_config = default_task_config
    simulator_config = default_simulator_config
    experiment_config = default_experiment_config
    
    print("="*60)
    print("实验配置")
    print("="*60)
    print(f"集群: {cluster_config.num_racks}机架 x {cluster_config.gpus_per_rack}GPU, "
          f"每GPU {cluster_config.gpu_memory}GB")
    print(f"任务: {task_config.num_tasks}个, "
          f"GPU范围[{task_config.min_gpus}, {task_config.max_gpus}], "
          f"内存范围[{task_config.min_memory}, {task_config.max_memory}]GB")
    print("="*60)
    
    if args.run_all:
        # 运行所有调度算法
        schedulers_to_test = ['first-fit', 'best-fit', 'rack-aware', 'min-gpu-time']
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
            scheduler = get_scheduler(scheduler_name, cluster)
            
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
            metrics.save_to_tables(
                experiment_config.output_dir, 
                scheduler_name.replace('-', '_')
            )
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
        scheduler = get_scheduler(args.scheduler, cluster)
        
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
        metrics.save_to_tables(
            experiment_config.output_dir, 
            args.scheduler.replace('-', '_')
        )
    
    print("\n实验完成！")


if __name__ == '__main__':
    main()
