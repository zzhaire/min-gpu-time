"""
示例脚本：展示如何使用调度实验平台
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cluster import Cluster
from schedulers import (
    FirstFitScheduler, BestFitScheduler, 
    RackAwareScheduler, MinGPUTimeScheduler
)
from simulator import Simulator
from utils.task_generator import TaskGenerator
from core.task import Task
from config.config import (
    default_cluster_config, default_task_config,
    default_simulator_config, default_experiment_config
)


def example_basic_usage():
    """基本使用示例"""
    print("="*60)
    print("示例1: 基本使用")
    print("="*60)
    
    # 使用默认配置创建集群
    cluster_config = default_cluster_config
    cluster = Cluster(
        num_racks=cluster_config.num_racks,
        gpus_per_rack=cluster_config.gpus_per_rack,
        gpu_memory=cluster_config.gpu_memory,
        intra_rack_penalty=cluster_config.intra_rack_penalty,
        inter_rack_penalty=cluster_config.inter_rack_penalty
    )
    
    # 创建调度器
    scheduler = MinGPUTimeScheduler(cluster)
    
    # 生成任务
    task_config = default_task_config
    generator = TaskGenerator(seed=default_experiment_config.seed)
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
    simulator = Simulator(cluster, scheduler, default_simulator_config)
    metrics = simulator.run(tasks)
    
    # 打印结果
    metrics.print_summary()
    metrics.print_task_table()
    
    return metrics


def example_compare_schedulers():
    """对比不同调度算法"""
    print("\n" + "="*60)
    print("示例2: 对比不同调度算法")
    print("="*60)
    
    # 使用相同的任务集
    task_config = default_task_config
    generator = TaskGenerator(seed=default_experiment_config.seed)
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
    
    schedulers = {
        'First-Fit': FirstFitScheduler,
        'Best-Fit': BestFitScheduler,
        'Rack-Aware': RackAwareScheduler,
        'Min-GPU-Time': MinGPUTimeScheduler
    }
    
    results = {}
    
    for name, SchedulerClass in schedulers.items():
        print(f"\n运行 {name} 调度器...")
        
        # 创建新的集群和调度器
        cluster_config = default_cluster_config
        cluster = Cluster(
            num_racks=cluster_config.num_racks,
            gpus_per_rack=cluster_config.gpus_per_rack,
            gpu_memory=cluster_config.gpu_memory,
            intra_rack_penalty=cluster_config.intra_rack_penalty,
            inter_rack_penalty=cluster_config.inter_rack_penalty
        )
        scheduler = SchedulerClass(cluster)
        
        # 创建新任务副本（因为任务状态会被修改）
        import copy
        tasks_copy = []
        for task in tasks:
            new_task = Task(
                task_id=task.task_id,
                num_gpus=task.num_gpus,
                memory_per_gpu=task.memory_per_gpu,
                submission_time=task.submission_time,
                estimated_duration=task.estimated_duration
            )
            tasks_copy.append(new_task)
        
        # 运行模拟
        simulator = Simulator(cluster, scheduler, default_simulator_config)
        metrics = simulator.run(tasks_copy)
        
        summary = metrics.get_summary()
        results[name] = summary
        
        print(f"  - 总GPU时间: {summary['total_gpu_time']:.2f} 秒")
        print(f"  - 平均JCT: {summary['average_jct']:.2f} 秒" if summary['average_jct'] else "  - 平均JCT: N/A")
        print(f"  - 饿死任务数: {summary['starved_tasks']}")
    
    # 打印对比结果
    print("\n" + "="*60)
    print("调度算法对比结果")
    print("="*60)
    print(f"{'调度器':<15} {'总GPU时间':<15} {'平均JCT':<15} {'饿死任务':<10}")
    print("-"*60)
    for name, summary in results.items():
        avg_jct = f"{summary['average_jct']:.2f}" if summary['average_jct'] else "N/A"
        print(f"{name:<15} {summary['total_gpu_time']:<15.2f} {avg_jct:<15} {summary['starved_tasks']:<10}")


def example_custom_tasks():
    """自定义任务示例"""
    print("\n" + "="*60)
    print("示例3: 使用自定义任务")
    print("="*60)
    
    # 创建集群
    cluster_config = default_cluster_config
    cluster = Cluster(
        num_racks=cluster_config.num_racks,
        gpus_per_rack=cluster_config.gpus_per_rack,
        gpu_memory=cluster_config.gpu_memory
    )
    
    # 创建调度器
    scheduler = RackAwareScheduler(cluster)
    
    # 手动创建任务
    tasks = [
        Task(
            task_id="task-0",
            num_gpus=2,
            memory_per_gpu=8.0,
            submission_time=0.0,
            estimated_duration=200.0
        ),
        Task(
            task_id="task-1",
            num_gpus=1,
            memory_per_gpu=12.0,
            submission_time=50.0,
            estimated_duration=150.0
        ),
        Task(
            task_id="task-2",
            num_gpus=3,
            memory_per_gpu=6.0,
            submission_time=100.0,
            estimated_duration=300.0
        ),
    ]
    
    print(f"创建了 {len(tasks)} 个自定义任务")
    for task in tasks:
        print(f"  - {task.task_id}: {task.num_gpus}个GPU, "
              f"{task.memory_per_gpu}GB/GPU, "
              f"提交时间={task.submission_time}秒, "
              f"预估时间={task.estimated_duration}秒")
    
    # 运行模拟
    simulator = Simulator(cluster, scheduler, default_simulator_config)
    metrics = simulator.run(tasks)
    
    metrics.print_summary()
    metrics.print_task_table()
    
    return metrics


if __name__ == '__main__':
    # 运行示例
    example_basic_usage()
    example_compare_schedulers()
    example_custom_tasks()
