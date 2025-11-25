"""
任务生成器：随机生成任务
"""
from typing import List
from core.task import Task
import random


class TaskGenerator:
    """任务生成器"""
    
    def __init__(self, seed: int = None):
        """
        初始化任务生成器
        
        Args:
            seed: 随机种子
        """
        if seed is not None:
            random.seed(seed)
    
    def generate_tasks(self, num_tasks: int, 
                      min_gpus: int = 1, max_gpus: int = 8,
                      min_memory: float = 4.0, max_memory: float = 24.0,
                      min_duration: float = 100.0, max_duration: float = 3600.0,
                      submission_window: float = 3600.0) -> List[Task]:
        """
        生成随机任务
        
        Args:
            num_tasks: 任务数量
            min_gpus: 最小GPU数量
            max_gpus: 最大GPU数量
            min_memory: 每个GPU最小内存（GB）
            max_memory: 每个GPU最大内存（GB）
            min_duration: 最小执行时间（秒）
            max_duration: 最大执行时间（秒）
            submission_window: 提交时间窗口（秒），任务在此时间范围内随机提交
            
        Returns:
            任务列表
        """
        tasks = []
        
        for i in range(num_tasks):
            task_id = f"task-{i}"
            num_gpus = random.randint(min_gpus, max_gpus)
            memory_per_gpu = random.uniform(min_memory, max_memory)
            estimated_duration = random.uniform(min_duration, max_duration)
            submission_time = random.uniform(0, submission_window)
            
            task = Task(
                task_id=task_id,
                num_gpus=num_gpus,
                memory_per_gpu=memory_per_gpu,
                submission_time=submission_time,
                estimated_duration=estimated_duration
            )
            
            tasks.append(task)
        
        return tasks
    
    def generate_uniform_tasks(self, num_tasks: int,
                              num_gpus: int, memory_per_gpu: float,
                              duration: float,
                              submission_window: float = 3600.0) -> List[Task]:
        """
        生成统一配置的任务
        
        Args:
            num_tasks: 任务数量
            num_gpus: GPU数量
            memory_per_gpu: 每个GPU内存（GB）
            duration: 执行时间（秒）
            submission_window: 提交时间窗口（秒）
            
        Returns:
            任务列表
        """
        tasks = []
        
        for i in range(num_tasks):
            task_id = f"task-{i}"
            submission_time = random.uniform(0, submission_window)
            
            task = Task(
                task_id=task_id,
                num_gpus=num_gpus,
                memory_per_gpu=memory_per_gpu,
                submission_time=submission_time,
                estimated_duration=duration
            )
            
            tasks.append(task)
        
        return tasks

