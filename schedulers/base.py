"""
调度器基类
"""

from abc import ABC, abstractmethod
from typing import List, Dict
from core.cluster import Cluster
from core.task import Task


class Scheduler(ABC):
    """调度器基类"""

    def __init__(self, cluster: Cluster):
        self.cluster = cluster

    @abstractmethod
    def schedule(
        self, pending_tasks: List[Task], current_time: float
    ) -> Dict[str, List[str]]:
        """
        调度任务

        Args:
            pending_tasks: 待调度的任务列表
            current_time: 当前时间

        Returns:
            分配结果：{task_id: [gpu_id1, gpu_id2, ...]}
        """
        pass

    def can_allocate(self, task: Task, gpu_ids: List[str]) -> bool:
        """检查是否可以在指定GPU上分配任务"""
        if len(gpu_ids) != task.num_gpus:
            return False

        for gpu_id in gpu_ids:
            gpu = self.cluster.get_gpu(gpu_id)
            if gpu is None or not gpu.can_allocate(task.memory_per_gpu):
                return False

        return True

    def allocate(self, task: Task, gpu_ids: List[str]) -> bool:
        """在指定GPU上分配任务"""
        if not self.can_allocate(task, gpu_ids):
            return False

        for gpu_id in gpu_ids:
            gpu = self.cluster.get_gpu(gpu_id)
            gpu.allocate(task.task_id, task.memory_per_gpu)

        return True

    def deallocate(self, task: Task):
        """释放任务占用的GPU资源"""
        for gpu_id in task.allocated_gpus:
            gpu = self.cluster.get_gpu(gpu_id)
            if gpu:
                gpu.deallocate(task.task_id, task.memory_per_gpu)
