"""
Pollux调度器：自适应资源调整
核心思想：不再固定任务的 GPU 数量，而是根据当前的集群状态，
在 min_gpus 和 max_gpus 之间动态选择一个“性价比”最高的数量。
策略：
1. 搜索所有可能的 GPU 数量 n (min <= n <= max)。
2. 对每个 n，找到当前最佳的放置位置（最小惩罚）。
3. 选择一个 n，使得 (Speedup^alpha / Cost) 最大化。
   或者简单策略：在惩罚系数较低（如同机架）的前提下，尽可能多地分配 GPU。
"""

from typing import List, Dict
from core.task import Task
from .base import Scheduler


class PolluxScheduler(Scheduler):
    """
    Pollux 自适应调度器 (原生版：无等待)
    """

    def __init__(self, cluster, alpha=0.5, **kwargs):
        super().__init__(cluster)
        self.alpha = alpha  # 0.0 = 纯 MinCost, 1.0 = 纯 MaxSpeed
        # 兼容性参数，这里忽略 patience_threshold 和 starvation_limit

    def allocate(self, task: Task, gpu_ids: List[str]) -> bool:
        """
        重写分配方法，允许动态 GPU 数量
        (不检查 len(gpu_ids) == task.num_gpus)
        """
        # 1. 检查资源
        for gpu_id in gpu_ids:
            gpu = self.cluster.get_gpu(gpu_id)
            if gpu is None or not gpu.can_allocate(task.memory_per_gpu):
                return False

        # 2. 执行分配
        for gpu_id in gpu_ids:
            gpu = self.cluster.get_gpu(gpu_id)
            gpu.allocate(task.task_id, task.memory_per_gpu)

        return True

    def schedule(
        self, pending_tasks: List[Task], current_time: float
    ) -> Dict[str, List[str]]:
        allocations = {}
        occupied_gpus = set()

        for task in pending_tasks:
            if task.status.value != "pending":
                continue

            max_n = task.num_gpus
            min_n = 1

            best_n = -1
            best_allocation = None
            best_score = -float("inf")

            available_gpus = [
                g
                for g in self.cluster.get_all_gpus()
                if g.gpu_id not in occupied_gpus and g.can_allocate(task.memory_per_gpu)
            ]

            if len(available_gpus) < min_n:
                continue

            candidates_n = range(min_n, min(max_n, len(available_gpus)) + 1)

            for n in candidates_n:
                # 寻找 n 个 GPU 的最佳放置
                current_allocation = []
                current_penalty = float("inf")

                # 1. 尝试单机架
                found_rack = False
                for rack in self.cluster.racks:
                    rack_free = [
                        g.gpu_id
                        for g in rack.get_available_gpus()
                        if g.gpu_id not in occupied_gpus
                        and g.can_allocate(task.memory_per_gpu)
                    ]
                    if len(rack_free) >= n:
                        current_allocation = rack_free[:n]
                        current_penalty = self.cluster.calculate_penalty(
                            current_allocation
                        )
                        found_rack = True
                        break

                if not found_rack:
                    # 2. 跨机架
                    current_allocation = [g.gpu_id for g in available_gpus[:n]]
                    current_penalty = self.cluster.calculate_penalty(current_allocation)

                # 计算 Score
                current_penalty = max(1.0, current_penalty)
                score = (n**self.alpha) / current_penalty

                if score > best_score:
                    best_score = score
                    best_allocation = current_allocation
                    best_n = n

            # 决策逻辑：原生 Pollux 是贪心的，只要有可用资源且分数最大，就立即分配
            # 不进行等待
            if best_allocation:
                if self.allocate(task, best_allocation):
                    allocations[task.task_id] = best_allocation
                    occupied_gpus.update(best_allocation)

        return allocations
