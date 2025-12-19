"""
最小化GPU时间调度器：耐心策略
核心思想：如果当前的分配方案导致惩罚系数太高（比如跨机架），
且任务没有面临饿死风险，则推迟调度，等待更好的资源碎片合并。
"""
from typing import List, Dict
from core.task import Task
from .base import Scheduler


class MinGPUTimeScheduler(Scheduler):
    """
    最小化GPU时间调度器：耐心策略
    """

    def __init__(self, cluster, patience_threshold=1.1, starvation_limit=2000.0):
        super().__init__(cluster)
        self.patience_threshold = patience_threshold
        # 必须小于模拟器 starvation_threshold，确保任务在被“饿死”前还能被强制执行
        self.starvation_limit = starvation_limit

    def schedule(self, pending_tasks: List[Task], current_time: float) -> Dict[str, List[str]]:
        allocations = {}

        for task in pending_tasks:
            if task.status.value != "pending":
                continue

            # 1. 寻找当前最佳分配方案
            best_allocation = None
            min_penalty = float('inf')

            # --- 策略：先看全机架（优先尝试机架内分配） ---
            for rack in self.cluster.racks:
                rack_free_gpus = [g.gpu_id for g in rack.get_available_gpus()
                                  if g.gpu_id not in [a for alloc in allocations.values() for a in alloc]]

                valid_rack_gpus = []
                for gid in rack_free_gpus:
                    gpu = self.cluster.get_gpu(gid)
                    if gpu.can_allocate(task.memory_per_gpu):
                        valid_rack_gpus.append(gid)

                if len(valid_rack_gpus) >= task.num_gpus:
                    candidate = valid_rack_gpus[:task.num_gpus]
                    penalty = self.cluster.calculate_penalty(candidate)
                    if penalty < min_penalty:
                        min_penalty = penalty
                        best_allocation = candidate

            # --- 策略：如果没有单机架，看全局混合 (跨机架) ---
            if best_allocation is None:
                all_free_gpus = [g for g in self.cluster.get_available_gpus()
                                 if g.gpu_id not in [a for alloc in allocations.values() for a in alloc]]

                valid_global_gpus = []
                for gpu in all_free_gpus:
                    if gpu.can_allocate(task.memory_per_gpu):
                        valid_global_gpus.append(gpu.gpu_id)

                if len(valid_global_gpus) >= task.num_gpus:
                    candidate = valid_global_gpus[:task.num_gpus]
                    penalty = self.cluster.calculate_penalty(candidate)
                    if penalty < min_penalty:
                        min_penalty = penalty
                        best_allocation = candidate

            # 2. 决策逻辑
            if best_allocation:
                wait_time = current_time - task.submission_time

                is_good_placement = min_penalty <= self.patience_threshold
                is_starving = wait_time > self.starvation_limit

                if is_good_placement or is_starving:
                    if self.allocate(task, best_allocation):
                        allocations[task.task_id] = best_allocation
                else:
                    # 忍耐：虽然有资源但位置不好，且没饿死，选择等待
                    pass

        return allocations
