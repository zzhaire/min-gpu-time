"""
机架感知调度器：优先在同一机架内分配，减少跨机架惩罚
"""

from typing import List, Dict
from core.task import Task
from .base import Scheduler


class RackAwareScheduler(Scheduler):
    """机架感知调度器：优先在同一机架内分配，减少跨机架惩罚"""

    def schedule(
        self, pending_tasks: List[Task], current_time: float
    ) -> Dict[str, List[str]]:
        allocations = {}

        # 按任务大小排序（小任务优先）
        sorted_tasks = sorted(pending_tasks, key=lambda t: t.num_gpus)

        for task in sorted_tasks:
            if task.status.value != "pending":
                continue

            # 优先尝试在同一机架内分配
            best_allocation = None
            best_penalty = float("inf")

            # 遍历所有机架
            for rack in self.cluster.racks:
                available_gpus_in_rack = [
                    gpu
                    for gpu in rack.get_available_gpus()
                    if gpu.gpu_id
                    not in [a for alloc in allocations.values() for a in alloc]
                ]

                if len(available_gpus_in_rack) >= task.num_gpus:
                    # 检查是否可以分配
                    candidate_gpus = []
                    for gpu in available_gpus_in_rack:
                        if gpu.can_allocate(task.memory_per_gpu):
                            candidate_gpus.append(gpu.gpu_id)
                            if len(candidate_gpus) == task.num_gpus:
                                break

                    if len(candidate_gpus) == task.num_gpus:
                        penalty = self.cluster.calculate_penalty(candidate_gpus)
                        if penalty < best_penalty:
                            best_penalty = penalty
                            best_allocation = candidate_gpus

            # 如果同一机架内无法分配，尝试跨机架
            if best_allocation is None:
                available_gpus = self.cluster.get_available_gpus()
                available_gpus = [
                    g
                    for g in available_gpus
                    if g.gpu_id
                    not in [a for alloc in allocations.values() for a in alloc]
                ]

                candidate_gpus = []
                for gpu in available_gpus:
                    if gpu.can_allocate(task.memory_per_gpu):
                        candidate_gpus.append(gpu.gpu_id)
                        if len(candidate_gpus) == task.num_gpus:
                            break

                if len(candidate_gpus) == task.num_gpus:
                    best_allocation = candidate_gpus

            if best_allocation and self.allocate(task, best_allocation):
                allocations[task.task_id] = best_allocation

        return allocations
