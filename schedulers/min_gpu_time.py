"""
最小化GPU时间调度器：考虑惩罚系数，最小化总GPU时间
"""
from typing import List, Dict
from core.task import Task
from .base import Scheduler


class MinGPUTimeScheduler(Scheduler):
    """最小化GPU时间调度器：考虑惩罚系数，最小化总GPU时间"""
    
    def schedule(self, pending_tasks: List[Task], current_time: float) -> Dict[str, List[str]]:
        """
        最小化GPU时间的调度策略
        目标：最小化 sum(gpu_time * penalty)
        """
        allocations = {}
        
        # 按任务大小和预估时间排序（小任务、短时间优先）
        sorted_tasks = sorted(pending_tasks, 
                            key=lambda t: (t.num_gpus, t.estimated_duration))
        
        for task in sorted_tasks:
            if task.status.value != "pending":
                continue
            
            # 找到最优的GPU分配
            best_allocation = None
            best_cost = float('inf')
            
            available_gpus = [g for g in self.cluster.get_available_gpus()
                            if g.gpu_id not in [a for alloc in allocations.values() for a in alloc]]
            
            # 尝试所有可能的GPU组合（贪心策略：优先选择利用率高的）
            if len(available_gpus) >= task.num_gpus:
                # 按利用率排序
                sorted_available = sorted(available_gpus,
                                        key=lambda g: g.get_utilization(),
                                        reverse=True)
                
                candidate_gpus = []
                for gpu in sorted_available:
                    if gpu.can_allocate(task.memory_per_gpu):
                        candidate_gpus.append(gpu.gpu_id)
                        if len(candidate_gpus) == task.num_gpus:
                            break
                
                if len(candidate_gpus) == task.num_gpus:
                    # 计算成本：考虑惩罚系数
                    penalty = self.cluster.calculate_penalty(candidate_gpus)
                    # 成本 = 预估时间 * 惩罚系数
                    cost = task.estimated_duration * penalty
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_allocation = candidate_gpus
            
            if best_allocation and self.allocate(task, best_allocation):
                allocations[task.task_id] = best_allocation
        
        return allocations

