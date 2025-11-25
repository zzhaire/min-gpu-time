"""
Best-Fit调度器：选择显存利用率最高的GPU组合
"""
from typing import List, Dict
from core.task import Task
from .base import Scheduler


class BestFitScheduler(Scheduler):
    """Best-Fit调度器：选择显存利用率最高的GPU组合"""
    
    def schedule(self, pending_tasks: List[Task], current_time: float) -> Dict[str, List[str]]:
        allocations = {}
        available_gpus = self.cluster.get_available_gpus()
        
        # 按利用率排序（从高到低）
        sorted_gpus = sorted(available_gpus, 
                           key=lambda g: g.get_utilization(), 
                           reverse=True)
        
        for task in pending_tasks:
            if task.status.value != "pending":
                continue
            
            # 尝试找到足够的GPU
            allocated_gpus = []
            for gpu in sorted_gpus:
                if gpu.gpu_id in [a for alloc in allocations.values() for a in alloc]:
                    continue  # GPU已被分配
                if gpu.can_allocate(task.memory_per_gpu):
                    allocated_gpus.append(gpu.gpu_id)
                    if len(allocated_gpus) == task.num_gpus:
                        break
            
            if len(allocated_gpus) == task.num_gpus:
                if self.allocate(task, allocated_gpus):
                    allocations[task.task_id] = allocated_gpus
        
        return allocations

