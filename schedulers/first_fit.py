"""
First-Fit调度器：按顺序分配第一个可用的GPU组合
"""
from typing import List, Dict
from core.task import Task
from .base import Scheduler


class FirstFitScheduler(Scheduler):
    """First-Fit调度器：按顺序分配第一个可用的GPU组合"""
    
    def schedule(self, pending_tasks: List[Task], current_time: float) -> Dict[str, List[str]]:
        allocations = {}
        available_gpus = self.cluster.get_available_gpus()
        
        for task in pending_tasks:
            if task.status.value != "pending":
                continue
            
            # 尝试找到足够的GPU
            allocated_gpus = []
            for gpu in available_gpus:
                if gpu.can_allocate(task.memory_per_gpu):
                    allocated_gpus.append(gpu.gpu_id)
                    if len(allocated_gpus) == task.num_gpus:
                        break
            
            if len(allocated_gpus) == task.num_gpus:
                if self.allocate(task, allocated_gpus):
                    allocations[task.task_id] = allocated_gpus
                    # 更新可用GPU列表（移除已分配的）
                    available_gpus = [g for g in available_gpus 
                                    if g.gpu_id not in allocated_gpus]
        
        return allocations

