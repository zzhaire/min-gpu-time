"""
GPU类：表示单个GPU，支持多任务共享
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class GPU:
    """单个GPU，支持多任务共享"""
    gpu_id: str  # GPU的唯一标识符，格式：rack_id-gpu_index
    rack_id: str  # 所属机架ID
    total_memory: float  # 总显存大小（GB）
    used_memory: float = 0.0  # 已使用的显存（GB）
    running_tasks: List[str] = field(default_factory=list)  # 正在运行的任务ID列表
    total_time: float = 0.0  # 累计运行时间（秒）
    
    def can_allocate(self, memory_required: float) -> bool:
        """检查是否可以分配指定大小的显存"""
        return (self.used_memory + memory_required) <= self.total_memory
    
    def allocate(self, task_id: str, memory_required: float) -> bool:
        """分配显存给任务"""
        if not self.can_allocate(memory_required):
            return False
        self.used_memory += memory_required
        if task_id not in self.running_tasks:
            self.running_tasks.append(task_id)
        return True
    
    def deallocate(self, task_id: str, memory_required: float):
        """释放任务占用的显存"""
        if task_id in self.running_tasks:
            self.running_tasks.remove(task_id)
        self.used_memory = max(0.0, self.used_memory - memory_required)
    
    def is_idle(self) -> bool:
        """检查GPU是否空闲"""
        return len(self.running_tasks) == 0
    
    def get_available_memory(self) -> float:
        """获取可用显存"""
        return self.total_memory - self.used_memory
    
    def add_time(self, seconds: float):
        """累加GPU运行时间"""
        if len(self.running_tasks) > 0:
            self.total_time += seconds
    
    def get_utilization(self) -> float:
        """获取GPU利用率（基于显存）"""
        return self.used_memory / self.total_memory if self.total_memory > 0 else 0.0

