"""
任务类：表示需要调度的任务
"""
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"  # 等待调度
    RUNNING = "running"  # 运行中
    COMPLETED = "completed"  # 已完成
    STARVED = "starved"  # 饿死（长时间等待）


@dataclass
class Task:
    """任务，包含GPU和内存需求"""
    task_id: str  # 任务唯一标识符
    num_gpus: int  # 需要的GPU数量
    memory_per_gpu: float  # 每个GPU需要的内存（GB）
    submission_time: float  # 提交时间（秒）
    estimated_duration: float  # 预估执行时间（秒）
    
    # 运行时状态
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None  # 开始执行时间
    completion_time: Optional[float] = None  # 完成时间
    allocated_gpus: List[str] = field(default_factory=list)  # 分配的GPU ID列表
    actual_duration: Optional[float] = None  # 实际执行时间
    
    def get_total_memory_required(self) -> float:
        """获取总内存需求"""
        return self.num_gpus * self.memory_per_gpu
    
    def get_jct(self) -> Optional[float]:
        """获取作业完成时间（Job Completion Time）"""
        if self.completion_time is not None and self.submission_time is not None:
            return self.completion_time - self.submission_time
        return None
    
    def get_wait_time(self) -> Optional[float]:
        """获取等待时间"""
        if self.start_time is not None:
            return self.start_time - self.submission_time
        return None
    
    def is_completed(self) -> bool:
        """检查任务是否完成"""
        return self.status == TaskStatus.COMPLETED
    
    def start(self, current_time: float, allocated_gpus: List[str]):
        """开始执行任务"""
        self.status = TaskStatus.RUNNING
        self.start_time = current_time
        self.allocated_gpus = allocated_gpus
    
    def complete(self, current_time: float):
        """完成任务"""
        self.status = TaskStatus.COMPLETED
        self.completion_time = current_time
        if self.start_time is not None:
            self.actual_duration = current_time - self.start_time
    
    def mark_starved(self):
        """标记为饿死"""
        self.status = TaskStatus.STARVED

