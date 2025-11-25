"""
配置文件：定义集群环境参数和实验参数
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class ClusterConfig:
    """集群环境配置（固定参数）"""
    num_racks: int = 5              # 机架数量
    gpus_per_rack: int = 8          # 每个机架的GPU数量
    gpu_memory: float = 80.0        # 每个GPU的显存大小（GB）
    intra_rack_penalty: float = 1.4  # 同一机架内GPU惩罚系数
    inter_rack_penalty: float = 2.1  # 跨机架GPU惩罚系数


@dataclass
class TaskConfig:
    """任务生成配置"""
    num_tasks: int = 50              # 任务数量
    min_gpus: int = 1               # 最小GPU数量
    max_gpus: int = 16               # 最大GPU数量
    min_memory: float = 2.0          # 每个GPU最小内存（GB）
    max_memory: float = 60.0         # 每个GPU最大内存（GB）
    min_duration: float = 10.0      # 最小执行时间（秒）
    max_duration: float = 1800.0     # 最大执行时间（秒）
    submission_window: float = 1800.0  # 提交时间窗口（秒）


@dataclass
class SimulatorConfig:
    """模拟器配置"""
    max_time: float = 86400.0        # 最大运行时间（秒），默认24小时
    starvation_threshold: float = 7200.0  # 饿死阈值（秒）
    time_step: float = 1.0           # 时间步长（秒）
    timeline_interval: float = 60.0  # 时间线记录间隔（秒）
    sharing_penalty_map: Dict[int, float] = field(
        default_factory=lambda: {
            1: 1.0,  # 无共享
            2: 0.9,  # 两个任务 -> 90%效率
            3: 0.8,  # 三个任务 -> 80%效率
        }
    )
    sharing_penalty_floor: float = 0.5  # 共享任务过多时的效率下限
    sharing_penalty_fn: Optional[Callable[[int], float]] = None  # 自定义共享惩罚函数
    sharing_penalty_aggregation: str = "min"  # "min" 或 "average"


@dataclass
class ExperimentConfig:
    """实验配置"""
    seed: int = 42                   # 随机种子
    output_dir: str = "results"     # 输出目录


# 默认配置实例
default_cluster_config = ClusterConfig()
default_task_config = TaskConfig()
default_simulator_config = SimulatorConfig()
default_experiment_config = ExperimentConfig()
