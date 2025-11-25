"""
机架类：包含多个GPU，同一机架内的GPU有惩罚系数
"""
from typing import List, Dict, Optional
from .gpu import GPU


class Rack:
    """机架，包含多个GPU"""
    
    def __init__(self, rack_id: str, num_gpus: int, gpu_memory: float, 
                 intra_rack_penalty: float = 1.0):
        """
        初始化机架
        
        Args:
            rack_id: 机架ID
            num_gpus: GPU数量
            gpu_memory: 每个GPU的显存大小（GB）
            intra_rack_penalty: 同一机架内不同GPU的惩罚系数（默认1.0，无惩罚）
        """
        self.rack_id = rack_id
        self.num_gpus = num_gpus
        self.gpu_memory = gpu_memory
        self.intra_rack_penalty = intra_rack_penalty
        
        # 创建GPU列表
        self.gpus: List[GPU] = []
        for i in range(num_gpus):
            gpu_id = f"{rack_id}-{i}"
            self.gpus.append(GPU(gpu_id=gpu_id, rack_id=rack_id, 
                                total_memory=gpu_memory))
    
    def get_gpu(self, gpu_index: int) -> Optional[GPU]:
        """获取指定索引的GPU"""
        if 0 <= gpu_index < len(self.gpus):
            return self.gpus[gpu_index]
        return None
    
    def get_all_gpus(self) -> List[GPU]:
        """获取所有GPU"""
        return self.gpus
    
    def get_available_gpus(self) -> List[GPU]:
        """获取所有有可用显存的GPU"""
        return [gpu for gpu in self.gpus if gpu.get_available_memory() > 0]
    
    def get_total_memory(self) -> float:
        """获取机架总显存"""
        return self.num_gpus * self.gpu_memory
    
    def get_used_memory(self) -> float:
        """获取机架已使用显存"""
        return sum(gpu.used_memory for gpu in self.gpus)
    
    def get_total_time(self) -> float:
        """获取机架所有GPU的累计时间"""
        return sum(gpu.total_time for gpu in self.gpus)
    
    def get_utilization(self) -> float:
        """获取机架利用率"""
        total = self.get_total_memory()
        used = self.get_used_memory()
        return used / total if total > 0 else 0.0

