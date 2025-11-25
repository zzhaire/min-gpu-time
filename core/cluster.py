"""
集群类：包含多个机架，跨机架有惩罚系数
"""
from typing import List, Dict, Optional, Tuple
from .rack import Rack
from .gpu import GPU


class Cluster:
    """集群，包含多个机架"""
    
    def __init__(self, num_racks: int, gpus_per_rack: int, gpu_memory: float,
                 intra_rack_penalty: float = 1.0, inter_rack_penalty: float = 1.5):
        """
        初始化集群
        
        Args:
            num_racks: 机架数量
            gpus_per_rack: 每个机架的GPU数量
            gpu_memory: 每个GPU的显存大小（GB）
            intra_rack_penalty: 同一机架内不同GPU的惩罚系数
            inter_rack_penalty: 跨机架GPU的惩罚系数
        """
        self.num_racks = num_racks
        self.gpus_per_rack = gpus_per_rack
        self.gpu_memory = gpu_memory
        self.intra_rack_penalty = intra_rack_penalty
        self.inter_rack_penalty = inter_rack_penalty
        
        # 创建机架列表
        self.racks: List[Rack] = []
        for i in range(num_racks):
            rack_id = f"rack-{i}"
            self.racks.append(Rack(rack_id=rack_id, num_gpus=gpus_per_rack,
                                  gpu_memory=gpu_memory, 
                                  intra_rack_penalty=intra_rack_penalty))
        
        # GPU映射：gpu_id -> GPU对象
        self.gpu_map: Dict[str, GPU] = {}
        for rack in self.racks:
            for gpu in rack.get_all_gpus():
                self.gpu_map[gpu.gpu_id] = gpu
    
    def get_rack(self, rack_id: str) -> Optional[Rack]:
        """根据机架ID获取机架"""
        for rack in self.racks:
            if rack.rack_id == rack_id:
                return rack
        return None
    
    def get_gpu(self, gpu_id: str) -> Optional[GPU]:
        """根据GPU ID获取GPU"""
        return self.gpu_map.get(gpu_id)
    
    def get_all_gpus(self) -> List[GPU]:
        """获取所有GPU"""
        return list(self.gpu_map.values())
    
    def get_available_gpus(self) -> List[GPU]:
        """获取所有有可用显存的GPU"""
        return [gpu for gpu in self.gpu_map.values() 
                if gpu.get_available_memory() > 0]
    
    def calculate_penalty(self, gpu_ids: List[str]) -> float:
        """
        计算分配GPU列表的惩罚系数
        
        Args:
            gpu_ids: GPU ID列表
            
        Returns:
            惩罚系数（>= 1.0）
        """
        if len(gpu_ids) <= 1:
            return 1.0
        
        # 按机架分组
        racks_used = set()
        for gpu_id in gpu_ids:
            gpu = self.get_gpu(gpu_id)
            if gpu:
                racks_used.add(gpu.rack_id)
        
        # 如果所有GPU在同一机架
        if len(racks_used) == 1:
            return self.intra_rack_penalty
        
        # 如果跨机架
        return self.inter_rack_penalty
    
    def get_total_gpus(self) -> int:
        """获取总GPU数量"""
        return self.num_racks * self.gpus_per_rack
    
    def get_total_memory(self) -> float:
        """获取集群总显存"""
        return self.get_total_gpus() * self.gpu_memory
    
    def get_used_memory(self) -> float:
        """获取集群已使用显存"""
        return sum(gpu.used_memory for gpu in self.gpu_map.values())
    
    def get_total_time(self) -> float:
        """获取集群所有GPU的累计时间"""
        return sum(gpu.total_time for gpu in self.gpu_map.values())
    
    def get_utilization(self) -> float:
        """获取集群利用率"""
        total = self.get_total_memory()
        used = self.get_used_memory()
        return used / total if total > 0 else 0.0
    
    def get_gpu_by_rack_and_index(self, rack_id: str, gpu_index: int) -> Optional[GPU]:
        """根据机架ID和GPU索引获取GPU"""
        rack = self.get_rack(rack_id)
        if rack:
            return rack.get_gpu(gpu_index)
        return None

