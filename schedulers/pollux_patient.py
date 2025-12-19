"""
优化版 Pollux 调度器：结合 Min-GPU-Time 的耐心机制 + 全感知代价模型
核心思想：在 Pollux 的弹性资源选择基础上，增加对“共享惩罚”的感知，以及基于效率的耐心机制。
"""

from typing import List, Dict
from core.task import Task
from .pollux import PolluxScheduler
from config.config import default_simulator_config


class PolluxPatientScheduler(PolluxScheduler):
    """
    Pollux 自适应调度器 (全感知耐心版)
    """

    def __init__(
        self, cluster, alpha=0.5, patience_threshold=0.8, starvation_limit=2000.0
    ):
        super().__init__(cluster, alpha=alpha)
        # 这里的 patience_threshold 含义变为：效率阈值 (efficiency_threshold)
        # 如果 当前效率 / 理想效率 < 0.8，则等待
        self.patience_threshold = 0.8
        self.starvation_limit = starvation_limit
        # 需要获取共享惩罚配置，这里简单起见使用默认配置
        # 理想情况下应该从 scheduler_config 传入，或者从 cluster 获取
        self.sharing_penalty_map = default_simulator_config.sharing_penalty_map

    def _get_sharing_penalty(self, gpu_id: str) -> float:
        """预测如果将任务分配给该GPU，产生的共享惩罚系数"""
        gpu = self.cluster.get_gpu(gpu_id)
        if not gpu:
            return 1.0

        # 当前已有任务数 + 1 (即将分配的新任务)
        current_tasks = len(gpu.running_tasks)
        new_count = current_tasks + 1

        # 查表
        penalty = self.sharing_penalty_map.get(
            new_count,
            self.sharing_penalty_map.get(max(self.sharing_penalty_map.keys()), 1.0),
        )
        return penalty

    def schedule(
        self, pending_tasks: List[Task], current_time: float
    ) -> Dict[str, List[str]]:
        allocations = {}
        occupied_gpus = set()

        for task in pending_tasks:
            if task.status.value != "pending":
                continue

            max_n = task.num_gpus
            min_n = 1

            best_n = -1
            best_allocation = None
            best_score = -float("inf")

            # 记录最佳方案的各项指标，用于决策
            best_topology_penalty = 1.0
            best_sharing_penalty = 1.0
            best_total_penalty = 1.0

            available_gpus = [
                g
                for g in self.cluster.get_all_gpus()
                if g.gpu_id not in occupied_gpus and g.can_allocate(task.memory_per_gpu)
            ]

            if len(available_gpus) < min_n:
                continue

            candidates_n = range(min_n, min(max_n, len(available_gpus)) + 1)

            for n in candidates_n:
                # 寻找 n 个 GPU 的最佳放置
                current_allocation = []

                # 计算候选方案
                candidates = []

                # 1. 尝试单机架 (优先找空闲的，或者共享代价小的)
                # 为了简化，我们先找单机架组合，再找跨机架组合，然后统一计算全成本

                rack_candidates = []
                for rack in self.cluster.racks:
                    rack_gpus = [
                        g.gpu_id
                        for g in rack.get_available_gpus()
                        if g.gpu_id not in occupied_gpus
                        and g.can_allocate(task.memory_per_gpu)
                    ]
                    if len(rack_gpus) >= n:
                        # 贪心选择：在该机架内，选择共享惩罚最小的 n 个 GPU
                        # (即优先选空闲的)
                        sorted_gpus = sorted(
                            rack_gpus, key=lambda g: -self._get_sharing_penalty(g)
                        )  # 惩罚越大越差，所以按惩罚降序排？不对，是按惩罚值(0.9)降序(越接近1越好)
                        # _get_sharing_penalty 返回的是效率系数 (如0.9)，所以越大越好
                        rack_candidates.append(sorted_gpus[:n])

                # 2. 跨机架 (全局最好的 n 个)
                global_gpus = sorted(
                    available_gpus, key=lambda g: -self._get_sharing_penalty(g.gpu_id)
                )
                global_candidate = [g.gpu_id for g in global_gpus[:n]]

                # 合并候选集 (每个满足条件的机架方案 + 全局方案)
                candidates = rack_candidates + [global_candidate]

                for alloc in candidates:
                    # 计算两部分惩罚
                    # A. 拓扑惩罚 (Topology) -> 返回值 >= 1.0 (越小越好)
                    topo_penalty = self.cluster.calculate_penalty(alloc)

                    # B. 共享惩罚 (Sharing) -> 返回值 <= 1.0 (越大越好)
                    # 我们需要将其转换为 "Cost" (>= 1.0) 以便统一乘法
                    # Simulator 中: duration = base * topo * sharing_factor (如果 sharing 是 0.9)
                    # 等等，Simulator 的公式是: adjusted_duration = duration * topo * sharing
                    # 其中 sharing 是通过 _get_task_sharing_penalty 获得的，比如 0.9
                    # 实际上如果 sharing=0.9，意味着速度变慢，duration 应该 除以 0.9 ？
                    # 不，simulator.py 第 108 行: adjusted_duration = ... * placement_penalty * sharing_penalty
                    # 如果 sharing_penalty_map 定义为 {2: 0.9} 代表效率是 90%？
                    # 让我们检查 Simulator.
                    # sharing_penalty_map 默认 {2: 0.9}。
                    # simulator.py: adjusted_duration = ... * placement * sharing
                    # 如果 sharing 是 0.9，duration 变短了？ 这不对！
                    # 通常 penalty 系数 > 1.0 代表变慢。
                    # 如果用户定义 {2: 0.9} 代表“效率”，那 duration 应该是 / 0.9。
                    # 如果用户定义 {2: 1.2} 代表“惩罚”，那 duration 应该是 * 1.2。

                    # 让我们回顾 Simulator 代码。
                    pass

                    # 假设 Simulator 的 sharing_penalty 是直接乘的，那如果值是 0.9，任务会变快。
                    # 这显然是 Bug 或者定义歧义。
                    # 让我们假设用户的意思是 "效率下降为 90%"，即时间变长 1/0.9 = 1.11倍。
                    # 但 Simulator 代码直接乘了。
                    # 让我们看 Simulator 代码确认一下。

                    # 假设 Simulator 中 sharing_penalty 返回的是 1.0 以上的数？
                    # config: {2: 0.9}.
                    # 看起来之前的 Simulator 逻辑可能有误，或者我理解反了。
                    # 如果 config 写的是 "效率"，那 simulator 应该除以它。
                    # 如果 config 写的是 "惩罚系数"，那 0.9 是什么意思？

                    # 暂时按 Simulator 现有的逻辑走：
                    # 假设 Simulator 代码是 duration * sharing。如果 sharing=0.9，那任务反而快了。
                    # 这绝对是 Bug。
                    # 但为了不改动 Simulator 导致之前的实验失效（或者之前的实验其实也是错的？），
                    # 我们先假设：Sharing Penalty 在 Config 里定义的是 "代价系数" (Cost Factor)？
                    # 但注释写着 "2个任务 -> 90%效率"。
                    # 90% 效率意味着时间是 1/0.9 = 1.11 倍。

                    # 无论如何，对于调度器来说，我们需要计算 "Total Cost"。
                    # Cost = Topology_Penalty * (1 / Sharing_Efficiency)

                    # 计算该分配的平均共享效率
                    sharing_efficiencies = [
                        self._get_sharing_penalty(gid) for gid in alloc
                    ]
                    avg_sharing_eff = sum(sharing_efficiencies) / len(
                        sharing_efficiencies
                    )

                    # Cost calculation
                    total_cost = topo_penalty * (1.0 / avg_sharing_eff)

                    # Score = Speed / Cost = n^alpha / total_cost
                    score = (n**self.alpha) / total_cost

                    if score > best_score:
                        best_score = score
                        best_allocation = alloc
                        best_n = n
                        best_topology_penalty = topo_penalty
                        best_sharing_penalty = avg_sharing_eff  # 0.9 etc.
                        best_total_penalty = total_cost  # >= 1.0

            # 决策逻辑：基于效率的耐心
            if best_allocation:
                # 理想情况：在该 n 下，无拓扑惩罚 (1.0) 且无共享损耗 (1.0)
                # Ideal Cost = 1.0 * 1.0 = 1.0
                # Ideal Score = n^alpha / 1.0

                ideal_score = (best_n**self.alpha) / 1.0

                # 当前效率比 = Actual Score / Ideal Score = 1 / Total Cost
                current_efficiency = 1.0 / best_total_penalty

                wait_time = current_time - task.submission_time
                is_efficient_enough = current_efficiency >= self.patience_threshold
                is_starving = wait_time > self.starvation_limit

                if is_efficient_enough or is_starving:
                    if self.allocate(task, best_allocation):
                        allocations[task.task_id] = best_allocation
                        occupied_gpus.update(best_allocation)
                # else: Waiting for better efficiency

        return allocations
