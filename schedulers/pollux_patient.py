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
        self, cluster, alpha=0.5, patience_threshold=1.1, starvation_limit=2000.0
    ):
        super().__init__(cluster, alpha=alpha)
        # 这里的 patience_threshold 改回针对 Topology Penalty
        # 我们只讨厌“跨机架” (Penalty > 1.0)
        # 我们喜欢“共享” (Saving GPU Hours)
        self.patience_threshold = patience_threshold
        self.starvation_limit = starvation_limit
        # 需要获取共享惩罚配置，这里简单起见使用默认配置
        self.sharing_penalty_map = default_simulator_config.sharing_penalty_map

    def _get_sharing_penalty(self, gpu_id: str) -> float:
        """预测如果将任务分配给该GPU，产生的共享效率系数 (0.0-1.0)"""
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

    def _get_resource_cost(self, gpu_id: str) -> float:
        """
        计算该GPU对新任务的摊销资源成本。
        Cost = 1 / (Efficiency * TaskCount)
        越低越好。
        """
        gpu = self.cluster.get_gpu(gpu_id)
        if not gpu:
            return 1.0

        current_count = len(gpu.running_tasks)
        new_count = current_count + 1
        efficiency = self._get_sharing_penalty(gpu_id)

        if efficiency <= 1e-6:
            return float("inf")

        return 1.0 / (efficiency * new_count)

    def schedule(
        self, pending_tasks: List[Task], current_time: float
    ) -> Dict[str, List[str]]:
        allocations = {}

        # 队列优化：不再单纯使用 FIFO (按提交时间)。
        # 为了最大化“装箱”密度 (Bin Packing Density)，我们采用 "Best Fit Decreasing" 策略。
        # 1. 优先调度“大石头” (内存需求大、GPU数量多)，因为它们最难安插。
        # 2. “小沙子” (小任务) 可以很容易地填充到大任务留下的缝隙中 (Sharing)。
        # 排序键: (-Memory, -NumGPUs, +SubmissionTime)
        sorted_pending_tasks = sorted(
            pending_tasks,
            key=lambda t: (-t.memory_per_gpu, -t.num_gpus, t.submission_time),
        )

        for task in sorted_pending_tasks:
            if task.status.value != "pending":
                continue

            max_n = task.num_gpus
            min_n = 1

            # 筛选出的合规方案（拓扑惩罚低）
            valid_candidates = []
            # 所有方案（用于保底）
            all_candidates = []

            # BUG FIX: 不再排除 occupied_gpus。
            # self.allocate() 会实时更新 GPU 的 used_memory。
            # 只要 g.can_allocate() 返回 True，就说明还有空间，完全可以共享。
            available_gpus = [
                g
                for g in self.cluster.get_all_gpus()
                if g.can_allocate(task.memory_per_gpu)
            ]

            if len(available_gpus) < min_n:
                continue

            candidates_n = range(min_n, min(max_n, len(available_gpus)) + 1)

            for n in candidates_n:
                # 寻找 n 个 GPU 的最佳放置
                # 1. 尝试单机架 (优先找空闲的，或者共享代价小的)
                rack_candidates = []
                for rack in self.cluster.racks:
                    rack_gpus = [
                        g.gpu_id
                        for g in rack.get_available_gpus()
                        if g.can_allocate(task.memory_per_gpu)
                    ]
                    if len(rack_gpus) >= n:
                        # 贪心选择：在该机架内，选择摊销成本最小的 n 个 GPU
                        # 使用 _get_resource_cost 统一排序标准
                        sorted_gpus = sorted(
                            rack_gpus, key=lambda g: self._get_resource_cost(g)
                        )
                        rack_candidates.append(sorted_gpus[:n])

                # 2. 跨机架 (全局最好的 n 个)
                # 同理，优先选摊销成本最小的
                global_gpus = sorted(
                    available_gpus, key=lambda g: self._get_resource_cost(g.gpu_id)
                )
                global_candidate = [g.gpu_id for g in global_gpus[:n]]

                # 合并候选集
                current_candidates = rack_candidates + [global_candidate]

                for alloc in current_candidates:
                    # A. 拓扑惩罚 (Topology) >= 1.0 (越小越好)
                    topo_penalty = self.cluster.calculate_penalty(alloc)

                    # B. 资源成本系数 (Resource Cost Factor)
                    # Cost = 1 / (Efficiency * TaskCount)
                    # 使用统一的辅助函数计算
                    resource_costs = [self._get_resource_cost(gid) for gid in alloc]

                    avg_resource_cost = (
                        sum(resource_costs) / len(resource_costs)
                        if resource_costs
                        else 1.0
                    )

                    # Total Cost = TopologyPenalty * ResourceCost
                    total_cost = topo_penalty * avg_resource_cost
                    score = (n**self.alpha) / total_cost

                    candidate_info = {
                        "allocation": alloc,
                        "n": n,
                        "score": score,
                        "topo_penalty": topo_penalty,
                    }

                    all_candidates.append(candidate_info)

                    # 核心修改：只对 Topology (跨机架) 进行耐心等待
                    if topo_penalty <= self.patience_threshold:
                        valid_candidates.append(candidate_info)

            # 决策逻辑
            # 1. 优先从 valid_candidates (位置好) 中选 Score 最高的
            selected_candidate = None

            if valid_candidates:
                selected_candidate = max(valid_candidates, key=lambda x: x["score"])
            else:
                # 2. 如果没有位置好的，检查是否饿死
                wait_time = current_time - task.submission_time
                if wait_time > self.starvation_limit:
                    # 饿死了，不得不跑，选 all_candidates 中分最高的
                    if all_candidates:
                        selected_candidate = max(
                            all_candidates, key=lambda x: x["score"]
                        )

                # 3. 既没好位置也没饿死 -> 等待 (不分配)
                pass

            if selected_candidate:
                if self.allocate(task, selected_candidate["allocation"]):
                    allocations[task.task_id] = selected_candidate["allocation"]

        return allocations
