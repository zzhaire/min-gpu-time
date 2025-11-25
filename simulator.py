"""
模拟器：运行调度实验
"""
from typing import List
from core.cluster import Cluster
from core.task import Task, TaskStatus
from schedulers.base import Scheduler
from utils.metrics import Metrics
from config.config import SimulatorConfig


class Simulator:
    """调度模拟器"""

    def __init__(self, cluster: Cluster, scheduler: Scheduler,
                 config: SimulatorConfig = None):
        """
        初始化模拟器

        Args:
            cluster: 集群对象
            scheduler: 调度器对象
            config: 模拟器配置
        """
        self.cluster = cluster
        self.scheduler = scheduler
        if config is None:
            from config.config import default_simulator_config
            config = default_simulator_config
        self.config = config
        self.sharing_penalty_fn = config.sharing_penalty_fn
        self.starvation_threshold = config.starvation_threshold
        self.metrics = Metrics()
        self.current_time = 0.0
        self.time_step = config.time_step

    def run(self, tasks: List[Task], max_time: float = None):
        """
        运行模拟

        Args:
            tasks: 任务列表
            max_time: 最大运行时间（秒），如果为None则使用配置中的值
        """
        if max_time is None:
            max_time = self.config.max_time

        print(f"开始模拟，共 {len(tasks)} 个任务")
        print(f"集群配置: {self.cluster.num_racks} 个机架, "
              f"每机架 {self.cluster.gpus_per_rack} 个GPU, "
              f"每GPU {self.cluster.gpu_memory}GB 显存")

        # 按提交时间排序
        sorted_tasks = sorted(tasks, key=lambda t: t.submission_time)

        while self.current_time < max_time:
            # 获取当前时间点的待调度任务
            pending_tasks = [t for t in sorted_tasks
                             if t.submission_time <= self.current_time
                             and t.status == TaskStatus.PENDING]

            # 获取运行中的任务
            running_tasks = [t for t in sorted_tasks
                             if t.status == TaskStatus.RUNNING]

            # 检查饿死任务
            for task in pending_tasks:
                wait_time = self.current_time - task.submission_time
                if wait_time > self.starvation_threshold:
                    task.mark_starved()
                    self.metrics.record_task_completion(task)

            # 调度新任务
            allocations = self.scheduler.schedule(
                pending_tasks, self.current_time)

            # 启动新分配的任务
            for task_id, gpu_ids in allocations.items():
                task = next(
                    (t for t in pending_tasks if t.task_id == task_id), None)
                if task:
                    task.start(self.current_time, gpu_ids)

            # 更新运行中的任务
            for task in running_tasks:
                # 检查任务是否完成
                if task.start_time is not None:
                    elapsed = self.current_time - task.start_time
                    # 考虑惩罚系数调整实际执行时间
                    if task.allocated_gpus:
                        placement_penalty = self.cluster.calculate_penalty(
                            task.allocated_gpus)
                        sharing_penalty = self._get_task_sharing_penalty(task)
                        adjusted_duration = task.estimated_duration * placement_penalty * sharing_penalty
                        if elapsed >= adjusted_duration:
                            task.complete(self.current_time)
                            # 释放资源
                            self.scheduler.deallocate(task)
                            self.metrics.record_task_completion(task)

            # 更新GPU时间
            for gpu in self.cluster.get_all_gpus():
                if len(gpu.running_tasks) > 0:
                    gpu.add_time(self.time_step)

            # 更新总GPU时间
            self.metrics.update_total_gpu_time(self.cluster)

            # 记录时间线
            if int(self.current_time) % int(self.config.timeline_interval) == 0:
                self.metrics.record_timeline(self.current_time, self.cluster,
                                             running_tasks, pending_tasks)

            # 检查是否所有任务都完成或饿死
            all_done = all(t.status in [TaskStatus.COMPLETED, TaskStatus.STARVED]
                           for t in sorted_tasks)
            if all_done:
                print(f"所有任务已完成或饿死，当前时间: {self.current_time:.2f} 秒")
                break

            # 推进时间
            self.current_time += self.time_step

        # 记录剩余未完成的任务
        for task in sorted_tasks:
            if task.status == TaskStatus.PENDING:
                task.mark_starved()
                self.metrics.record_task_completion(task)

        print(f"模拟完成，总运行时间: {self.current_time:.2f} 秒")
        return self.metrics

    def _get_task_sharing_penalty(self, task: Task) -> float:
        """根据任务占用的 GPU 获得共享惩罚"""
        if not task.allocated_gpus:
            return 1.0

        penalties = []
        for gpu_id in task.allocated_gpus:
            gpu = self.cluster.get_gpu(gpu_id)
            if gpu:
                penalties.append(self._get_gpu_sharing_penalty(
                    len(gpu.running_tasks)))

        if not penalties:
            return 1.0

        aggregation = (
            self.config.sharing_penalty_aggregation or "min").lower()
        if aggregation == "average":
            return sum(penalties) / len(penalties)
        return min(penalties)

    def _get_gpu_sharing_penalty(self, task_count: int) -> float:
        """获取单个 GPU 的共享惩罚系数"""
        task_count = max(1, task_count)

        if self.sharing_penalty_fn:
            value = self.sharing_penalty_fn(task_count)
        else:
            penalty_map = self.config.sharing_penalty_map
            if task_count in penalty_map:
                value = penalty_map[task_count]
            elif penalty_map:
                max_key = max(penalty_map.keys())
                value = penalty_map[max_key]
            else:
                value = 1.0

        floor = getattr(self.config, "sharing_penalty_floor", 0.0)
        if value is None:
            value = 1.0
        return max(floor, min(1.0, value))
