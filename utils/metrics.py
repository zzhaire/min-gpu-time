"""
测量类：记录总GPU时间和任务完成时间
"""
from typing import List, Dict, Optional
import csv
import os
from core.task import Task
from core.cluster import Cluster


class Metrics:
    """测量和记录指标"""
    
    def __init__(self):
        self.total_gpu_time = 0.0  # 总GPU时间（累加所有GPU的运行时间）
        self.task_metrics: List[Dict] = []  # 任务指标记录
        self.timeline: List[Dict] = []  # 时间线记录
        
    def record_task_completion(self, task: Task):
        """记录任务完成"""
        jct = task.get_jct()
        wait_time = task.get_wait_time()
        
        metric = {
            "task_id": task.task_id,
            "submission_time": task.submission_time,
            "start_time": task.start_time,
            "completion_time": task.completion_time,
            "estimated_duration": task.estimated_duration,
            "actual_duration": task.actual_duration,
            "jct": jct,
            "wait_time": wait_time,
            "num_gpus": task.num_gpus,
            "memory_per_gpu": task.memory_per_gpu,
            "allocated_gpus": ",".join(task.allocated_gpus) if task.allocated_gpus else "",
            "status": task.status.value
        }
        
        self.task_metrics.append(metric)
    
    def update_total_gpu_time(self, cluster: Cluster):
        """更新总GPU时间（从每个GPU累加）"""
        self.total_gpu_time = cluster.get_total_time()
    
    def record_timeline(self, current_time: float, cluster: Cluster, 
                       running_tasks: List[Task], pending_tasks: List[Task]):
        """记录时间线快照"""
        snapshot = {
            "time": current_time,
            "total_gpu_time": cluster.get_total_time(),
            "cluster_utilization": cluster.get_utilization(),
            "running_tasks": len(running_tasks),
            "pending_tasks": len(pending_tasks),
            "completed_tasks": len([t for t in self.task_metrics if t["status"] == "completed"])
        }
        self.timeline.append(snapshot)
    
    def get_average_jct(self) -> Optional[float]:
        """获取平均JCT"""
        completed_tasks = [m for m in self.task_metrics if m["jct"] is not None]
        if len(completed_tasks) == 0:
            return None
        return sum(m["jct"] for m in completed_tasks) / len(completed_tasks)
    
    def get_average_wait_time(self) -> Optional[float]:
        """获取平均等待时间"""
        tasks_with_wait = [m for m in self.task_metrics if m["wait_time"] is not None]
        if len(tasks_with_wait) == 0:
            return None
        return sum(m["wait_time"] for m in tasks_with_wait) / len(tasks_with_wait)
    
    def get_starved_tasks(self) -> List[Dict]:
        """获取饿死的任务"""
        return [m for m in self.task_metrics if m["status"] == "starved"]
    
    def get_summary(self) -> Dict:
        """获取汇总统计"""
        completed = [m for m in self.task_metrics if m["status"] == "completed"]
        starved = [m for m in self.task_metrics if m["status"] == "starved"]
        
        return {
            "total_tasks": len(self.task_metrics),
            "completed_tasks": len(completed),
            "starved_tasks": len(starved),
            "total_gpu_time": self.total_gpu_time,
            "average_jct": self.get_average_jct(),
            "average_wait_time": self.get_average_wait_time(),
            "total_jct": sum(m["jct"] for m in completed) if completed else 0.0
        }
    
    def save_to_tables(self, output_dir: str, scheduler_name: str = "unknown"):
        """
        保存指标到CSV表格文件
        
        Args:
            output_dir: 输出目录
            scheduler_name: 调度器名称
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存汇总统计表格
        summary_file = os.path.join(output_dir, f"summary_{scheduler_name}.csv")
        summary = self.get_summary()
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['指标', '值'])
            writer.writerow(['调度器', scheduler_name])
            writer.writerow(['总任务数', summary['total_tasks']])
            writer.writerow(['完成任务数', summary['completed_tasks']])
            writer.writerow(['饿死任务数', summary['starved_tasks']])
            writer.writerow(['总GPU时间(秒)', f"{summary['total_gpu_time']:.2f}"])
            writer.writerow(['平均JCT(秒)', f"{summary['average_jct']:.2f}" if summary['average_jct'] else "N/A"])
            writer.writerow(['平均等待时间(秒)', f"{summary['average_wait_time']:.2f}" if summary['average_wait_time'] else "N/A"])
            writer.writerow(['总JCT(秒)', f"{summary['total_jct']:.2f}"])
        
        print(f"汇总统计已保存到: {summary_file}")
        
        # 2. 保存任务详情表格
        tasks_file = os.path.join(output_dir, f"tasks_{scheduler_name}.csv")
        
        with open(tasks_file, 'w', newline='', encoding='utf-8') as f:
            if len(self.task_metrics) > 0:
                fieldnames = [
                    'task_id', 'status', 'num_gpus', 'memory_per_gpu',
                    'submission_time', 'start_time', 'completion_time',
                    'estimated_duration', 'actual_duration', 'jct', 'wait_time',
                    'allocated_gpus'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for metric in self.task_metrics:
                    row = {k: (f"{v:.2f}" if isinstance(v, float) else str(v)) 
                          for k, v in metric.items() if k in fieldnames}
                    writer.writerow(row)
        
        print(f"任务详情已保存到: {tasks_file}")
        
        # 3. 保存时间线表格（如果存在）
        if len(self.timeline) > 0:
            timeline_file = os.path.join(output_dir, f"timeline_{scheduler_name}.csv")
            
            with open(timeline_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['time', 'total_gpu_time', 'cluster_utilization', 
                            'running_tasks', 'pending_tasks', 'completed_tasks']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for snapshot in self.timeline:
                    row = {k: (f"{v:.2f}" if isinstance(v, float) else str(v)) 
                          for k, v in snapshot.items() if k in fieldnames}
                    writer.writerow(row)
            
            print(f"时间线已保存到: {timeline_file}")
    
    def print_summary(self):
        """打印汇总统计"""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("实验统计汇总")
        print("="*50)
        print(f"总任务数: {summary['total_tasks']}")
        print(f"完成任务数: {summary['completed_tasks']}")
        print(f"饿死任务数: {summary['starved_tasks']}")
        print(f"总GPU时间: {summary['total_gpu_time']:.2f} 秒")
        print(f"平均JCT: {summary['average_jct']:.2f} 秒" if summary['average_jct'] else "平均JCT: N/A")
        print(f"平均等待时间: {summary['average_wait_time']:.2f} 秒" if summary['average_wait_time'] else "平均等待时间: N/A")
        print(f"总JCT: {summary['total_jct']:.2f} 秒")
        print("="*50)

    def print_task_table(self):
        """打印任务详情表格"""
        if len(self.task_metrics) == 0:
            print("没有任务数据")
            return
        
        print("\n" + "="*120)
        print("任务详情表")
        print("="*120)
        print(f"{'任务ID':<12} {'状态':<10} {'GPU数':<6} {'内存/GPU':<10} "
              f"{'提交时间':<10} {'开始时间':<10} {'完成时间':<10} "
              f"{'JCT':<10} {'等待时间':<10}")
        print("-"*120)
        
        for metric in self.task_metrics:
            task_id = metric['task_id']
            status = metric['status']
            num_gpus = metric['num_gpus']
            memory = f"{metric['memory_per_gpu']:.1f}"
            sub_time = f"{metric['submission_time']:.1f}" if metric['submission_time'] is not None else "N/A"
            start_time = f"{metric['start_time']:.1f}" if metric['start_time'] is not None else "N/A"
            comp_time = f"{metric['completion_time']:.1f}" if metric['completion_time'] is not None else "N/A"
            jct = f"{metric['jct']:.1f}" if metric['jct'] is not None else "N/A"
            wait = f"{metric['wait_time']:.1f}" if metric['wait_time'] is not None else "N/A"
            
            print(f"{task_id:<12} {status:<10} {num_gpus:<6} {memory:<10} "
                  f"{sub_time:<10} {start_time:<10} {comp_time:<10} "
                  f"{jct:<10} {wait:<10}")
        
        print("="*120)
