"""
绘图工具：绘制任务调度的时空图（Gantt Chart）
"""
import csv
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple


class Plotter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.colors = {}

    def _get_color(self, task_id: str) -> str:
        """为每个任务生成固定的随机颜色"""
        if task_id not in self.colors:
            # 生成柔和的随机颜色
            r = random.random() * 0.5 + 0.5
            g = random.random() * 0.5 + 0.5
            b = random.random() * 0.5 + 0.5
            self.colors[task_id] = (r, g, b)
        return self.colors[task_id]

    def plot_gantt_chart(self, tasks_csv_path: str, output_filename: str, title: str):
        """
        根据任务CSV数据绘制时空图
        采用子轨道（Sub-track）布局，避免重叠
        """
        if not os.path.exists(tasks_csv_path):
            print(f"错误：找不到数据文件 {tasks_csv_path}")
            return

        tasks = []
        with open(tasks_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['status'] == 'completed' and row['allocated_gpus']:
                    # 预处理：解析 GPU ID 列表
                    row['gpu_list'] = [gid.strip() for gid in row['allocated_gpus'].replace('"', '').split(',') if gid.strip()]
                    row['start'] = float(row['start_time'])
                    row['end'] = float(row['completion_time'])
                    tasks.append(row)

        if not tasks:
            print(f"警告：{tasks_csv_path} 中没有已完成的任务数据，跳过绘图")
            return

        # 1. 收集所有 GPU 并排序
        all_gpus = set()
        for task in tasks:
            all_gpus.update(task['gpu_list'])
        
        sorted_gpus = sorted(list(all_gpus), key=lambda x: (
            int(x.split('-')[1]) if len(x.split('-')) > 1 else 0,
            int(x.split('-')[2]) if len(x.split('-')) > 2 else 0
        ))
        
        gpu_to_y = {gid: i for i, gid in enumerate(sorted_gpus)}
        
        # 2. 计算每个 GPU 上的任务轨道分配
        # gpu_tracks[gid] = [ [(start, end, task_idx), ...], ... ] (每个子列表是一个轨道)
        gpu_tracks: Dict[str, List[List[int]]] = {gid: [] for gid in sorted_gpus}
        
        # 任务按开始时间排序
        tasks.sort(key=lambda x: x['start'])
        
        # 记录每个任务在每个 GPU 上分配到的轨道索引
        # task_placements[task_idx][gpu_id] = track_index
        task_placements: Dict[int, Dict[str, int]] = {}

        for i, task in enumerate(tasks):
            task_placements[i] = {}
            for gid in task['gpu_list']:
                if gid not in gpu_tracks: 
                    continue
                
                # 寻找该 GPU 上可用的第一个轨道
                tracks = gpu_tracks[gid]
                assigned_track = -1
                
                for track_idx, track in enumerate(tracks):
                    # 检查当前轨道是否有冲突
                    conflict = False
                    for t_idx in track:
                        other_task = tasks[t_idx]
                        # 检查时间交集：not (end1 <= start2 or start1 >= end2)
                        if not (task['end'] <= other_task['start'] or task['start'] >= other_task['end']):
                            conflict = True
                            break
                    if not conflict:
                        assigned_track = track_idx
                        track.append(i)
                        break
                
                # 如果没有可用轨道，创建新轨道
                if assigned_track == -1:
                    assigned_track = len(tracks)
                    tracks.append([i])
                
                task_placements[i][gid] = assigned_track

        # 3. 确定每行的高度和布局
        # 计算每个 GPU 需要多少个轨道
        gpu_track_counts = {gid: len(tracks) for gid, tracks in gpu_tracks.items()}
        # 为了美观，每行至少保留 1 个轨道的高度，即使没有任务
        for gid in gpu_track_counts:
            gpu_track_counts[gid] = max(1, gpu_track_counts[gid])

        # 4. 绘图
        # 动态调整画布高度：轨道越多，图越高
        total_tracks = sum(gpu_track_counts.values())
        fig, ax = plt.subplots(figsize=(15, max(6, total_tracks * 0.3)))
        
        min_time = min(t['start'] for t in tasks) if tasks else 0
        max_time = max(t['end'] for t in tasks) if tasks else 100

        # 绘制辅助背景行（区分不同 GPU）
        current_y_base = 0
        y_ticks = []
        y_labels = []
        
        for gid in sorted_gpus:
            num_tracks = gpu_track_counts[gid]
            # 每个 GPU 占据的高度区间：[current_y_base, current_y_base + num_tracks]
            # 绘制斑马纹背景
            if gpu_to_y[gid] % 2 == 0:
                rect = patches.Rectangle(
                    (min_time - 100, current_y_base), 
                    max_time - min_time + 200, 
                    num_tracks, 
                    facecolor='#f0f0f0', 
                    edgecolor='none', 
                    zorder=0
                )
                ax.add_patch(rect)
            
            # 记录 Y 轴标签位置（居中）
            y_ticks.append(current_y_base + num_tracks / 2)
            y_labels.append(gid)
            
            # 绘制分隔线
            ax.axhline(y=current_y_base, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            current_y_base += num_tracks

        ax.axhline(y=current_y_base, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        # 重新遍历任务进行绘制
        # 我们需要根据 current_y_base 反推每个 GPU 的起始 Y 坐标
        # 建立 gpu_y_start 映射
        gpu_y_start = {}
        curr = 0
        for gid in sorted_gpus:
            gpu_y_start[gid] = curr
            curr += gpu_track_counts[gid]

        for i, task in enumerate(tasks):
            task_id = task['task_id']
            duration = task['end'] - task['start']
            color = self._get_color(task_id)
            
            for gid in task['gpu_list']:
                if gid in gpu_y_start and gid in task_placements[i]:
                    base_y = gpu_y_start[gid]
                    track_idx = task_placements[i][gid]
                    
                    # 绘制矩形
                    # 高度设为 0.8，留出 0.1 的上下间隙
                    rect = patches.Rectangle(
                        (task['start'], base_y + track_idx + 0.1), 
                        duration, 
                        0.8, 
                        linewidth=1, 
                        edgecolor='black', 
                        facecolor=color,
                        alpha=0.8,
                        zorder=10
                    )
                    ax.add_patch(rect)
                    
                    # 只有当格子足够宽时才显示文字
                    if duration > (max_time - min_time) * 0.02:
                        ax.text(
                            task['start'] + duration/2, 
                            base_y + track_idx + 0.5, 
                            task_id.split('-')[-1], 
                            ha='center', 
                            va='center', 
                            fontsize=7,
                            color='black',
                            fontweight='bold',
                            zorder=20
                        )

        # 设置轴
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("GPU ID")
        ax.set_title(f"Task Schedule - {title}")
        
        ax.set_xlim(max(0, min_time - 50), max_time + 50)
        ax.set_ylim(0, current_y_base)
        
        # 5. 保存
        output_path = os.path.join(self.output_dir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"时空图已保存到: {output_path}")
