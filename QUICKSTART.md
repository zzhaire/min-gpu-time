# 快速开始指南

## 快速运行

最简单的运行方式：

```bash
python main.py
```

这将使用默认配置运行实验：
- 4个机架，每机架8个GPU，每GPU 24GB显存
- 20个随机任务
- 最小化GPU时间调度器

## 常用场景

### 1. 小规模测试

```bash
python main.py --num-tasks 10 --num-racks 2 --gpus-per-rack 4
```

### 2. 大规模实验

```bash
python main.py --num-tasks 50 --num-racks 8 --gpus-per-rack 16 --max-time 72000
```

### 3. 对比不同调度算法

```bash
# First-Fit
python main.py --scheduler first-fit --output results_ff.json

# Best-Fit
python main.py --scheduler best-fit --output results_bf.json

# Rack-Aware
python main.py --scheduler rack-aware --output results_ra.json

# Min-GPU-Time
python main.py --scheduler min-gpu-time --output results_mgt.json
```

### 4. 自定义任务配置

```bash
python main.py \
    --num-tasks 30 \
    --min-gpus 2 \
    --max-gpus 8 \
    --min-memory 8.0 \
    --max-memory 20.0 \
    --min-duration 200.0 \
    --max-duration 1200.0
```

### 5. 研究饿死情况

```bash
# 设置较低的饿死阈值
python main.py --starvation-threshold 600 --num-tasks 30

# 设置较高的任务负载
python main.py --num-tasks 50 --gpus-per-rack 4
```

## 理解输出结果

### 控制台输出

实验运行时会显示：
- 集群配置信息
- 任务生成信息
- 模拟进度
- 最终统计汇总

### JSON输出文件

结果文件包含三个部分：

1. **summary**: 汇总统计
   ```json
   {
     "total_tasks": 20,
     "completed_tasks": 18,
     "starved_tasks": 2,
     "total_gpu_time": 12345.67,
     "average_jct": 234.56,
     "average_wait_time": 12.34,
     "total_jct": 4222.08
   }
   ```

2. **task_metrics**: 每个任务的详细信息
   ```json
   {
     "task_id": "task-0",
     "submission_time": 0.0,
     "start_time": 0.0,
     "completion_time": 200.5,
     "jct": 200.5,
     "wait_time": 0.0,
     "allocated_gpus": ["rack-0-0", "rack-0-1"],
     "status": "completed"
   }
   ```

3. **timeline**: 时间线快照（每60秒）
   ```json
   {
     "time": 60.0,
     "total_gpu_time": 120.0,
     "cluster_utilization": 0.75,
     "running_tasks": 5,
     "pending_tasks": 3,
     "completed_tasks": 2
   }
   ```

## 关键指标说明

### 总GPU时间 (total_gpu_time)
- **定义**: 所有GPU的累计运行时间之和
- **计算**: 从每个GPU累加其运行时间
- **意义**: 反映资源使用总量，越小越好

### 平均JCT (average_jct)
- **定义**: 平均作业完成时间（Job Completion Time）
- **计算**: (完成时间 - 提交时间) 的平均值
- **意义**: 反映任务响应速度，越小越好

### 饿死任务数 (starved_tasks)
- **定义**: 等待时间超过阈值的任务数
- **意义**: 反映调度公平性，越少越好

### 集群利用率 (cluster_utilization)
- **定义**: 已使用显存 / 总显存
- **意义**: 反映资源利用效率，越高越好（在保证公平性的前提下）

## 实验设计建议

### 1. 对比实验
- 使用相同的随机种子 (`--seed`)
- 使用相同的任务配置
- 只改变调度算法

### 2. 参数调优
- 调整惩罚系数，观察对总GPU时间的影响
- 调整任务负载，观察饿死情况
- 调整集群规模，观察可扩展性

### 3. 结果分析
- 对比不同调度算法的总GPU时间
- 分析JCT分布
- 检查饿死任务的共同特征

## 常见问题

### Q: 为什么总GPU时间比运行时间大？
A: 总GPU时间是所有GPU时间的累加。如果多个GPU同时运行，总时间会大于实际运行时间。

### Q: 如何判断调度算法好坏？
A: 综合考虑：
- 总GPU时间（越小越好）
- 平均JCT（越小越好）
- 饿死任务数（越少越好）
- 集群利用率（越高越好）

### Q: 如何自定义调度算法？
A: 在 `scheduler.py` 中创建新类，继承 `Scheduler` 基类，实现 `schedule()` 方法。

### Q: 如何分析结果？
A: 可以使用Python脚本读取JSON文件，进行数据分析和可视化。

## 下一步

- 查看 `example.py` 了解更详细的使用示例
- 阅读 `README.md` 了解完整功能
- 修改 `scheduler.py` 实现自己的调度算法

