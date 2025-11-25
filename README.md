# 最小化GPU时间调度算法实验平台

这是一个用于研究和实验最小化GPU时间调度算法的实验平台。平台支持多任务共享GPU、机架感知调度、惩罚系数等功能。

## 项目结构

```
min-gpu-time/
├── config/              # 配置文件
│   └── config.py       # 集群、任务、模拟器配置
├── core/               # 核心模块
│   ├── __init__.py
│   ├── gpu.py          # GPU类
│   ├── rack.py         # 机架类
│   ├── cluster.py      # 集群类
│   └── task.py         # 任务类
├── schedulers/         # 调度器模块
│   ├── __init__.py
│   ├── base.py         # 调度器基类
│   ├── first_fit.py    # First-Fit调度器
│   ├── best_fit.py     # Best-Fit调度器
│   ├── rack_aware.py   # 机架感知调度器
│   └── min_gpu_time.py # 最小化GPU时间调度器
├── utils/              # 工具模块
│   ├── __init__.py
│   ├── task_generator.py # 任务生成器
│   └── metrics.py      # 指标测量（输出CSV表格）
├── simulator.py        # 模拟器
├── main.py             # 主运行脚本
├── example.py          # 使用示例
├── requirements.txt    # 依赖
└── README.md           # 说明文档
```

## 功能特性

1. **多任务共享GPU**：同一个GPU可以同时运行多个任务（基于显存限制）
2. **机架感知**：支持同一机架内和跨机架的惩罚系数
3. **同构GPU**：所有GPU具有相同的显存大小
4. **任务随机化**：可以随机生成不同配置的任务
5. **指标测量**：
   - 总GPU时间：从每个GPU累加运行时间
   - 任务完成时间（JCT）：每个任务的提交时间和完成时间
   - 饿死检测：长时间等待的任务会被标记为饿死
6. **CSV表格输出**：结果以CSV格式保存，便于分析

## 配置说明

所有环境参数都在 `config/config.py` 中配置：

### 集群配置（ClusterConfig）
- `num_racks`: 机架数量（默认：4）
- `gpus_per_rack`: 每个机架的GPU数量（默认：8）
- `gpu_memory`: 每个GPU的显存大小，GB（默认：24.0）
- `intra_rack_penalty`: 同一机架内GPU惩罚系数（默认：1.0）
- `inter_rack_penalty`: 跨机架GPU惩罚系数（默认：1.5）

### 任务配置（TaskConfig）
- `num_tasks`: 任务数量（默认：20）
- `min_gpus` / `max_gpus`: GPU数量范围（默认：1-4）
- `min_memory` / `max_memory`: 每个GPU内存范围，GB（默认：4.0-16.0）
- `min_duration` / `max_duration`: 执行时间范围，秒（默认：100.0-1800.0）
- `submission_window`: 提交时间窗口，秒（默认：1800.0）

### 模拟器配置（SimulatorConfig）
- `max_time`: 最大运行时间，秒（默认：86400.0，24小时）
- `starvation_threshold`: 饿死阈值，秒（默认：3600.0）
- `time_step`: 时间步长，秒（默认：1.0）
- `timeline_interval`: 时间线记录间隔，秒（默认：60.0）

### 实验配置（ExperimentConfig）
- `seed`: 随机种子（默认：42）
- `output_dir`: 输出目录（默认："results"）

## 使用方法

### 1. 修改配置

编辑 `config/config.py` 文件，修改你需要的环境参数：

```python
# 例如：修改集群配置
default_cluster_config = ClusterConfig(
    num_racks=8,           # 8个机架
    gpus_per_rack=16,      # 每机架16个GPU
    gpu_memory=32.0,       # 每GPU 32GB
    intra_rack_penalty=1.0,
    inter_rack_penalty=1.5
)
```

### 2. 运行单个调度算法

```bash
python main.py --scheduler min-gpu-time
```

可选调度器：
- `first-fit`: First-Fit调度器
- `best-fit`: Best-Fit调度器
- `rack-aware`: 机架感知调度器
- `min-gpu-time`: 最小化GPU时间调度器（默认）

### 3. 运行所有调度算法并对比

```bash
python main.py --run-all
```

这将运行所有调度算法，并生成对比结果。

### 4. 运行示例

```bash
python example.py
```

## 输出结果

结果保存在 `results/` 目录下，包含：

### 1. 汇总统计表格（summary_*.csv）
包含实验的整体统计信息：
- 调度器名称
- 总任务数、完成任务数、饿死任务数
- 总GPU时间、平均JCT、平均等待时间、总JCT

### 2. 任务详情表格（tasks_*.csv）
包含每个任务的详细信息：
- task_id, status, num_gpus, memory_per_gpu
- submission_time, start_time, completion_time
- estimated_duration, actual_duration, jct, wait_time
- allocated_gpus

### 3. 时间线表格（timeline_*.csv）
包含时间线快照（每60秒记录一次）：
- time, total_gpu_time, cluster_utilization
- running_tasks, pending_tasks, completed_tasks

## 调度算法说明

### 1. First-Fit调度器
按顺序分配第一个可用的GPU组合。

### 2. Best-Fit调度器
选择显存利用率最高的GPU组合。

### 3. 机架感知调度器
优先在同一机架内分配，减少跨机架惩罚。

### 4. 最小化GPU时间调度器
考虑惩罚系数，最小化总GPU时间（`sum(gpu_time * penalty)`）。

## 扩展开发

### 添加新的调度算法

1. 在 `schedulers/` 目录下创建新文件，例如 `my_scheduler.py`
2. 继承 `Scheduler` 基类，实现 `schedule()` 方法
3. 在 `schedulers/__init__.py` 中导入并导出
4. 在 `main.py` 中添加新的调度器选项

### 自定义任务生成

修改 `config/config.py` 中的 `TaskConfig`，或直接创建 `Task` 对象。

## 注意事项

1. 所有GPU都是同构的，显存大小相同
2. 任务的实际执行时间会乘以惩罚系数（如果跨机架或跨GPU）
3. 饿死阈值用于检测长时间等待的任务
4. 时间步长为1秒，可以在配置中调整
5. 结果以CSV格式保存，可以用Excel或其他工具打开分析

## 许可证

本项目用于研究和实验目的。
