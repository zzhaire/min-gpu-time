"""
全局绘图颜色和标签配置
用于统一所有绘图脚本中的调度器颜色和显示名称
"""

# 调度器颜色配置 (固定颜色)
SCHEDULER_COLORS = {
    # PACE (Ours) - 红色
    "pollux_patient": "#F47F72",
    "pollux-patient": "#F47F72",
    "pace": "#F47F72",
    
    # Pollux - 橙黄色
    "pollux": "#FBB463",
    
    # Rack Aware - 浅蓝色
    "rack_aware": "#80B1D3",
    "rack-aware": "#80B1D3",
    
    # Min GPU Time - 浅青色
    "min_gpu_time": "#8DD1C6",
    "min-gpu-time": "#8DD1C6",
    
    # First Fit - 灰紫色
    "first_fit": "#BDBADB",
    "first-fit": "#BDBADB",
    
    # Best Fit - 浅黄色
    "best_fit": "#FBF8B4",
    "best-fit": "#FBF8B4",
}

# 调度器显示名称映射
SCHEDULER_DISPLAY_NAMES = {
    # PACE (Ours)
    "pollux_patient": "PACE (Ours)",
    "pollux-patient": "PACE (Ours)",
    "pace": "PACE (Ours)",
    
    # Pollux
    "pollux": "Pollux",
    
    # Rack Aware
    "rack_aware": "Rack Aware",
    "rack-aware": "Rack Aware",
    
    # Min GPU Time
    "min_gpu_time": "Min GPU Time",
    "min-gpu-time": "Min GPU Time",
    
    # First Fit
    "first_fit": "First Fit",
    "first-fit": "First Fit",
    
    # Best Fit
    "best_fit": "Best Fit",
    "best-fit": "Best Fit",
}

# 默认颜色 (当调度器不在配置中时使用)
DEFAULT_COLOR = "#808080"  # 灰色

# 颜色列表 (按优先级排序，用于没有预设颜色时)
COLOR_PALETTE = [
    "#F47F72",  # 红 (PACE)
    "#FBB463",  # 橙黄 (Pollux)
    "#80B1D3",  # 浅蓝 (Rack Aware)
    "#8DD1C6",  # 浅青 (Min GPU Time)
    "#BDBADB",  # 灰紫 (First Fit)
    "#FBF8B4",  # 浅黄 (Best Fit)
]


def get_scheduler_color(scheduler_key: str) -> str:
    """
    获取调度器的颜色
    
    Args:
        scheduler_key: 调度器键名 (如 'pollux_patient', 'pollux' 等)
    
    Returns:
        颜色 hex 字符串
    """
    # 标准化键名
    key = scheduler_key.lower().replace("-", "_")
    return SCHEDULER_COLORS.get(key, DEFAULT_COLOR)


def get_scheduler_display_name(scheduler_key: str) -> str:
    """
    获取调度器的显示名称
    
    Args:
        scheduler_key: 调度器键名
    
    Returns:
        显示名称字符串
    """
    # 标准化键名
    key = scheduler_key.lower().replace("-", "_")
    return SCHEDULER_DISPLAY_NAMES.get(key, scheduler_key)


def get_color_palette(index: int) -> str:
    """
    从调色板获取颜色
    
    Args:
        index: 颜色索引
    
    Returns:
        颜色 hex 字符串
    """
    return COLOR_PALETTE[index % len(COLOR_PALETTE)]
