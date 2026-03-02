# -*- coding: utf-8 -*-
"""
集群管理工具：GPU解析、空闲检测、端口查找等
"""
import re

# 默认空闲阈值：显存利用率、GPU利用率低于此值视为空闲
DEFAULT_MEM_THRESHOLD = 20   # 显存使用率 %
DEFAULT_UTIL_THRESHOLD = 15  # GPU算力使用率 %

def _strip_ansi(text):
    """移除 gpustat 输出中的 ANSI 颜色转义码"""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def parse_gpustat_output(gpu_status_text):
    """
    解析 gpustat 输出，返回每块 GPU 的详细信息
    返回: [{'index': 0, 'name': '...', 'mem_used': 0, 'mem_total': 0, 'util': 0, 'temp': 0, 'processes': '...'}, ...]
    """
    result = []
    gpu_status_text = _strip_ansi(gpu_status_text or '')
    lines = gpu_status_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or 'not found' in line.lower() or 'command not found' in line.lower():
            continue
        
        # gpustat 格式: [0] NVIDIA GeForce RTX 3090 | 45°C, 12% | 1024 / 24576 MiB | python(1234)
        # 或: 45'C, 87 % | 5120 / 12288 MB  （支持 °/' 以及 数字和 % 之间的空格）
        match = re.search(r'\[(\d+)\]\s+(.*?)\s*\|\s*(\d+)(?:[°\'])?C,?\s*(\d+)\s*%\s*\|\s*(\d+)\s*/\s*(\d+)\s*M[iI]?B\s*(?:\|\s*(.*))?$', line)
        if match:
            idx, name, temp, util, mem_used, mem_total, processes = match.groups()
            mem_used = int(mem_used)
            mem_total = int(mem_total)
            util = int(util)
            temp = int(temp)
            mem_percent = round(mem_used / mem_total * 100) if mem_total > 0 else 0
            result.append({
                'index': int(idx),
                'name': name.strip(),
                'mem_used': mem_used,
                'mem_total': mem_total,
                'mem_percent': mem_percent,
                'util': util,
                'temp': temp,
                'processes': (processes or '').strip()
            })
    
    return result


def find_idle_gpus(gpu_status_text, mem_threshold=None, util_threshold=None, reserved_gpu_count=0):
    """
    从 gpustat 输出中找出空闲的 GPU
    reserved_gpu_count: 每台服务器保留的空闲卡数量（用于模型调度时的冗余，默认0）
    """
    mem_threshold = mem_threshold or DEFAULT_MEM_THRESHOLD
    util_threshold = util_threshold or DEFAULT_UTIL_THRESHOLD
    
    gpus = parse_gpustat_output(gpu_status_text)
    idle = []
    for g in gpus:
        if g['mem_percent'] <= mem_threshold and g['util'] <= util_threshold:
            idle.append(g['index'])
    
    # 若需要保留冗余卡，则只返回 (总空闲数 - 保留数) 的卡
    if reserved_gpu_count > 0 and len(idle) > reserved_gpu_count:
        idle = idle[:len(idle) - reserved_gpu_count]
    
    return idle


def parse_used_ports(port_list_output):
    """解析 netstat 或 ss 输出的已用端口列表"""
    used = set()
    for line in port_list_output.split('\n'):
        line = line.strip()
        if not line:
            continue
        # 匹配 :PORT 或 *:PORT 或 0.0.0.0:PORT
        m = re.search(r':(\d{4,5})\s', line) or re.search(r'\*:(\d{4,5})\b', line) or re.search(r'0\.0\.0\.0:(\d{4,5})\b', line)
        if m:
            p = int(m.group(1))
            if 1024 <= p <= 65535:
                used.add(p)
    return used


def find_available_port(used_ports, base_port=18000, max_try=100):
    """在 base_port 起寻找一个未占用的端口"""
    for i in range(max_try):
        port = base_port + i
        if port not in used_ports and port <= 65535:
            return port
    return None


# 常用端口范围，避免与已知服务冲突
PORT_RANGE_START = 18000
PORT_RANGE_END = 19000
