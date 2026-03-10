from flask import Flask, render_template, jsonify, request, send_file, session, redirect, url_for, Response
import json
import logging
import paramiko
import socket
import time
import threading
import os
import tempfile
import re
import shlex
import secrets
import hashlib
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from database import DatabaseManager
from cluster_utils import parse_gpustat_output, find_idle_gpus, find_available_port, parse_used_ports

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.disabled = True

# 日志器（替代散落的 print 调试输出）
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler()
    # 显示时间、级别、文件名和代码行号，便于快速定位
    _formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s'
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)

# 初始化数据库管理器
db = DatabaseManager()

# 全局变量存储服务器状态
server_status = {}

# 存储用户私钥的临时文件
user_keys = {}

# 存储授权码
auth_codes = {}

# 东八区时区
CST = timezone(timedelta(hours=8))

# 集群默认配置 key
CONFIG_CODE_PATH = 'code_path'
CONFIG_DATA_PATH = 'data_path'
CONFIG_MEM_THRESHOLD = 'gpu_mem_threshold'
CONFIG_UTIL_THRESHOLD = 'gpu_util_threshold'
CONFIG_TRAIN_MEM_THRESHOLD = 'train_gpu_mem_threshold'
CONFIG_TRAIN_UTIL_THRESHOLD = 'train_gpu_util_threshold'
CONFIG_TEST_MEM_THRESHOLD = 'test_gpu_mem_threshold'
CONFIG_TEST_UTIL_THRESHOLD = 'test_gpu_util_threshold'
CONFIG_RESERVED_GPU = 'reserved_gpu_count'
CONFIG_RESERVED_GPU_PER_SERVER = 'reserved_gpu_per_server'  # 各服务器独立的保留卡数配置（JSON格式）
CONFIG_GPU_LOCK_TTL_TRAIN = 'gpu_lock_ttl_train_seconds'  # 训练GPU预占锁超时（秒）
CONFIG_GPU_LOCK_TTL_DEPLOY = 'gpu_lock_ttl_deploy_seconds'  # 部署GPU预占锁超时（秒）
CONFIG_SERVER_GROUPS = 'server_groups'  # 服务器分组配置

# 验证管理员权限的装饰器
def require_admin(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return jsonify({'success': False, 'error': '需要管理员权限'}), 401
        return f(*args, **kwargs)
    return decorated_function

def load_servers():
    """从数据库加载服务器列表"""
    return db.get_all_servers()


def _ssh_connect_params(server):
    """根据服务器配置返回 SSH 连接参数，支持密码或密钥"""
    params = {
        'hostname': server['ip'],
        'port': server.get('port', 22),
        'username': server['username'],
        'timeout': 10,
    }
    auth_type = server.get('auth_type') or 'password'
    key_path = (server.get('key_path') or '').strip()
    if auth_type == 'key' and key_path:
        path = os.path.expanduser(key_path)
        if os.path.isfile(path):
            passphrase = (server.get('key_passphrase') or '').strip() or None
            for key_class in (paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey):
                try:
                    params['pkey'] = key_class.from_private_key_file(path, password=passphrase or '')
                    params['look_for_keys'] = False
                    return params
                except (paramiko.ssh_exception.SSHException, paramiko.ssh_exception.PasswordRequiredException):
                    continue
            try:
                params['pkey'] = paramiko.PKey.from_private_key_file(path, password=passphrase or '')
                params['look_for_keys'] = False
                return params
            except paramiko.ssh_exception.PasswordRequiredException:
                pass  # 需要密码但未提供，fallback 到 password
            except Exception:
                pass
            params['key_filename'] = path
            params['look_for_keys'] = False
            return params
    params['password'] = server.get('password', '')
    return params


def execute_ssh_command(server, command):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**_ssh_connect_params(server))
        
        #stdin, stdout, stderr = ssh.exec_command(command)
        full_command = f'source ~/.bashrc 2>/dev/null; source ~/.profile 2>/dev/null; {command}'
        stdin, stdout, stderr = ssh.exec_command(full_command)
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        
        ssh.close()
        
        if error:
            return f"Error: {error}"
        return output
    except Exception as e:
        return f"Connection Error: {str(e)}"

def get_server_status(server):
    status = {
        'name': server['name'],
        'ip': server['ip'],
        'port': server.get('port', 22),
        'timestamp': datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S'),
        'disk_usage': 'Loading...',
        'gpu_status': 'Loading...'
    }
    
    # 获取磁盘使用情况
    disk_output = execute_ssh_command(server, 'df -h')
    status['disk_usage'] = disk_output
    
    # 检查并安装 gpustat（如果未安装）
    check_cmd = 'gpustat --version 2>&1 || which gpustat 2>&1 || command -v gpustat 2>&1'
    check_output = execute_ssh_command(server, check_cmd)
    
    # 如果检查失败（没有找到 gpustat），尝试安装
    if not check_output.strip() or ('not found' in check_output.lower() or 'command not found' in check_output.lower()):
        # gpustat 未安装，尝试使用 pip 安装
        install_cmd = 'pip install gpustat 2>&1 || pip3 install gpustat 2>&1 || python -m pip install gpustat 2>&1 || python3 -m pip install gpustat 2>&1'
        install_output = execute_ssh_command(server, install_cmd)
        # 如果安装失败，记录错误信息
        if 'successfully' not in install_output.lower() and ('error' in install_output.lower() or 'permission denied' in install_output.lower()):
            status['gpu_status'] = f'安装 gpustat 失败: {install_output}'
            return status
    
    # 获取GPU状态
    gpu_output = execute_ssh_command(server, 'gpustat')
    status['gpu_status'] = gpu_output
    
    return status

def update_single_server(server):
    """更新单个服务器状态（用于并发执行）"""
    try:
        status = get_server_status(server)
        return (server['name'], status)
    except Exception as e:
        # 如果更新失败，返回错误状态
        return (server['name'], {
            'name': server['name'],
            'ip': server['ip'],
            'port': server.get('port', 22),
            'timestamp': datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S'),
            'disk_usage': f'Error: {str(e)}',
            'gpu_status': f'Error: {str(e)}'
        })

def update_all_servers():
    global server_status
    
    # 创建线程池（最多10个并发线程，在循环外创建以提高效率）
    executor = ThreadPoolExecutor(max_workers=10)
    
    try:
        while True:
            servers = load_servers()  # 每次循环重新加载服务器列表
            current_server_names = {server['name'] for server in servers}
            
            # 清理已删除的服务器状态
            for server_name in list(server_status.keys()):
                if server_name not in current_server_names:
                    del server_status[server_name]
            
            # 并发更新所有服务器状态
            future_to_server = {executor.submit(update_single_server, server): server for server in servers}
            
            # 收集更新结果
            for future in as_completed(future_to_server):
                try:
                    server_name, status = future.result()
                    server_status[server_name] = status
                    # 内存报警：显存使用率>90%时记录
                    try:
                        gpus = parse_gpustat_output(status.get('gpu_status', ''))
                        for g in gpus:
                            if g.get('mem_percent', 0) > 90:
                                db.add_memory_alert(server_name, 'gpu_memory', f"GPU {g['index']} 显存使用率 {g['mem_percent']}%")
                    except Exception:
                        pass
                except Exception as e:
                    server = future_to_server[future]
                    logger.info(f"更新服务器 {server['name']} 时发生错误: {str(e)}")
            
            time.sleep(30)  # 每30秒更新一次
    finally:
        executor.shutdown(wait=True)


# ========== 集群任务调度辅助 ==========
def execute_ssh_command_silent(server, command, timeout=60):
    """执行 SSH 命令，返回 (success, output)。会打印执行日志便于排查失败原因。"""
    server_name = server.get('name', '?')
    code_path = server.get('code_path') or db.get_config(CONFIG_CODE_PATH, '/home')
    full_cmd = f'source ~/.bashrc 2>/dev/null; source ~/.profile 2>/dev/null; cd {code_path} 2>/dev/null; {command}'
    logger.info(f"[SSH] server={server_name} code_path={code_path}")
    logger.info(f"[SSH] command= {full_cmd[:500]}{'...' if len(full_cmd) > 500 else ''}")

    out = ''
    err = ''
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**_ssh_connect_params(server))
        stdin, stdout, stderr = ssh.exec_command(full_cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode('utf-8', errors='replace')
        ssh.close()
        combined = out + (('\n' + err) if err else '')
        logger.info(f"[SSH] server={server_name} ok=True output_preview= {repr(combined[:400])}{'...' if len(combined) > 400 else ''}")
        return True, combined
    except Exception as e:
        # 超时特殊处理：命令多半已在远端成功执行，这里按“成功”处理，避免重复执行
        if isinstance(e, (TimeoutError, socket.timeout)):
            err_msg = f"{type(e).__name__}: {repr(e)}"
            logger.warning(
                f"[SSH] server={server_name} timeout, "
                f"treat as success. error={err_msg} command={command}"
            )
            combined = out + (('\n' + err) if err else '')
            suffix = f"[SSH_TIMEOUT_TREATED_AS_SUCCESS] {err_msg}"
            return True, (combined + '\n' if combined else '') + suffix

        # 其他异常：按失败处理，返回 False
        err_msg = f"{type(e).__name__}: {repr(e)}"
        logger.exception(f"[SSH] server={server_name} exception while running command={command}")
        return False, err_msg


def kill_process_group(server, pid, timeout=15):
    """
    在远程服务器上尽量干净地杀掉该 PID 及其所有子进程：
    - 递归查出该 PID 的所有子进程（包括多级子进程）
    - 对收集到的所有 PID 执行 kill -9
    - 最后通过 ps 校验是否还有这些 PID 存活
    """
    cmd = (
        f'root={pid}; '
        'collect_children() { '
        '  local p=$1; '
        '  echo "$p"; '
        '  for c in $(ps -o pid= --ppid "$p" 2>/dev/null); do '
        '    collect_children "$c"; '
        '  done; '
        '}; '
        'pids=$(collect_children "$root" 2>/dev/null | tr "\\n" " "); '
        'kill -9 $pids 2>/dev/null || true; '
        'sleep 1; '
        'for p in $pids; do ps -o pid= -p "$p" 2>/dev/null; done | tr -d " \\n "'
    )
    ok, out = execute_ssh_command_silent(server, cmd, timeout=timeout)
    if not ok:
        return False, False, out
    alive = bool((out or '').strip())
    return True, not alive, out


def freeze_training_script_for_task(task_id, script_path, allowed_servers):
    """
    提交训练任务时冻结脚本：
    - 在每台相关服务器的代码目录下创建 .cluster_snapshots 目录
    - 将当前脚本拷贝为带 task_id+时间戳 的新文件
    - 更新训练任务的 script_path 指向该快照文件
    """
    # 选择需要冻结的服务器
    servers = []
    if allowed_servers:
        names = [str(s).strip() for s in (allowed_servers or []) if str(s).strip()]
        for name in names:
            srv = db.get_server_by_name(name)
            if srv:
                servers.append(srv)
    else:
        servers = load_servers()
    if not servers:
        return False, '没有可用服务器用于冻结脚本'

    base = os.path.basename(script_path)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = '.sh'
    ts = int(time.time())
    snapshot_rel = f".cluster_snapshots/{root}_task{task_id}_{ts}{ext}"

    errors = []
    for srv in servers:
        # 在对应服务器代码目录下复制当前脚本到快照路径
        cmd = f"mkdir -p .cluster_snapshots && cp -f {script_path} {snapshot_rel}"
        logger.info(f"[Freeze] server={srv['name']} copy cmd= {cmd}")
        ok, out = execute_ssh_command_silent(srv, cmd, timeout=20)
        if not ok:
            errors.append(f"{srv['name']}: {out}")
            # 若为源文件不存在，发出警报
            out_lower = (out or '').lower()
            if 'no such file or directory' in out_lower or 'cannot stat' in out_lower:
                db.add_memory_alert(
                    srv['name'], 'script_not_found',
                    f'训练脚本不存在，无法冻结: {script_path}。错误: {(out or "").strip()[:300]}'
                )
        else:
            logger.info(f"[Freeze] server={srv['name']} ok -> snapshot {snapshot_rel}")

    if errors:
        return False, '; '.join(errors)

    # 更新任务的脚本路径为快照路径
    db.update_training_task(task_id, script_path=snapshot_rel)
    return True, snapshot_rel


def get_used_ports_on_server(server):
    """获取服务器上已占用的端口"""
    ok, out = execute_ssh_command_silent(server, 'ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null || netstat -an 2>/dev/null')
    if not ok:
        return set()
    return parse_used_ports(out)


def find_idle_server_and_gpus(gpu_count=1, reserved=0, task_type='train', allowed_servers=None):
    """在集群中找一台有空闲 GPU 的服务器，返回 (server, [gpu_ids]) 或 (None, [])
    task_type: 'train' 用训练阈值(显存/算力要求高)，'deploy' 用部署阈值(原测试阈值)
    allowed_servers: 可选，服务器名列表；为空或 None 时考虑所有服务器
    """
    if task_type in ('deploy', 'test'):
        mem_th = int(db.get_config(CONFIG_TEST_MEM_THRESHOLD) or db.get_config(CONFIG_MEM_THRESHOLD, 20))
        util_th = int(db.get_config(CONFIG_TEST_UTIL_THRESHOLD) or db.get_config(CONFIG_UTIL_THRESHOLD, 15))
        lock_ttl = int(db.get_config(CONFIG_GPU_LOCK_TTL_DEPLOY, 300))
    else:
        mem_th = int(db.get_config(CONFIG_TRAIN_MEM_THRESHOLD) or db.get_config(CONFIG_MEM_THRESHOLD, 20))
        util_th = int(db.get_config(CONFIG_TRAIN_UTIL_THRESHOLD) or db.get_config(CONFIG_UTIL_THRESHOLD, 15))
        lock_ttl = int(db.get_config(CONFIG_GPU_LOCK_TTL_TRAIN, 300))
    # 读取保留卡数配置（全局默认值 + 各服务器独立配置）
    default_reserved = int(db.get_config(CONFIG_RESERVED_GPU, 0))
    reserved_per_server_str = db.get_config(CONFIG_RESERVED_GPU_PER_SERVER, '{}')
    try:
        reserved_per_server = json.loads(reserved_per_server_str) if reserved_per_server_str else {}
    except json.JSONDecodeError:
        reserved_per_server = {}

    servers = load_servers()
    logger.info(f"[Scheduler] find_idle_server_and_gpus task_type={task_type} gpu_count={gpu_count} default_reserved={default_reserved} allowed={allowed_servers}")
    if allowed_servers:
        allow_set = set(s.strip() for s in allowed_servers if s and str(s).strip())
        servers = [s for s in servers if s['name'] in allow_set]
    for srv in servers:
        st = server_status.get(srv['name'], {})
        gpu_text = st.get('gpu_status', '')
        # 获取该服务器的保留卡数（优先使用独立配置，否则使用默认值）
        srv_reserved = reserved_per_server.get(srv['name'], default_reserved)
        idle = find_idle_gpus(gpu_text, mem_threshold=mem_th, util_threshold=util_th, reserved_gpu_count=srv_reserved)

        # 过滤掉被预占锁的GPU（锁文件中已包含独立的到期时间戳，无需传入TTL）
        available = []
        for gpu_id in idle:
            is_locked = check_gpu_locked(srv, gpu_id)
            if is_locked:
                logger.info(f"[Scheduler]   server={srv['name']} gpu={gpu_id} is locked, skip")
            else:
                available.append(gpu_id)

        logger.info(f"[Scheduler]   server={srv['name']} reserved={srv_reserved} idle_gpus={idle} available_gpus={available} locked_count={len(idle)-len(available)} (need {gpu_count})")
        if len(available) >= gpu_count:
            chosen = available[:gpu_count]
            logger.info(f"[Scheduler]   -> choose server={srv['name']} gpus={chosen}")
            return srv, chosen
    logger.info("[Scheduler]   -> no server has enough GPUs")
    return None, []


def find_available_server(allowed_servers=None):
    """找一台可用服务器（不依赖 GPU）。"""
    servers = load_servers()
    if allowed_servers:
        allow_set = set(s.strip() for s in allowed_servers if s and str(s).strip())
        servers = [s for s in servers if s['name'] in allow_set]
    if not servers:
        return None

    # 优先挑当前状态看起来连接正常的服务器
    for srv in servers:
        st = server_status.get(srv['name'], {}) or {}
        gpu_text = (st.get('gpu_status') or '').strip()
        if not gpu_text:
            return srv
        if 'Connection Error' in gpu_text or gpu_text.startswith('Error:') or gpu_text == 'Loading...':
            continue
        return srv
    return servers[0]


def find_available_port_on_server(server, base=18000):
    used = get_used_ports_on_server(server)
    return find_available_port(used, base_port=base)


def try_preempt_gpu(server, gpu_id, task_id, ttl_seconds=None, task_type='train'):
    """
    尝试远程预占GPU，TTL时间内其他调度器会跳过
    锁文件格式: task_id expire_timestamp
    返回: (success: bool, error_msg: str)
    """
    if ttl_seconds is None:
        if task_type in ('deploy', 'test'):
            ttl_seconds = int(db.get_config(CONFIG_GPU_LOCK_TTL_DEPLOY, 300))
        else:
            ttl_seconds = int(db.get_config(CONFIG_GPU_LOCK_TTL_TRAIN, 300))
    
    lock_file = f"/tmp/gpu_scheduler_locks/{server['name']}_gpu_{gpu_id}.lock"
    
    cmd = f'''
    mkdir -p /tmp/gpu_scheduler_locks
    now=$(date +%s)
    expire=$((now + {ttl_seconds}))
    
    # 检查锁是否存在且未过期（比较expire时间戳）
    if [ -f "{lock_file}" ]; then
        lock_expire=$(cat "{lock_file}" | awk '{{print $2}}')
        if [ "$lock_expire" -gt "$now" ] 2>/dev/null; then
            remain=$((lock_expire - now))
            echo "locked $remain"
            exit 0
        fi
    fi
    
    # 创建新锁（原子操作），写入到期时间戳
    echo "{task_id} $expire" > "{lock_file}" && echo "success"
    '''
    
    ok, out = execute_ssh_command_silent(server, cmd, timeout=10)
    
    if "success" in out:
        return True, ""
    elif "locked" in out:
        remain = out.strip().split()[-1] if len(out.strip().split()) > 1 else "unknown"
        return False, f"GPU {gpu_id} 被预占（剩余{remain}秒）"
    return False, f"预占检查失败: {out}"


def release_gpu_lock(server, gpu_id):
    """主动释放GPU预占锁"""
    lock_file = f"/tmp/gpu_scheduler_locks/{server['name']}_gpu_{gpu_id}.lock"
    cmd = f"rm -f {lock_file}"
    execute_ssh_command_silent(server, cmd, timeout=5)


def cleanup_expired_locks(server):
    """清理超时的GPU锁文件
    锁文件格式: task_id expire_timestamp
    直接比较expire_timestamp和当前时间
    """
    cmd = '''
    if [ -d /tmp/gpu_scheduler_locks ]; then
        now=$(date +%s)
        for lock_file in /tmp/gpu_scheduler_locks/*.lock; do
            [ -f "$lock_file" ] || continue
            lock_expire=$(cat "$lock_file" | awk '{print $2}')
            if [ "$lock_expire" -le "$now" ] 2>/dev/null; then
                rm -f "$lock_file"
            fi
        done
    fi
    echo "done"
    '''
    execute_ssh_command_silent(server, cmd, timeout=10)


def check_gpu_locked(server, gpu_id):
    """检查GPU是否被预占（不创建锁，只检查）
    锁文件格式: task_id expire_timestamp
    返回True表示被锁定，False表示可用
    """
    lock_file = f"/tmp/gpu_scheduler_locks/{server['name']}_gpu_{gpu_id}.lock"
    
    cmd = f'''
    if [ -f "{lock_file}" ]; then
        lock_expire=$(cat "{lock_file}" | awk '{{print $2}}')
        now=$(date +%s)
        if [ "$lock_expire" -gt "$now" ] 2>/dev/null; then
            echo "locked"
            exit 0
        fi
    fi
    echo "available"
    '''
    
    ok, out = execute_ssh_command_silent(server, cmd, timeout=10)
    is_locked = "locked" in out if ok else False
    logger.info(f"[LockCheck] server={server['name']} gpu={gpu_id} locked={is_locked} out={out!r}")
    return is_locked


def extract_port_from_url(url):
    """从 URL 里提取端口，提取失败返回 None。"""
    if not url:
        return None
    m = re.search(r':(\d+)(?:/|$)', str(url).strip())
    if not m:
        return None
    try:
        p = int(m.group(1))
        if 1 <= p <= 65535:
            return p
    except (TypeError, ValueError):
        return None
    return None


def extract_save_folder_from_output(output_text):
    """仅从返回文本中提取 `save_folder: xxx`。"""
    text = output_text or ''
    m = re.search(r'save_folder\s*:\s*(.+)', text, flags=re.IGNORECASE)
    if not m:
        return ''
    raw = (m.group(1) or '').strip()
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()
    if raw.startswith('./'):
        raw = raw[2:]
    return raw


def _read_remote_log_with_limit(server, log_path, lines=None, is_tail=True):
    """读取远程日志；lines 为正整数时仅返回最后 N 行。"""
    path = str(log_path or '').strip()
    if not path:
        return False, '日志路径为空'
    qpath = shlex.quote(path)

    n = None
    if lines is not None:
        try:
            n = int(lines)
        except (TypeError, ValueError):
            n = None
    if n is not None and n > 0:
        cmd = f'if [ -f {qpath} ]; then {"tail" if is_tail else "head"} -n {n} {qpath}; else echo "(日志文件不存在)"; fi'
    else:
        cmd = f'if [ -f {qpath} ]; then cat {qpath}; else echo "(日志文件不存在)"; fi'
    return execute_ssh_command_silent(server, cmd)


def _build_log_preamble_cmd(log_path, lines):
    """生成写入日志头的 shell 片段，先清空日志再逐行写入。"""
    qlog = shlex.quote(str(log_path))
    cmds = [f': > {qlog}']
    for line in (lines or []):
        cmds.append(f"printf '%s\\n' {shlex.quote(str(line))} >> {qlog}")
    cmds.append(f"printf '%s\\n' '' >> {qlog}")
    return '; '.join(cmds)


def is_remote_pid_alive(server, pid, timeout=12):
    """检查远程 PID 是否仍存活：True=存活，False=不存在，None=检查失败。"""
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False

    check_cmd = f'ps -p {pid_int} -o pid= 2>/dev/null | tr -d "[:space:]"'
    ok, out = execute_ssh_command_silent(server, check_cmd, timeout=timeout)
    if not ok:
        return None
    return (out or '').strip() == str(pid_int)


def extract_weight_from_log(log_content):
    """从日志内容中提取权重路径，支持格式: [save_folder] /path/to/folder"""
    if not log_content:
        return None
    # 匹配 [save_folder] 开头的行
    m = re.search(r'^\[save_folder\]\s*(.+)$', log_content, re.MULTILINE | re.IGNORECASE)
    if m:
        path = m.group(1).strip()
        # 去除可能的引号
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1].strip()
        return path if path else None
    return None


def extract_port_from_log(log_content):
    """从日志内容中提取端口号，支持格式: Starting API server on port XXXX"""
    if not log_content:
        return None
    # 匹配 "Starting API server on port XXXX" 格式
    m = re.search(r'Starting API server on port\s+(\d+)', log_content, re.IGNORECASE)
    if m:
        port = int(m.group(1))
        if 1 <= port <= 65535:
            return port
    return None


def reconcile_running_tasks():
    """巡检 running 任务，若远程 PID 已结束则自动置为 done。
    同时自动从日志提取权重路径和端口号（如果尚未记录）。"""
    now = datetime.now(CST).isoformat()

    for task in db.get_running_training_tasks():
        server_name = (task.get('server_name') or '').strip()
        pid = task.get('pid')
        if not server_name or not pid:
            continue
        server = db.get_server_by_name(server_name)
        if not server:
            continue
        alive = is_remote_pid_alive(server, pid)
        if alive is False:
            db.update_training_task(task['id'], status='done', finished_at=now, pid=None)
            logger.info(f"[Health] training task_id={task['id']} pid={pid} ended -> done")
        elif alive is None:
            logger.info(f"[Health] training task_id={task['id']} pid={pid} check failed, keep running")

        # 自动从日志提取权重路径（如果尚未记录）
        if alive is not False and not (task.get('weight_path') or '').strip():
            log_path = task.get('log_path')
            if log_path:
                ok, content = _read_remote_log_with_limit(server, log_path, lines=100, is_tail=False)
                if ok:
                    weight_path = extract_weight_from_log(content)
                    if weight_path:
                        db.update_training_task(task['id'], weight_path=weight_path)
                        db.upsert_model_weight_for_task(
                            task['id'],
                            task.get('name', 'task') + '_' + str(task['id']),
                            weight_path,
                            server_name
                        )
                        logger.info(f"[AutoExtract] training task_id={task['id']} weight_path={weight_path}")

    for task in db.get_running_deploy_tasks():
        server_name = (task.get('server_name') or '').strip()
        pid = task.get('pid')
        if not server_name or not pid:
            continue
        server = db.get_server_by_name(server_name)
        if not server:
            continue
        alive = is_remote_pid_alive(server, pid)
        if alive is False:
            db.update_deploy_task(task['id'], status='done', finished_at=now, pid=None)
            logger.info(f"[Health] deploy task_id={task['id']} pid={pid} ended -> done")
        elif alive is None:
            logger.info(f"[Health] deploy task_id={task['id']} pid={pid} check failed, keep running")

        # 自动从日志提取端口号（如果尚未记录）
        if alive is not False and not task.get('port'):
            log_path = task.get('log_path')
            if log_path:
                ok, content = _read_remote_log_with_limit(server, log_path, lines=100, is_tail=False)
                if ok:
                    port = extract_port_from_log(content)
                    if port:
                        db.update_deploy_task(task['id'], port=port)
                        logger.info(f"[AutoExtract] deploy task_id={task['id']} port={port}")

    for task in db.get_running_test_tasks():
        server_name = (task.get('server_name') or '').strip()
        pid = task.get('pid')
        if not server_name or not pid:
            continue
        server = db.get_server_by_name(server_name)
        if not server:
            continue
        alive = is_remote_pid_alive(server, pid)
        if alive is False:
            db.update_test_task(task['id'], status='done', finished_at=now, pid=None)
            logger.info(f"[Health] test task_id={task['id']} pid={pid} ended -> done")
        elif alive is None:
            logger.info(f"[Health] test task_id={task['id']} pid={pid} check failed, keep running")


def run_training_on_server(task, server, gpu_ids):
    """在指定服务器上启动训练任务（支持 .sh 与 .py）"""
    code_path = server.get('code_path') or db.get_config(CONFIG_CODE_PATH, '/home')
    script = task['script_path']
    args = task.get('script_args') or ''
    task_name = (task.get('task_name') or '').strip()
    gpu_str = ','.join(map(str, gpu_ids))
    log_path = f"{str(code_path).rstrip('/')}/outputs/logs/train_{task['id']}_{int(time.time())}.log"
    runner = 'bash' if script.lower().endswith('.sh') else 'python'
    env_prefix = f'CUDA_VISIBLE_DEVICES={gpu_str}' if gpu_str else ''
    has_task_name_arg = bool(re.search(r'(^|\s)--task_name(?:\s|=)', args))
    task_name_part = f' --task_name {shlex.quote(task_name)}' if task_name and not has_task_name_arg else ''
    launch_core = f'{runner} {script} {args}{task_name_part}'.strip()
    launch_cmd = f'{env_prefix} {launch_core}'.strip()
    exec_cmd = f'env {env_prefix} {launch_core}'.strip() if env_prefix else launch_core
    preamble_cmd = _build_log_preamble_cmd(log_path, [
        f'[START_AT] {datetime.now(CST).isoformat()}',
        f'[TASK_TYPE] train',
        f'[TASK_ID] {task["id"]}',
        f'[SERVER] {server["name"]}',
        f'[WORKDIR] {code_path}',
        f'[LAUNCH_CMD] {launch_cmd}'
    ])
    cmd = f'mkdir -p outputs/logs; {preamble_cmd}; nohup {exec_cmd} >> {shlex.quote(log_path)} 2>&1 & echo $!'
    logger.info(f"[RunTrain] task_id={task['id']} server={server['name']} code_path={code_path} script={script} log={log_path}")
    logger.info(f"[RunTrain] full_cmd= {cmd}")
    ok, out = execute_ssh_command_silent(server, cmd, timeout=30)
    if not ok:
        return False, out

    # 从输出中严格解析 PID（最后一行必须是纯数字）
    lines = [ln.strip() for ln in (out or '').split('\n') if ln.strip()]
    if not lines:
        return False, f'启动训练失败，未获取到 PID，输出为空: {out}'
    last = lines[-1]
    pid_match = re.fullmatch(r'(\d+)', last)
    if not pid_match:
        return False, f'启动训练失败，未获取到有效 PID，输出: {out}'
    pid = int(pid_match.group(1))
    db.update_training_task(task['id'], server_name=server['name'], gpu_ids=','.join(map(str, gpu_ids)),
                            status='running', log_path=log_path, pid=pid,
                            started_at=datetime.now(CST).isoformat())

    # 启动成功后，仅从返回文本中提取 save_folder 并记录权重路径
    try:
        if not (task.get('weight_path') or '').strip():
            auto_weight_path = extract_save_folder_from_output(out)
            if auto_weight_path:
                db.update_training_task(task['id'], weight_path=auto_weight_path)
                db.upsert_model_weight_for_task(
                    task['id'],
                    task.get('name', 'task') + '_' + str(task['id']),
                    auto_weight_path,
                    server.get('name', '')
                )
                logger.info(f"[RunTrain] task_id={task['id']} auto weight_path from returned save_folder: {auto_weight_path}")
    except Exception as e:
        logger.info(f"[RunTrain] task_id={task['id']} auto record weight_path skipped: {e}")

    return True, {'pid': pid, 'log_path': log_path}


def run_test_on_server(task, server, gpu_ids, port):
    """在指定服务器上启动测试任务（mock 或真机）"""
    code_path = server.get('code_path') or db.get_config(CONFIG_CODE_PATH, '/home')
    test_code_path = task.get('test_code_path') or code_path
    script = task.get('script_path') or ''
    args = task.get('script_args') or ''
    task_type = task.get('task_type') or 'mock'
    mock_url = task.get('mock_url') or ''
    mock_task_name = task.get('mock_task_name') or ''
    user_token = task.get('user_token') or ''
    run_id = task.get('run_id') or ''
    action_nums = task.get('action_nums')
    gpu_str = ','.join(map(str, gpu_ids)) if gpu_ids else ''
    env = f'CUDA_VISIBLE_DEVICES={gpu_str}' if gpu_str else ''
    log_path = f"{str(code_path).rstrip('/')}/outputs/logs/test_{task['id']}_{int(time.time())}.log"
    runner = 'bash' if script.lower().endswith('.sh') else 'python'

    # 先记录日志路径，保证即使启动失败也能在列表里查看日志
    db.update_test_task(
        task['id'],
        server_name=server['name'],
        gpu_ids=','.join(map(str, gpu_ids)) if gpu_ids else '',
        port=port,
        log_path=log_path
    )
    extra_parts = [f'']
    if mock_url:
        extra_parts.append(f'--url {shlex.quote(str(mock_url))}')
    if mock_task_name:
        extra_parts.append(f'--task_name {shlex.quote(str(mock_task_name))}')
    if task_type == 'real':
        if user_token:
            extra_parts.append(f'--user_token {shlex.quote(str(user_token))}')
        if run_id:
            extra_parts.append(f'--run_id {shlex.quote(str(run_id))}')
        if action_nums not in (None, ''):
            try:
                extra_parts.append(f'--action_nums {int(action_nums)}')
            except (TypeError, ValueError):
                logger.info(f"[RunTest] task_id={task['id']} invalid action_nums={action_nums}, skip")
    extra_parts.append(f'--test_type {task_type}')
    extra = ' '.join(extra_parts)
    launch_core = f'{runner} {script} {args} {extra}'.strip()
    launch_cmd = f'{env} {launch_core}'.strip()
    exec_cmd = f'env {env} {launch_core}'.strip() if env else launch_core
    preamble_cmd = _build_log_preamble_cmd(log_path, [
        f'[START_AT] {datetime.now(CST).isoformat()}',
        f'[TASK_TYPE] test',
        f'[TASK_ID] {task["id"]}',
        f'[SERVER] {server["name"]}',
        f'[WORKDIR] {test_code_path}',
        f'[LAUNCH_CMD] {launch_cmd}'
    ])
    cmd = (
        f'mkdir -p outputs/logs; '
        f'cd {shlex.quote(str(test_code_path))} 2>/dev/null; '
        f'{preamble_cmd}; '
        f'nohup {exec_cmd} >> {shlex.quote(log_path)} 2>&1 & echo $!'
    )
    ok, out = execute_ssh_command_silent(server, cmd, timeout=30)
    if not ok:
        return False, out

    lines = [ln.strip() for ln in (out or '').split('\n') if ln.strip()]
    if not lines:
        return False, f'启动测试失败，未获取到 PID，输出为空: {out}'
    last = lines[-1]
    pid_match = re.fullmatch(r'(\d+)', last)
    if not pid_match:
        return False, f'启动测试失败，未获取到有效 PID，输出: {out}'
    pid = int(pid_match.group(1))
    db.update_test_task(task['id'], server_name=server['name'], gpu_ids=','.join(map(str, gpu_ids)) if gpu_ids else '',
                        port=port, status='running', pid=pid, log_path=log_path, started_at=datetime.now(CST).isoformat())
    return True, {'pid': pid, 'port': port, 'log_path': log_path}


def run_deploy_on_server(task, server, gpu_ids):
    """在指定服务器上启动部署任务（支持 .sh 与 .py）"""
    code_path = server.get('code_path') or db.get_config(CONFIG_CODE_PATH, '/home')
    script = task.get('script_path') or ''
    weight = task.get('weight_path') or ''
    port = task.get('port')
    use_norm = task.get('use_norm', False)
    gpu_str = ','.join(map(str, gpu_ids)) if gpu_ids else ''
    env = f'CUDA_VISIBLE_DEVICES={gpu_str}' if gpu_str else ''
    log_path = f"{str(code_path).rstrip('/')}/outputs/logs/deploy_{task['id']}_{int(time.time())}.log"
    runner = 'bash' if script.lower().endswith('.sh') else 'python'
    weight_arg = f' --weight {weight}' if weight else ''
    port_arg = f' --port {port}' if port not in (None, '') else ''
    norm_arg = ' --norm' if use_norm else ''
    launch_core = f'{runner} {script}{weight_arg}{port_arg}{norm_arg}'.strip()
    launch_cmd = f'{env} {launch_core}'.strip()
    exec_cmd = f'env {env} {launch_core}'.strip() if env else launch_core
    preamble_cmd = _build_log_preamble_cmd(log_path, [
        f'[START_AT] {datetime.now(CST).isoformat()}',
        f'[TASK_TYPE] deploy',
        f'[TASK_ID] {task["id"]}',
        f'[SERVER] {server["name"]}',
        f'[WORKDIR] {code_path}',
        f'[LAUNCH_CMD] {launch_cmd}'
    ])
    cmd = f'mkdir -p outputs/logs; {preamble_cmd}; nohup {exec_cmd} >> {shlex.quote(log_path)} 2>&1 & echo $!'
    logger.info(f"[RunDeploy] task_id={task['id']} server={server['name']} script={script} weight={weight} port={port} use_norm={use_norm} log={log_path}")
    logger.info(f"[RunDeploy] full_cmd= {cmd}")
    ok, out = execute_ssh_command_silent(server, cmd, timeout=30)
    if not ok:
        return False, out

    lines = [ln.strip() for ln in (out or '').split('\n') if ln.strip()]
    if not lines:
        return False, f'启动部署失败，未获取到 PID，输出为空: {out}'
    last = lines[-1]
    pid_match = re.fullmatch(r'(\d+)', last)
    if not pid_match:
        return False, f'启动部署失败，未获取到有效 PID，输出: {out}'
    pid = int(pid_match.group(1))
    db.update_deploy_task(
        task['id'],
        server_name=server['name'],
        gpu_ids=','.join(map(str, gpu_ids)) if gpu_ids else '',
        status='running',
        log_path=log_path,
        pid=pid,
        started_at=datetime.now(CST).isoformat()
    )
    return True, {'pid': pid, 'log_path': log_path}


def cluster_task_scheduler():
    """后台调度：处理待执行的训练和测试任务"""
    cleanup_counter = 0
    while True:
        try:
            # 先巡检 running 任务，避免进程结束后状态长期停留在 running
            reconcile_running_tasks()

            # 至少有一台服务器配置了 code_path 才调度
            servers = load_servers()
            has_any_code_path = any(s.get('code_path') for s in servers) or db.get_config(CONFIG_CODE_PATH)
            if not has_any_code_path:
                time.sleep(15)
                continue
            
            # 每隔几次循环清理一次过期锁（减少SSH频率）
            cleanup_counter += 1
            if cleanup_counter >= 6:  # 约60秒清理一次
                cleanup_counter = 0
                # 清理时直接读取每个锁的expire_timestamp判断，无需传入TTL
                for srv in servers:
                    cleanup_expired_locks(srv)
                    logger.info(f"[Scheduler] cleanup expired GPU locks on {srv['name']}")
            # 训练任务调度（使用训练用阈值，根据任务自己的 GPU 数量等待）
            for task in db.get_pending_training_tasks():
                allowed = task.get('allowed_servers') or []
                try:
                    task_gpu_count = int(task.get('gpu_count') or 1)
                except (TypeError, ValueError):
                    task_gpu_count = 1
                if task_gpu_count < 1:
                    task_gpu_count = 1
                logger.info(f"[Scheduler] try schedule train_task id={task['id']} name={task['name']} gpu_count={task_gpu_count} allowed={allowed}")
                server, gpu_ids = find_idle_server_and_gpus(
                    gpu_count=task_gpu_count,
                    reserved=0,
                    task_type='train',
                    allowed_servers=allowed if allowed else None
                )
                if server and gpu_ids:
                    # 先预占GPU，防止并发冲突
                    preempt_ok = True
                    for gpu_id in gpu_ids:
                        ok, err = try_preempt_gpu(server, gpu_id, f"train_{task['id']}", task_type='train')
                        if not ok:
                            logger.info(f"[Scheduler]   preempt GPU {gpu_id} failed: {err}")
                            preempt_ok = False
                            break
                    
                    if not preempt_ok:
                        # 有GPU被抢占了，跳过，下次调度重试
                        logger.info(f"[Scheduler]   skip train_task id={task['id']} due to GPU lock conflict")
                        continue
                    
                    logger.info(f"[Scheduler]   start train_task id={task['id']} on {server['name']} gpus={gpu_ids} OK")
                    ok, msg = run_training_on_server(task, server, gpu_ids)
                    if not ok:
                        logger.info(f"[Scheduler]   start train_task id={task['id']} FAILED: {msg}")
                        db.update_training_task(task['id'], status='failed', error_message=str(msg))
                        # 启动失败，释放锁（让其他调度器可以尝试）
                        for gpu_id in gpu_ids:
                            release_gpu_lock(server, gpu_id)
                    else:
                        logger.info(f"[Scheduler]   start train_task id={task['id']} on {server['name']} gpus={gpu_ids} OK")
                    time.sleep(2)  # 避免连续提交过快

            # 部署任务调度（使用部署阈值找 GPU）
            for task in db.get_pending_deploy_tasks():
                allowed = task.get('allowed_servers') or []
                server, gpu_ids = find_idle_server_and_gpus(
                    gpu_count=1,
                    reserved=0,
                    task_type='deploy',
                    allowed_servers=allowed if allowed else None
                )
                if not server:
                    server, gpu_ids = find_idle_server_and_gpus(
                        gpu_count=0,
                        reserved=0,
                        task_type='deploy',
                        allowed_servers=allowed if allowed else None
                    )
                    gpu_ids = []
                if server:
                    # 有GPU时先预占
                    preempt_ok = True
                    for gpu_id in gpu_ids:
                        ok, err = try_preempt_gpu(server, gpu_id, f"deploy_{task['id']}", task_type='deploy')
                        if not ok:
                            logger.info(f"[Scheduler]   preempt GPU {gpu_id} failed: {err}")
                            preempt_ok = False
                            break
                    
                    if not preempt_ok:
                        logger.info(f"[Scheduler]   skip deploy_task id={task['id']} due to GPU lock conflict")
                        continue
                    
                    logger.info(f"[Scheduler]   start deploy_task id={task['id']} on {server['name']} gpus={gpu_ids}")
                    ok, msg = run_deploy_on_server(task, server, gpu_ids)
                    if not ok:
                        logger.info(f"[Scheduler]   start deploy_task id={task['id']} FAILED: {msg}")
                        db.update_deploy_task(task['id'], status='failed', result=str(msg))
                        # 启动失败，释放锁
                        for gpu_id in gpu_ids:
                            release_gpu_lock(server, gpu_id)
                    else:
                        logger.info(f"[Scheduler]   start deploy_task id={task['id']} on {server['name']} OK")
                    time.sleep(2)

            # 测试任务调度（不依赖 GPU）
            for task in db.get_pending_test_tasks():
                fixed_server_name = (task.get('server_name') or '').strip()
                server = db.get_server_by_name(fixed_server_name) if fixed_server_name else None
                gpu_ids = []
                if server: 
                    port = extract_port_from_url(task.get('mock_url'))
                    if port:
                        ok, msg = run_test_on_server(task, server, gpu_ids, port)
                    else:
                        db.update_test_task(task['id'], status='failed', result='未找到可用端口')
                    time.sleep(2)
                else:
                    db.update_test_task(task['id'], status='failed', result='指定服务器不存在')
                    time.sleep(2)
        except Exception as e:
            logger.info(f"[Scheduler] Error: {e}")
        time.sleep(10)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin_panel():
    """管理员面板"""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('admin.html')

@app.route('/admin/login')
def admin_login():
    """管理员登录页面"""
    return render_template('admin_login.html')

@app.route('/api/admin/login', methods=['POST'])
def admin_login_api():
    """管理员登录API"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'error': '用户名和密码不能为空'})
    
    if db.verify_admin(username, password):
        session['admin_logged_in'] = True
        session['admin_username'] = username
        return jsonify({'success': True, 'message': '登录成功'})
    else:
        return jsonify({'success': False, 'error': '用户名或密码错误'})

@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    """管理员退出登录"""
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    return jsonify({'success': True, 'message': '退出成功'})

@app.route('/api/admin/change-password', methods=['POST'])
@require_admin
def change_admin_password():
    """修改管理员密码"""
    data = request.get_json()
    old_password = data.get('old_password')
    new_password = data.get('new_password')
    
    if not old_password or not new_password:
        return jsonify({'success': False, 'error': '旧密码和新密码不能为空'})
    
    # 验证旧密码
    if not db.verify_admin('admin', old_password):
        return jsonify({'success': False, 'error': '当前密码不正确'})
    
    # 验证新密码长度
    if len(new_password) < 6:
        return jsonify({'success': False, 'error': '新密码长度至少6位'})
    
    # 更新密码
    success, message = db.update_admin_password('admin', new_password)
    
    if success:
        return jsonify({'success': True, 'message': '密码修改成功'})
    else:
        return jsonify({'success': False, 'error': message})

@app.route('/api/admin/generate-auth-code', methods=['POST'])
@require_admin
def admin_generate_auth_code():
    """管理员生成授权码（无需再次验证密码）"""
    # 生成授权码
    auth_code = secrets.token_hex(16)
    auth_codes[auth_code] = {
        'created_at': datetime.now(CST),
        'expires_in': 1800  # 30分钟 = 1800秒
    }
    
    return jsonify({
        'success': True, 
        'auth_code': auth_code,
        'expires_in': 1800,
        'message': '授权码生成成功，有效期30分钟'
    })

@app.route('/api/status')
def get_status():
    return jsonify(server_status)



def verify_auth_code(auth_code):
    """验证授权码是否有效"""
    if auth_code not in auth_codes:
        return False
    
    code_info = auth_codes[auth_code]
    current_time = datetime.now(CST)
    
    # 检查是否过期
    time_diff = (current_time - code_info['created_at']).total_seconds()
    if time_diff > code_info['expires_in']:
        # 删除过期的授权码
        del auth_codes[auth_code]
        return False
    
    return True

@app.route('/api/get-auth-code', methods=['POST'])
def get_auth_code():
    data = request.get_json()
    admin_password = data.get('admin_password')
    
    # 验证管理员密码
    if not db.verify_admin('admin', admin_password):
        return jsonify({'success': False, 'error': '管理员密码错误'})
    
    # 生成授权码
    auth_code = secrets.token_hex(16)
    auth_codes[auth_code] = {
        'created_at': datetime.now(CST),
        'expires_in': 1800  # 30分钟 = 1800秒
    }
    
    return jsonify({
        'success': True, 
        'auth_code': auth_code,
        'expires_in': 1800,
        'message': '授权码生成成功，有效期30分钟'
    })

def create_user_on_server(server, username, admin_password):
    """在指定服务器上创建用户"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**_ssh_connect_params(server))
        
        # 创建用户命令序列
        commands = [
            f'echo "{admin_password}" | sudo -S useradd -s /bin/bash -m {username}',
            f'echo "{admin_password}" | sudo -S -u {username} ssh-keygen -t rsa -m PEM -f /home/{username}/.ssh/id_rsa -N ""',
            f'echo "{admin_password}" | sudo -S -u {username} cp /home/{username}/.ssh/id_rsa.pub /home/{username}/.ssh/authorized_keys',
            f'echo "{admin_password}" | sudo -S chmod 600 /home/{username}/.ssh/authorized_keys',
            f'echo "{admin_password}" | sudo -S chown {username}:{username} /home/{username}/.ssh/authorized_keys',
            f'echo "{admin_password}" | sudo -S bash -c "echo \'{username}:{username}\' | chpasswd"'  # 正确的密码设置方式
        ]
        
        results = []
        for cmd in commands:
            stdin, stdout, stderr = ssh.exec_command(cmd)
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            results.append(f"Command: {cmd}\nOutput: {output}\nError: {error}\n")
        
        # 获取私钥内容
        stdin, stdout, stderr = ssh.exec_command(f'echo "{admin_password}" | sudo -S cat /home/{username}/.ssh/id_rsa')
        private_key = stdout.read().decode('utf-8')
        key_error = stderr.read().decode('utf-8')
        
        ssh.close()
        
        if key_error and 'No such file' in key_error:
            return False, f"私钥文件未找到: {key_error}", None
        
        if private_key:
            return True, "用户创建成功", private_key
        else:
            return False, f"无法获取私钥: {key_error}", None
            
    except Exception as e:
        return False, f"连接错误: {str(e)}", None

@app.route('/api/create-user', methods=['POST'])
def create_user():
    data = request.get_json()
    server_name = data.get('server_name')
    username = data.get('username')
    auth_code = data.get('auth_code')
    
    # 查找服务器配置
    server = db.get_server_by_name(server_name)
    
    if not server:
        return jsonify({'success': False, 'error': '服务器未找到'})
    
    # 检查是否设置了专用密码
    if server.get('dedicated_password'):
        # 如果设置了专用密码，验证输入的授权码是否等于专用密码
        if auth_code != server.get('dedicated_password'):
            return jsonify({'success': False, 'error': '专用密码错误'})
    else:
        # 如果没有设置专用密码，使用正常的授权码验证
        if not verify_auth_code(auth_code):
            return jsonify({'success': False, 'error': '授权码无效或已过期'})
        
        # 立即删除授权码（单次有效）
        if auth_code in auth_codes:
            del auth_codes[auth_code]
    
    # 验证用户名格式
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]{0,31}$', username):
        return jsonify({'success': False, 'error': '用户名格式无效'})
    
    # 创建用户（使用固定的管理员密码）
    success, message, private_key = create_user_on_server(server, username, '123456')
    
    if success and private_key:
        # 存储私钥到临时文件
        key_id = f"{server_name}_{username}"
        user_keys[key_id] = private_key
        
        # 生成私钥文件名
        filename = f"id_rsa-{server_name}-{username}"
        
        return jsonify({
            'success': True, 
            'message': message,
            'ssh_command': f'ssh {username}@{server["ip"]} -p {server.get("port", 22)}',
            'key_filename': filename
        })
    else:
        return jsonify({'success': False, 'error': message})

@app.route('/api/get-users/<server_name>', methods=['POST'])
def get_users(server_name):
    """获取指定服务器的用户列表"""
    data = request.get_json()
    admin_password = data.get('admin_password')
    
    # 验证管理员密码
    if not db.verify_admin('admin', admin_password):
        return jsonify({'success': False, 'error': '管理员密码错误'})
    
    # 查找服务器配置
    server = db.get_server_by_name(server_name)
    
    if not server:
        return jsonify({'success': False, 'error': '服务器未找到'})
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**_ssh_connect_params(server))
        
        # 获取用户列表（排除系统用户）
        stdin, stdout, stderr = ssh.exec_command("getent passwd | awk -F: '$3 >= 1000 && $3 != 65534 {print $1}' | sort")
        users_output = stdout.read().decode('utf-8').strip()
        
        if stderr.read().decode('utf-8'):
            ssh.close()
            return jsonify({'success': False, 'error': '获取用户列表失败'})
        
        users = []
        if users_output:
            user_names = users_output.split('\n')
            
            # 检查每个用户的sudo权限
            for username in user_names:
                if username.strip():
                    # 检查用户是否在sudo组或sudoers文件中
                    stdin, stdout, stderr = ssh.exec_command(f"groups {username} | grep -q sudo && echo 'sudo' || echo 'normal'")
                    sudo_status = stdout.read().decode('utf-8').strip()
                    
                    # 如果不在sudo组，检查sudoers文件
                    if sudo_status == 'normal':
                        stdin, stdout, stderr = ssh.exec_command(f"sudo grep -q '^{username}.*ALL=(ALL.*) ALL' /etc/sudoers /etc/sudoers.d/* 2>/dev/null && echo 'sudo' || echo 'normal'")
                        sudoers_status = stdout.read().decode('utf-8').strip()
                        if sudoers_status == 'sudo':
                            sudo_status = 'sudo'
                    
                    users.append({
                        'username': username,
                        'has_sudo': sudo_status == 'sudo'
                    })
        
        ssh.close()
        return jsonify({'success': True, 'users': users})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'连接服务器失败: {str(e)}'})

@app.route('/api/delete-user', methods=['POST'])
def delete_user():
    """删除用户"""
    data = request.get_json()
    server_name = data.get('server_name')
    username = data.get('username')
    admin_password = data.get('admin_password')
    
    # 验证管理员密码
    if not db.verify_admin('admin', admin_password):
        return jsonify({'success': False, 'error': '管理员密码错误'})
    
    # 验证用户名
    if not username or username in ['root', 'admin', 'administrator']:
        return jsonify({'success': False, 'error': '无法删除系统用户或管理员用户'})
    
    # 查找服务器配置
    server = db.get_server_by_name(server_name)
    
    if not server:
        return jsonify({'success': False, 'error': '服务器未找到'})
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**_ssh_connect_params(server))
        
        # 先检查用户是否存在
        stdin, stdout, stderr = ssh.exec_command(f'id {username}')
        if stdout.read().decode('utf-8').strip() == '':
            ssh.close()
            return jsonify({'success': False, 'error': '用户不存在'})
        
        # 删除用户及其家目录，忽略邮件目录错误
        stdin, stdout, stderr = ssh.exec_command(f'sudo userdel -r {username} 2>&1')
        delete_output = stdout.read().decode('utf-8')
        delete_error = stderr.read().decode('utf-8')
        
        ssh.close()
        
        # 检查是否成功删除用户（忽略邮件目录相关警告）
        if 'No such user' in delete_error or 'does not exist' in delete_error:
            return jsonify({'success': False, 'error': '用户不存在'})
        elif 'userdel: user' in delete_error and 'deleted' in delete_error:
            # 成功删除，即使有邮件目录警告
            return jsonify({'success': True, 'message': f'用户 {username} 删除成功'})
        elif 'mail spool' in delete_error or 'not found' in delete_error:
            # 邮件目录相关警告，但用户可能已成功删除，再次检查
            try:
                ssh2 = paramiko.SSHClient()
                ssh2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                params = _ssh_connect_params(server)
                params['timeout'] = 5
                ssh2.connect(**params)
                
                stdin, stdout, stderr = ssh2.exec_command(f'id {username}')
                if stdout.read().decode('utf-8').strip() == '':
                    ssh2.close()
                    return jsonify({'success': True, 'message': f'用户 {username} 删除成功'})
                else:
                    ssh2.close()
                    return jsonify({'success': False, 'error': f'删除用户失败: 用户仍然存在'})
            except:
                # 如果无法再次连接，假设删除成功（因为邮件目录错误通常是警告）
                return jsonify({'success': True, 'message': f'用户 {username} 删除成功'})
        elif delete_error:
            return jsonify({'success': False, 'error': f'删除用户失败: {delete_error}'})
        else:
            return jsonify({'success': True, 'message': f'用户 {username} 删除成功'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'连接服务器失败: {str(e)}'})

@app.route('/api/manage-sudo', methods=['POST'])
def manage_sudo():
    """管理用户sudo权限"""
    data = request.get_json()
    server_name = data.get('server_name')
    username = data.get('username')
    action = data.get('action')  # 'grant' 或 'revoke'
    admin_password = data.get('admin_password')
    
    # 验证管理员密码
    if not db.verify_admin('admin', admin_password):
        return jsonify({'success': False, 'error': '管理员密码错误'})
    
    # 验证参数
    if not username or not action or action not in ['grant', 'revoke']:
        return jsonify({'success': False, 'error': '参数错误'})
    
    # 不能修改root用户权限
    if username == 'root':
        return jsonify({'success': False, 'error': '无法修改root用户权限'})
    
    # 查找服务器配置
    server = db.get_server_by_name(server_name)
    
    if not server:
        return jsonify({'success': False, 'error': '服务器未找到'})
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**_ssh_connect_params(server))
        
        if action == 'grant':
            # 授予sudo权限：将用户添加到sudo组
            stdin, stdout, stderr = ssh.exec_command(f'sudo usermod -a -G sudo {username}')
            action_text = '授予'
        else:
            # 移除sudo权限：使用更可靠的方法
            stdin, stdout, stderr = ssh.exec_command(f'sudo deluser {username} sudo 2>/dev/null || sudo gpasswd -d {username} sudo')
            action_text = '移除'
        
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        
        ssh.close()
        
        if error and 'does not exist' in error:
            return jsonify({'success': False, 'error': '用户不存在'})
        elif error and 'not a member' in error:
            return jsonify({'success': False, 'error': '用户不在sudo组中'})
        elif error:
            return jsonify({'success': False, 'error': f'操作失败: {error}'})
        else:
            return jsonify({'success': True, 'message': f'成功{action_text}用户 {username} 的sudo权限'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'连接服务器失败: {str(e)}'})

@app.route('/api/admin/servers', methods=['GET'])
@require_admin
def get_admin_servers():
    """获取所有服务器（管理员）"""
    servers = db.get_all_servers()
    return jsonify({'success': True, 'servers': servers})

@app.route('/api/admin/servers', methods=['POST'])
@require_admin
def add_server():
    """添加服务器（管理员）"""
    data = request.get_json()
    name = data.get('name')
    ip = data.get('ip')
    username = data.get('username')
    password = data.get('password')
    port = data.get('port', 22)
    description = data.get('description', '')
    dedicated_password = data.get('dedicated_password')
    code_path = data.get('code_path', '')
    data_path = data.get('data_path', '')
    auth_type = data.get('auth_type', 'password')
    key_path = data.get('key_path', '')
    server_group = data.get('server_group', '')
    
    # 验证必填字段
    if not all([name, ip, username]):
        return jsonify({'success': False, 'error': '名称、IP、用户名必填'})
    if auth_type == 'password' and not password:
        return jsonify({'success': False, 'error': '密码认证时密码必填'})
    if auth_type == 'key' and not key_path:
        return jsonify({'success': False, 'error': '密钥认证时密钥路径必填'})
    
    # 验证IP地址格式
    ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    if not ip_pattern.match(ip):
        return jsonify({'success': False, 'error': '无效的IP地址格式'})
    
    # 验证端口号
    if not isinstance(port, int) or port < 1 or port > 65535:
        return jsonify({'success': False, 'error': '端口号必须在1-65535之间'})
    
    success, message = db.add_server(
        name,
        ip,
        port,
        username,
        password or '',
        description,
        dedicated_password,
        code_path,
        data_path,
        auth_type,
        key_path,
        server_group
    )
    return jsonify({'success': success, 'message': message if success else message})

@app.route('/api/admin/servers/<int:server_id>', methods=['PUT'])
@require_admin
def update_server(server_id):
    """更新服务器（管理员）"""
    data = request.get_json()
    
    # 验证IP地址格式（如果提供）
    if 'ip' in data:
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
        if not ip_pattern.match(data['ip']):
            return jsonify({'success': False, 'error': '无效的IP地址格式'})
    
    # 验证端口号（如果提供）
    if 'port' in data:
        port = data['port']
        if not isinstance(port, int) or port < 1 or port > 65535:
            return jsonify({'success': False, 'error': '端口号必须在1-65535之间'})
    
    success, message = db.update_server(
        server_id,
        name=data.get('name'),
        ip=data.get('ip'),
        port=data.get('port'),
        username=data.get('username'),
        password=data.get('password'),
        description=data.get('description'),
        dedicated_password=data.get('dedicated_password'),
        code_path=data.get('code_path') if 'code_path' in data else None,
        data_path=data.get('data_path') if 'data_path' in data else None,
        auth_type=data.get('auth_type') if 'auth_type' in data else None,
        key_path=data.get('key_path') if 'key_path' in data else None,
        server_group=data.get('server_group') if 'server_group' in data else None
    )
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/admin/servers/<int:server_id>', methods=['DELETE'])
@require_admin
def delete_server_endpoint(server_id):
    """删除服务器（管理员）"""
    success, message = db.delete_server(server_id)
    return jsonify({'success': success, 'message': message})

@app.route('/api/admin/servers/<server_name>/users', methods=['GET'])
@require_admin
def get_server_users(server_name):
    """获取指定服务器的用户列表（管理员）"""
    # 查找服务器配置
    server = db.get_server_by_name(server_name)
    
    if not server:
        return jsonify({'success': False, 'error': '服务器未找到'})
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**_ssh_connect_params(server))
        
        # 获取用户列表（排除系统用户）
        stdin, stdout, stderr = ssh.exec_command("getent passwd | awk -F: '$3 >= 1000 && $3 != 65534 {print $1}' | sort")
        users_output = stdout.read().decode('utf-8').strip()
        
        if stderr.read().decode('utf-8'):
            ssh.close()
            return jsonify({'success': False, 'error': '获取用户列表失败'})
        
        users = []
        if users_output:
            user_names = users_output.split('\n')
            
            # 检查每个用户的sudo权限
            for username in user_names:
                if username.strip():
                    # 检查用户是否在sudo组或sudoers文件中
                    stdin, stdout, stderr = ssh.exec_command(f"groups {username} | grep -q sudo && echo 'sudo' || echo 'normal'")
                    sudo_status = stdout.read().decode('utf-8').strip()
                    
                    # 如果不在sudo组，检查sudoers文件
                    if sudo_status == 'normal':
                        stdin, stdout, stderr = ssh.exec_command(f"sudo grep -q '^{username}.*ALL=(ALL.*) ALL' /etc/sudoers /etc/sudoers.d/* 2>/dev/null && echo 'sudo' || echo 'normal'")
                        sudoers_status = stdout.read().decode('utf-8').strip()
                        if sudoers_status == 'sudo':
                            sudo_status = 'sudo'
                    
                    users.append({
                        'username': username,
                        'has_sudo': sudo_status == 'sudo'
                    })
        
        ssh.close()
        return jsonify({'success': True, 'users': users})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'连接服务器失败: {str(e)}'})

@app.route('/api/admin/servers/<server_name>/users/<username>', methods=['DELETE'])
@require_admin
def delete_server_user(server_name, username):
    """删除服务器用户（管理员）"""
    # 验证用户名
    if not username or username in ['root', 'admin', 'administrator']:
        return jsonify({'success': False, 'error': '无法删除系统用户或管理员用户'})
    
    # 查找服务器配置
    server = db.get_server_by_name(server_name)
    
    if not server:
        return jsonify({'success': False, 'error': '服务器未找到'})
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**_ssh_connect_params(server))
        
        # 先检查用户是否存在
        stdin, stdout, stderr = ssh.exec_command(f'id {username}')
        if stdout.read().decode('utf-8').strip() == '':
            ssh.close()
            return jsonify({'success': False, 'error': '用户不存在'})
        
        # 删除用户及其家目录，忽略邮件目录错误
        stdin, stdout, stderr = ssh.exec_command(f'sudo userdel -r {username} 2>&1')
        delete_output = stdout.read().decode('utf-8')
        delete_error = stderr.read().decode('utf-8')
        
        ssh.close()
        
        # 检查是否成功删除用户（忽略邮件目录相关警告）
        if 'No such user' in delete_error or 'does not exist' in delete_error:
            return jsonify({'success': False, 'error': '用户不存在'})
        elif 'userdel: user' in delete_error and 'deleted' in delete_error:
            return jsonify({'success': True, 'message': f'用户 {username} 删除成功'})
        elif 'mail spool' in delete_error or 'not found' in delete_error:
            # 邮件目录相关警告，但用户可能已成功删除
            return jsonify({'success': True, 'message': f'用户 {username} 删除成功'})
        elif delete_error:
            return jsonify({'success': False, 'error': f'删除用户失败: {delete_error}'})
        else:
            return jsonify({'success': True, 'message': f'用户 {username} 删除成功'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'连接服务器失败: {str(e)}'})

@app.route('/api/admin/servers/<server_name>/users/<username>/sudo', methods=['PUT'])
@require_admin
def manage_server_user_sudo(server_name, username):
    """管理服务器用户sudo权限（管理员）"""
    data = request.get_json()
    action = data.get('action')  # 'grant' 或 'revoke'
    
    # 验证参数
    if not username or not action or action not in ['grant', 'revoke']:
        return jsonify({'success': False, 'error': '参数错误'})
    
    # 不能修改root用户权限
    if username == 'root':
        return jsonify({'success': False, 'error': '无法修改root用户权限'})
    
    # 查找服务器配置
    server = db.get_server_by_name(server_name)
    
    if not server:
        return jsonify({'success': False, 'error': '服务器未找到'})
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**_ssh_connect_params(server))
        
        if action == 'grant':
            # 授予sudo权限：将用户添加到sudo组，同时设置密码为用户名
            commands = [
                f'sudo usermod -a -G sudo {username}',
                f'echo "{username}:{username}" | sudo chpasswd'  # 使用chpasswd设置密码
            ]
            action_text = '授予'
            
            # 执行所有命令
            all_success = True
            error_messages = []
            
            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                output = stdout.read().decode('utf-8')
                error = stderr.read().decode('utf-8')
                
                if error and 'does not exist' in error:
                    ssh.close()
                    return jsonify({'success': False, 'error': '用户不存在'})
                elif error and 'chpasswd' not in cmd:  # 非密码设置命令的错误
                    all_success = False
                    error_messages.append(error)
            
            ssh.close()
            
            if all_success:
                return jsonify({'success': True, 'message': f'成功{action_text}用户 {username} 的sudo权限并设置密码为用户名'})
            else:
                return jsonify({'success': True, 'message': f'成功{action_text}用户 {username} 的sudo权限，但密码设置可能失败'})
                
        else:
            # 移除sudo权限：使用更可靠的方法
            stdin, stdout, stderr = ssh.exec_command(f'sudo deluser {username} sudo 2>/dev/null || sudo gpasswd -d {username} sudo')
            action_text = '移除'
            
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            
            ssh.close()
            
            if error and 'does not exist' in error:
                return jsonify({'success': False, 'error': '用户不存在'})
            elif error and 'not a member' in error:
                return jsonify({'success': False, 'error': '用户不在sudo组中'})
            elif error:
                return jsonify({'success': False, 'error': f'操作失败: {error}'})
            else:
                return jsonify({'success': True, 'message': f'成功{action_text}用户 {username} 的sudo权限'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'连接服务器失败: {str(e)}'})

# ========== 集群配置 API ==========
@app.route('/api/cluster/config', methods=['GET'])
@require_admin
def get_cluster_config():
    return jsonify({
        'success': True,
        'config': {
            'code_path': db.get_config(CONFIG_CODE_PATH, ''),
            'data_path': db.get_config(CONFIG_DATA_PATH, ''),
            'gpu_mem_threshold': db.get_config(CONFIG_MEM_THRESHOLD, '20'),
            'gpu_util_threshold': db.get_config(CONFIG_UTIL_THRESHOLD, '15'),
            'train_gpu_mem_threshold': db.get_config(CONFIG_TRAIN_MEM_THRESHOLD) or db.get_config(CONFIG_MEM_THRESHOLD, '20'),
            'train_gpu_util_threshold': db.get_config(CONFIG_TRAIN_UTIL_THRESHOLD) or db.get_config(CONFIG_UTIL_THRESHOLD, '15'),
            'test_gpu_mem_threshold': db.get_config(CONFIG_TEST_MEM_THRESHOLD) or db.get_config(CONFIG_MEM_THRESHOLD, '20'),
            'test_gpu_util_threshold': db.get_config(CONFIG_TEST_UTIL_THRESHOLD) or db.get_config(CONFIG_UTIL_THRESHOLD, '15'),
            'reserved_gpu_count': db.get_config(CONFIG_RESERVED_GPU, '0'),
            'reserved_gpu_per_server': db.get_config(CONFIG_RESERVED_GPU_PER_SERVER, ''),
            'gpu_lock_ttl_train_seconds': db.get_config(CONFIG_GPU_LOCK_TTL_TRAIN, '300'),
            'gpu_lock_ttl_deploy_seconds': db.get_config(CONFIG_GPU_LOCK_TTL_DEPLOY, '300'),
            'server_groups': db.get_config(CONFIG_SERVER_GROUPS, ''),
        }
    })


@app.route('/api/cluster/config', methods=['POST'])
@require_admin
def set_cluster_config():
    data = request.get_json() or {}
    keys = [CONFIG_CODE_PATH, CONFIG_DATA_PATH, CONFIG_MEM_THRESHOLD, CONFIG_UTIL_THRESHOLD,
            CONFIG_TRAIN_MEM_THRESHOLD, CONFIG_TRAIN_UTIL_THRESHOLD, CONFIG_TEST_MEM_THRESHOLD, CONFIG_TEST_UTIL_THRESHOLD,
            CONFIG_RESERVED_GPU, CONFIG_GPU_LOCK_TTL_TRAIN, CONFIG_GPU_LOCK_TTL_DEPLOY, CONFIG_SERVER_GROUPS,
            CONFIG_RESERVED_GPU_PER_SERVER]
    for k in keys:
        if k in data:
            db.set_config(k, str(data[k]) if data[k] is not None else '')
    return jsonify({'success': True})


# ========== 训练任务 API ==========
@app.route('/api/cluster/training/submit', methods=['POST'])
@require_admin
def submit_training_task():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    task_name = data.get('task_name', '').strip()
    script_path = data.get('script_path', '').strip()
    script_args = data.get('script_args', '')
    priority = int(data.get('priority', 5))
    gpu_count = data.get('gpu_count', 1)
    allowed_servers = data.get('allowed_servers')  # 可选，服务器名列表；空/不传表示所有服务器
    logger.info(f"[API] submit_training_task name={name} task_name={task_name} script={script_path} args={script_args} priority={priority} gpu_count={gpu_count} allowed_servers={allowed_servers}")
    if not name or not script_path:
        return jsonify({'success': False, 'error': '任务名和脚本路径必填'})
    if not task_name:
        return jsonify({'success': False, 'error': 'task_name 必填'})
    ok, result = db.add_training_task(
        name, script_path, script_args, priority,
        gpu_count=gpu_count, allowed_servers=allowed_servers, task_name=task_name
    )
    if not ok:
        return jsonify({'success': False, 'error': str(result)})
    task_id = result
    # 提交时冻结脚本：为该任务创建脚本快照
    freeze_ok, freeze_info = freeze_training_script_for_task(task_id, script_path, allowed_servers)
    if not freeze_ok:
        db.update_training_task(task_id, status='failed', error_message=str(freeze_info))
        return jsonify({'success': False, 'error': f'冻结脚本失败: {freeze_info}'})
    return jsonify({'success': True, 'task_id': task_id})


@app.route('/api/cluster/training/list')
@require_admin
def list_training_tasks():
    limit = request.args.get('limit', 100, type=int)
    tasks = db.get_all_training_tasks(limit=limit)
    return jsonify({'success': True, 'tasks': tasks})


@app.route('/api/cluster/training/<int:task_id>/log')
@require_admin
def get_training_log(task_id):
    task = db.get_training_task(task_id)
    if not task or not task.get('server_name') or not task.get('log_path'):
        return jsonify({'success': False, 'error': '任务或日志不存在'})
    server = db.get_server_by_name(task['server_name'])
    if not server:
        return jsonify({'success': False, 'error': '服务器不存在'})
    log_path = task['log_path']
    lines = request.args.get('lines', type=int)
    # 先在内容前面标注日志路径，方便排查
    prefix = f'[日志路径] {log_path}\\n\\n'
    if lines and lines > 0:
        prefix += f'[显示模式] 最近 {lines} 行\\n\\n'
    # 远程读取日志文件；可按行数截取尾部
    ok, content = _read_remote_log_with_limit(server, log_path, lines=lines)
    if not ok:
        return jsonify({'success': False, 'error': content})
    return Response(prefix + (content or ''), mimetype='text/plain; charset=utf-8')


@app.route('/api/cluster/training/<int:task_id>/kill', methods=['POST'])
@require_admin
def kill_training_task(task_id):
    task = db.get_training_task(task_id)
    if not task or task.get('status') != 'running':
        return jsonify({'success': False, 'error': '任务未在运行'})
    server = db.get_server_by_name(task['server_name'])
    if not server:
        return jsonify({'success': False, 'error': '服务器不存在'})
    pid = task.get('pid')
    if not pid:
        return jsonify({'success': False, 'error': '无 PID'})
    ok, killed, out = kill_process_group(server, pid)
    if not ok:
        return jsonify({'success': False, 'error': f'发送 kill 失败: {out}'})
    if not killed:
        return jsonify({'success': False, 'error': f'进程仍在运行 (pid={pid})，请手动检查'})
    db.update_training_task(task_id, status='killed', finished_at=datetime.now(CST).isoformat())
    return jsonify({'success': True})


@app.route('/api/cluster/training/<int:task_id>/delete', methods=['POST'])
@require_admin
def delete_training_task_api(task_id):
    task = db.get_training_task(task_id)
    if not task:
        return jsonify({'success': False, 'error': '任务不存在'})
    if task.get('status') == 'running':
        return jsonify({'success': False, 'error': '运行中的任务不能删除，请先停止'})
    ok = db.delete_training_task(task_id)
    if not ok:
        return jsonify({'success': False, 'error': '删除失败'})
    return jsonify({'success': True})


@app.route('/api/cluster/training/<int:task_id>/weight', methods=['POST'])
@require_admin
def record_training_weight(task_id):
    data = request.get_json() or {}
    path = data.get('path', '').strip()
    if not path:
        return jsonify({'success': False, 'error': '权重路径必填'})
    task = db.get_training_task(task_id)
    if not task:
        return jsonify({'success': False, 'error': '任务不存在'})
    ok = db.update_training_task(task_id, weight_path=path)
    if not ok:
        return jsonify({'success': False, 'error': '更新训练任务权重路径失败'})

    server_name = (task.get('server_name') or '').strip()
    if not server_name:
        return jsonify({'success': False, 'error': '该任务尚未绑定服务器，无法记录权重，请等待任务启动后再记录'})

    ok, msg = db.upsert_model_weight_for_task(
        task_id,
        task.get('name', 'task') + '_' + str(task_id),
        path,
        server_name
    )
    if not ok:
        return jsonify({'success': False, 'error': f'写入权重记录失败: {msg}'})
    return jsonify({'success': True})


# ========== 部署任务 API ==========
@app.route('/api/cluster/deploy/submit', methods=['POST'])
@require_admin
def submit_deploy_task():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    script_path = data.get('script_path', '').strip()
    weight_path = data.get('weight_path', '').strip()
    priority = int(data.get('priority', 5))
    port = data.get('port')
    training_task_id = data.get('training_task_id')
    use_norm = data.get('use_norm', False)
    try:
        port = int(port) if port not in (None, '') else None
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': '端口必须为数字'})
    # 处理关联的训练任务ID
    if training_task_id:
        try:
            training_task_id = int(training_task_id)
            # 获取训练任务信息，自动填充权重路径
            train_task = db.get_training_task(training_task_id)
            if train_task:
                if train_task.get('weight_path') and not weight_path:
                    weight_path = train_task['weight_path']
            else:
                return jsonify({'success': False, 'error': f'训练任务 {training_task_id} 不存在'})
        except (TypeError, ValueError):
            training_task_id = None
    allowed_servers = data.get('allowed_servers')
    if not name or not script_path:
        return jsonify({'success': False, 'error': '任务名和脚本路径必填'})
    ok, result = db.add_deploy_task(
        name=name,
        script_path=script_path,
        weight_path=weight_path,
        priority=priority,
        allowed_servers=allowed_servers,
        port=port,
        training_task_id=training_task_id,
        use_norm=use_norm
    )
    if not ok:
        return jsonify({'success': False, 'error': str(result)})
    return jsonify({'success': True, 'task_id': result})


@app.route('/api/cluster/deploy/list')
@require_admin
def list_deploy_tasks():
    limit = request.args.get('limit', 100, type=int)
    tasks = db.get_all_deploy_tasks(limit=limit)
    return jsonify({'success': True, 'tasks': tasks})


@app.route('/api/cluster/deploy/<int:task_id>/log')
@require_admin
def get_deploy_log(task_id):
    task = db.get_deploy_task(task_id)
    if not task or not task.get('server_name') or not task.get('log_path'):
        return jsonify({'success': False, 'error': '任务或日志不存在'})
    server = db.get_server_by_name(task['server_name'])
    if not server:
        return jsonify({'success': False, 'error': '服务器不存在'})
    log_path = task['log_path']
    lines = request.args.get('lines', type=int)
    prefix = f'[日志路径] {log_path}\\n\\n'
    if lines and lines > 0:
        prefix += f'[显示模式] 最近 {lines} 行\\n\\n'
    ok, content = _read_remote_log_with_limit(server, log_path, lines=lines)
    if not ok:
        return jsonify({'success': False, 'error': content})
    return Response(prefix + (content or ''), mimetype='text/plain; charset=utf-8')


@app.route('/api/cluster/deploy/<int:task_id>/kill', methods=['POST'])
@require_admin
def kill_deploy_task(task_id):
    task = db.get_deploy_task(task_id)
    if not task or task.get('status') != 'running':
        return jsonify({'success': False, 'error': '任务未在运行'})
    server = db.get_server_by_name(task['server_name'])
    if not server:
        return jsonify({'success': False, 'error': '服务器不存在'})
    pid = task.get('pid')
    if not pid:
        return jsonify({'success': False, 'error': '无 PID'})
    ok, killed, out = kill_process_group(server, pid)
    if not ok:
        return jsonify({'success': False, 'error': f'发送 kill 失败: {out}'})
    if not killed:
        return jsonify({'success': False, 'error': f'进程仍在运行 (pid={pid})，请手动检查'})
    db.update_deploy_task(task_id, status='killed', finished_at=datetime.now(CST).isoformat())
    return jsonify({'success': True})


@app.route('/api/cluster/deploy/<int:task_id>/delete', methods=['POST'])
@require_admin
def delete_deploy_task_api(task_id):
    task = db.get_deploy_task(task_id)
    if not task:
        return jsonify({'success': False, 'error': '任务不存在'})
    if task.get('status') == 'running':
        return jsonify({'success': False, 'error': '运行中的任务不能删除，请先停止'})
    ok = db.delete_deploy_task(task_id)
    if not ok:
        return jsonify({'success': False, 'error': '删除失败'})
    return jsonify({'success': True})


@app.route('/api/cluster/deploy/<int:task_id>/port', methods=['POST'])
@require_admin
def record_deploy_port(task_id):
    data = request.get_json() or {}
    port = data.get('port')
    try:
        port = int(port)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': '端口必须为数字'})
    if port < 1 or port > 65535:
        return jsonify({'success': False, 'error': '端口范围应为 1-65535'})

    task = db.get_deploy_task(task_id)
    if not task:
        return jsonify({'success': False, 'error': '任务不存在'})

    ok = db.update_deploy_task(task_id, port=port)
    if not ok:
        return jsonify({'success': False, 'error': '更新端口失败'})
    return jsonify({'success': True, 'port': port})


# ========== 测试任务 API ==========
@app.route('/api/cluster/test/submit', methods=['POST'])
@require_admin
def submit_test_task():
    data = request.get_json() or {}
    logger.info(f"[submit_test_task] 收到请求 data_keys={list(data.keys())} task_type={data.get('task_type')}")
    name = data.get('name', '').strip()
    task_type = data.get('task_type', 'mock')  # mock 或 real
    server_name = data.get('server_name', '').strip()
    script_path = data.get('script_path', '')
    script_args = data.get('script_args', '')
    test_code_path = data.get('test_code_path', '').strip()
    mock_url = data.get('mock_url', '').strip()
    mock_task_name = data.get('mock_task_name', '').strip()
    user_token = data.get('user_token', '').strip()
    run_id = data.get('run_id', '').strip()
    action_nums_raw = data.get('action_nums')
    action_nums = None
    if action_nums_raw not in (None, ''):
        try:
            action_nums = int(action_nums_raw)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'action_nums 必须为整数'})
        if action_nums < 1:
            return jsonify({'success': False, 'error': 'action_nums 必须 >= 1'})
    deploy_task_id_raw = data.get('deploy_task_id')
    deploy_task_id = None
    if deploy_task_id_raw not in (None, ''):
        try:
            deploy_task_id = int(deploy_task_id_raw)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': '部署任务ID格式错误'})
    # 选择了部署任务ID时，自动回填服务器、task_name、url
    if deploy_task_id is not None:
        deploy_task = db.get_deploy_task(deploy_task_id)
        if not deploy_task:
            return jsonify({'success': False, 'error': '指定部署任务不存在'})
        if deploy_task.get('server_name'):
            server_name = (deploy_task.get('server_name') or '').strip()
        if deploy_task.get('name'):
            mock_task_name = (deploy_task.get('name') or '').strip()
        if deploy_task.get('port'):
            mock_url = f"http://127.0.0.1:{int(deploy_task.get('port'))}"

    if not name:
        return jsonify({'success': False, 'error': '任务名必填'})
    if task_type not in ('mock', 'real'):
        logger.info(f"[submit_test_task] 校验失败 task_type 非法")
        return jsonify({'success': False, 'error': 'task_type 须为 mock 或 real'})
    if not server_name:
        logger.info(f"[submit_test_task] 校验失败 测试服务器必填")
        return jsonify({'success': False, 'error': '测试服务器必填'})
    if not db.get_server_by_name(server_name):
        return jsonify({'success': False, 'error': '指定测试服务器不存在'})
    if not script_path:
        return jsonify({'success': False, 'error': '脚本路径必填'})
    if not test_code_path:
        return jsonify({'success': False, 'error': '测试代码路径必填'})
    if not mock_url or not mock_task_name:
        return jsonify({'success': False, 'error': 'url 和 task_name 必填'})
    if task_type == 'real':
        if not user_token or not run_id:
            logger.info(f"[submit_test_task] 校验失败 real 缺少 user_token/run_id")
            return jsonify({'success': False, 'error': 'real 模式需填写 user_token 和 run_id'})
        if action_nums is None:
            logger.info(f"[submit_test_task] 校验失败 real 缺少 action_nums")
            return jsonify({'success': False, 'error': 'real 模式需填写 action_nums'})

    logger.info(f"[submit_test_task] 校验通过 写入 DB name={name} task_type={task_type} server_name={server_name}")
    ok, result = db.add_test_task(
        name=name,
        task_type=task_type,
        server_name=server_name,
        script_path=script_path,
        script_args=script_args,
        test_code_path=test_code_path,
        mock_url=mock_url,
        mock_task_name=mock_task_name,
        user_token=user_token,
        run_id=run_id,
        action_nums=action_nums,
        deploy_task_id=deploy_task_id
    )
    if ok:
        logger.info(f"[submit_test_task] 成功 task_id={result}")
        return jsonify({'success': True, 'task_id': result})
    logger.info(f"[submit_test_task] DB 写入失败 result={result}")
    return jsonify({'success': False, 'error': str(result)})


@app.route('/api/cluster/test/list')
@require_admin
def list_test_tasks():
    limit = request.args.get('limit', 100, type=int)
    tasks = db.get_all_test_tasks(limit=limit)
    return jsonify({'success': True, 'tasks': tasks})


@app.route('/api/cluster/test/<int:task_id>/log')
@require_admin
def get_test_log(task_id):
    task = db.get_test_task(task_id)
    if not task or not task.get('server_name') or not task.get('log_path'):
        return jsonify({'success': False, 'error': '任务或日志不存在'})
    server = db.get_server_by_name(task['server_name'])
    if not server:
        return jsonify({'success': False, 'error': '服务器不存在'})
    log_path = task['log_path']
    lines = request.args.get('lines', type=int)
    prefix = f'[日志路径] {log_path}\\n\\n'
    if lines and lines > 0:
        prefix += f'[显示模式] 最近 {lines} 行\\n\\n'
    ok, content = _read_remote_log_with_limit(server, log_path, lines=lines)
    if not ok:
        return jsonify({'success': False, 'error': content})
    return Response(prefix + (content or ''), mimetype='text/plain; charset=utf-8')


@app.route('/api/cluster/test/<int:task_id>/kill', methods=['POST'])
@require_admin
def kill_test_task(task_id):
    task = db.get_test_task(task_id)
    if not task or task.get('status') != 'running':
        return jsonify({'success': False, 'error': '任务未在运行'})
    server = db.get_server_by_name(task['server_name'])
    if not server:
        return jsonify({'success': False, 'error': '服务器不存在'})
    pid = task.get('pid')
    if not pid:
        return jsonify({'success': False, 'error': '无 PID'})
    ok, killed, out = kill_process_group(server, pid)
    if not ok:
        return jsonify({'success': False, 'error': f'发送 kill 失败: {out}'})
    if not killed:
        return jsonify({'success': False, 'error': f'进程仍在运行 (pid={pid})，请手动检查'})
    db.update_test_task(task_id, status='killed', finished_at=datetime.now(CST).isoformat())
    return jsonify({'success': True})


@app.route('/api/cluster/test/<int:task_id>/delete', methods=['POST'])
@require_admin
def delete_test_task_api(task_id):
    task = db.get_test_task(task_id)
    if not task:
        return jsonify({'success': False, 'error': '任务不存在'})
    if task.get('status') == 'running':
        return jsonify({'success': False, 'error': '运行中的任务不能删除，请先停止'})
    ok = db.delete_test_task(task_id)
    if not ok:
        return jsonify({'success': False, 'error': '删除失败'})
    return jsonify({'success': True})


@app.route('/api/cluster/test/<int:task_id>/result', methods=['POST'])
@require_admin
def record_test_result(task_id):
    """记录测试任务的成功率和分数"""
    data = request.get_json() or {}
    success_rate = data.get('success_rate')
    score = data.get('score')
    
    task = db.get_test_task(task_id)
    if not task:
        return jsonify({'success': False, 'error': '任务不存在'})
    
    # 验证输入
    if success_rate is not None:
        try:
            success_rate = float(success_rate)
            if success_rate < 0 or success_rate > 100:
                return jsonify({'success': False, 'error': '成功率应在 0-100 之间'})
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': '成功率必须是数字'})
    
    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': '分数必须是数字'})
    
    update_data = {}
    if success_rate is not None:
        update_data['success_rate'] = success_rate
    if score is not None:
        update_data['score'] = score
    
    if not update_data:
        return jsonify({'success': False, 'error': '请提供成功率或分数'})
    
    ok = db.update_test_task(task_id, **update_data)
    if not ok:
        return jsonify({'success': False, 'error': '更新结果失败'})
    return jsonify({'success': True})


# ========== 模型权重 API ==========
@app.route('/api/cluster/weights')
@require_admin
def list_model_weights():
    weights = db.get_all_model_weights()
    return jsonify({'success': True, 'weights': weights})


def _sftp_transfer(src_server, dst_server, remote_path):
    """通过管理节点中转：从 src 下载到临时文件，再上传到 dst"""
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix='.transfer')
        os.close(fd)
        # 从 src 下载
        ssh_src = paramiko.SSHClient()
        ssh_src.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_src.connect(**_ssh_connect_params(src_server))
        sftp_src = ssh_src.open_sftp()
        sftp_src.get(remote_path, tmp_path)
        sftp_src.close()
        ssh_src.close()
        # 上传到 dst
        ssh_dst = paramiko.SSHClient()
        ssh_dst.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_dst.connect(**_ssh_connect_params(dst_server))
        sftp_dst = ssh_dst.open_sftp()
        try:
            sftp_dst.put(tmp_path, remote_path)
        finally:
            sftp_dst.close()
            ssh_dst.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return True, None
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        return False, str(e)


@app.route('/api/cluster/weights/transfer', methods=['POST'])
@require_admin
def transfer_weight():
    data = request.get_json() or {}
    src_server = data.get('src_server')
    dst_server = data.get('dst_server')
    path = data.get('path', '').strip()
    if not all([src_server, dst_server, path]):
        return jsonify({'success': False, 'error': 'src_server, dst_server, path 必填'})
    srv_src = db.get_server_by_name(src_server)
    srv_dst = db.get_server_by_name(dst_server)
    if not srv_src or not srv_dst:
        return jsonify({'success': False, 'error': '服务器不存在'})
    ok, err = _sftp_transfer(srv_src, srv_dst, path)
    if ok:
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': err})


# ========== 内存报警 API ==========
@app.route('/api/cluster/alerts')
@require_admin
def list_alerts():
    alerts = db.get_recent_alerts()
    return jsonify({'success': True, 'alerts': alerts})


@app.route('/api/cluster/servers/idle-gpus')
@require_admin
def get_idle_gpus():
    """获取各服务器空闲 GPU 信息（分别展示训练用、部署用空闲卡），并返回服务器分组信息"""
    train_mem = int(db.get_config(CONFIG_TRAIN_MEM_THRESHOLD) or db.get_config(CONFIG_MEM_THRESHOLD, 20))
    train_util = int(db.get_config(CONFIG_TRAIN_UTIL_THRESHOLD) or db.get_config(CONFIG_UTIL_THRESHOLD, 15))
    deploy_mem = int(db.get_config(CONFIG_TEST_MEM_THRESHOLD) or db.get_config(CONFIG_MEM_THRESHOLD, 20))
    deploy_util = int(db.get_config(CONFIG_TEST_UTIL_THRESHOLD) or db.get_config(CONFIG_UTIL_THRESHOLD, 15))

    # 读取全局保留卡数和各服务器独立配置
    default_reserved = int(db.get_config(CONFIG_RESERVED_GPU, 0))
    reserved_per_server_str = db.get_config(CONFIG_RESERVED_GPU_PER_SERVER, '{}')
    try:
        reserved_per_server = json.loads(reserved_per_server_str) if reserved_per_server_str else {}
    except json.JSONDecodeError:
        reserved_per_server = {}

    # 读取服务器分组信息
    all_servers = db.get_all_servers()
    name_to_group = {s['name']: (s.get('server_group') or '') for s in all_servers}

    result = []
    for srv_name, st in server_status.items():
        gpu_text = st.get('gpu_status', '') or ''
        is_error = 'Connection Error' in gpu_text or gpu_text.strip().startswith('Error:') or 'Loading...' in gpu_text
        gpus = parse_gpustat_output(gpu_text)
        # 获取该服务器的保留卡数（优先使用独立配置，否则使用默认值）
        srv_reserved = reserved_per_server.get(srv_name, default_reserved)
        train_idle = find_idle_gpus(gpu_text, train_mem, train_util, reserved_gpu_count=srv_reserved)
        deploy_idle = find_idle_gpus(gpu_text, deploy_mem, deploy_util, reserved_gpu_count=srv_reserved)
        result.append({
            'server': srv_name,
            'idle_gpus': deploy_idle,
            'train_idle_gpus': train_idle,
            'deploy_idle_gpus': deploy_idle,
            'test_idle_gpus': deploy_idle,  # 兼容旧前端字段
            'total_gpus': len(gpus),
            'gpus': gpus,
            'connection_error': is_error,
            'gpu_status_preview': gpu_text[:80].replace('\n', ' ') if is_error else '',
            'server_group': name_to_group.get(srv_name, ''),
            'reserved_gpu_count': srv_reserved  # 返回该服务器的保留卡数配置
        })
    return jsonify({'success': True, 'servers': result})


# ========== 集群管理页面 ==========
@app.route('/cluster')
def cluster_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('cluster.html')


@app.route('/api/download-key/<server_name>/<username>')
def download_key(server_name, username):
    key_id = f"{server_name}_{username}"
    
    if key_id not in user_keys:
        return "私钥文件未找到", 404
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_file.write(user_keys[key_id])
    temp_file.close()
    
    # 生成文件名
    filename = f"id_rsa-{server_name}-{username}"
    
    # 下载完成后清理
    def cleanup():
        try:
            os.unlink(temp_file.name)
            if key_id in user_keys:
                del user_keys[key_id]
        except:
            pass
    
    # 延迟清理（给下载一些时间）
    threading.Timer(60, cleanup).start()
    
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=filename,
        mimetype='application/octet-stream'
    )

if __name__ == '__main__':
    # 启动后台线程进行定期更新
    update_thread = threading.Thread(target=update_all_servers, daemon=True)
    update_thread.start()
    # 启动集群任务调度器
    scheduler_thread = threading.Thread(target=cluster_task_scheduler, daemon=True)
    scheduler_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False) 
