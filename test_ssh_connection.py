#!/usr/bin/env python3
"""
SSH 连接诊断脚本 - 用于排查 "Error reading SSH protocol banner" 等问题
"""
import socket
import sys
import os

# 添加当前目录以导入 database
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager
import paramiko


def check_port(host, port, timeout=5):
    """检查端口是否可达"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0, "端口开放" if result == 0 else f"端口关闭/不可达 (err={result})"
    except socket.gaierror as e:
        return False, f"DNS解析失败: {e}"
    except Exception as e:
        return False, str(e)


def _ssh_connect_params(server):
    """与 app.py 一致，支持密码或密钥"""
    params = {
        "hostname": server["ip"],
        "port": int(server.get("port") or 22),
        "username": server["username"],
        "timeout": 15,
    }
    auth_type = server.get("auth_type") or "password"
    key_path = (server.get("key_path") or "").strip()
    if auth_type == "key" and key_path:
        path = os.path.expanduser(key_path)
        if os.path.isfile(path):
            params["key_filename"] = path
            params["look_for_keys"] = False
            return params
    params["password"] = server.get("password", "")
    return params


def test_ssh(server, command="echo ok", timeout=15):
    """尝试 SSH 连接并执行简单命令"""
    name = server.get("name", "?")
    ip = server.get("ip", "")
    port = server.get("port", 22)
    username = server.get("username", "")

    # 确保 port 是整数
    try:
        port = int(port) if port is not None else 22
    except (ValueError, TypeError):
        port = 22

    print(f"\n{'='*60}")
    print(f"服务器: {name} | {ip}:{port} | 用户: {username}")
    print("=" * 60)

    # 1. 端口连通性
    print("\n[1] 端口连通性检查...")
    ok, msg = check_port(ip, port, timeout=5)
    print(f"    结果: {msg}")
    if not ok:
        print("    ⚠ 端口不通，SSH 连接必然失败。请检查: 网络、防火墙、sshd 是否监听")
        return

    # 2. Paramiko SSH 连接
    auth_info = "密钥" if (server.get("auth_type") == "key" and server.get("key_path")) else "密码"
    print("\n[2] Paramiko SSH 连接 (timeout={}s, {})...".format(timeout, auth_info))
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**_ssh_connect_params(server))
        print("    连接成功 ✓")

        stdin, stdout, stderr = ssh.exec_command(command, timeout=10)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        ssh.close()

        if err:
            print(f"    命令 stderr: {err[:200]}")
        print(f"    命令输出: {out.strip() or '(空)'}")

    except paramiko.SSHException as e:
        print(f"    ❌ SSHException: {e}")
        if "banner" in str(e).lower():
            print("    可能原因: sshd 响应慢、端口被代理/负载均衡、非SSH服务占用了该端口")
    except socket.timeout:
        print("    ❌ 连接超时")
    except Exception as e:
        print(f"    ❌ {type(e).__name__}: {e}")


def main():
    print("SSH 连接诊断工具")
    print("=" * 60)

    db = DatabaseManager()
    servers = db.get_all_servers()

    if not servers:
        print("数据库中没有配置任何服务器")
        return

    print(f"共 {len(servers)} 台服务器\n")

    for srv in servers:
        test_ssh(srv)

    print("\n" + "=" * 60)
    print("诊断完成")
    print("\n常见 'Error reading SSH protocol banner' 原因:")
    print("  - 端口实际运行的不是 SSH 服务")
    print("  - 中间有代理/负载均衡，协议不兼容")
    print("  - 目标 sshd 启动慢或负载高，需增大 timeout")
    print("  - 防火墙/安全组限制")
    print("  - 目标 IP/端口配置错误")


if __name__ == "__main__":
    main()
