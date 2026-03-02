#!/usr/bin/env python3
"""
SSH 密钥连接测试脚本 - 用于排查 "Invalid key" 等密钥认证问题

用法:
  python test_ssh_key.py                      # 测试数据库中所有使用密钥的服务器
  python test_ssh_key.py A100_47              # 测试指定服务器
  python test_ssh_key.py /path/to/id_rsa user@host  # 测试指定密钥和主机
"""
import socket
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager
import paramiko


def check_key_file(path):
    """检查密钥文件"""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return False, f"文件不存在: {path}"
    if not os.path.isfile(path):
        return False, f"不是文件: {path}"
    if os.stat(path).st_mode & 0o077:
        return False, f"权限过宽 (应为 600): {path}"
    with open(path, 'rb') as f:
        head = f.read(50).decode('utf-8', errors='replace')
    if 'BEGIN' in head:
        return True, head.split('\n')[0]
    return False, "无法识别密钥格式"


def load_key(path, passphrase=''):
    """尝试多种方式加载私钥"""
    path = os.path.expanduser(path)
    results = []
    key_types = [('RSA', paramiko.RSAKey), ('Ed25519', paramiko.Ed25519Key)]
    if hasattr(paramiko, 'ECDSAKey'):
        key_types.append(('ECDSA', paramiko.ECDSAKey))
    for name, key_class in key_types:
        try:
            key = key_class.from_private_key_file(path, password=passphrase or '')
            return True, key, f"{name} 加载成功"
        except paramiko.ssh_exception.PasswordRequiredException as e:
            results.append(f"{name}: 需要密码")
        except paramiko.ssh_exception.SSHException as e:
            results.append(f"{name}: {e}")
        except Exception as e:
            results.append(f"{name}: {type(e).__name__} {e}")
    try:
        key = paramiko.PKey.from_private_key_file(path, password=passphrase or '')
        return True, key, "PKey 加载成功"
    except Exception as e:
        results.append(f"PKey: {e}")
    return False, None, "; ".join(results)


def test_key_connection(host, port, username, key_path, key_passphrase='', timeout=15):
    """测试密钥连接"""
    print("\n" + "=" * 60)
    print(f"目标: {username}@{host}:{port}")
    print(f"密钥: {key_path}")
    print("=" * 60)

    path = os.path.expanduser(key_path)
    if not os.path.isfile(path):
        print(f"❌ 密钥文件不存在: {path}")
        return

    print("\n[1] 密钥文件检查...")
    ok, msg = check_key_file(path)
    print(f"    结果: {msg}")
    if not ok:
        print("    建议: chmod 600 私钥文件")
        return

    print("\n[2] 密钥加载...")
    ok, pkey, msg = load_key(path, key_passphrase)
    if not ok:
        print(f"    ❌ 加载失败: {msg}")
        print("    可能原因: 密钥格式不兼容、需要密码(passphrase)")
        return
    print(f"    ✓ {msg}")

    print("\n[3] 端口连通性...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        r = sock.connect_ex((host, port))
        sock.close()
        if r != 0:
            print(f"    ❌ 端口不可达")
            return
        print("    ✓ 端口开放")
    except Exception as e:
        print(f"    ❌ {e}")
        return

    print("\n[4] SSH 连接...")
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=host,
            port=port,
            username=username,
            pkey=pkey,
            timeout=timeout,
            banner_timeout=30,
            look_for_keys=False,
        )
        print("    ✓ 连接成功")

        stdin, stdout, stderr = ssh.exec_command("echo ok", timeout=5)
        out = stdout.read().decode().strip()
        ssh.close()
        print(f"    执行: echo ok -> {out}")

    except paramiko.AuthenticationException as e:
        print(f"    ❌ 认证失败: {e}")
        print("    建议: 确保公钥已加入目标机 ~/.ssh/authorized_keys")
        print("          在目标机执行: ssh-copy-id -i 公钥 user@目标")
    except Exception as e:
        print(f"    ❌ {type(e).__name__}: {e}")


def main():
    args = sys.argv[1:]
    db = DatabaseManager()

    if len(args) == 0:
        # 从数据库读取所有使用密钥的服务器
        servers = [s for s in db.get_all_servers() if (s.get('auth_type') == 'key') and (s.get('key_path'))]
        if not servers:
            print("数据库中没有配置密钥认证的服务器")
            print("用法: python test_ssh_key.py 服务器名")
            print("      python test_ssh_key.py /path/to/key user@host")
            return
        print(f"测试 {len(servers)} 台使用密钥的服务器\n")
        for s in servers:
            test_key_connection(
                host=s['ip'],
                port=int(s.get('port') or 22),
                username=s['username'],
                key_path=s.get('key_path', ''),
                key_passphrase=s.get('key_passphrase', ''),
            )

    elif len(args) == 1:
        # 按服务器名从数据库查找
        name = args[0]
        server = db.get_server_by_name(name)
        if not server:
            print(f"未找到服务器: {name}")
            return
        if server.get('auth_type') != 'key' or not server.get('key_path'):
            print(f"服务器 {name} 未配置密钥认证")
            return
        test_key_connection(
            host=server['ip'],
            port=int(server.get('port') or 22),
            username=server['username'],
            key_path=server.get('key_path', ''),
            key_passphrase=server.get('key_passphrase', ''),
        )

    elif len(args) >= 2:
        # 直接指定: 密钥路径 user@host
        key_path = args[0]
        user_host = args[1]
        if '@' in user_host:
            username, host = user_host.split('@', 1)
        else:
            username = os.environ.get('USER', 'root')
            host = user_host
        port = int(args[2]) if len(args) > 2 else 22
        test_key_connection(host=host, port=port, username=username, key_path=key_path)

    else:
        print("用法:")
        print("  python test_ssh_key.py                     # 测试所有密钥服务器")
        print("  python test_ssh_key.py A100_47             # 测试指定服务器")
        print("  python test_ssh_key.py ~/.ssh/id_rsa user@36.163.20.47")


if __name__ == "__main__":
    main()
