#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from database import DatabaseManager
from app import _ssh_connect_params   # 复用你项目里现成的 SSH 参数构造逻辑
import paramiko

SERVER_NAME = 'wujie_new_3'

# 这里把你要测试的命令写完整（和面板里的保持一致）
COMMAND = (
    "source ~/.bashrc 2>/dev/null; "
    "source ~/.profile 2>/dev/null; "
    "cd /data/zhangkaidong/A1 2>/dev/null; "
    "conda activate a1 && "
    "CUDA_VISIBLE_DEVICES=2 nohup bash .cluster_snapshots/train_rc_task14_1772594271.sh "
    "> /tmp/train_14_1772594277.log 2>&1 & echo $!"
)

def main():
    db = DatabaseManager()
    server = db.get_server_by_name(SERVER_NAME)
    if not server:
        print(f"服务器 {SERVER_NAME} 不存在")
        return

    params = _ssh_connect_params(server)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print("连接参数:", {k: v for k, v in params.items() if k != "password" and k != "pkey"})

    ssh.connect(**params)
    stdin, stdout, stderr = ssh.exec_command(COMMAND, timeout=60)
    exit_status = stdout.channel.recv_exit_status()

    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")

    ssh.close()

    print("==== exit_status ====")
    print(exit_status)
    print("==== STDOUT ====")
    print(out)
    print("==== STDERR ====")
    print(err)

if __name__ == "__main__":
    main()