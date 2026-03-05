import sqlite3
import json
from cryptography.fernet import Fernet
import os
import hashlib

class DatabaseManager:
    def __init__(self, db_path='server_management.db', key_file='encryption.key'):
        self.db_path = db_path
        self.key_file = key_file
        self.fernet = self._get_or_create_fernet_key()
        self.init_database()
    
    def _get_or_create_fernet_key(self):
        """获取或创建加密密钥"""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
        return Fernet(key)
    
    def encrypt_data(self, data):
        """加密数据"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return self.fernet.encrypt(data).decode('utf-8')
    
    def decrypt_data(self, encrypted_data):
        """解密数据"""
        if isinstance(encrypted_data, str):
            encrypted_data = encrypted_data.encode('utf-8')
        return self.fernet.decrypt(encrypted_data).decode('utf-8')
    
    def _safe_decrypt(self, data):
        """安全解密，如果解密失败则返回原数据"""
        if not data:
            return None
        try:
            return self.decrypt_data(data)
        except:
            # 如果解密失败，可能是明文数据，直接返回
            return data
    
    def init_database(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建服务器表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS servers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    ip TEXT NOT NULL,
                    port INTEGER DEFAULT 22,
                    username TEXT NOT NULL,
                    password TEXT NOT NULL,
                    dedicated_password TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    code_path TEXT DEFAULT '',
                    data_path TEXT DEFAULT '',
                    auth_type TEXT DEFAULT 'password',
                    key_path TEXT DEFAULT ''
                )
            ''')
            # 迁移：为旧表添加 code_path, data_path 列
            cursor.execute("PRAGMA table_info(servers)")
            cols = [r[1] for r in cursor.fetchall()]
            if 'code_path' not in cols:
                cursor.execute("ALTER TABLE servers ADD COLUMN code_path TEXT DEFAULT ''")
            if 'data_path' not in cols:
                cursor.execute("ALTER TABLE servers ADD COLUMN data_path TEXT DEFAULT ''")
            if 'auth_type' not in cols:
                cursor.execute("ALTER TABLE servers ADD COLUMN auth_type TEXT DEFAULT 'password'")
            if 'key_path' not in cols:
                cursor.execute("ALTER TABLE servers ADD COLUMN key_path TEXT DEFAULT ''")
            
            # 创建管理员表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS admin_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 全局配置表（代码路径、数据路径、空闲GPU阈值等）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cluster_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT UNIQUE NOT NULL,
                    config_value TEXT NOT NULL,
                    description TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 训练任务表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    script_path TEXT NOT NULL,
                    script_args TEXT,
                    server_name TEXT,
                    gpu_ids TEXT,
                    priority INTEGER DEFAULT 5,
                    status TEXT DEFAULT 'pending',
                    log_path TEXT,
                    weight_path TEXT,
                    pid INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    finished_at TIMESTAMP,
                    error_message TEXT
                )
            ''')
            # 迁移：为训练任务表添加 allowed_servers（可选服务器白名单，空表示所有服务器）和 gpu_count（所需 GPU 数）
            cursor.execute("PRAGMA table_info(training_tasks)")
            train_cols = [r[1] for r in cursor.fetchall()]
            if 'allowed_servers' not in train_cols:
                cursor.execute("ALTER TABLE training_tasks ADD COLUMN allowed_servers TEXT DEFAULT ''")
            if 'gpu_count' not in train_cols:
                cursor.execute("ALTER TABLE training_tasks ADD COLUMN gpu_count INTEGER DEFAULT 1")
            
            # 测试任务表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    server_name TEXT,
                    port INTEGER,
                    gpu_ids TEXT,
                    weight_path TEXT,
                    script_path TEXT,
                    script_args TEXT,
                    training_task_id INTEGER,
                    status TEXT DEFAULT 'pending',
                    pid INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    finished_at TIMESTAMP,
                    result TEXT,
                    test_code_path TEXT,
                    mock_url TEXT,
                    mock_task_name TEXT,
                    user_token TEXT,
                    run_id TEXT,
                    FOREIGN KEY (training_task_id) REFERENCES training_tasks(id)
                )
            ''')
            # 迁移：为测试任务表添加新参数列
            cursor.execute("PRAGMA table_info(test_tasks)")
            test_cols = [r[1] for r in cursor.fetchall()]
            if 'test_code_path' not in test_cols:
                cursor.execute("ALTER TABLE test_tasks ADD COLUMN test_code_path TEXT DEFAULT ''")
            if 'mock_url' not in test_cols:
                cursor.execute("ALTER TABLE test_tasks ADD COLUMN mock_url TEXT DEFAULT ''")
            if 'mock_task_name' not in test_cols:
                cursor.execute("ALTER TABLE test_tasks ADD COLUMN mock_task_name TEXT DEFAULT ''")
            if 'user_token' not in test_cols:
                cursor.execute("ALTER TABLE test_tasks ADD COLUMN user_token TEXT DEFAULT ''")
            if 'run_id' not in test_cols:
                cursor.execute("ALTER TABLE test_tasks ADD COLUMN run_id TEXT DEFAULT ''")

            # 部署任务表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS deploy_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    script_path TEXT NOT NULL,
                    weight_path TEXT,
                    server_name TEXT,
                    gpu_ids TEXT,
                    priority INTEGER DEFAULT 5,
                    allowed_servers TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    log_path TEXT,
                    pid INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    finished_at TIMESTAMP,
                    result TEXT,
                    port INTEGER
                )
            ''')
            # 迁移：为部署任务表添加 port
            cursor.execute("PRAGMA table_info(deploy_tasks)")
            deploy_cols = [r[1] for r in cursor.fetchall()]
            if 'port' not in deploy_cols:
                cursor.execute("ALTER TABLE deploy_tasks ADD COLUMN port INTEGER")
            
            # 模型权重记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    server_name TEXT NOT NULL,
                    training_task_id INTEGER,
                    size_mb REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (training_task_id) REFERENCES training_tasks(id)
                )
            ''')
            
            # 内存报警记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    server_name TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def add_server(self, name, ip, port, username, password, description='', dedicated_password=None, code_path='', data_path='', auth_type='password', key_path=''):
        """添加服务器"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 只加密用户名，密码和专用密码明文存储
                encrypted_username = self.encrypt_data(username)
                
                cursor.execute('''
                    INSERT INTO servers (name, ip, port, username, password, dedicated_password, description, code_path, data_path, auth_type, key_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (name, ip, port, encrypted_username, password, dedicated_password, description, code_path or '', data_path or '', auth_type or 'password', key_path or ''))
                
                conn.commit()
                return True, "服务器添加成功"
        except sqlite3.IntegrityError:
            return False, "服务器名称已存在"
        except Exception as e:
            return False, f"添加服务器失败: {str(e)}"
    
    def get_all_servers(self):
        """获取所有服务器（解密后）"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM servers ORDER BY name')
                rows = cursor.fetchall()
                
                servers = []
                for row in rows:
                    server = {
                        'id': row[0],
                        'name': row[1],
                        'ip': row[2],
                        'port': row[3],
                        'username': self.decrypt_data(row[4]),
                        'password': row[5],  # 明文存储，直接返回
                        'description': row[6],
                        'created_at': row[7],
                        'updated_at': row[8],
                        'dedicated_password': row[9] if len(row) > 9 and row[9] else None,
                        'code_path': row[10] if len(row) > 10 and row[10] else '',
                        'data_path': row[11] if len(row) > 11 and row[11] else '',
                        'auth_type': row[12] if len(row) > 12 and row[12] else 'password',
                        'key_path': row[13] if len(row) > 13 and row[13] else '',
                    }
                    servers.append(server)
                
                return servers
        except Exception as e:
            print(f"获取服务器列表失败: {str(e)}")
            return []
    
    def get_server_by_name(self, name):
        """根据名称获取服务器"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM servers WHERE name = ?', (name,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'ip': row[2],
                        'port': row[3],
                        'username': self.decrypt_data(row[4]),
                        'password': row[5],
                        'description': row[6],
                        'created_at': row[7],
                        'updated_at': row[8],
                        'dedicated_password': row[9] if len(row) > 9 and row[9] else None,
                        'code_path': row[10] if len(row) > 10 and row[10] else '',
                        'data_path': row[11] if len(row) > 11 and row[11] else '',
                        'auth_type': row[12] if len(row) > 12 and row[12] else 'password',
                        'key_path': row[13] if len(row) > 13 and row[13] else '',
                    }
                return None
        except Exception as e:
            print(f"获取服务器失败: {str(e)}")
            return None
    
    def update_server(self, server_id, name=None, ip=None, port=None, username=None, password=None, description=None, dedicated_password=None, code_path=None, data_path=None, auth_type=None, key_path=None):
        """更新服务器信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 构建更新语句
                update_fields = []
                values = []
                
                if name is not None:
                    update_fields.append('name = ?')
                    values.append(name)
                
                if ip is not None:
                    update_fields.append('ip = ?')
                    values.append(ip)
                
                if port is not None:
                    update_fields.append('port = ?')
                    values.append(port)
                
                if username is not None:
                    update_fields.append('username = ?')
                    values.append(self.encrypt_data(username))
                
                if password is not None:
                    update_fields.append('password = ?')
                    values.append(password)  # 明文存储
                
                if dedicated_password is not None:
                    if dedicated_password == '':
                        update_fields.append('dedicated_password = NULL')
                    else:
                        update_fields.append('dedicated_password = ?')
                        values.append(dedicated_password)  # 明文存储
                
                if description is not None:
                    update_fields.append('description = ?')
                    values.append(description)
                
                if code_path is not None:
                    update_fields.append('code_path = ?')
                    values.append(code_path)
                
                if data_path is not None:
                    update_fields.append('data_path = ?')
                    values.append(data_path)
                
                if auth_type is not None:
                    update_fields.append('auth_type = ?')
                    values.append(auth_type)
                
                if key_path is not None:
                    update_fields.append('key_path = ?')
                    values.append(key_path)
                
                update_fields.append('updated_at = CURRENT_TIMESTAMP')
                values.append(server_id)
                
                if update_fields:
                    sql = f"UPDATE servers SET {', '.join(update_fields)} WHERE id = ?"
                    cursor.execute(sql, values)
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        return True, "服务器更新成功"
                    else:
                        return False, "服务器不存在"
                
                return False, "没有提供更新字段"
        
        except sqlite3.IntegrityError:
            return False, "服务器名称已存在"
        except Exception as e:
            return False, f"更新服务器失败: {str(e)}"
    
    def delete_server(self, server_id):
        """删除服务器"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM servers WHERE id = ?', (server_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    return True, "服务器删除成功"
                else:
                    return False, "服务器不存在"
        
        except Exception as e:
            return False, f"删除服务器失败: {str(e)}"
    
    def migrate_from_json(self, json_file='servers.json'):
        """从JSON文件迁移数据到数据库"""
        try:
            if not os.path.exists(json_file):
                return False, "JSON文件不存在"
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            servers = data.get('servers', [])
            success_count = 0
            error_messages = []
            
            for server in servers:
                success, message = self.add_server(
                    name=server.get('name', ''),
                    ip=server.get('ip', ''),
                    port=server.get('port', 22),
                    username=server.get('username', ''),
                    password=server.get('password', ''),
                    description=server.get('description', ''),
                    dedicated_password=server.get('dedicated_password')
                )
                
                if success:
                    success_count += 1
                else:
                    error_messages.append(f"服务器 {server.get('name', 'Unknown')}: {message}")
            
            if success_count > 0:
                return True, f"成功迁移 {success_count} 个服务器"
            else:
                return False, f"迁移失败: {'; '.join(error_messages)}"
        
        except Exception as e:
            return False, f"迁移失败: {str(e)}"
    
    def create_admin_user(self, username, password):
        """创建管理员用户"""
        try:
            # 使用SHA-256哈希密码
            password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO admin_users (username, password_hash)
                    VALUES (?, ?)
                ''', (username, password_hash))
                conn.commit()
                return True, "管理员用户创建成功"
        
        except sqlite3.IntegrityError:
            return False, "管理员用户名已存在"
        except Exception as e:
            return False, f"创建管理员失败: {str(e)}"
    
    def verify_admin(self, username, password):
        """验证管理员用户"""
        try:
            password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id FROM admin_users 
                    WHERE username = ? AND password_hash = ?
                ''', (username, password_hash))
                
                return cursor.fetchone() is not None
        
        except Exception as e:
            print(f"验证管理员失败: {str(e)}")
            return False
    
    def update_admin_password(self, username, new_password):
        """更新管理员密码"""
        try:
            password_hash = hashlib.sha256(new_password.encode('utf-8')).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE admin_users 
                    SET password_hash = ? 
                    WHERE username = ?
                ''', (password_hash, username))
                conn.commit()
                
                if cursor.rowcount > 0:
                    return True, "密码更新成功"
                else:
                    return False, "用户不存在"
        
        except Exception as e:
            return False, f"更新密码失败: {str(e)}"
    
    # ========== 集群配置 ==========
    def get_config(self, key, default=None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT config_value FROM cluster_config WHERE config_key = ?', (key,))
                row = cursor.fetchone()
                return row[0] if row else default
        except Exception:
            return default
    
    def set_config(self, key, value, description=''):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO cluster_config (config_key, config_value, description, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(config_key) DO UPDATE SET config_value=?, description=?, updated_at=CURRENT_TIMESTAMP
                ''', (key, str(value), description, str(value), description))
                conn.commit()
                return True
        except Exception as e:
            print(f"设置配置失败: {e}")
            return False
    
    # ========== 训练任务 ==========
    def add_training_task(self, name, script_path, script_args='', priority=5, gpu_count=1, allowed_servers=None):
        """allowed_servers: 可选服务器名列表，空/None 表示所有服务器可用；gpu_count: 任务需要的 GPU 数量"""
        try:
            # 规范化 gpu_count
            try:
                gpu_count_int = int(gpu_count)
            except (TypeError, ValueError):
                gpu_count_int = 1
            if gpu_count_int < 1:
                gpu_count_int = 1

            allowed_str = json.dumps(allowed_servers or []) if allowed_servers is not None else '[]'
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO training_tasks (name, script_path, script_args, priority, gpu_ids, status, log_path, weight_path, pid, created_at, started_at, finished_at, error_message, server_name, allowed_servers, gpu_count)
                    VALUES (?, ?, ?, ?, NULL, 'pending', NULL, NULL, NULL, CURRENT_TIMESTAMP, NULL, NULL, NULL, NULL, ?, ?)
                ''', (name, script_path, script_args, priority, allowed_str, gpu_count_int))
                conn.commit()
                return True, cursor.lastrowid
        except Exception as e:
            return False, str(e)
    
    def update_training_task(self, task_id, **kwargs):
        try:
            allowed = {'server_name', 'gpu_ids', 'status', 'log_path', 'weight_path', 'pid', 'started_at', 'finished_at', 'error_message', 'script_path', 'gpu_count'}
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                updates = []
                values = []
                for k, v in kwargs.items():
                    if k in allowed:
                        updates.append(f'{k}=?')
                        values.append(v)
                if not updates:
                    return False
                values.append(task_id)
                cursor.execute(f'UPDATE training_tasks SET {", ".join(updates)} WHERE id=?', values)
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False
    
    def get_pending_training_tasks(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM training_tasks WHERE status="pending" ORDER BY priority DESC, created_at ASC')
                return [self._row_to_training_task(row) for row in cursor.fetchall()]
        except Exception:
            return []
    
    def get_training_task(self, task_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM training_tasks WHERE id=?', (task_id,))
                row = cursor.fetchone()
                return self._row_to_training_task(row) if row else None
        except Exception:
            return None
    
    def get_all_training_tasks(self, limit=100):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM training_tasks ORDER BY id DESC LIMIT ?', (limit,))
                return [self._row_to_training_task(row) for row in cursor.fetchall()]
        except Exception:
            return []

    def delete_training_task(self, task_id):
        """删除训练任务记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM training_tasks WHERE id=?', (task_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False
    
    def _row_to_training_task(self, row):
        allowed = []
        gpu_count = 1
        # 目前 training_tasks 基础列为 15 个，之后按顺序追加 allowed_servers, gpu_count
        if len(row) > 15 and row[15]:
            try:
                allowed = json.loads(row[15])
            except (TypeError, ValueError):
                allowed = []
        if len(row) > 16 and row[16] is not None:
            try:
                gpu_count = int(row[16])
            except (TypeError, ValueError):
                gpu_count = 1
        return {
            'id': row[0], 'name': row[1], 'script_path': row[2], 'script_args': row[3] or '',
            'server_name': row[4], 'gpu_ids': row[5], 'priority': row[6], 'status': row[7],
            'log_path': row[8], 'weight_path': row[9], 'pid': row[10],
            'created_at': row[11], 'started_at': row[12], 'finished_at': row[13], 'error_message': row[14],
            'allowed_servers': allowed,
            'gpu_count': gpu_count
        }
    
    # ========== 测试任务 ==========
    def add_test_task(self, name, task_type, script_path='', script_args='', training_task_id=None,
                      test_code_path='', mock_url='', mock_task_name='', user_token='', run_id=''):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO test_tasks (
                        name, task_type, script_path, script_args, training_task_id, status,
                        test_code_path, mock_url, mock_task_name, user_token, run_id
                    )
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                ''', (name, task_type, script_path, script_args, training_task_id,
                      test_code_path, mock_url, mock_task_name, user_token, run_id))
                conn.commit()
                return True, cursor.lastrowid
        except Exception as e:
            return False, str(e)
    
    def update_test_task(self, task_id, **kwargs):
        try:
            allowed = {'server_name', 'port', 'gpu_ids', 'status', 'pid', 'started_at', 'finished_at', 'result'}
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                updates = []
                values = []
                for k, v in kwargs.items():
                    if k in allowed:
                        updates.append(f'{k}=?')
                        values.append(v)
                if not updates:
                    return False
                values.append(task_id)
                cursor.execute(f'UPDATE test_tasks SET {", ".join(updates)} WHERE id=?', values)
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False
    
    def get_test_task(self, task_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM test_tasks WHERE id=?', (task_id,))
                row = cursor.fetchone()
                return self._row_to_test_task(row) if row else None
        except Exception:
            return None
    
    def get_all_test_tasks(self, limit=100):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM test_tasks ORDER BY id DESC LIMIT ?', (limit,))
                return [self._row_to_test_task(row) for row in cursor.fetchall()]
        except Exception:
            return []

    def delete_test_task(self, task_id):
        """删除测试任务记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM test_tasks WHERE id=?', (task_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False
    
    def get_pending_test_tasks(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM test_tasks WHERE status="pending" ORDER BY created_at ASC')
                return [self._row_to_test_task(row) for row in cursor.fetchall()]
        except Exception:
            return []
    
    def _row_to_test_task(self, row):
        return {
            'id': row[0], 'name': row[1], 'task_type': row[2], 'server_name': row[3], 'port': row[4],
            'gpu_ids': row[5], 'weight_path': row[6], 'script_path': row[7], 'script_args': row[8],
            'training_task_id': row[9], 'status': row[10], 'pid': row[11],
            'created_at': row[12], 'started_at': row[13], 'finished_at': row[14], 'result': row[15],
            'test_code_path': row[16] if len(row) > 16 else '',
            'mock_url': row[17] if len(row) > 17 else '',
            'mock_task_name': row[18] if len(row) > 18 else '',
            'user_token': row[19] if len(row) > 19 else '',
            'run_id': row[20] if len(row) > 20 else '',
        }

    # ========== 部署任务 ==========
    def add_deploy_task(self, name, script_path, weight_path='', priority=5, allowed_servers=None, port=None):
        try:
            allowed_str = json.dumps(allowed_servers or []) if allowed_servers is not None else '[]'
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO deploy_tasks (name, script_path, weight_path, priority, allowed_servers, status, port)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?)
                ''', (name, script_path, weight_path, int(priority), allowed_str, port))
                conn.commit()
                return True, cursor.lastrowid
        except Exception as e:
            return False, str(e)

    def update_deploy_task(self, task_id, **kwargs):
        try:
            allowed = {'server_name', 'gpu_ids', 'status', 'log_path', 'pid', 'started_at', 'finished_at', 'result', 'weight_path', 'script_path', 'priority', 'port'}
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                updates = []
                values = []
                for k, v in kwargs.items():
                    if k in allowed:
                        updates.append(f'{k}=?')
                        values.append(v)
                if not updates:
                    return False
                values.append(task_id)
                cursor.execute(f'UPDATE deploy_tasks SET {", ".join(updates)} WHERE id=?', values)
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False

    def get_pending_deploy_tasks(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM deploy_tasks WHERE status="pending" ORDER BY priority DESC, created_at ASC')
                return [self._row_to_deploy_task(row) for row in cursor.fetchall()]
        except Exception:
            return []

    def get_deploy_task(self, task_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM deploy_tasks WHERE id=?', (task_id,))
                row = cursor.fetchone()
                return self._row_to_deploy_task(row) if row else None
        except Exception:
            return None

    def get_all_deploy_tasks(self, limit=100):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM deploy_tasks ORDER BY id DESC LIMIT ?', (limit,))
                return [self._row_to_deploy_task(row) for row in cursor.fetchall()]
        except Exception:
            return []

    def delete_deploy_task(self, task_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM deploy_tasks WHERE id=?', (task_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False

    def _row_to_deploy_task(self, row):
        allowed = []
        if len(row) > 7 and row[7]:
            try:
                allowed = json.loads(row[7])
            except (TypeError, ValueError):
                allowed = []
        return {
            'id': row[0], 'name': row[1], 'script_path': row[2], 'weight_path': row[3] or '',
            'server_name': row[4], 'gpu_ids': row[5], 'priority': row[6], 'allowed_servers': allowed,
            'status': row[8], 'log_path': row[9], 'pid': row[10], 'created_at': row[11],
            'started_at': row[12], 'finished_at': row[13], 'result': row[14],
            'port': row[15] if len(row) > 15 else None
        }
    
    # ========== 模型权重 ==========
    def add_model_weight(self, name, path, server_name, training_task_id=None, size_mb=None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_weights (name, path, server_name, training_task_id, size_mb)
                    VALUES (?, ?, ?, ?, ?)
                ''', (name, path, server_name, training_task_id, size_mb))
                conn.commit()
                return True, cursor.lastrowid
        except Exception as e:
            return False, str(e)

    def upsert_model_weight_for_task(self, training_task_id, name, path, server_name, size_mb=None):
        """按 training_task_id 更新权重记录；不存在则新增。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT id FROM model_weights WHERE training_task_id=? ORDER BY id DESC LIMIT 1',
                    (training_task_id,)
                )
                row = cursor.fetchone()
                if row:
                    cursor.execute(
                        'UPDATE model_weights SET name=?, path=?, server_name=?, size_mb=? WHERE id=?',
                        (name, path, server_name, size_mb, row[0])
                    )
                    conn.commit()
                    return True, row[0]

                cursor.execute('''
                    INSERT INTO model_weights (name, path, server_name, training_task_id, size_mb)
                    VALUES (?, ?, ?, ?, ?)
                ''', (name, path, server_name, training_task_id, size_mb))
                conn.commit()
                return True, cursor.lastrowid
        except Exception as e:
            return False, str(e)
    
    def get_all_model_weights(self, limit=200):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM model_weights ORDER BY created_at DESC LIMIT ?', (limit,))
                rows = cursor.fetchall()
                return [{'id': r[0], 'name': r[1], 'path': r[2], 'server_name': r[3],
                         'training_task_id': r[4], 'size_mb': r[5], 'created_at': r[6]} for r in rows]
        except Exception:
            return []
    
    # ========== 内存报警 ==========
    def add_memory_alert(self, server_name, alert_type, message):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO memory_alerts (server_name, alert_type, message) VALUES (?, ?, ?)',
                               (server_name, alert_type, message))
                conn.commit()
                return True
        except Exception:
            return False
    
    def get_recent_alerts(self, limit=50):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM memory_alerts ORDER BY created_at DESC LIMIT ?', (limit,))
                rows = cursor.fetchall()
                return [{'id': r[0], 'server_name': r[1], 'alert_type': r[2], 'message': r[3], 'created_at': r[4]} for r in rows]
        except Exception:
            return []