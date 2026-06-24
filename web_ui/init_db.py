"""
初始化 MySQL 数据库：建库、建表、插入默认管理员
"""
import pymysql
from werkzeug.security import generate_password_hash
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

# 1. 建库
conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD)
cur = conn.cursor()
cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci")
cur.execute(f"USE `{DB_NAME}`")

# 2. users 表
cur.execute("""
CREATE TABLE IF NOT EXISTS `users` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `username` VARCHAR(50) NOT NULL UNIQUE,
    `password` VARCHAR(256) NOT NULL,
    `role` ENUM('admin','user') NOT NULL DEFAULT 'user',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")

# 3. face_db 表
cur.execute("""
CREATE TABLE IF NOT EXISTS `face_db` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `name` VARCHAR(100) NOT NULL,
    `image_path` VARCHAR(500),
    `feature_path` VARCHAR(500),
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")

# 4. face_recognition_log 表 — 先建后用 ALTER 添加新的事件类型
cur.execute("""
CREATE TABLE IF NOT EXISTS `face_recognition_log` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `name` VARCHAR(100),
    `event_type` VARCHAR(20) NOT NULL DEFAULT 'unknown',
    `confidence` FLOAT,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")

# 5. 插入默认管理员（如果不存在）
admin_pwd = generate_password_hash("123456")
cur.execute("SELECT COUNT(*) FROM `users` WHERE `username`='admin'")
if cur.fetchone()[0] == 0:
    cur.execute("INSERT INTO `users` (`username`,`password`,`role`) VALUES (%s,%s,%s)",
                ("admin", admin_pwd, "admin"))
    print("[init_db] 已创建管理员: admin / 123456")
else:
    print("[init_db] 管理员已存在，跳过")

conn.commit()
cur.close()
conn.close()
print("[init_db] 数据库初始化完成")
