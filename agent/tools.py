"""
智能体工具集 — MVP 版本
每个 tool 对应一个可被 LLM 调用的函数
"""

# ─── 工具 Schema（给 DeepSeek function calling 用） ───

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_records",
            "description": "查询人脸识别记录。支持按时间范围、姓名过滤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "按姓名过滤，留空则查询全部"
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["today", "yesterday", "week", "all"],
                        "description": "时间范围：today=今天, yesterday=昨天, week=本周, all=全部"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回条数上限，设为 0 则不限制"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_faces",
            "description": "查看人脸数据库中已注册的所有人员名单",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": "获取系统运行状态：FPS、检测人脸数、数据库人数等",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_frame",
            "description": "分析当前摄像头画面，回答关于画面内容的问题，如人数、场景描述等。需要视觉能力。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "关于画面的问题，例如：现在门口有几个人？、画面里有什么？"
                    }
                },
                "required": ["question"]
            }
        }
    }
]


# ─── 工具实现 ───

def query_records(db_query, name=None, time_range="today", limit=0):
    """查询识别记录"""
    conditions = []
    args = []

    if name:
        conditions.append("name LIKE %s")
        args.append(f"%{name}%")

    if time_range == "today":
        conditions.append("DATE(created_at) = CURDATE()")
    elif time_range == "yesterday":
        conditions.append("DATE(created_at) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)")
    elif time_range == "week":
        conditions.append("created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    if limit > 0:
        sql = f"SELECT name, event_type, confidence, created_at FROM face_recognition_log {where} ORDER BY created_at DESC LIMIT %s"
        args.append(limit)
        rows = db_query(sql, tuple(args))
    else:
        sql = f"SELECT name, event_type, confidence, created_at FROM face_recognition_log {where} ORDER BY created_at DESC"
        rows = db_query(sql, tuple(args) if args else None)
    if not rows:
        return {"count": 0, "records": [], "message": "没有找到相关记录"}

    records = []
    for r in rows:
        records.append({
            "name": r["name"],
            "event_type": r["event_type"],
            "confidence": round(float(r["confidence"]), 2) if r["confidence"] else None,
            "time": r["created_at"].strftime("%Y-%m-%d %H:%M:%S") if r["created_at"] else None
        })

    return {"count": len(records), "records": records}


def list_faces(db_query):
    """列出人脸数据库"""
    rows = db_query("SELECT id, name, created_at FROM face_db ORDER BY id")
    if not rows:
        return {"count": 0, "faces": [], "message": "人脸数据库为空"}

    faces = []
    for r in rows:
        faces.append({
            "id": r["id"],
            "name": r["name"],
            "added_at": r["created_at"].strftime("%Y-%m-%d %H:%M:%S") if r["created_at"] else None
        })

    return {"count": len(faces), "faces": faces}


def get_system_status(get_stats_fn):
    """获取系统状态"""
    return get_stats_fn()


def analyze_frame(frame, question, vision_analyze_fn):
    """分析当前画面"""
    if frame is None:
        return {"error": "摄像头未打开，无法获取画面"}
    return vision_analyze_fn(frame, question)
