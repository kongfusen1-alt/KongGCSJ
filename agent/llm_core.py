"""
智能体核心 — DeepSeek + Function Calling
"""
import json
import base64
import cv2
from openai import OpenAI
from agent.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from agent.tools import (
    TOOLS_SCHEMA, query_records, list_faces,
    get_system_status, analyze_frame
)

SYSTEM_PROMPT = """你是一个安防人脸识别系统的智能助手。你的名字叫 FaceID 助手。

你可以帮用户：
1. 查询人脸识别记录（"今天谁来过"、"查一下张三的记录"）
2. 查看已注册的人脸名单（"数据库里有谁"）
3. 获取系统运行状态（"系统正常吗"、"FPS多少"）
4. 分析当前摄像头画面（"门口有几个人"、"画面里有什么"）—— 注意视觉分析功能需要模型支持多模态，若不支持会返回错误

回答要求：
- 简洁明了，不要啰嗦
- 查到记录时用列表展示
- 没查到就明确说没有
- 不确定的事情不要编造
"""


def _get_client():
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


# ─── 视觉分析 ───

def _vision_analyze(frame, question):
    """把当前画面编码发给 LLM，回答视觉问题"""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf).decode()

    try:
        resp = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }}
                ]
            }],
            max_tokens=500
        )
        return {"answer": resp.choices[0].message.content}
    except Exception as e:
        return {"answer": f"视觉分析暂不可用（当前模型不支持多模态）：{e}"}


# ─── 主智能体类 ───

class FaceAgent:
    def __init__(self, db_query, db_insert, get_stats_fn):
        self.db_query = db_query
        self.db_insert = db_insert
        self.get_stats_fn = get_stats_fn
        self.client = _get_client()
        self.history = []

    def _exec_tool(self, name, args, frame=None):
        """执行工具调用"""
        if name == "query_records":
            return query_records(
                self.db_query,
                name=args.get("name"),
                time_range=args.get("time_range", "today"),
                limit=args.get("limit", 0)
            )
        elif name == "list_faces":
            return list_faces(self.db_query)
        elif name == "get_system_status":
            return get_system_status(self.get_stats_fn)
        elif name == "analyze_frame":
            return analyze_frame(
                frame,
                args.get("question", "描述一下画面"),
                _vision_analyze
            )
        return {"error": f"未知工具: {name}"}

    def chat(self, user_message, frame=None):
        """处理一轮对话，支持 function calling 多轮循环"""
        self.history.append({"role": "user", "content": user_message})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history[-10:]
        ]

        for _ in range(5):
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                max_tokens=1024
            )

            msg = resp.choices[0].message

            if not msg.tool_calls:
                reply = msg.content or ""
                self.history.append({"role": "assistant", "content": reply})
                if len(self.history) > 40:
                    self.history = self.history[-20:]
                return reply

            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = self._exec_tool(fn_name, fn_args, frame=frame)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })

        reply = "抱歉，处理过程太复杂了，请换个方式提问。"
        self.history.append({"role": "assistant", "content": reply})
        return reply
