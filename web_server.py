"""
分镜AI导演 - Web 前端服务 (SSE Streaming)
=================================
启动: uv run python web_server.py
访问: http://localhost:8000

特性:
- SSE 流式推送 Agent 中间步骤 (思考/工具调用/工具结果)
- 前端实时展示 Agent 处理进度
"""

import asyncio
import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import uvicorn

from fenjing_agent import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    build_graph,
    clear_web_ask_callback,
    set_web_ask_callback,
)

app = FastAPI(title="分镜AI导演")

# ── CORS 中间件 (允许其他 IP / 设备访问) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 目录准备 ──
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================
# Session 会话管理 (SSE Streaming)
# ============================================================

class AgentSession:
    """管理单个用户与 Agent 的会话状态，支持 SSE 流式推送中间步骤。"""

    def __init__(self):
        self.agent_app = build_graph()
        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self.event_queue: queue.Queue = queue.Queue()
        self.answer_queue: queue.Queue = queue.Queue()
        self.agent_thread: threading.Thread | None = None
        self.uploaded_files: list[dict] = []
        self.is_waiting_answer: bool = False

    # ── SSE 事件辅助 ──
    @staticmethod
    def _sse(event_type: str, data: dict) -> dict:
        """构造一条 SSE 事件字典。"""
        return {"event": event_type, "data": json.dumps(data, ensure_ascii=False)}

    # ── ask_user 回调 (Web 模式) ──
    def _ask_user_handler(self, question: str) -> str:
        """将问题推送为 SSE question 事件，阻塞等待用户回答。"""
        self.event_queue.put(self._sse("question", {"content": question}))
        self.is_waiting_answer = True
        try:
            answer = self.answer_queue.get(timeout=300)
        except queue.Empty:
            answer = ""
        finally:
            self.is_waiting_answer = False
        return answer

    # ── 启动流式 Agent ──
    def start_streaming(self, text: str, file_urls: list[str] | None = None):
        """启动 Agent 线程，使用 stream() 逐步推送中间事件。"""
        parts: list[str] = []
        if file_urls:
            parts.append("【用户上传了以下参考素材】")
            for url in file_urls:
                parts.append(f"- {url}")
            parts.append("")
        parts.append(text)
        full_message = "\n".join(parts)

        self.agent_thread = threading.Thread(
            target=self._run_agent_streaming,
            args=(full_message,),
            daemon=True,
        )
        self.agent_thread.start()

    # ── Agent 流式执行核心逻辑 ──
    def _run_agent_streaming(self, user_message: str):
        """在子线程中运行 Agent，通过 event_queue 推送 SSE 事件。"""
        set_web_ask_callback(self._ask_user_handler)
        put = self.event_queue.put
        sse = self._sse
        try:
            for chunk in self.agent_app.stream(
                {"messages": [HumanMessage(content=user_message)]},
                self.config,
                stream_mode="updates",
            ):
                for node_name, updates in chunk.items():
                    if node_name == "agent":
                        for msg in updates.get("messages", []):
                            if not isinstance(msg, AIMessage):
                                continue
                            # ── 提取模型思考过程 (thought parts) ──
                            original = msg.additional_kwargs.get("_genai_content")
                            if original and hasattr(original, "parts") and original.parts:
                                for part in original.parts:
                                    if (
                                        getattr(part, "thought", False)
                                        and hasattr(part, "text")
                                        and part.text
                                    ):
                                        put(sse("thinking", {"content": part.text}))

                            # ── 中间文本 (伴随 tool_calls 的说明文字) ──
                            if msg.content and msg.tool_calls:
                                put(sse("thinking", {"content": msg.content}))

                            # ── 工具调用事件 ──
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    put(sse("tool_call", {
                                        "name": tc["name"],
                                        "args": tc["args"],
                                    }))

                            # ── 最终文本回复 (无 tool_calls) ──
                            if msg.content and not msg.tool_calls:
                                put(sse("response", {"content": msg.content}))

                    elif node_name == "tools":
                        for msg in updates.get("messages", []):
                            if not isinstance(msg, ToolMessage):
                                continue
                            # ask_user 结果不推送 (其内容已通过 question 事件展示)
                            if msg.name == "ask_user":
                                continue
                            put(sse("tool_result", {
                                "name": msg.name,
                                "content": str(msg.content)[:500],
                            }))

            put(sse("done", {}))
        except Exception as e:
            print(f"[Agent Error] {e}")
            put(sse("error", {"content": str(e)}))
        finally:
            clear_web_ask_callback()

    # ── SSE 事件生成器 ──
    async def stream_events(self):
        """异步生成 SSE 事件流，直到 done/error 事件。"""
        loop = asyncio.get_event_loop()
        while True:
            try:
                evt = await loop.run_in_executor(
                    None, lambda: self.event_queue.get(timeout=2)
                )
                yield f"event: {evt['event']}\ndata: {evt['data']}\n\n"
                if evt["event"] in ("done", "error"):
                    break
            except queue.Empty:
                # 发送 keepalive 避免连接超时
                yield ": keepalive\n\n"

    # ── 获取工作流状态 ──
    def get_state(self) -> dict:
        try:
            snapshot = self.agent_app.get_state(self.config)
            s = snapshot.values
            return {
                "story_concept": s.get("story_concept"),
                "duration": s.get("duration"),
                "ratio": s.get("ratio"),
                "style": s.get("style"),
                "reference_images": s.get("reference_images", []),
                "storyboard_content": s.get("storyboard_content"),
                "video_task_id": s.get("video_task_id"),
            }
        except Exception:
            return {}


# 全局 session 存储
_sessions: dict[str, AgentSession] = {}


def _get_session(session_id: str) -> AgentSession:
    if session_id not in _sessions:
        _sessions[session_id] = AgentSession()
    return _sessions[session_id]


# ============================================================
# API 路由
# ============================================================

@app.post("/api/chat/stream")
async def chat_stream(payload: dict):
    """SSE 流式聊天：推送思考过程、工具调用、最终回复等中间事件。"""
    session_id = payload.get("session_id", "default")
    message = payload.get("message", "")
    file_urls = payload.get("file_urls", [])
    if not message:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)
    session = _get_session(session_id)
    session.start_streaming(message, file_urls or None)
    return StreamingResponse(
        session.stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/answer")
def chat_answer(payload: dict):
    """回答 ask_user 的追问（将用户输入传入 Agent 阻塞等待的回调）。
    支持在回答时附带上传的图片 URL。"""
    session_id = payload.get("session_id", "default")
    answer = payload.get("answer", "")
    file_urls = payload.get("file_urls", [])
    session = _get_session(session_id)
    # 如果回答中附带了图片，拼入答案文本
    if file_urls:
        parts = ["【用户上传了以下参考素材】"]
        for url in file_urls:
            parts.append(f"- {url}")
        parts.append("")
        parts.append(answer)
        answer = "\n".join(parts)
    session.answer_queue.put(answer)
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_files(
    session_id: str = Form("default"),
    files: list[UploadFile] = File(...),
):
    """上传参考素材图片。"""
    print(f"[Upload] session={session_id}, files={len(files)}")
    session = _get_session(session_id)
    uploaded = []
    for file in files:
        ext = Path(file.filename or "img.png").suffix or ".png"
        new_name = f"{uuid.uuid4().hex[:12]}{ext}"
        file_path = UPLOAD_DIR / new_name
        content = await file.read()
        print(f"[Upload] {file.filename} -> {new_name} ({len(content)} bytes)")
        with open(file_path, "wb") as f:
            f.write(content)
        url = f"/uploads/{new_name}"
        session.uploaded_files.append({"name": file.filename, "url": url})
        uploaded.append({"name": file.filename, "url": url})
    print(f"[Upload] done, {len(uploaded)} file(s)")
    return {"files": uploaded}


@app.get("/api/state/{session_id}")
def get_state(session_id: str):
    """获取当前工作流状态。"""
    return _get_session(session_id).get_state()


@app.get("/")
def index():
    """返回前端页面。"""
    return FileResponse("static/index.html")


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    print()
    print("=" * 50)
    print("  Fenjing AI Director - Web Server (SSE)")
    print("=" * 50)
    print("  URL: http://localhost:8000")
    print("=" * 50)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
