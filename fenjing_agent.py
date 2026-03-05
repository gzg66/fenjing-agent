"""
游戏营销与 UGC 视频自动化生成 Agent
===================================
基于 LangGraph ReAct 架构，实现视频分镜脚本自动化生成流水线。

工作流：意图收集 → 深度挖掘 → 生成参考图 → 构建分镜 → 提交视频任务

特性：
- MemorySaver 持久化多轮对话状态（历史记忆）
- Human-in-the-loop（ask_user 工具实现人类在环）
- 结构化 State 业务字段追踪工作流进度

使用方式：
  1. 复制 .env.example 为 .env，填入 GOOGLE_API_KEY
  2. uv sync
  3. uv run python fenjing_agent.py
"""

import json
import os
import threading
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


# ============================================================
# Web 模式回调支持 (线程安全)
# ============================================================
_tls = threading.local()


def set_web_ask_callback(callback):
    """为当前线程设置 ask_user 的 Web 回调函数。callback(question: str) -> str"""
    _tls.ask_user_callback = callback


def clear_web_ask_callback():
    """清除当前线程的 ask_user Web 回调。"""
    _tls.ask_user_callback = None


# ============================================================
# 1. 状态定义 (State)
#    - messages：使用 add_messages reducer，自动追加消息
#    - 其余业务字段：使用"最新值覆盖"策略，追踪工作流进度
# ============================================================

class VideoAgentState(TypedDict):
    """Agent 状态定义，包含消息历史和结构化业务字段。"""
    messages: Annotated[list[BaseMessage], add_messages]  # 多轮对话消息列表
    story_concept: str          # 故事概念（一句话描述）
    duration: int               # 视频时长 (4-15 秒)
    ratio: str                  # 画幅比例 (16:9 / 9:16 / 1:1 / 21:9 / 4:3 / 3:4)
    style: str                  # 视觉风格
    storyboard_content: str     # 生成的分镜脚本内容
    reference_images: list[str] # 参考图片 URL 列表
    video_task_id: str          # Seedance 2.0 视频任务 ID


# ============================================================
# 2. 核心工具 (Tools)
#    - ask_user: 向用户追问缺失信息（Human-in-the-loop）
#    - generate_reference_images: 模拟 Seedream 4.5 生图
#    - build_storyboard: 构建结构化分镜脚本
#    - submit_seedance_task: 模拟 Seedance 2.0 提交视频任务
# ============================================================

@tool
def ask_user(question: str) -> str:
    """当缺少时长、画幅、风格、故事细节等必要信息时，调用此工具向用户提问。
    在 CLI 环境下会阻塞等待用户输入并将输入作为 Observation 返回。

    Args:
        question: 要向用户提出的问题，请使用友好的中文表述。
    """
    # Web 模式：通过线程安全回调与前端交互
    callback = getattr(_tls, "ask_user_callback", None)
    if callback is not None:
        answer = callback(question)
        if not answer:
            return "用户未输入任何内容，请换个方式再问一次。"
        return f"用户回答：{answer}"

    # CLI 模式：在终端打印问题并阻塞等待用户输入
    print(f"\n🤖 Agent 追问: {question}")
    answer = input("👤 用户回答: ").strip()
    if not answer:
        return "用户未输入任何内容，请换个方式再问一次。"
    return f"用户回答：{answer}"


@tool
def generate_reference_images(prompt: str, image_size: str, num_images: int = 2) -> str:
    """调用 Seedream 4.5 API 生成参考图片（当前为 Mock 实现）。

    Args:
        prompt: 英文图片描述，应包含角色外貌、服装、姿势、表情、环境等细节。
        image_size: 图片尺寸。可选值：landscape_16_9 / portrait_16_9 / landscape_4_3 / portrait_4_3 / square_hd。
        num_images: 生成数量，1-6 张，默认 2。
    """
    # ====== Mock 实现：模拟 Seedream 4.5 API 调用 ======
    print("\n🎨 [Seedream 4.5 Mock] 生成参考图中...")
    print(f"   📝 Prompt : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"   📐 Size   : {image_size}  |  🔢 Count: {num_images}")

    mock_urls = [
        f"https://mock-cdn.seedream.ai/img/{uuid.uuid4().hex[:8]}.png"
        for _ in range(num_images)
    ]

    print("   ✅ 生成完成！")
    for i, url in enumerate(mock_urls, 1):
        print(f"      @image_file_{i} → {url}")

    return json.dumps({
        "status": "success",
        "image_urls": mock_urls,
        "message": f"已成功生成 {num_images} 张参考图",
    }, ensure_ascii=False)


@tool
def build_storyboard(
    story_concept: str,
    duration: int,
    ratio: str,
    style: str,
    storyboard_text: str,
    sound_design: str = "",
    reference_notes: str = "",
) -> str:
    """根据收集到的所有参数，生成包含时间轴、运镜和声音设计的结构化分镜脚本，并更新到 State。

    Args:
        story_concept: 一句话故事概念描述。
        duration: 视频时长（秒），范围 4-15。
        ratio: 画幅比例，如 16:9 / 9:16 / 1:1。
        style: 视觉风格描述，如"电影级科幻写实"。
        storyboard_text: 按时间轴排列的分镜内容，格式为多行 "0-X秒：[镜头运动] + [画面内容] + [动作描述]"。
        sound_design: 声音设计说明（配乐 + 音效 + 对白），可选。
        reference_notes: 参考素材说明（@image_file_N 等引用），可选。
    """
    divider = "=" * 60
    storyboard = f"""{divider}
📹  专业分镜脚本  |  Seedance 2.0 Prompt
{divider}

【风格】{style}，{duration}秒，{ratio}
【故事】{story_concept}

【时间轴】
{storyboard_text}

【声音设计】
{sound_design if sound_design else '（待补充）'}

【参考素材】
{reference_notes if reference_notes else '（无）'}

{divider}"""

    print("\n📋 [分镜构建器] 分镜脚本已生成：")
    print(storyboard)
    return storyboard


@tool
def submit_seedance_task(
    prompt: str,
    ratio: str,
    duration: int,
    image_files: list[str] | None = None,
    video_files: list[str] | None = None,
    audio_files: list[str] | None = None,
) -> str:
    """将最终分镜 Prompt 和参考素材提交到 Seedance 2.0 API 生成视频（当前为 Mock 实现）。

    Args:
        prompt: 完整的分镜 Prompt 文本（来自 build_storyboard 的输出）。
        ratio: 画幅比例 (16:9 / 9:16 / 1:1 / 21:9 / 4:3 / 3:4)。
        duration: 视频时长（4-15 秒）。
        image_files: 参考图片 URL 列表（可选，最多 9 张）。
        video_files: 参考视频 URL 列表（可选，最多 3 个）。
        audio_files: 参考音频 URL 列表（可选，最多 3 个）。
    """
    image_files = image_files or []
    video_files = video_files or []
    audio_files = audio_files or []

    # ====== Mock 实现：模拟 Seedance 2.0 API 调用 ======
    task_id = f"seedance-{uuid.uuid4().hex[:12]}"

    print("\n🎬 [Seedance 2.0 Mock] 提交视频生成任务...")
    print(f"   🆔 Task ID  : {task_id}")
    print(f"   📐 画幅     : {ratio}")
    print(f"   ⏱️  时长     : {duration}秒")
    print(f"   🖼️  图片素材 : {len(image_files)} 张")
    print(f"   🎥 视频素材 : {len(video_files)} 个")
    print(f"   🔊 音频素材 : {len(audio_files)} 个")
    print("   ✅ 任务已成功提交！预计 3-5 分钟生成完成。")

    return json.dumps({
        "status": "submitted",
        "task_id": task_id,
        "estimated_time": "3-5 minutes",
        "message": f"视频生成任务已成功提交，Task ID: {task_id}",
    }, ensure_ascii=False)


# ============================================================
# 3. 工具注册 & Gemini 客户端延迟初始化
# ============================================================

TOOLS = [ask_user, generate_reference_images, build_storyboard, submit_seedance_task]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# 延迟初始化：避免导入模块时就校验 API Key，仅在实际运行时创建客户端
_genai_client = None

# 默认使用 gemini-3-flash-preview 模型
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-flash-preview")


def get_client() -> genai.Client:
    """获取 genai Client 实例（首次调用时初始化，采用图片中的 Vertex AI 模式）。"""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(
            vertexai=True,
            api_key=os.getenv("GOOGLE_API_KEY", ""),
            http_options=types.HttpOptions(api_version='v1'),
        )
    return _genai_client


def _build_genai_tool_declarations() -> list[types.Tool]:
    """从 LangChain @tool 定义自动构建 genai FunctionDeclaration 列表。

    会自动处理 pydantic schema → genai Schema 的类型映射，
    包括 anyOf（如 list[str] | None）等复杂类型。
    """
    _type_map = {
        "string": "STRING", "integer": "INTEGER", "number": "NUMBER",
        "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT",
    }
    declarations = []
    for t in TOOLS:
        schema = t.args_schema.model_json_schema()
        properties = {}
        for pname, pinfo in schema.get("properties", {}).items():
            # ── 处理 anyOf 类型（如 list[str] | None → 取非 null 的变体）
            if "anyOf" in pinfo:
                for variant in pinfo["anyOf"]:
                    if variant.get("type") != "null":
                        pinfo = {**pinfo, **variant}
                        break
            json_type = pinfo.get("type", "string")
            genai_type = _type_map.get(json_type, "STRING")
            prop_kwargs: dict = {
                "type": genai_type,
                "description": pinfo.get("description", ""),
            }
            # 数组类型需要声明 items
            if genai_type == "ARRAY" and "items" in pinfo:
                items_type = _type_map.get(
                    pinfo["items"].get("type", "string"), "STRING"
                )
                prop_kwargs["items"] = types.Schema(type=items_type)
            properties[pname] = types.Schema(**prop_kwargs)

        declarations.append(types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters=types.Schema(
                type="OBJECT",
                properties=properties,
                required=schema.get("required", []),
            ),
        ))
    return [types.Tool(function_declarations=declarations)]


def _messages_to_genai(messages: list[BaseMessage]) -> list[types.Content]:
    """将 LangChain 消息列表转换为 genai Content 格式。

    映射规则：
      HumanMessage  → role="user",  Part(text=...)
      AIMessage     → role="model", Part(text=...) / Part(function_call=...)
      ToolMessage   → Part(function_response=...)
      SystemMessage → 跳过（通过 config.system_instruction 传递）
    """
    contents: list[types.Content] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=str(msg.content))],
            ))
        elif isinstance(msg, AIMessage):
            # ── 优先使用原始 genai Content（保留 thought_signature 等元数据）──
            original = msg.additional_kwargs.get("_genai_content")
            if original is not None:
                contents.append(original)
            else:
                # 如果没有原始 Content（如手工构造的 AIMessage），则从字段重构
                parts: list[types.Part] = []
                if msg.content:
                    parts.append(types.Part.from_text(text=str(msg.content)))
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(types.Part.from_function_call(
                            name=tc["name"], args=tc["args"],
                        ))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
        elif isinstance(msg, ToolMessage):
            # 工具返回结果 → genai function_response
            fn_name = getattr(msg, "name", None) or "unknown"
            contents.append(types.Content(
                parts=[types.Part.from_function_response(
                    name=fn_name,
                    response={"result": str(msg.content)},
                )],
            ))
    return contents


def _genai_response_to_message(response) -> AIMessage:
    """将 genai GenerateContentResponse 转换为 LangChain AIMessage。

    解析响应中的文本片段和函数调用，统一封装为 AIMessage，
    保持与 LangGraph tool_calls 格式兼容。

    关键：通过 additional_kwargs["_genai_content"] 保存原始 Content 对象，
    以便下次调用时原样回传给 Gemini（保留 thought_signature 等必要元数据）。
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    original_content = None  # 保存完整的原始 Content 对象

    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content:
            # ── 保存原始 Content，下次回传时原样使用 ──
            original_content = candidate.content
            if candidate.content.parts:
                for part in candidate.content.parts:
                    # 跳过思考部分（thought=True），不展示给用户
                    if getattr(part, "thought", False):
                        continue
                    # 文本片段
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                    # 函数调用 → 转为 LangChain tool_calls 格式
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        tool_calls.append({
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                            "id": uuid.uuid4().hex[:8],
                            "type": "tool_call",
                        })

    return AIMessage(
        content="\n".join(text_parts) if text_parts else "",
        tool_calls=tool_calls,
        additional_kwargs={"_genai_content": original_content},
    )


# ============================================================
# 4. 系统提示词 (System Prompt)
#    明确告知 LLM 身份、工作步骤和分镜模板
# ============================================================

SYSTEM_PROMPT = """\
你是一位专业的全栈视频制作人兼 AI 导演，专注于游戏营销与 UGC 短视频的自动化生成。
你擅长将用户的创意想法转化为专业的视频分镜脚本，并对接 AI 视频生成服务。

## 你的工作流程（必须严格按顺序执行）

### 步骤一：基础信息收集
在开始创作前，你必须确认以下 4 项关键信息（缺一不可）：
1. **故事概念** — 一句话描述视频内容
2. **时长** — 4 到 15 秒之间
3. **画幅比例** — 可选：16:9（横屏） / 9:16（竖屏） / 1:1（方形） / 21:9（超宽屏） / 4:3 / 3:4
4. **视觉风格** — 如：写实 / 动画 / 水墨 / 科幻 / 赛博朋克 / 电影级 等

⚠️ 如果用户没有明确提供上述任何一项，你**必须**调用 `ask_user` 工具逐项询问。
⚠️ 在所有 4 项信息收集完毕之前，**绝对不能**进入后续步骤或调用其他工具。

### 步骤二：角色与素材确认
基础信息收齐后，你**必须**进一步追问以下内容：
1. **角色人物** — 视频中出现哪些角色？每个角色的外貌、服饰、特征描述
2. **图片素材** — 用户是否已有角色参考图或场景素材？如果有，请用户提供；如果没有，后续步骤会为其生成

⚠️ 必须明确知道角色信息后才能进入下一步。

### 步骤三：深度挖掘（可选）
在角色信息明确后，可以通过 `ask_user` 进一步了解以下维度，让分镜更专业：
1. **场景细节** — 具体的场景氛围、环境元素
2. **视觉** — 光影、色调、质感、情绪氛围
3. **运镜** — 推/拉/摇/移/跟/环绕/升降
4. **动作** — 主体的动作节奏和细节
5. **声音** — 配乐风格、音效、对白

如果用户已经提供了足够细节，你可以跳过追问，直接为用户补充专业建议。

### 步骤四：构建分镜脚本（先给用户审阅）
调用 `build_storyboard` 工具，基于已收集的所有信息撰写专业分镜脚本。
分镜必须包含：
- 完整的时间轴（0-X秒：[镜头运动] + [画面内容] + [动作描述]）
- 声音设计（配乐 + 音效 + 对白）

⚠️ 构建完分镜后，你**必须**调用 `ask_user` 将分镜脚本展示给用户，让用户审阅并确认。
⚠️ 用户可能会提出修改意见，你需要根据反馈调整分镜，直到用户满意为止。
⚠️ **修改分镜时，必须输出修改后的完整分镜脚本（所有时间段），严禁省略未修改的部分。** 用户需要看到完整的上下文才能做出准确判断。
⚠️ 在用户确认分镜脚本之前，**绝对不能**生成参考图或提交视频任务。

### 步骤五：生成首帧参考图
用户确认分镜脚本后，如果用户没有现成素材，调用 `generate_reference_images` 工具为分镜中的角色和场景生成首帧参考图。
- 提示词使用英文效果更佳
- 图片尺寸需与视频画幅匹配：
  16:9 → landscape_16_9 | 9:16 → portrait_16_9 | 1:1 → square_hd
  4:3 → landscape_4_3   | 3:4 → portrait_4_3   | 21:9 → landscape_16_9
- 生成后将图片展示给用户确认，并在分镜中标注引用（使用 @image_file_N 格式）

### 步骤六：提交视频任务
分镜和首帧图都确认无误后，调用 `submit_seedance_task` 工具将分镜脚本和参考素材提交给 Seedance 2.0。

## 分镜模板参考

根据创作场景选择合适的模板：

**叙事故事类**（情感叙事、微电影）：
0-3秒：[镜头运动]，[场景建立]，[主体引入]
3-7秒：[镜头运动]，[情节发展]，[动作描述]
7-11秒：[镜头运动]，[高潮/冲突]，[情绪爆发]
11-13秒：[镜头运动]，[转折/过渡]
13-15秒：[镜头运动]，[结尾/落版]

**产品展示类**（品牌广告、电商视频）：
0-2秒：开场抓眼球，产品特写或悬念设置
2-5秒：产品全景展示，运镜环绕/推拉
5-8秒：产品细节特写，材质/工艺展示
8-12秒：使用场景展示
12-15秒：品牌落版，slogan展示

**角色动作类**（武侠/科幻/舞蹈动作）：
0-3秒：角色亮相，定格或缓慢展示造型
3-6秒：动作起始，准备姿势
6-11秒：核心动作展示
11-13秒：动作收尾，pose定格
13-15秒：特效/氛围强化，画面落版

**风景旅拍类**（风景纪录、旅拍 Vlog）：
0-3秒：大景别建立镜头，展示环境全貌
3-6秒：中景推进，引入人物或细节
6-10秒：多角度切换
10-13秒：特写细节，光影变化
13-15秒：回到大景别或意境落版

## 镜头运动词汇
推镜头(push in)、拉镜头(pull out)、摇镜头(pan)、移镜头(dolly)、跟镜头(follow)、\
环绕镜头(orbit)、升降镜头(crane)、希区柯克变焦(dolly zoom)、手持晃动(handheld)、一镜到底(one shot)

## 多模态引用语法
- 图片：@image_file_1, @image_file_2, ...（对应 image_files 数组顺序）
- 视频：@video_file_1, ...
- 音频：@audio_file_1, ...

## 重要规则
1. 始终使用中文与用户交流，保持友好专业的 AI 导演视角
2. 信息不完整时必须先调用 ask_user，绝不能假设或跳过
3. 调用 ask_user 时不要同时调用其他工具
4. 每个步骤完成后简要告知用户进度
5. 分镜脚本必须专业、详细，包含明确的镜头语言
6. **严格遵守流程顺序**：信息收集 → 角色确认 → 分镜脚本 → 用户确认 → 首帧图 → 提交
7. 分镜脚本必须先让用户审阅确认，**未经用户确认不得进入后续环节**
8. **用户要求修改分镜时，回复中必须包含修改后的完整分镜脚本（所有时间段全部列出），严禁用"保持原有…"等方式省略未修改的部分**
"""


# ============================================================
# 5. 图节点定义
#    - agent_node: 调用 LLM 做决策（输出文本回复或工具调用）
#    - tool_node:  执行工具并同步更新 State 业务字段
# ============================================================

def agent_node(state: VideoAgentState) -> dict:
    """
    Agent 节点：调用 Gemini 模型进行决策。
    将完整历史消息转换为 genai Content 格式后发送，系统提示词通过 system_instruction 传递。
    Gemini 会返回两种结果之一：
      1. 纯文本回复 → 直接展示给用户
      2. function_call → 转换为 AIMessage.tool_calls，交由 tool_node 执行
    """
    # 将 LangGraph 消息转换为 genai Content 格式
    contents = _messages_to_genai(state["messages"])

    # 调用 Gemini 模型（系统提示词通过 system_instruction 注入，不放在 contents 中）
    client = get_client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=_build_genai_tool_declarations(),
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
        ),
    )

    # 将 genai 响应转换回 LangChain AIMessage（兼容 LangGraph 的 tool_calls 格式）
    ai_message = _genai_response_to_message(response)
    return {"messages": [ai_message]}


def tool_node(state: VideoAgentState) -> dict:
    """
    工具执行节点：
      1. 从 LLM 最新回复中提取 tool_calls 列表
      2. 逐个执行工具，收集 ToolMessage 结果
      3. 根据工具类型，将关键数据同步更新到 State 的业务字段中
         ── build_storyboard → 更新 storyboard_content 及核心参数
         ── generate_reference_images → 追加 reference_images
         ── submit_seedance_task → 更新 video_task_id
         ── ask_user → 仅返回用户回答，不更新业务字段
    """
    last_message = state["messages"][-1]
    tool_results: list[ToolMessage] = []
    state_updates: dict = {}

    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_fn = TOOLS_BY_NAME.get(tool_name)

        if tool_fn is None:
            # 工具不存在时返回错误信息，避免图流转中断
            tool_results.append(ToolMessage(
                content=f"错误：未知工具 '{tool_name}'",
                tool_call_id=tc["id"],
                name=tool_name,
            ))
            continue

        # 执行工具调用
        result = tool_fn.invoke(tool_args)

        # ────────── 根据工具类型同步更新 State 业务字段 ──────────
        if tool_name == "build_storyboard":
            # 将完整分镜脚本保存到 state
            state_updates["storyboard_content"] = result
            # 同时同步核心参数到 state，便于后续查看进度
            if tool_args.get("story_concept"):
                state_updates["story_concept"] = tool_args["story_concept"]
            if tool_args.get("duration"):
                state_updates["duration"] = tool_args["duration"]
            if tool_args.get("ratio"):
                state_updates["ratio"] = tool_args["ratio"]
            if tool_args.get("style"):
                state_updates["style"] = tool_args["style"]

        elif tool_name == "generate_reference_images":
            # 解析 Mock 返回的图片 URL 列表，追加到已有参考图中
            try:
                data = json.loads(result)
                new_urls = data.get("image_urls", [])
                existing = list(state.get("reference_images", []) or [])
                state_updates["reference_images"] = existing + new_urls
            except (json.JSONDecodeError, TypeError):
                pass

        elif tool_name == "submit_seedance_task":
            # 保存视频生成任务 ID
            try:
                data = json.loads(result)
                state_updates["video_task_id"] = data.get("task_id", "")
            except (json.JSONDecodeError, TypeError):
                pass

        # ask_user 的返回值直接作为 ToolMessage 传回 LLM，不需要额外更新业务字段

        tool_results.append(ToolMessage(
            content=str(result),
            tool_call_id=tc["id"],
            name=tool_name,
        ))

    return {"messages": tool_results, **state_updates}


# ============================================================
# 6. 路由函数
#    决定 LLM 是继续调用工具，还是已经给出最终回复
# ============================================================

def should_continue(state: VideoAgentState) -> str:
    """
    条件路由：检查 LLM 最新回复中是否包含 tool_calls。
    ┌─ 有 tool_calls → 路由到 "tools" 节点，执行工具
    └─ 无 tool_calls → 路由到 END，本轮对话结束，挂起等待用户下一条输入
    """
    last_message = state["messages"][-1]
    # AIMessage 有 tool_calls 属性时，说明 LLM 想调用工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # 否则 LLM 直接给出了文本回复，本轮结束
    return END


# ============================================================
# 7. 构建 StateGraph 并编译
# ============================================================

def build_graph():
    """
    构建 LangGraph StateGraph，流转逻辑如下：

    ┌─────────┐     有 tool_calls     ┌─────────┐
    │  agent  │ ─────────────────────→ │  tools  │
    │  (LLM)  │                        │  (执行)  │
    └────┬────┘                        └────┬────┘
         │ 无 tool_calls                     │
         ↓                                   │
       [END]                                 │
         ↑                                   │
         └───────────────────────────────────┘
                   工具执行完毕后回到 agent
    """
    # 创建 StateGraph，指定状态类型
    graph = StateGraph(VideoAgentState)

    # ── 添加节点 ──
    graph.add_node("agent", agent_node)    # LLM 决策节点
    graph.add_node("tools", tool_node)     # 工具执行节点

    # ── 设置入口点 ──
    graph.set_entry_point("agent")

    # ── 添加条件边：agent 根据是否有 tool_calls 决定下一步 ──
    graph.add_conditional_edges(
        "agent",            # 源节点
        should_continue,    # 路由函数
        {
            "tools": "tools",  # 有工具调用 → 执行工具
            END: END,          # 无工具调用 → 结束本轮
        },
    )

    # ── 工具执行完毕后回到 agent，让 LLM 处理工具返回结果 ──
    graph.add_edge("tools", "agent")

    # ── 使用 MemorySaver 作为 Checkpointer，持久化多轮对话状态 ──
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ============================================================
# 8. CLI 交互主循环
# ============================================================

def print_state(state: dict):
    """打印当前工作流状态摘要。"""
    print("\n" + "─" * 50)
    print("📊 当前工作流状态")
    print("─" * 50)
    concept = state.get("story_concept")
    print(f"  📖 故事概念 : {concept if concept else '❌ 未设定'}")
    dur = state.get("duration")
    print(f"  ⏱️  时长     : {f'{dur}秒' if dur else '❌ 未设定'}")
    ratio = state.get("ratio")
    print(f"  📐 画幅比例 : {ratio if ratio else '❌ 未设定'}")
    style = state.get("style")
    print(f"  🎨 视觉风格 : {style if style else '❌ 未设定'}")
    imgs = state.get("reference_images") or []
    print(f"  🖼️  参考图片 : {len(imgs)} 张")
    sb = state.get("storyboard_content")
    print(f"  📋 分镜脚本 : {'✅ 已生成' if sb else '❌ 未生成'}")
    tid = state.get("video_task_id")
    print(f"  🎬 任务 ID  : {tid if tid else '❌ 未提交'}")
    print("─" * 50 + "\n")


def main():
    """CLI 交互主循环。"""
    app = build_graph()

    # 每次运行创建新的会话线程，MemorySaver 会自动管理该线程的对话历史
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print()
    print("=" * 58)
    print("  🎬  游戏营销 & UGC 视频分镜生成 Agent  🎬")
    print("=" * 58)
    print()
    print("  输入你的视频创意，我来帮你生成专业的分镜脚本！")
    print("  ─────────────────────────────────────────")
    print("  💡 命令：'quit' 退出 | 'state' 查看进度")
    print()

    while True:
        try:
            user_input = input("👤 用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n再见！期待你的下一个创意！👋🎬")
            break

        if user_input.lower() == "state":
            # 从 Checkpointer 读取当前状态快照并展示
            snapshot = app.get_state(config)
            print_state(snapshot.values)
            continue

        # 将用户消息发送给 Agent
        # invoke 会自动走完整个图流转（agent → tools → agent → ... → END）
        # 当 ask_user 工具被调用时，会在 tool_node 内阻塞等待 input()
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        # 从返回的消息列表中找到最后一条 AI 文本回复并展示
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\n🤖 Agent: {msg.content}\n")
                break


if __name__ == "__main__":
    main()
