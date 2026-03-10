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

import base64
import json
import mimetypes
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

import requests

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
#    - generate_reference_images: 调用 Gemini 图片生成模型生成参考图
#    - build_storyboard: 构建结构化分镜脚本
#    - submit_seedance_task: 调用 RunningHub API 图生视频
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
def generate_reference_images(
    prompt: str,
    image_size: str,
    num_images: int = 2,
    reference_image_urls: list[str] | None = None,
) -> str:
    """调用 Gemini 图片生成模型生成参考图片。

    Args:
        prompt: 英文图片描述，应包含角色外貌、服装、姿势、表情、环境等细节。
        image_size: 图片尺寸。可选值：landscape_16_9 / portrait_16_9 / landscape_4_3 / portrait_4_3 / square_hd。
        num_images: 生成数量，1-6 张，默认 2。
        reference_image_urls: 用户上传的参考素材图片 URL 列表（如 /uploads/xxx.png），生成时将作为角色 / 风格参考输入给模型。
    """
    # ── 画幅提示映射 ──
    size_hints = {
        "landscape_16_9": "16:9 landscape wide format",
        "portrait_16_9": "9:16 portrait tall format",
        "landscape_4_3": "4:3 landscape format",
        "portrait_4_3": "3:4 portrait format",
        "square_hd": "1:1 square format",
    }
    aspect_hint = size_hints.get(image_size, "16:9 landscape wide format")

    print(f"\n[Image Gen] Model: {IMAGE_MODEL_NAME}")
    print(f"[Image Gen] Prompt: {prompt[:100]}...")
    print(f"[Image Gen] Size: {image_size} ({aspect_hint}) | Count: {num_images}")

    # ── 读取用户上传的参考素材作为多模态输入 ──
    _MIME_MAP = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    }
    ref_parts: list[types.Part] = []
    if reference_image_urls:
        for url in reference_image_urls:
            filename = url.split("/")[-1]
            filepath = UPLOAD_DIR / filename
            if filepath.exists():
                image_data = filepath.read_bytes()
                mime = _MIME_MAP.get(filepath.suffix.lower(), "image/png")
                ref_parts.append(types.Part.from_bytes(data=image_data, mime_type=mime))
                print(f"[Image Gen] Loaded reference: {filename} ({len(image_data)} bytes)")
            else:
                print(f"[Image Gen] Warning: reference file not found: {filepath}")
    if ref_parts:
        print(f"[Image Gen] Using {len(ref_parts)} reference image(s) as input")

    client = get_image_client()
    image_urls: list[str] = []

    for i in range(num_images):
        try:
            full_prompt = (
                f"Generate a high-quality cinematic reference image in {aspect_hint}. "
                f"{prompt}"
            )

            # ── 构建多模态 contents：参考图 + 文字提示 ──
            contents: list[types.Part] = []
            if ref_parts:
                contents.extend(ref_parts)
                contents.append(types.Part.from_text(
                    text=(
                        f"Using the above images as character and style reference, "
                        f"generate a new image that keeps the same characters' appearance: "
                        f"{full_prompt}"
                    )
                ))
            else:
                contents.append(types.Part.from_text(text=full_prompt))

            response = client.models.generate_content(
                model=IMAGE_MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )

            # ── 从响应中提取图片数据 ──
            found_image = False
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data is not None:
                        # 根据 MIME 类型确定文件扩展名
                        mime_type = getattr(part.inline_data, "mime_type", "") or "image/png"
                        ext = ".png"
                        if "jpeg" in mime_type or "jpg" in mime_type:
                            ext = ".jpg"
                        elif "webp" in mime_type:
                            ext = ".webp"

                        # 保存图片到 uploads 目录
                        filename = f"gen_{uuid.uuid4().hex[:12]}{ext}"
                        filepath = UPLOAD_DIR / filename
                        with open(filepath, "wb") as f:
                            f.write(part.inline_data.data)

                        url = f"/uploads/{filename}"
                        image_urls.append(url)
                        print(f"[Image Gen] Saved image {i + 1}: {filename} ({len(part.inline_data.data)} bytes)")
                        found_image = True
                        break  # 每次调用只取第一张图

            if not found_image:
                print(f"[Image Gen] Warning: No image data in response for image {i + 1}")

        except Exception as e:
            print(f"[Image Gen] Error generating image {i + 1}: {e}")
            continue

    if not image_urls:
        return json.dumps({
            "status": "error",
            "message": "图片生成失败，请检查 API Key 和模型配置后重试",
        }, ensure_ascii=False)

    print(f"[Image Gen] Done, generated {len(image_urls)}/{num_images} images")
    return json.dumps({
        "status": "success",
        "image_urls": image_urls,
        "message": f"已成功生成 {len(image_urls)} 张参考图",
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
    negative_prompt: str = "",
    image_files: list[str] | None = None,
    video_files: list[str] | None = None,
    audio_files: list[str] | None = None,
) -> str:
    """将最终分镜 Prompt 和首帧参考图提交到 RunningHub 图生视频 API 生成视频。
    工具会自动轮询任务状态，直到视频生成完成或超时。

    Args:
        prompt: 完整的分镜 Prompt 文本（来自 build_storyboard 的输出）。
        ratio: 画幅比例 (16:9 / 9:16 / 1:1 / 21:9 / 4:3 / 3:4)。
        duration: 视频时长（4-15 秒）。
        negative_prompt: 负面提示词，描述不想在视频中出现的内容，可为空。
        image_files: 参考图片 URL 列表（第一张将作为首帧图输入），如 /uploads/gen_xxx.png。
        video_files: 参考视频 URL 列表（可选，当前未使用）。
        audio_files: 参考音频 URL 列表（可选，当前未使用）。
    """
    image_files = image_files or []
    video_files = video_files or []
    audio_files = audio_files or []

    # ── 检查 API Key ──
    api_key = RUNNINGHUB_API_KEY or os.getenv("RUNNINGHUB_API_KEY", "")
    if not api_key:
        return json.dumps({
            "status": "error",
            "message": "未配置 RUNNINGHUB_API_KEY 环境变量，无法提交视频生成任务",
        }, ensure_ascii=False)

    # ── 获取首帧图片并转为 Base64 data URI ──
    if not image_files:
        return json.dumps({
            "status": "error",
            "message": "缺少首帧参考图片，请先通过 generate_reference_images 生成首帧图",
        }, ensure_ascii=False)

    first_image_url = image_files[0]
    filename = first_image_url.split("/")[-1]
    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        return json.dumps({
            "status": "error",
            "message": f"首帧图片文件不存在: {filepath}",
        }, ensure_ascii=False)

    print(f"\n[RunningHub] Converting first-frame image to Base64: {filename}")
    image_base64 = _image_to_base64_uri(str(filepath))
    print(f"[RunningHub] Base64 length: {len(image_base64)} chars")

    # ── 构建请求 ──
    submit_url = f"{RUNNINGHUB_API_BASE}/alibaba/wan-2.6/image-to-video"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    resolution = "720p"
    payload = {
        "imageUrl": image_base64,
        "prompt": prompt,
        "negativePrompt": negative_prompt,
        "resolution": resolution,
        "duration": str(duration),
        "shotType": "multi",
    }

    print(f"[RunningHub] Submitting video generation task...")
    print(f"[RunningHub] Ratio: {ratio} | Duration: {duration}s | Resolution: {resolution}")
    print(f"[RunningHub] Image files: {len(image_files)} | First frame: {filename}")

    # ── 提交任务 ──
    try:
        response = requests.post(submit_url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return json.dumps({
                "status": "error",
                "message": f"提交失败: HTTP {response.status_code}, {response.text[:500]}",
            }, ensure_ascii=False)

        result = response.json()
        task_id = result.get("taskId")
        if not task_id:
            return json.dumps({
                "status": "error",
                "message": f"提交失败: 未获取到 taskId, 响应: {json.dumps(result, ensure_ascii=False)[:500]}",
            }, ensure_ascii=False)

        print(f"[RunningHub] Task submitted successfully. ID: {task_id}")
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"提交请求异常: {e}",
        }, ensure_ascii=False)

    # ── 轮询任务状态，直到完成或超时 ──
    query_url = f"{RUNNINGHUB_API_BASE}/query"
    max_wait = 600      # 最长等待 10 分钟
    poll_interval = 5   # 每 5 秒查询一次
    elapsed = 0
    begin = time.time()

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed = int(time.time() - begin)

        try:
            resp = requests.post(
                query_url, headers=headers,
                json={"taskId": task_id}, timeout=15,
            )
            if resp.status_code != 200:
                print(f"[RunningHub] Query error: HTTP {resp.status_code}")
                continue

            data = resp.json()
            status = data.get("status", "UNKNOWN")

            if status == "SUCCESS":
                video_url = ""
                if data.get("results") and len(data["results"]) > 0:
                    video_url = data["results"][0].get("url", "")

                print(f"[RunningHub] Task completed in ~{elapsed}s. Video URL: {video_url}")

                # ── 下载视频到本地 uploads 目录 ──
                local_video_path = ""
                if video_url:
                    try:
                        video_filename = f"video_{task_id}.mp4"
                        video_filepath = UPLOAD_DIR / video_filename
                        print(f"[RunningHub] Downloading video to {video_filename}...")
                        video_resp = requests.get(video_url, timeout=120)
                        if video_resp.status_code == 200:
                            with open(video_filepath, "wb") as f:
                                f.write(video_resp.content)
                            local_video_path = f"/uploads/{video_filename}"
                            print(f"[RunningHub] Video saved: {video_filename} ({len(video_resp.content)} bytes)")
                        else:
                            print(f"[RunningHub] Download failed: HTTP {video_resp.status_code}")
                    except Exception as dl_err:
                        print(f"[RunningHub] Warning: Failed to download video: {dl_err}")

                return json.dumps({
                    "status": "success",
                    "task_id": task_id,
                    "video_url": video_url,
                    "local_video_path": local_video_path,
                    "elapsed_seconds": elapsed,
                    "message": f"视频生成完成! 耗时约 {elapsed} 秒。",
                }, ensure_ascii=False)

            elif status in ("RUNNING", "QUEUED"):
                print(f"[RunningHub] Status: {status} (waited {elapsed}s / {max_wait}s)")
                continue

            else:
                error_msg = data.get("errorMessage", "未知错误")
                print(f"[RunningHub] Task failed: {error_msg}")
                return json.dumps({
                    "status": "error",
                    "task_id": task_id,
                    "message": f"视频生成失败: {error_msg}",
                }, ensure_ascii=False)

        except Exception as e:
            print(f"[RunningHub] Query exception: {e}")
            continue

    # ── 超时 ──
    return json.dumps({
        "status": "timeout",
        "task_id": task_id,
        "message": f"视频生成超时 (等待超过 {max_wait} 秒)，任务 ID: {task_id}，可稍后手动查询。",
    }, ensure_ascii=False)


@tool
def submit_kling_video_task(
    prompt: str,
    negative_prompt: str = "",
    image_files: list[str] | None = None,
    duration: str = "15",
    cfg_scale: float = 0.5,
    sound: bool = True,
) -> str:
    """调用 RunningHub Kling V3.0 Pro 图生视频 API，根据首帧图片和 Prompt 生成视频。
    工具会自动轮询任务状态，直到视频生成完成或超时。

    Args:
        prompt: 视频生成的提示词，描述视频内容、场景、动作、对白等。
        negative_prompt: 负面提示词，描述不想在视频中出现的内容，可为空。
        image_files: 参考图片 URL 列表（第一张将作为首帧图输入），如 /uploads/gen_xxx.png。
        duration: 视频时长（秒），可选 "5" / "10" / "15"，默认 "15"。
        cfg_scale: CFG Scale 参数，控制生成一致性，范围 0-1，默认 0.5。
        sound: 是否生成声音，默认为 True。
    """
    image_files = image_files or []

    # ── 检查 API Key ──
    api_key = RUNNINGHUB_API_KEY or os.getenv("RUNNINGHUB_API_KEY", "")
    if not api_key:
        return json.dumps({
            "status": "error",
            "message": "未配置 RUNNINGHUB_API_KEY 环境变量，无法提交视频生成任务",
        }, ensure_ascii=False)

    # ── 获取首帧图片并转为 Base64 data URI ──
    if not image_files:
        return json.dumps({
            "status": "error",
            "message": "缺少首帧参考图片，请先通过 generate_reference_images 生成首帧图",
        }, ensure_ascii=False)

    first_image_url = image_files[0]
    filename = first_image_url.split("/")[-1]
    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        return json.dumps({
            "status": "error",
            "message": f"首帧图片文件不存在: {filepath}",
        }, ensure_ascii=False)

    print(f"\n[Kling V3.0 Pro] Converting first-frame image to Base64: {filename}")
    image_base64 = _image_to_base64_uri(str(filepath))
    print(f"[Kling V3.0 Pro] Base64 length: {len(image_base64)} chars")

    # ── 构建请求 ──
    submit_url = f"{RUNNINGHUB_API_BASE}/kling-v3.0-pro/image-to-video"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "prompt": prompt,
        "negativePrompt": negative_prompt,
        "firstImageUrl": image_base64,
        "lastImageUrl": "",
        "duration": str(duration),
        "cfgScale": cfg_scale,
        "sound": sound,
    }

    print(f"[Kling V3.0 Pro] Submitting video generation task...")
    print(f"[Kling V3.0 Pro] Duration: {duration}s | CFG Scale: {cfg_scale} | Sound: {sound}")
    print(f"[Kling V3.0 Pro] Image files: {len(image_files)} | First frame: {filename}")

    # ── 提交任务 ──
    begin = time.time()
    try:
        response = requests.post(submit_url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return json.dumps({
                "status": "error",
                "message": f"提交失败: HTTP {response.status_code}, {response.text[:500]}",
            }, ensure_ascii=False)

        result = response.json()
        task_id = result.get("taskId")
        if not task_id:
            return json.dumps({
                "status": "error",
                "message": f"提交失败: 未获取到 taskId, 响应: {json.dumps(result, ensure_ascii=False)[:500]}",
            }, ensure_ascii=False)

        print(f"[Kling V3.0 Pro] Task submitted successfully. ID: {task_id}")
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"提交请求异常: {e}",
        }, ensure_ascii=False)

    # ── 轮询任务状态，直到完成或超时 ──
    query_url = f"{RUNNINGHUB_API_BASE}/query"
    max_wait = 600      # 最长等待 10 分钟
    poll_interval = 5   # 每 5 秒查询一次
    elapsed = 0

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed = int(time.time() - begin)

        try:
            resp = requests.post(
                query_url, headers=headers,
                json={"taskId": task_id}, timeout=15,
            )
            if resp.status_code != 200:
                print(f"[Kling V3.0 Pro] Query error: HTTP {resp.status_code}")
                continue

            data = resp.json()
            status = data.get("status", "UNKNOWN")

            if status == "SUCCESS":
                video_url = ""
                if data.get("results") and len(data["results"]) > 0:
                    video_url = data["results"][0].get("url", "")

                print(f"[Kling V3.0 Pro] Task completed in ~{elapsed}s. Video URL: {video_url}")

                # ── 下载视频到本地 uploads 目录 ──
                local_video_path = ""
                if video_url:
                    try:
                        video_filename = f"video_kling_{task_id}.mp4"
                        video_filepath = UPLOAD_DIR / video_filename
                        print(f"[Kling V3.0 Pro] Downloading video to {video_filename}...")
                        video_resp = requests.get(video_url, timeout=120)
                        if video_resp.status_code == 200:
                            with open(video_filepath, "wb") as f:
                                f.write(video_resp.content)
                            local_video_path = f"/uploads/{video_filename}"
                            print(f"[Kling V3.0 Pro] Video saved: {video_filename} ({len(video_resp.content)} bytes)")
                        else:
                            print(f"[Kling V3.0 Pro] Download failed: HTTP {video_resp.status_code}")
                    except Exception as dl_err:
                        print(f"[Kling V3.0 Pro] Warning: Failed to download video: {dl_err}")

                return json.dumps({
                    "status": "success",
                    "task_id": task_id,
                    "video_url": video_url,
                    "local_video_path": local_video_path,
                    "elapsed_seconds": elapsed,
                    "message": f"视频生成完成! 耗时约 {elapsed} 秒。",
                }, ensure_ascii=False)

            elif status in ("RUNNING", "QUEUED"):
                print(f"[Kling V3.0 Pro] Status: {status} (waited {elapsed}s / {max_wait}s)")
                continue

            else:
                error_msg = data.get("errorMessage", "未知错误")
                print(f"[Kling V3.0 Pro] Task failed: {error_msg}")
                return json.dumps({
                    "status": "error",
                    "task_id": task_id,
                    "message": f"视频生成失败: {error_msg}",
                }, ensure_ascii=False)

        except Exception as e:
            print(f"[Kling V3.0 Pro] Query exception: {e}")
            continue

    # ── 超时 ──
    return json.dumps({
        "status": "timeout",
        "task_id": task_id,
        "message": f"视频生成超时 (等待超过 {max_wait} 秒)，任务 ID: {task_id}，可稍后手动查询。",
    }, ensure_ascii=False)


# ============================================================
# 3. 技能系统 (Skills) — 动态发现与调用
#    自动扫描 .cursor/skills 目录，无论未来放入什么 skill
#    只要包含 SKILL.md，即可被 Agent 自动发现和调用。
# ============================================================

SKILLS_DIR = Path(".cursor/skills")


def _parse_frontmatter(content: str) -> dict:
    """解析 SKILL.md 开头的 YAML frontmatter，提取 name / description 等元数据。"""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    result = {}
    for line in content[3:end].strip().split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                result[key] = value
    return result


def load_skills_catalog() -> dict[str, dict]:
    """扫描 SKILLS_DIR 下所有子目录，构建技能索引。

    返回结构：{skill_name: {name, description, dir, files, scripts}}
    - 任何新增 skill 只要放入目录、包含 SKILL.md，下次启动即自动发现
    """
    catalog: dict[str, dict] = {}
    if not SKILLS_DIR.exists():
        return catalog

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        content = skill_md.read_text(encoding="utf-8")
        meta = _parse_frontmatter(content)
        name = meta.get("name", skill_dir.name)

        # 收集 skill 下所有可访问文件（排除缓存和元数据）
        files = []
        for f in sorted(skill_dir.rglob("*")):
            if (f.is_file()
                    and "__pycache__" not in str(f)
                    and not f.suffix == ".pyc"
                    and f.name != "_meta.json"):
                files.append(str(f.relative_to(skill_dir)).replace("\\", "/"))

        # 收集 scripts/ 下的可执行脚本
        scripts_dir = skill_dir / "scripts"
        scripts = []
        if scripts_dir.is_dir():
            scripts = [
                f.name for f in sorted(scripts_dir.iterdir())
                if f.is_file() and not f.name.startswith("__") and f.suffix != ".pyc"
            ]

        catalog[name] = {
            "name": name,
            "description": meta.get("description", ""),
            "dir": str(skill_dir),
            "files": files,
            "scripts": scripts,
        }

    return catalog


# 模块加载时扫描一次；若运行期间新增了 skill，重启即可发现
SKILLS_CATALOG: dict[str, dict] = load_skills_catalog()


def _build_skills_summary() -> str:
    """构建可用技能摘要（嵌入系统提示词，帮助 Agent 决定调用哪个 skill）。"""
    if not SKILLS_CATALOG:
        return "（当前没有可用的技能）"
    lines = []
    for name, info in SKILLS_CATALOG.items():
        desc = info["description"]
        extras = []
        if info["scripts"]:
            extras.append(f"含脚本: {', '.join(info['scripts'])}")
        ref_count = sum(1 for f in info["files"] if f != "SKILL.md")
        if ref_count:
            extras.append(f"{ref_count} 个附属文件")
        suffix = f"（{'; '.join(extras)}）" if extras else ""
        lines.append(f"- **{name}**：{desc} {suffix}")
    return "\n".join(lines)


@tool
def read_skill(skill_name: str, file_path: str | None = None) -> str:
    """读取指定技能的文档内容。
    不指定 file_path 时返回技能主文档（SKILL.md）并列出所有可用文件；
    指定 file_path 时读取技能目录下对应的参考文档、模板、示例等。

    Args:
        skill_name: 技能名称，对应 SKILL.md 中 frontmatter 的 name 字段
        file_path: 可选，技能目录内的文件相对路径（如 "reference.md"、"references/tools-guide.md"）
    """
    skill = SKILLS_CATALOG.get(skill_name)
    if not skill:
        available = ", ".join(SKILLS_CATALOG.keys()) or "无"
        return f"错误：未找到技能 '{skill_name}'。当前可用技能：{available}"

    skill_dir = Path(skill["dir"])

    if file_path:
        target = skill_dir / file_path
        # 安全校验：防止路径穿越
        try:
            target.resolve().relative_to(skill_dir.resolve())
        except ValueError:
            return "错误：不允许访问技能目录之外的文件。"
        if not target.exists() or not target.is_file():
            return (
                f"错误：文件 '{file_path}' 不存在于技能 '{skill_name}' 中。\n"
                f"可用文件：\n" + "\n".join(f"  - {f}" for f in skill["files"])
            )
        return target.read_text(encoding="utf-8")

    # 默认：读取 SKILL.md 并附加文件清单
    content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

    other_files = [f for f in skill["files"] if f != "SKILL.md"]
    if other_files:
        content += "\n\n---\n📁 该技能还包含以下文件（可通过 file_path 参数读取）：\n"
        content += "\n".join(f"  - {f}" for f in other_files)
    if skill["scripts"]:
        content += "\n\n🔧 可执行脚本（可通过 run_skill_script 工具调用）：\n"
        content += "\n".join(f"  - {s}" for s in skill["scripts"])

    return content


@tool
def run_skill_script(skill_name: str, script_name: str, arguments: str = "") -> str:
    """执行指定技能 scripts 目录下的脚本文件，支持传入命令行参数。

    Args:
        skill_name: 技能名称
        script_name: scripts 目录下的脚本文件名（如 "seedance_api.py"）
        arguments: 传给脚本的命令行参数字符串
    """
    skill = SKILLS_CATALOG.get(skill_name)
    if not skill:
        available = ", ".join(SKILLS_CATALOG.keys()) or "无"
        return f"错误：未找到技能 '{skill_name}'。当前可用技能：{available}"

    if script_name not in skill["scripts"]:
        avail = ", ".join(skill["scripts"]) if skill["scripts"] else "无"
        return f"错误：脚本 '{script_name}' 不存在于技能 '{skill_name}' 中。可用脚本：{avail}"

    skill_dir = Path(skill["dir"])
    script_path = skill_dir / "scripts" / script_name

    # 根据文件后缀选择执行方式
    if script_name.endswith(".py"):
        cmd = f'python "{script_path}" {arguments}'
    elif script_name.endswith(".sh"):
        cmd = f'bash "{script_path}" {arguments}'
    else:
        cmd = f'"{script_path}" {arguments}'

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=300, cwd=str(skill_dir),
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code]: {result.returncode}"
        return output or "（脚本执行完毕，无输出）"
    except subprocess.TimeoutExpired:
        return "错误：脚本执行超时（300 秒限制）"
    except Exception as e:
        return f"错误：脚本执行失败 — {e}"


# ============================================================
# 4. 工具注册 & Gemini 客户端延迟初始化
# ============================================================

TOOLS = [
    ask_user, generate_reference_images, build_storyboard,
    submit_seedance_task, submit_kling_video_task,
    read_skill, run_skill_script,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# 延迟初始化：避免导入模块时就校验 API Key，仅在实际运行时创建客户端
_genai_client = None

# 默认使用 gemini-3-flash-preview 模型
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-flash-preview")

# ── 图片生成模型配置 ──
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "gemini-3-pro-image-preview")
_image_genai_client = None

# ── 图片/视频保存目录（与 web_server.py 共享） ──
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── RunningHub 图生视频 API 配置 ──
RUNNINGHUB_API_KEY = os.getenv("RUNNINGHUB_API_KEY", "")
RUNNINGHUB_API_BASE = "https://www.runninghub.cn/openapi/v2"


def _image_to_base64_uri(image_path: str) -> str:
    """将本地图片文件转换为 Base64 data URI 格式（用于 RunningHub API）。"""
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def get_client() -> genai.Client:
    """获取 genai Client 实例（首次调用时初始化，使用标准 Google AI API）。"""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(
            vertexai=True,
            api_key=os.getenv("GOOGLE_API_KEY", ""),
            http_options=types.HttpOptions(api_version='v1')
        )
    return _genai_client


def get_image_client() -> genai.Client:
    """获取图片生成专用 genai Client（使用标准 API，支持 response_modalities=IMAGE）。"""
    global _image_genai_client
    if _image_genai_client is None:
        _image_genai_client = genai.Client(
            vertexai=True,
            api_key=os.getenv("GOOGLE_API_KEY", ""),
            http_options=types.HttpOptions(api_version='v1')
        )
    return _image_genai_client  


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
      ToolMessage   → Part(function_response=...)  ※ 连续的合并为同一 Content
      SystemMessage → 跳过（通过 config.system_instruction 传递）

    重要：Gemini API 要求同一轮所有 function_response 合并在一个 Content 中，
    数量必须等于对应 function_call 的数量，否则报 400 INVALID_ARGUMENT。
    """
    contents: list[types.Content] = []
    # ── 缓冲区：累积连续的 ToolMessage 对应的 function_response Parts ──
    pending_tool_parts: list[types.Part] = []

    def _flush_tool_parts():
        """将累积的 function_response Parts 合并为一个 Content 并清空缓冲。"""
        if pending_tool_parts:
            contents.append(types.Content(parts=list(pending_tool_parts)))
            pending_tool_parts.clear()

    for msg in messages:
        if isinstance(msg, ToolMessage):
            # 累积 function_response，不立即 append
            fn_name = getattr(msg, "name", None) or "unknown"
            pending_tool_parts.append(types.Part.from_function_response(
                name=fn_name,
                response={"result": str(msg.content)},
            ))
        else:
            # 遇到非 ToolMessage 时先刷出缓冲的工具结果
            _flush_tool_parts()

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

    # 消息列表末尾可能还有未刷出的 ToolMessage
    _flush_tool_parts()

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
#    核心身份 + 技能系统说明（详细工作流由 skill 文档提供）
# ============================================================

SYSTEM_PROMPT = f"""\
你是一位专业的全栈视频制作人兼 AI 导演，专注于游戏营销与 UGC 短视频的自动化生成。
你擅长将用户的创意想法转化为专业的视频分镜脚本，并对接 AI 视频生成服务。

## 技能系统 (Skills)

你具备动态加载「技能」的能力。技能是模块化的专业知识包，包含完整的工作流程指导、参考文档和可执行脚本。
通过技能系统，你可以在不重启的情况下获取任何领域的专业指导。

### 可用工具

- `read_skill(skill_name)` — 加载技能主文档（SKILL.md），获取工作流程、参数说明和操作指南
- `read_skill(skill_name, file_path)` — 读取技能附带的参考文档、模板或示例文件
- `run_skill_script(skill_name, script_name, arguments)` — 执行技能自带的脚本

### 当前可用技能

{_build_skills_summary()}

## 工作规则

1. **任务前先加载技能**：收到用户请求后，先判断需要哪些技能，调用 `read_skill` 加载技能文档，然后严格按照文档中的工作流程执行
2. 始终使用中文与用户交流，保持友好专业的 AI 导演视角
3. 信息不完整时必须先调用 `ask_user` 向用户提问，绝不能假设或跳过
4. 调用 `ask_user` 时不要同时调用其他工具
5. 每个步骤完成后简要告知用户进度
6. 如果技能文档中引用了其他文件（如 reference.md），可通过 `read_skill` 的 `file_path` 参数按需读取

## 多模态引用语法

- 图片：@image_file_1, @image_file_2, ...（对应 image_files 数组顺序）
- 视频：@video_file_1, ...
- 音频：@audio_file_1, ...
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

            # ── 方式B：首帧角色一致性兜底验证 ──
            # 在工具返回结果中追加自检提醒，让 LLM 在呈现给用户前先验证首帧是否包含主角
            result += (
                "\n\n[⚠️ 角色一致性自检] 请检查以上分镜脚本的**第一段时间轴**（即首帧画面描述）"
                "是否包含主角/核心角色。"
                "若首帧画面中缺少主角，你必须修改第一段时间轴的描述，"
                "将主角自然融入首帧画面（例如：主角从远处飞来、主角站在画面前景等），"
                "然后重新调用 build_storyboard 生成修正后的完整分镜。"
                "原因：首帧图是视频生成的唯一视觉参考，缺少主角会导致后续画面人物一致性无法保证。"
            )

        elif tool_name == "generate_reference_images":
            # 解析图片生成返回的 URL 列表，追加到已有参考图中
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

        elif tool_name == "submit_kling_video_task":
            # 保存 Kling 视频生成任务 ID
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
