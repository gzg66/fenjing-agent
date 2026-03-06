"""测试 Google AI Studio API Key 是否有效"""
import os
from google import genai
from google.genai import types

api_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
print(f"API Key: {api_key[:8]}...{api_key[-4:]}  (length={len(api_key)})")

client = genai.Client(
    vertexai=True,
    api_key=api_key,
    http_options=types.HttpOptions(api_version='v1'),
)

# ── 测试1: 列出可用模型 ──
print("\n[Test 1] Listing models...")
try:
    models = list(client.models.list())
    print(f"  OK - found {len(models)} models")
    for m in models[:5]:
        print(f"    - {m.name}")
    print(f"    ... (showing first 5)")
except Exception as e:
    print(f"  FAILED: {e}")

# ── 测试2: 简单文本生成 ──
print("\n[Test 2] Generate text with gemini-2.0-flash...")
try:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Say hello in Chinese, one sentence only.",
    )
    print(f"  OK - Response: {response.text}")
except Exception as e:
    print(f"  FAILED: {e}")

# ── 测试3: 用项目配置的模型 ──
model_name = os.getenv("MODEL_NAME", "gemini-3-flash-preview")
print(f"\n[Test 3] Generate text with {model_name}...")
try:
    response = client.models.generate_content(
        model=model_name,
        contents="Say hello in Chinese, one sentence only.",
    )
    print(f"  OK - Response: {response.text}")
except Exception as e:
    print(f"  FAILED: {e}")

# ── 测试4: 图片生成模型 ──
image_model = os.getenv("IMAGE_MODEL_NAME", "gemini-3-pro-image-preview")
print(f"\n[Test 4] Image generation with {image_model}...")
try:
    response = client.models.generate_content(
        model=image_model,
        contents="Generate a simple red circle on white background",
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )
    found = False
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                print(f"  OK - Got image: {part.inline_data.mime_type}, {len(part.inline_data.data)} bytes")
                found = True
                break
            if hasattr(part, "text") and part.text:
                print(f"  Text response: {part.text[:100]}")
    if not found:
        print("  WARNING - No image data in response")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n--- Done ---")
