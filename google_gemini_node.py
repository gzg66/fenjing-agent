import io as io_module
from typing_extensions import override

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import ComfyExtension, io
from google import genai
from google.genai import types


class Gemini3ImageNode(io.ComfyNode):
    """
    Gemini 3 图像生成节点

    使用 google-generativeai 库调用 Google Gemini 3 API 生成图像
    重要说明:
    1. google-genai 1.60.0 版本已支持 imageSize 参数!
    2. 参数名使用驼峰命名: aspectRatio, imageSize (不是 image_size)
    3. imageSize 支持的值: "1K", "2K", "4K" (必须大写K)
    4. 必须将图片包含在 contents 参数中: contents=[prompt, image]
    5. 支持同时生成 1、2、4 张图片（仅限 gemini-3-pro-image-preview 模型）
    注意：由于 API 限制，多张图片通过多次 API 调用实现
    """

    @staticmethod
    def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        """将 ComfyUI 的 IMAGE 格式 (torch.Tensor) 转换为 PIL Image"""
        # ComfyUI IMAGE 格式: [batch, height, width, channels] 或 [height, width, channels]
        # 值范围: 0.0 - 1.0
        if image_tensor.dim() == 4:
            # 如果是批次，取第一张
            image_tensor = image_tensor[0]

        # 转换为 numpy 数组
        image_np = image_tensor.cpu().numpy()

        # 确保值范围在 0-255
        if image_np.max() <= 1.0:
            image_np = (image_np * 255.0).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        # 转换为 PIL Image
        return Image.fromarray(image_np)

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
        返回包含节点全部信息的 schema。
        常见类型："Model", "Vae", "Clip", "Conditioning", "Latent", "Image", "Int", "String", "Float", "Combo"。
        输出使用 "io.Model.Output"，输入使用 "io.Model.Input"。
        类型可以是 "Combo" - 表示下拉选项列表。
        """
        # 构建可选图片输入（最多9张）
        optional_inputs = []
        for i in range(1, 10):
            optional_inputs.append(io.Image.Input(f"image_{i}", optional=True))

        return io.Schema(
            node_id="Gemini3ImageNode",
            display_name="Gemini 3 Image (Google API)",
            category="LLM/Google",
            inputs=[
                io.String.Input("api_key", multiline=False),
                io.String.Input("system_prompt", multiline=True, default=""),
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input(
                    "model",
                    options=["gemini-3-pro-image-preview", "gemini-3-flash-preview"],
                    default="gemini-3-pro-image-preview",
                ),
                io.Combo.Input(
                    "aspect_ratio",
                    options=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    default="16:9",
                ),
                io.Combo.Input(
                    "image_size",
                    options=["1K", "2K", "4K"],
                    default="1K",
                ),
                io.Combo.Input(
                    "image_count",
                    options=["1", "2", "4"],
                    default="1",
                ),
                *optional_inputs,  # 添加9个可选图片输入
            ],
            outputs=[
                io.Image.Output("image"),
                io.String.Output("text"),
            ],
        )

    @classmethod
    def check_lazy_status(
        cls,
        api_key,
        system_prompt,
        prompt,
        model,
        aspect_ratio,
        image_size,
        image_count,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
    ):
        """
        返回需要被计算的输入名称列表。

        当存在 lazy 输入尚未计算时会调用该函数。只要返回至少一个
        未计算字段（且仍有更多未计算字段），当请求字段可用时此函数会再次被调用。

        已计算的输入会作为参数传入；未计算的输入值为 None。
        """
        # 检查必需输入是否都已提供
        if not api_key or not prompt or not model:
            return ["api_key", "prompt", "model", "system_prompt", "aspect_ratio", "image_size", "image_count"]

        # 图片输入都是可选的，只要必需输入都已提供就可以执行
        return []

    @classmethod
    def execute(
        cls,
        api_key: str,
        system_prompt: str,
        prompt: str,
        model: str,
        aspect_ratio: str,
        image_size: str,
        image_count: str,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
    ) -> io.NodeOutput:
        if not api_key or not api_key.strip():
            raise ValueError("api_key 不能为空")
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt 不能为空")
        
        # 解析生成图片数量
        try:
            num_images = int(image_count)
            if num_images not in [1, 2, 4]:
                raise ValueError("image_count 必须是 1、2 或 4")
        except (ValueError, TypeError):
            raise ValueError("image_count 必须是 1、2 或 4")

        # 收集所有非空的图片输入
        input_images = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9]
        input_images = [img for img in input_images if img is not None]  # 自动过滤掉空值

        # 验证图片数量（小于10张）
        if len(input_images) >= 10:
            raise ValueError("图片数量不能超过9张")

        try:
            # 初始化客户端
            client = genai.Client(
                vertexai=True,
                api_key=api_key,
                http_options=types.HttpOptions(api_version="v1"),
            )

            # 构建 contents：必须包含提示词，图片是可选的
            contents = [prompt]

            # 将输入图片转换为 PIL Image 并添加到 contents（如果有图片输入）
            for img_tensor in input_images:
                if img_tensor is not None:
                    pil_image = cls._tensor_to_pil(img_tensor)
                    contents.append(pil_image)

            # 根据模型类型设置不同的输出模式
            # gemini-3-pro-image-preview 只支持图像输出
            # gemini-3-flash-preview 只支持文本输出
            is_image_model = "image" in model.lower()
            
            # 构建配置参数
            config_kwargs = {}
            
            if is_image_model:
                # 图像生成模型：只输出图像
                response_modalities = ["IMAGE"]
                config_kwargs = {
                    "response_modalities": response_modalities,
                    "image_config": types.ImageConfig(
                        aspectRatio=aspect_ratio,  # 宽高比
                        imageSize=image_size,  # ✅ 分辨率参数(1.60.0版本支持)
                    ),
                }
            else:
                # 文本模型：只输出文本
                response_modalities = ["TEXT"]
                config_kwargs = {
                    "response_modalities": response_modalities,
                }
            
            # 如果提供了系统提示词，添加到配置中
            if system_prompt and system_prompt.strip():
                config_kwargs["system_instruction"] = system_prompt.strip()
            
            config = types.GenerateContentConfig(**config_kwargs)

            # 从响应中提取图像数据和文本数据
            image_tensors = []  # 存储多张图片
            text_content = ""
            
            # 注意：Google API 对图像响应只支持一个候选结果
            # 如果需要生成多张图片，需要多次调用 API
            # 对于图像模型，根据 image_count 决定调用次数
            # 对于文本模型，只调用一次
            call_count = num_images if is_image_model else 1
            
            for i in range(call_count):
                # 调用生成内容 API
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                # 处理响应结果
                if not response.candidates:
                    if is_image_model:
                        raise RuntimeError(f"API 返回了空响应，没有候选结果（第 {i+1}/{num_images} 次调用）")
                    else:
                        raise RuntimeError("API 返回了空响应，没有候选结果")

                # 处理每个候选结果
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        for part in candidate.content.parts:
                            # 检查是否有文本数据
                            if hasattr(part, "text") and part.text:
                                text_content += part.text
                            # 检查是否有图像数据
                            image_bytes = None
                            if hasattr(part, "inline_data") and part.inline_data:
                                image_bytes = part.inline_data.data
                            # 兼容其他可能的图像数据格式
                            elif hasattr(part, "image") and part.image:
                                if hasattr(part.image, "data"):
                                    image_bytes = part.image.data
                            
                            # 处理图像数据
                            if image_bytes is not None:
                                # 将图像字节转换为 PIL Image
                                generated_image = Image.open(io_module.BytesIO(image_bytes)).convert("RGB")
                                # 转换为 ComfyUI 的 IMAGE 格式 (torch.Tensor)
                                image_np = np.array(generated_image).astype(np.float32) / 255.0
                                image_tensor = torch.from_numpy(image_np)
                                image_tensors.append(image_tensor)
                                break  # 每个候选结果只取第一张图片

            # 处理图像输出
            if image_tensors:
                # 将多张图片合并为批次格式 [batch, height, width, channels]
                image_tensor = torch.stack(image_tensors, dim=0)
            else:
                # 如果没有图像，创建一个空的占位图像
                # 创建一个 1x1 的黑色图像作为占位符
                image_np = np.zeros((1, 1, 3), dtype=np.float32)
                image_tensor = torch.from_numpy(image_np)[None,]

            # 如果没有文本内容，使用空字符串
            if not text_content:
                text_content = ""

            return io.NodeOutput(image_tensor, text_content)

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "NOT_FOUND" in error_msg:
                raise RuntimeError(
                    f"模型未找到: {model}。"
                    "请确保模型名称正确，例如: gemini-3-pro-image-preview"
                ) from e
            elif "401" in error_msg or "403" in error_msg or "UNAUTHENTICATED" in error_msg:
                raise RuntimeError("API key 无效或无权访问。请检查 API key 是否正确") from e
            else:
                raise RuntimeError(f"Google API 调用失败: {error_msg}") from e


class Gemini3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Gemini3ImageNode,
        ]


async def comfy_entrypoint() -> Gemini3Extension:  # ComfyUI 会调用该方法来加载扩展及其节点
    return Gemini3Extension()
