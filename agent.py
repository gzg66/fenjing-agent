"""
主逻辑类：StoryboardAgent
实现脚本分析、资产一致性管理和图像生成功能
"""

import json
import base64
import time
import io
from typing import Optional, Dict, List, Any
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from config import (
    ASSET_DESCRIPTION_PROMPT,
    SCRIPT_ANALYSIS_PROMPT,
    PROMPT_ENGINEERING_PROMPT,
    GEMINI_FLASH_MODEL,
    GEMINI_IMAGE_MODEL
)


class StoryboardAgent:
    """故事板生成代理类"""
    
    def __init__(self, api_key: str):
        """
        初始化代理
        
        Args:
            api_key: Google Generative AI API密钥
        """
        self.api_key = api_key
        # 使用vertexai=True和http_options指定API版本（根据图片示例）
        self.client = genai.Client(
            vertexai=True,
            api_key=api_key,
            http_options=types.HttpOptions(api_version='v1')
        )
        self.character_anchor: Optional[Dict[str, Any]] = None
        self.scene_anchor: Optional[Dict[str, Any]] = None
        self.character_image_path: Optional[str] = None
        self.scene_image_path: Optional[str] = None
        
    def _generate_content_with_retry(
        self,
        model: str,
        contents: Any,
        generation_config: Optional[Dict[str, Any]] = None,
        use_json_mode: bool = False
    ) -> str:
        """
        使用genai SDK发送API请求，带指数退避重试机制
        
        Args:
            model: 模型名称
            contents: 内容（可以是字符串、列表等）
            generation_config: 生成配置字典
            use_json_mode: 是否使用JSON模式（用于脚本分析）
            
        Returns:
            响应文本内容
            
        Raises:
            Exception: 如果所有重试都失败
        """
        # 构建配置对象
        config_dict = generation_config.copy() if generation_config else {}
        if use_json_mode:
            config_dict["response_mime_type"] = "application/json"
        
        config = types.GenerateContentConfig(**config_dict) if config_dict else None
        
        for attempt, delay in enumerate([1, 2, 4, 8, 16]):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
                return response.text
            except Exception as e:
                if attempt == 4:  # 最后一次重试
                    raise Exception(f"API请求失败，已重试5次: {str(e)}")
                print(f"请求失败，{delay}秒后重试 (尝试 {attempt + 1}/5)...")
                time.sleep(delay)
        
        raise Exception("API请求失败")
    
    
    def analyze_asset(self, image_path: str, is_character: bool = True) -> Dict[str, Any]:
        """
        Pass 1: 分析资产图片，提取视觉锚点
        
        Args:
            image_path: 图片文件路径
            is_character: True表示角色图片，False表示场景图片
            
        Returns:
            包含视觉锚点的字典
        """
        print(f"正在分析{'角色' if is_character else '场景'}资产: {image_path}")
        
        # 读取图片
        image = Image.open(image_path)
        
        # 构建内容（文本提示词 + 图片）
        contents = [ASSET_DESCRIPTION_PROMPT, image]
        
        # 生成配置
        generation_config = {
            "temperature": 0.3,
            "top_k": 40,
            "top_p": 0.95,
            "max_output_tokens": 2048
        }
        
        # 调用Gemini Flash模型
        content = self._generate_content_with_retry(
            model=GEMINI_FLASH_MODEL,
            contents=contents,
            generation_config=generation_config
        )
        
        # 解析响应
        # 处理可能的markdown代码块格式
        content_cleaned = content.strip()
        
        # 如果内容被markdown代码块包裹，提取JSON部分
        if content_cleaned.startswith("```"):
            # 查找第一个```后的内容和最后一个```前的内容
            lines = content_cleaned.split("\n")
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    json_lines.append(line)
            content_cleaned = "\n".join(json_lines)
        elif content_cleaned.startswith("```json"):
            # 处理```json代码块
            lines = content_cleaned.split("\n")
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    json_lines.append(line)
            content_cleaned = "\n".join(json_lines)
        
        try:
            anchor_data = json.loads(content_cleaned)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
            print(f"响应内容前500字符: {content[:500]}")
            raise Exception(f"无法解析资产分析结果，API返回的不是有效的JSON格式: {str(e)}")
        
        # 验证数据结构
        if not isinstance(anchor_data, dict):
            raise Exception(f"资产分析结果格式错误，期望字典类型，得到: {type(anchor_data)}")
        
        # 存储锚点
        if is_character:
            self.character_anchor = anchor_data
            print("角色锚点已提取并存储")
            print("  存储位置: 内存 (self.character_anchor)")
            anchor_preview = json.dumps(anchor_data, ensure_ascii=False, indent=2)[:200]
            print(f"  锚点内容预览: {anchor_preview}...")
        else:
            self.scene_anchor = anchor_data
            print("场景锚点已提取并存储")
            print("  存储位置: 内存 (self.scene_anchor)")
            anchor_preview = json.dumps(anchor_data, ensure_ascii=False, indent=2)[:200]
            print(f"  锚点内容预览: {anchor_preview}...")
        
        return anchor_data
    
    def analyze_script(self, script_text: str) -> List[Dict[str, Any]]:
        """
        分析脚本，分解为镜头列表
        
        Args:
            script_text: 剧本文本
            
        Returns:
            镜头对象列表，每个对象包含shotNumber, sceneDescription, characters, location, dialogue, visualStyle, duration
        """
        print("正在分析剧本...")
        
        # 构建内容
        contents = f"{SCRIPT_ANALYSIS_PROMPT}\n\nScript:\n{script_text}"
        
        # 生成配置
        generation_config = {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "max_output_tokens": 4096
        }
        
        # 调用API（使用JSON模式）
        content = self._generate_content_with_retry(
            model=GEMINI_FLASH_MODEL,
            contents=contents,
            generation_config=generation_config,
            use_json_mode=True
        )
        
        # 解析响应
        # 处理可能的markdown代码块格式
        content_cleaned = content.strip()
        
        # 如果内容被markdown代码块包裹，提取JSON部分
        if content_cleaned.startswith("```"):
            # 查找第一个```后的内容和最后一个```前的内容
            lines = content_cleaned.split("\n")
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    json_lines.append(line)
            content_cleaned = "\n".join(json_lines)
        elif content_cleaned.startswith("```json"):
            # 处理```json代码块
            lines = content_cleaned.split("\n")
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    json_lines.append(line)
            content_cleaned = "\n".join(json_lines)
        
        try:
            shots = json.loads(content_cleaned)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
            print(f"响应内容前500字符: {content[:500]}")
            raise Exception(f"无法解析脚本分析结果，API返回的不是有效的JSON格式: {str(e)}")
        
        if not isinstance(shots, list):
            raise Exception("脚本分析结果不是数组格式")
        
        print(f"剧本分析完成，共识别 {len(shots)} 个镜头")
        return shots
    
    def generate_image_prompt(self, shot: Dict[str, Any]) -> str:
        """
        Pass 2: 合成图像生成提示词，注入视觉锚点
        
        Args:
            shot: 镜头对象，包含sceneDescription, characters, location等信息
            
        Returns:
            优化后的英文图像生成提示词
        """
        # 构建上下文信息
        context_parts = []
        
        # 添加角色锚点（如果镜头中有角色且已设置角色锚点）
        if shot.get("characters") and self.character_anchor:
            anchor_info = json.dumps(self.character_anchor, ensure_ascii=False, indent=2)
            context_parts.append(f"**CHARACTER VISUAL ANCHORS (MUST BE PRESERVED EXACTLY)**:\n{anchor_info}")
            context_parts.append("**IMPORTANT**: The characters in this shot MUST match these visual anchors exactly. Include all appearance details, colors, clothing, and distinctive features from the anchors.")
        
        # 添加场景锚点（如果已设置）
        if self.scene_anchor:
            anchor_info = json.dumps(self.scene_anchor, ensure_ascii=False, indent=2)
            context_parts.append(f"**SCENE VISUAL ANCHORS (MUST BE PRESERVED EXACTLY)**:\n{anchor_info}")
            context_parts.append("**IMPORTANT**: The scene/environment in this shot MUST match these visual anchors exactly. Include all environmental details, colors, lighting, and atmosphere from the anchors.")
        
        # 添加镜头详情
        shot_info = json.dumps(shot, ensure_ascii=False, indent=2)
        context_parts.append(f"**SHOT DETAILS**:\n{shot_info}")
        
        # 添加静态图片强调
        context_parts.append("**CRITICAL: This is a STATIC SINGLE FRAME image prompt, NOT a video prompt.**")
        context_parts.append("**IMPORTANT**: Describe a frozen moment, a static snapshot. Avoid dynamic words like 'transitions', 'unfolding', 'moving', 'changing'. Use static descriptions like 'shows', 'displays', 'is' to describe the current state of the frame.")
        
        # 添加一致性强调
        consistency_note = []
        if shot.get("characters") and self.character_anchor:
            consistency_note.append("Character consistency is CRITICAL - use the character anchors provided above.")
        if self.scene_anchor:
            consistency_note.append("Scene consistency is CRITICAL - use the scene anchors provided above.")
        
        if consistency_note:
            context_parts.append(f"**CONSISTENCY REQUIREMENT**: {' '.join(consistency_note)}")
        
        context = "\n\n".join(context_parts)
        
        # 构建内容
        contents = f"{PROMPT_ENGINEERING_PROMPT}\n\n{context}"
        
        # 生成配置（降低temperature以提高一致性）
        generation_config = {
            "temperature": 0.6,  # 降低temperature以提高一致性
            "top_k": 40,
            "top_p": 0.9,  # 稍微降低top_p
            "max_output_tokens": 1024
        }
        
        # 调用API生成提示词
        prompt = self._generate_content_with_retry(
            model=GEMINI_FLASH_MODEL,
            contents=contents,
            generation_config=generation_config
        )
        
        return prompt.strip()
    
    def generate_image(
        self, 
        prompt: str, 
        output_path: Optional[str] = None,
        aspect_ratio: str = "16:9",
        image_size: str = "2K",
        character_image_path: Optional[str] = None,
        scene_image_path: Optional[str] = None
    ) -> str:
        """
        调用Gemini Pro Image生成图像
        
        Args:
            prompt: 图像生成提示词
            output_path: 输出文件路径（可选，如果不提供则返回base64）
            aspect_ratio: 宽高比，支持:"1:1","2:3","3:2","3:4","4:3","4:5", "5:4", "9:16", "16:9","21:9"
            image_size: 分辨率，支持:"1K", "2K", "4K"（必须大写K）
            character_image_path: 人物参考图片路径（可选）
            scene_image_path: 场景参考图片路径（可选）
            
        Returns:
            如果output_path为None，返回base64字符串；否则返回文件路径
        """
        print(f"正在生成图像: {prompt[:50]}...")
        
        # 构建 contents：必须包含提示词，参考图片是可选的（参照google_gemini_node.py的逻辑）
        # 为了确保一致性，如果有参考图片，需要将其添加到提示词中明确说明
        enhanced_prompt = prompt
        
        # 添加参考图片到 contents（如果有提供）
        reference_images = []
        if character_image_path:
            try:
                character_img = Image.open(character_image_path)
                # 在提示词前添加一致性说明
                if "character reference image" not in enhanced_prompt.lower():
                    enhanced_prompt = f"Use the character reference image provided to maintain exact visual consistency. {enhanced_prompt}"
                contents = [enhanced_prompt, character_img]
                reference_images.append("人物参考图")
                print(f"已添加人物参考图: {character_image_path}")
            except Exception as e:
                print(f"警告: 无法加载人物参考图 {character_image_path}: {str(e)}")
                contents = [prompt]
        else:
            contents = [prompt]
        
        if scene_image_path:
            try:
                scene_img = Image.open(scene_image_path)
                # 在提示词前添加一致性说明
                if "scene reference image" not in enhanced_prompt.lower():
                    enhanced_prompt = f"Use the scene reference image provided to maintain exact visual consistency. {enhanced_prompt}"
                # 如果已经有character_img，添加到列表；否则创建新列表
                if len(contents) == 1:
                    contents = [enhanced_prompt, scene_img]
                else:
                    contents.append(scene_img)
                reference_images.append("场景参考图")
                print(f"已添加场景参考图: {scene_image_path}")
            except Exception as e:
                print(f"警告: 无法加载场景参考图 {scene_image_path}: {str(e)}")
        
        if reference_images:
            print(f"使用参考图确保一致性: {', '.join(reference_images)}")
            print("提示词已增强以强调一致性要求")
        
        # 调用图像生成API（根据图片示例）
        for attempt, delay in enumerate([1, 2, 4, 8, 16]):
            try:
                # 使用types.GenerateContentConfig和types.ImageConfig（参照google_gemini_node.py）
                # 注意：参数使用驼峰命名：aspectRatio, imageSize（不是下划线命名）
                # 图像生成模型：只输出图像（参照google_gemini_node.py的逻辑）
                response = self.client.models.generate_content(
                    model=GEMINI_IMAGE_MODEL,
                    contents=contents,  # 包含提示词和参考图片
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],  # 只返回图像（参照google_gemini_node.py）
                        image_config=types.ImageConfig(
                            aspectRatio=aspect_ratio,  # 驼峰命名
                            imageSize=image_size  # 驼峰命名，支持"1K", "2K", "4K"（必须大写K）
                        )
                    )
                )
                
                # 提取图像数据（参照google_gemini_node.py的逻辑）
                if not response.candidates:
                    raise Exception("API返回了空响应，没有候选结果")
                
                # 处理每个候选结果
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        for part in candidate.content.parts:
                            # 检查是否有图像数据（参照google_gemini_node.py的逻辑）
                            image_bytes = None
                            if hasattr(part, "inline_data") and part.inline_data:
                                image_bytes = part.inline_data.data
                            # 兼容其他可能的图像数据格式
                            elif hasattr(part, "image") and part.image:
                                if hasattr(part.image, "data"):
                                    image_bytes = part.image.data
                            
                            # 处理图像数据
                            if image_bytes is not None:
                                # 将图像字节转换为 PIL Image（参照google_gemini_node.py的逻辑）
                                try:
                                    generated_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                                    
                                    if output_path:
                                        # 保存为PNG格式
                                        generated_image.save(output_path, format='PNG')
                                        file_size = len(image_bytes)
                                        print(f"图像已保存到: {output_path} (大小: {file_size} 字节)")
                                        return output_path
                                    else:
                                        # 转换为base64
                                        buffer = io.BytesIO()
                                        generated_image.save(buffer, format='PNG')
                                        return base64.b64encode(buffer.getvalue()).decode('utf-8')
                                except Exception as img_error:
                                    print(f"图像处理失败: {str(img_error)}")
                                    # 如果PIL处理失败，尝试直接保存原始字节
                                    if output_path:
                                        try:
                                            with open(output_path, "wb") as f:
                                                f.write(image_bytes)
                                            print(f"图像已保存到: {output_path} (原始字节，大小: {len(image_bytes)} 字节)")
                                            return output_path
                                        except Exception as save_error:
                                            raise Exception(f"保存图像失败: {str(save_error)}")
                                    else:
                                        return base64.b64encode(image_bytes).decode('utf-8') if isinstance(image_bytes, bytes) else image_bytes
                
                # 如果所有方法都失败，打印调试信息
                print("\n调试信息 - 无法提取图像数据")
                print(f"response类型: {type(response)}")
                if hasattr(response, 'candidates'):
                    print(f"candidates数量: {len(response.candidates)}")
                    if len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        print(f"candidate类型: {type(candidate)}")
                        if hasattr(candidate, 'content'):
                            print(f"content类型: {type(candidate.content)}")
                            if hasattr(candidate.content, 'parts'):
                                print(f"parts数量: {len(candidate.content.parts)}")
                                for i, part in enumerate(candidate.content.parts):
                                    print(f"part[{i}]类型: {type(part)}")
                                    print(f"part[{i}]属性: {[attr for attr in dir(part) if not attr.startswith('_')]}")
                
                raise Exception("API响应中未找到图像数据")
                
            except Exception as e:
                if attempt == 4:  # 最后一次重试
                    raise Exception(f"图像生成失败，已重试5次: {str(e)}")
                print(f"图像生成失败，{delay}秒后重试 (尝试 {attempt + 1}/5)...")
                time.sleep(delay)
        
        raise Exception("图像生成失败")
    
    def process_storyboard(
        self, 
        script_text: str, 
        output_dir: str = "output",
        character_image: Optional[str] = None,
        scene_image: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        完整工作流：处理故事板生成
        
        Args:
            script_text: 剧本文本
            output_dir: 输出目录
            character_image: 角色图片路径（可选）
            scene_image: 场景图片路径（可选）
            
        Returns:
            处理结果列表，包含每个镜头的信息和生成的图像路径
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Pass 1: 分析资产（如果提供）
        if character_image:
            self.character_image_path = character_image
            self.analyze_asset(character_image, is_character=True)
        
        if scene_image:
            self.scene_image_path = scene_image
            self.analyze_asset(scene_image, is_character=False)
        
        # 分析脚本
        shots = self.analyze_script(script_text)
        
        # 将分镜脚本保存到txt文件
        storyboard_file = output_path / "storyboard.txt"
        self._save_storyboard_to_file(shots, storyboard_file)
        print(f"\n分镜脚本已保存到: {storyboard_file}")
        
        # 为每个镜头生成图像
        results = []
        for shot in shots:
            shot_num = shot["shotNumber"]
            print(f"\n处理镜头 {shot_num}/{len(shots)}...")
            
            # 生成图像提示词（支持重新生成）
            max_regenerate_attempts = 5  # 最大重新生成次数
            regenerate_count = 0
            prompt = None
            
            while True:
                # 生成或重新生成提示词
                if prompt is None or regenerate_count > 0:
                    print(f"\n{'重新' if regenerate_count > 0 else ''}生成图像提示词...")
                    prompt = self.generate_image_prompt(shot)
                
                print("\n生成的提示词（完整）:")
                print("=" * 80)
                print(prompt)
                print("=" * 80)
                
                # 询问是否重新生成
                if regenerate_count < max_regenerate_attempts:
                    print("\n是否重新生成提示词？(y/n，直接按回车视为n): ", end="")
                    regenerate_choice = input().strip().lower()
                    
                    if regenerate_choice == 'y' or regenerate_choice == 'yes':
                        regenerate_count += 1
                        print(f"正在重新生成提示词（第 {regenerate_count} 次）...")
                        continue  # 重新生成提示词
                
                # 不重新生成，允许用户修改提示词
                print("\n提示：如果需要修改提示词，请输入新的提示词（直接按回车键使用当前提示词）:")
                user_input = input("请输入修改后的提示词（或直接按回车使用当前提示词）: ").strip()
                
                if user_input:
                    prompt = user_input
                    print("\n已使用修改后的提示词:")
                    print("=" * 80)
                    print(prompt)
                    print("=" * 80)
                else:
                    print("\n使用当前提示词")
                
                break  # 退出循环，使用当前提示词生成图像
            
            # 生成图像（传递参考图片路径）
            image_path = str(output_path / f"shot_{shot_num:03d}.png")
            try:
                self.generate_image(
                    prompt, 
                    image_path,
                    character_image_path=self.character_image_path,
                    scene_image_path=self.scene_image_path
                )
                results.append({
                    "shot": shot,
                    "prompt": prompt,
                    "imagePath": image_path
                })
            except Exception as e:
                print(f"镜头 {shot_num} 图像生成失败: {str(e)}")
                results.append({
                    "shot": shot,
                    "prompt": prompt,
                    "imagePath": None,
                    "error": str(e)
                })
        
        print(f"\n故事板生成完成！共处理 {len(results)} 个镜头")
        return results
    
    def _save_storyboard_to_file(self, shots: List[Dict[str, Any]], file_path: Path) -> None:
        """
        将分镜脚本保存到txt文件（全部使用中文）
        
        Args:
            shots: 镜头列表
            file_path: 输出文件路径
        """
        # 计算总时长
        total_duration = sum(shot.get('duration', 0) for shot in shots)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("分镜脚本\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"总镜头数: {len(shots)}\n")
            f.write(f"总时长: {total_duration}秒 ({total_duration // 60}分{total_duration % 60}秒)\n\n")
            
            for shot in shots:
                duration = shot.get('duration', 0)
                f.write("-" * 80 + "\n")
                f.write(f"镜头 {shot['shotNumber']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"场景描述: {shot['sceneDescription']}\n")
                f.write(f"位置: {shot['location']}\n")
                f.write(f"角色: {', '.join(shot['characters']) if shot['characters'] else '无'}\n")
                f.write(f"对话: {shot['dialogue'] if shot['dialogue'] else '无'}\n")
                f.write(f"视觉风格: {shot['visualStyle']}\n")
                f.write(f"时长: {duration}秒\n")
                f.write("\n")