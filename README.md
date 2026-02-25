# Storyboard Visionary Agent

一个基于Python的AI代理，用于自动化视频分镜脚本生成。该代理可以分析剧本、处理角色/场景图片以保持一致性，并生成每个镜头的第一帧图像。

## 功能特性

- **脚本分析**：使用Gemini 3 Flash Preview模型分析剧本，自动分解为镜头列表
- **一致性引擎**：通过"锚点系统"保持角色和场景的视觉一致性
  - Pass 1: 分析上传的角色/场景图片，提取关键视觉锚点
  - Pass 2: 生成图像时强制注入锚点信息
- **图像生成**：使用Gemini 3 Pro Image Preview模型生成高质量的分镜图像
- **错误处理**：实现指数退避重试机制，提高API调用的可靠性

## 技术栈

- Python 3.10+
- Google Generative AI REST API
  - Gemini 3 Flash Preview (文本/视觉分析)
  - Gemini 3 Pro Image Preview (图像生成)
- requests库 (HTTP请求)

## 安装

1. 克隆或下载项目到本地

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 获取Google Generative AI API密钥
   - 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
   - 创建API密钥

## 使用方法

### 基本用法

```bash
python main.py --api-key YOUR_API_KEY --script script.txt
```

### 完整用法（包含角色和场景一致性）

```bash
python main.py \
  --api-key YOUR_API_KEY \
  --script script.txt \
  --character-image character.jpg \
  --scene-image scene.jpg \
  --output-dir output
```

### 参数说明

- `--api-key`: **必需** - Google Generative AI API密钥
- `--script`: **必需** - 剧本文件路径（文本文件，UTF-8编码）
- `--character-image`: **可选** - 角色设计图片路径，用于保持角色视觉一致性
- `--scene-image`: **可选** - 场景设计图片路径，用于保持场景视觉一致性
- `--output-dir`: **可选** - 输出目录（默认: `output`）

## 项目结构

```
fenjing-agent/
├── config.py          # 系统提示词和配置常量
├── agent.py           # StoryboardAgent主逻辑类
├── main.py            # CLI入口点
├── requirements.txt   # Python依赖
└── README.md          # 项目说明文档
```

## 工作流程

1. **资产分析阶段**（如果提供了角色/场景图片）：
   - 使用Gemini Flash模型分析上传的图片
   - 提取关键视觉锚点（外观、颜色、风格、关键元素等）
   - 存储锚点信息供后续使用

2. **脚本分析阶段**：
   - 使用Gemini Flash模型分析剧本
   - 将剧本分解为镜头列表
   - 每个镜头包含：镜头编号、场景描述、角色、位置、对话、视觉风格

3. **图像生成阶段**：
   - 为每个镜头合成图像生成提示词
   - 将视觉锚点注入到提示词中（确保一致性）
   - 调用Gemini Pro Image模型生成图像
   - 保存图像到输出目录

## 输出格式

生成的图像文件命名格式：`shot_001.png`, `shot_002.png`, ...

每个镜头的信息包含：
- 镜头编号
- 场景描述
- 角色列表
- 位置
- 对话
- 视觉风格
- 生成的图像路径

## 注意事项

1. **API限制**：请注意Google Generative AI API的调用频率限制和配额
2. **图片格式**：支持的图片格式取决于API要求（通常支持JPEG、PNG）
3. **剧本格式**：剧本应为纯文本文件，UTF-8编码
4. **网络连接**：需要稳定的网络连接以访问Google API

## 错误处理

- 所有API调用都实现了指数退避重试机制（1s, 2s, 4s, 8s, 16s）
- 最多重试5次
- 如果所有重试都失败，程序会输出详细错误信息

## 扩展性

代码采用模块化设计，易于扩展：
- 可以轻松添加GUI界面
- 可以支持批量处理多个剧本
- 可以添加更多图像生成选项和参数
- 可以集成其他AI模型

## 许可证

本项目仅供学习和研究使用。
