"""
CLI入口点：运行故事板生成工作流
"""

import argparse
import sys
from pathlib import Path
from agent import StoryboardAgent


def main():
    """主函数：解析命令行参数并执行工作流"""
    parser = argparse.ArgumentParser(
        description="Storyboard Visionary Agent - 自动化视频分镜脚本生成工具"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Google Generative AI API密钥"
    )
    
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="剧本文件路径（文本文件）"
    )
    
    parser.add_argument(
        "--character-image",
        type=str,
        default=None,
        help="角色设计图片路径（可选，用于保持角色一致性）"
    )
    
    parser.add_argument(
        "--scene-image",
        type=str,
        default=None,
        help="场景设计图片路径（可选，用于保持场景一致性）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出目录（默认: output）"
    )
    
    args = parser.parse_args()
    
    # 验证输入文件
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"错误：剧本文件不存在: {args.script}")
        sys.exit(1)
    
    if args.character_image and not Path(args.character_image).exists():
        print(f"错误：角色图片文件不存在: {args.character_image}")
        sys.exit(1)
    
    if args.scene_image and not Path(args.scene_image).exists():
        print(f"错误：场景图片文件不存在: {args.scene_image}")
        sys.exit(1)
    
    # 读取剧本
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            script_text = f.read()
    except Exception as e:
        print(f"错误：无法读取剧本文件: {str(e)}")
        sys.exit(1)
    
    # 创建代理并执行工作流
    try:
        agent = StoryboardAgent(api_key=args.api_key)
        results = agent.process_storyboard(
            script_text=script_text,
            output_dir=args.output_dir,
            character_image=args.character_image,
            scene_image=args.scene_image
        )
        
        # 输出结果摘要
        print("\n" + "="*60)
        print("处理结果摘要")
        print("="*60)
        for result in results:
            shot = result["shot"]
            print(f"\n镜头 {shot['shotNumber']}:")
            print(f"  场景: {shot['location']}")
            print(f"  角色: {', '.join(shot['characters']) if shot['characters'] else '无'}")
            print(f"  视觉风格: {shot['visualStyle']}")
            if result.get("imagePath"):
                print(f"  图像: {result['imagePath']}")
            else:
                print(f"  图像: 生成失败 - {result.get('error', '未知错误')}")
        
        print("\n" + "="*60)
        print("工作流执行完成！")
        print("="*60)
        
    except Exception as e:
        print(f"错误：工作流执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
