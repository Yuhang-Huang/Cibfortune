#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B-Instruct Web界面启动器
提供多种界面选择
"""

import os
import sys
import subprocess
import webbrowser
import time

def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                Qwen3-VL-8B-Instruct 多模态大模型Web界面         ║
║                        启动器                                 ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    missing_deps = []
    
    try:
        import gradio
        print(f"✓ Gradio: {gradio.__version__}")
    except ImportError:
        missing_deps.append("gradio>=4.0.0")
        print("✗ Gradio未安装")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
        print("✗ PyTorch未安装")
    
    try:
        from transformers import Qwen3VLForConditionalGeneration
        print("✓ Transformers: 支持Qwen3VL")
    except ImportError:
        missing_deps.append("transformers")
        print("✗ Transformers未安装")
    
    if missing_deps:
        print(f"\n缺少依赖: {', '.join(missing_deps)}")
        return False
    
    print("✅ 所有依赖已安装")
    return True

def install_missing_deps():
    """安装缺失的依赖"""
    print("📦 安装缺失的依赖...")
    
    deps = ["gradio>=4.0.0", "torch", "transformers", "accelerate", "sentencepiece", "protobuf", "Pillow", "requests"]
    
    for dep in deps:
        try:
            print(f"安装 {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True, capture_output=True)
            print(f"✓ {dep} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"✗ {dep} 安装失败: {e}")
            return False
    
    return True

def check_model():
    """检查模型"""
    model_path = "/data/storage1/wulin/models/qwen3-vl-8b-instruct"
    
    print(f"🔍 检查模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    # 检查关键文件
    required_files = ["config.json", "tokenizer_config.json"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 模型文件不完整，缺少: {', '.join(missing_files)}")
        return False
    
    print("✅ 模型检查通过")
    return True

def show_interface_menu():
    """显示界面选择菜单"""
    print("\n" + "="*60)
    print("1. 智能助手 (端口7862)")
    print("   - 模式切换：通用版 / 专业版")
    print("   - 触屏优化：更大按钮与间距")
    print("")
    print("2. 检查系统状态")
    print("3. 安装依赖")
    print("0. 退出")
    print("="*60)

def check_system_status():
    """检查系统状态"""
    print("\n🔍 系统状态检查:")
    print("-" * 40)
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        print(f"系统内存: {total_gb:.1f}GB 总计, {available_gb:.1f}GB 可用")
    except ImportError:
        print("系统内存: 无法检测 (psutil未安装)")
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA: 可用 (设备数量: {torch.cuda.device_count()})")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA: 不可用")
    except ImportError:
        print("CUDA: 无法检测 (PyTorch未安装)")
    
    # 检查模型
    model_ok = check_model()
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    print("\n📊 状态总结:")
    print(f"模型: {'✅ 正常' if model_ok else '❌ 异常'}")
    print(f"依赖: {'✅ 正常' if deps_ok else '❌ 异常'}")
    
    if model_ok and deps_ok:
        print("🎉 系统状态良好，可以启动界面！")
    else:
        print("⚠️  系统状态异常，请先解决问题")

def main():
    """主函数"""
    print_banner()
    
    while True:
        show_interface_menu()
        
        choice = input("\n请输入选项 (0-3): ").strip()
        
        if choice == "0":
            print("👋 再见！")
            break
        
        elif choice == "1":
            if check_dependencies() and check_model():
                try:
                    from gradio_unified import main as unified_main
                    unified_main()
                except Exception as e:
                    print(f"❌ 启动失败: {e}")
            else:
                print("❌ 系统检查失败，请先解决问题")
        elif choice == "2":
            check_system_status()
        elif choice == "3":
            if install_missing_deps():
                print("✅ 依赖安装完成")
            else:
                print("❌ 依赖安装失败")
        else:
            print("❌ 无效选项，请重新选择")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()
