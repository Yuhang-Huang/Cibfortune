#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 PaddleOCR API 调用
直接使用 API 进行 OCR 识别，支持图片和PDF文件
"""

import os
import base64
import requests
from pathlib import Path
import json
import time

# API 配置
API_URL = "https://wdc9jbw9l1f8996b.aistudio-app.com/ocr"
TOKEN = "61236296494fb5e32ee89aef50d4d6aa99fa2ba7"


def format_ocr_result(result):
    """
    格式化 OCR 结果为字符串
    
    Args:
        result: OCR 结果，可能是字符串、字典或其他类型
        
    Returns:
        格式化后的字符串
    """
    if result is None:
        return ""
    
    if isinstance(result, str):
        return result
    elif isinstance(result, dict):
        # 如果是字典，尝试提取文本内容或转换为 JSON
        # 优先查找常见的文本字段
        if "text" in result:
            return str(result["text"])
        elif "content" in result:
            return str(result["content"])
        elif "result" in result:
            return format_ocr_result(result["result"])
        else:
            # 如果没有找到文本字段，转换为格式化的 JSON
            return json.dumps(result, ensure_ascii=False, indent=2)
    elif isinstance(result, list):
        # 如果是列表，尝试提取文本或转换为 JSON
        text_parts = []
        for item in result:
            text_parts.append(format_ocr_result(item))
        return "\n".join(text_parts)
    else:
        # 其他类型直接转换为字符串
        return str(result)


def test_image_ocr_api(image_path, output_dir="output"):
    """
    使用 API 测试图片 OCR 识别
    
    Args:
        image_path: 图片文件路径
        output_dir: 输出目录
    """
    print("=" * 80)
    print("图片 OCR API 测试")
    print("=" * 80)
    print(f"图片文件: {image_path}")
    print(f"输出目录: {output_dir}")
    
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在: {image_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取文件并编码
    print("\n正在读取文件...")
    with open(image_path, "rb") as file:
        file_bytes = file.read()
        file_data = base64.b64encode(file_bytes).decode("ascii")
    
    print(f"文件大小: {len(file_bytes) / 1024:.2f} KB")
    
    # 准备请求
    headers = {
        "Authorization": f"token {TOKEN}",
        "Content-Type": "application/json"
    }
    
    # 对于图片，fileType 设置为 1
    payload = {
        "file": file_data,
        "fileType": 1,  # 1 表示图片
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
    }
    
    # 发送请求
    print("\n正在调用 API...")
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=300)
        
        elapsed_time = time.time() - start_time
        print(f"API 响应时间: {elapsed_time:.2f}秒")
        
        if response.status_code != 200:
            print(f"❌ API 请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return
        
        result = response.json()
        
        if "result" not in result:
            print(f"❌ API 响应格式错误: {result}")
            return
        
        ocr_result = result["result"]
        
        # 获取输入文件名（不含扩展名）
        input_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # 处理 OCR 结果
        print(f"\n✅ 识别成功，共 {len(ocr_result.get('ocrResults', []))} 个结果")
        
        # 保存文本结果
        txt_file = os.path.join(output_dir, f"{input_filename}_ocr.txt")
        json_file = os.path.join(output_dir, f"{input_filename}_ocr.json")
        
        with open(txt_file, "w", encoding="utf-8") as f:
            for i, res in enumerate(ocr_result.get("ocrResults", [])):
                pruned_result = res.get("prunedResult", "")
                formatted_result = format_ocr_result(pruned_result)
                print(f"\n结果 {i + 1}:")
                print(formatted_result)
                f.write(f"\n{'='*60}\n")
                f.write(f"结果 {i + 1}\n")
                f.write(f"{'='*60}\n\n")
                f.write(formatted_result)
                f.write("\n\n")
        
        # 保存 JSON 结果
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 文本结果已保存到: {txt_file}")
        print(f"✅ JSON 结果已保存到: {json_file}")
        
        # 下载并保存图片
        print("\n正在下载 OCR 结果图片...")
        saved_images = 0
        for i, res in enumerate(ocr_result.get("ocrResults", [])):
            image_url = res.get("ocrImage")
            if image_url:
                try:
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        filename = os.path.join(output_dir, f"{input_filename}_{i}.jpg")
                        with open(filename, "wb") as f:
                            f.write(img_response.content)
                        print(f"  ✅ 图片 {i + 1} 已保存: {filename}")
                        saved_images += 1
                    else:
                        print(f"  ⚠️ 下载图片 {i + 1} 失败，状态码: {img_response.status_code}")
                except Exception as e:
                    print(f"  ⚠️ 下载图片 {i + 1} 时出错: {e}")
        
        if saved_images > 0:
            print(f"\n✅ 共保存 {saved_images} 张图片")
        
        print("\n✅ 图片处理完成！")
        
    except requests.exceptions.Timeout:
        print("❌ API 请求超时（超过5分钟）")
    except requests.exceptions.RequestException as e:
        print(f"❌ API 请求失败: {e}")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def test_pdf_ocr_api(pdf_path, output_dir="output"):
    """
    使用 API 测试 PDF OCR 识别
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
    """
    print("=" * 80)
    print("PDF OCR API 测试")
    print("=" * 80)
    print(f"PDF文件: {pdf_path}")
    print(f"输出目录: {output_dir}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取文件并编码
    print("\n正在读取PDF文件...")
    with open(pdf_path, "rb") as file:
        file_bytes = file.read()
        file_data = base64.b64encode(file_bytes).decode("ascii")
    
    file_size_mb = len(file_bytes) / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.2f} MB")
    
    # 准备请求
    headers = {
        "Authorization": f"token {TOKEN}",
        "Content-Type": "application/json"
    }
    
    # 对于 PDF 文档，fileType 设置为 0
    payload = {
        "file": file_data,
        "fileType": 0,  # 0 表示 PDF
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
    }
    
    # 发送请求
    print("\n正在调用 API（PDF处理可能需要较长时间）...")
    print("💡 提示: 大文件可能需要几分钟时间，请耐心等待...")
    start_time = time.time()
    
    try:
        # PDF 处理可能需要更长时间，设置更长的超时时间
        timeout = max(600, int(file_size_mb * 60))  # 根据文件大小动态调整超时时间
        print(f"超时设置: {timeout}秒")
        
        response = requests.post(API_URL, json=payload, headers=headers, timeout=timeout)
        
        elapsed_time = time.time() - start_time
        print(f"\nAPI 响应时间: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
        
        if response.status_code != 200:
            print(f"❌ API 请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text[:500]}")  # 只显示前500个字符
            return
        
        result = response.json()
        
        if "result" not in result:
            print(f"❌ API 响应格式错误")
            print(f"响应内容: {result}")
            return
        
        ocr_result = result["result"]
        
        # 获取输入文件名（不含扩展名）
        input_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 处理 OCR 结果
        ocr_results = ocr_result.get("ocrResults", [])
        print(f"\n✅ 识别成功，共 {len(ocr_results)} 页结果")
        
        # 保存文本结果（合并所有页面）
        txt_file = os.path.join(output_dir, f"{input_filename}_ocr.txt")
        md_file = os.path.join(output_dir, f"{input_filename}_ocr.md")
        json_file = os.path.join(output_dir, f"{input_filename}_ocr.json")
        
        with open(txt_file, "w", encoding="utf-8") as f:
            for i, res in enumerate(ocr_results):
                pruned_result = res.get("prunedResult", "")
                formatted_result = format_ocr_result(pruned_result)
                f.write(f"\n{'='*60}\n")
                f.write(f"第 {i + 1} 页\n")
                f.write(f"{'='*60}\n\n")
                f.write(formatted_result)
                f.write("\n\n")
        
        # 保存 Markdown 格式（合并所有页面）
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# {input_filename} OCR 结果\n\n")
            for i, res in enumerate(ocr_results):
                pruned_result = res.get("prunedResult", "")
                formatted_result = format_ocr_result(pruned_result)
                f.write(f"## 第 {i + 1} 页\n\n")
                # 如果是多行文本，使用代码块格式
                if "\n" in formatted_result:
                    f.write("```\n")
                    f.write(formatted_result)
                    f.write("\n```\n")
                else:
                    f.write(formatted_result)
                f.write("\n\n---\n\n")
        
        # 保存 JSON 结果
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 文本结果已保存到: {txt_file}")
        print(f"✅ Markdown 结果已保存到: {md_file}")
        print(f"✅ JSON 结果已保存到: {json_file}")
        
        # 下载并保存图片
        print("\n正在下载 OCR 结果图片...")
        saved_images = 0
        for i, res in enumerate(ocr_results):
            image_url = res.get("ocrImage")
            if image_url:
                try:
                    print(f"  正在下载第 {i + 1} 页图片...", end="", flush=True)
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        filename = os.path.join(output_dir, f"{input_filename}_page_{i + 1}.jpg")
                        with open(filename, "wb") as f:
                            f.write(img_response.content)
                        print(f" ✅")
                        saved_images += 1
                    else:
                        print(f" ⚠️ (状态码: {img_response.status_code})")
                except Exception as e:
                    print(f" ⚠️ (错误: {e})")
        
        if saved_images > 0:
            print(f"\n✅ 共保存 {saved_images} 张图片")
        
        print("\n✅ PDF处理完成！")
        
    except requests.exceptions.Timeout:
        print(f"\n❌ API 请求超时（超过 {timeout} 秒）")
        print("💡 提示: PDF文件可能太大，请尝试使用较小的文件或联系管理员")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API 请求失败: {e}")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 检查 requests 库
    try:
        import requests
    except ImportError:
        print("❌ 未安装 requests 库")
        print("💡 请运行: pip install requests")
        exit(1)
    
    print("PaddleOCR API 测试工具")
    print("=" * 80)
    print("1. 测试图片 OCR API")
    print("2. 测试 PDF OCR API")
    print("=" * 80)
    
    choice = input("\n请选择测试类型 [1/2]: ").strip()
    
    if choice == "1":
        image_path = input("请输入图片文件路径: ").strip()
        if not image_path:
            print("❌ 请提供图片文件路径")
            exit(1)
        test_image_ocr_api(image_path)
    elif choice == "2":
        pdf_path = input("请输入PDF文件路径: ").strip()
        if not pdf_path:
            print("❌ 请提供PDF文件路径")
            exit(1)
        test_pdf_ocr_api(pdf_path)
    else:
        print("❌ 无效的选择")
        exit(1)

