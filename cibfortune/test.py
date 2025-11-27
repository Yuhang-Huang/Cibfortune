#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试本地 PaddleOCR 效果
支持图片和PDF文件的OCR识别
"""

import os
import sys
from pathlib import Path

# 检查是否安装了 PaddleOCR
try:
    from paddleocr import PaddleOCR
    print("✅ PaddleOCR 已安装")
except ImportError:
    print("❌ 未安装 PaddleOCR")
    print("💡 请运行: pip install paddleocr")
    print("   或者: pip install paddlepaddle paddleocr")
    sys.exit(1)

def test_image_ocr(image_path, output_dir="output"):
    """
    测试图片 OCR 识别
    
    Args:
        image_path: 图片文件路径
        output_dir: 输出目录
    """
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在: {image_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"📷 开始识别图片: {image_path}")
    print(f"{'='*80}\n")
    
    # 初始化 PaddleOCR
    # 注意：新版本 PaddleOCR 不再支持 use_gpu 参数，会自动检测
    # use_angle_cls 已弃用，使用 use_textline_orientation 代替
    # 如果本地没有模型文件，不要指定模型目录，让 PaddleOCR 自动下载
    try:
        # 检查本地是否有模型目录，如果没有则不指定（让 PaddleOCR 自动下载）
        ocr_params = {
            "lang": "ch" , # 语言：ch（中文）、en（英文）等
        }
        
        # 可选：如果本地有模型文件，可以指定路径
        # 但需要确保目录存在，否则会报错
        if os.path.exists("ch_PP-OCRv4_det_infer"):
            ocr_params["det_model_dir"] = "ch_PP-OCRv4_det_infer"
        if os.path.exists("ch_PP-OCRv3_rec_infer"):
            ocr_params["rec_model_dir"] = "ch_PP-OCRv3_rec_infer"
        if os.path.exists("ch_ppocr_mobile_v2.0_cls_infer"):
            ocr_params["cls_model_dir"] = "ch_ppocr_mobile_v2.0_cls_infer"
            # 只有指定了 cls_model_dir 时才启用文本行方向分类
            try:
                ocr_params["use_textline_orientation"] = True
            except:
                pass  # 如果参数不支持，忽略
        
        ocr = PaddleOCR(**ocr_params)
    except Exception as e:
        # 如果初始化失败，使用最简单的初始化方式（会自动下载模型）
        print(f"⚠️  使用默认参数初始化（将自动下载模型）: {e}")
        ocr = PaddleOCR(lang="ch")
    
    print("✅ PaddleOCR 初始化完成\n")
    
    # 执行 OCR
    import time
    start_time = time.time()
    
    try:
        # 新版本推荐使用 predict 方法，但也可以使用 ocr 方法
        try:
            # 尝试使用新的 predict 方法
            result = ocr.predict(image_path)
        except (AttributeError, TypeError):
            # 如果不支持 predict，使用 ocr 方法
            try:
                result = ocr.ocr(image_path, cls=True)
            except TypeError:
                result = ocr.ocr(image_path)
        elapsed_time = time.time() - start_time
        print(f"⏱️  识别耗时: {elapsed_time:.2f} 秒\n")
    except Exception as e:
        print(f"❌ OCR 识别失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 处理不同版本的返回格式
    # 新版本的 predict 方法返回格式可能不同
    if hasattr(result, 'dt_polys') or isinstance(result, dict):
        # 新版本返回对象或字典
        if hasattr(result, 'rec_text'):
            # 新版本格式
            rec_texts = result.rec_text if hasattr(result, 'rec_text') else []
            if isinstance(rec_texts, list) and len(rec_texts) > 0:
                # 转换为旧格式以便后续处理
                result = [[None, (text, 1.0)] for text in rec_texts]
            else:
                result = [result] if result else []
        else:
            result = [result] if result else []
    elif not isinstance(result, list):
        result = [result] if result else []
    
    # 确保 result 是列表格式
    if not result or (isinstance(result, list) and len(result) > 0 and not result[0]):
        print("⚠️  未识别到任何内容")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取文件名（不含扩展名）
    input_filename = Path(image_path).stem
    
    # 保存结果
    txt_file = os.path.join(output_dir, f"{input_filename}_ocr.txt")
    json_file = os.path.join(output_dir, f"{input_filename}_ocr.json")
    
    # 收集所有识别的文本
    all_texts = []
    all_results = []
    
    print("📝 识别结果:\n")
    print("-" * 80)
    
    # 处理结果格式
    ocr_lines = result[0] if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) else result
    
    for line_idx, line in enumerate(ocr_lines, 1):
        if not line:
            continue
        
        # 处理不同的行格式
        try:
            if isinstance(line, list) and len(line) >= 2:
                # 标准格式：[[box], (text, conf)]
                box = line[0] if line[0] is not None else None
                text_info = line[1]
                if isinstance(text_info, tuple) and len(text_info) >= 1:
                    text = text_info[0]
                    confidence = text_info[1] if len(text_info) > 1 else 1.0
                else:
                    text = str(text_info)
                    confidence = 1.0
            elif isinstance(line, dict):
                # 字典格式：{'text': ..., 'confidence': ..., 'box': ...}
                text = line.get('text', str(line))
                confidence = line.get('confidence', 1.0)
                box = line.get('box', None)
            elif isinstance(line, str):
                # 如果直接是字符串
                text = line
                confidence = 1.0
                box = None
            elif isinstance(line, tuple) and len(line) >= 1:
                # 元组格式：(text, conf) 或 (text,)
                text = line[0]
                confidence = line[1] if len(line) > 1 else 1.0
                box = None
            else:
                # 其他格式，尝试转换
                text = str(line)
                confidence = 1.0
                box = None
            
            if text and str(text).strip():
                all_texts.append(str(text))
                all_results.append({
                    "line": line_idx,
                    "text": str(text),
                    "confidence": float(confidence),
                    "box": box
                })
                
                # 打印结果
                print(f"行 {line_idx}: {text} (置信度: {confidence:.4f})")
        except Exception as line_error:
            # 如果某一行处理失败，跳过并记录
            print(f"⚠️  跳过无法解析的行 {line_idx}: {line_error}")
            print(f"   行内容: {line}")
            continue
    
    print("-" * 80)
    print(f"\n✅ 共识别 {len(all_texts)} 行文本\n")
    
    # 保存为文本文件
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_texts))
    print(f"💾 文本结果已保存: {txt_file}")
    
    # 保存为 JSON 文件
    import json
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "image_path": image_path,
            "total_lines": len(all_texts),
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON结果已保存: {json_file}")


def test_pdf_ocr(pdf_path, output_dir="output"):
    """
    测试 PDF OCR 识别（逐页处理）
    
    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
    """
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        return
    
    # 检查是否安装了 PDF 处理库
    try:
        import fitz  # PyMuPDF
        pdf_lib = "pymupdf"
    except ImportError:
        try:
            from pdf2image import convert_from_path
            pdf_lib = "pdf2image"
        except ImportError:
            print("❌ 未安装 PDF 处理库")
            print("💡 请运行: pip install PyMuPDF 或 pip install pdf2image")
            return
    
    print(f"\n{'='*80}")
    print(f"📄 开始识别 PDF: {pdf_path}")
    print(f"{'='*80}\n")
    
    # 初始化 PaddleOCR
    # 注意：新版本 PaddleOCR 不再支持 use_gpu 参数，会自动检测
    # 如果本地没有模型文件，不要指定模型目录，让 PaddleOCR 自动下载
    try:
        ocr_params = {
            "lang": "ch"  # 语言：ch（中文）、en（英文）等
        }
        
        # 可选：如果本地有模型文件，可以指定路径
        # 但需要确保目录存在，否则会报错
        if os.path.exists("ch_ppocr_mobile_v2.0_cls_infer"):
            ocr_params["cls_model_dir"] = "ch_ppocr_mobile_v2.0_cls_infer"
            # 只有指定了 cls_model_dir 时才启用文本行方向分类
            try:
                ocr_params["use_textline_orientation"] = True
            except:
                pass  # 如果参数不支持，忽略
        
        ocr = PaddleOCR(**ocr_params)
    except Exception as e:
        # 如果初始化失败，使用最简单的初始化方式（会自动下载模型）
        print(f"⚠️  使用默认参数初始化（将自动下载模型）: {e}")
        ocr = PaddleOCR(lang="ch")
    
    print("✅ PaddleOCR 初始化完成\n")
    
    # 转换 PDF 为图片
    print("📄 正在转换 PDF 为图片...")
    images = []
    
    if pdf_lib == "pymupdf":
        import fitz
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2倍缩放提高清晰度
            from PIL import Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
    else:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=200)
    
    total_pages = len(images)
    print(f"✅ PDF 共 {total_pages} 页\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    input_filename = Path(pdf_path).stem
    
    # 保存所有页面的结果
    all_pages_text = []
    all_pages_results = []
    
    import time
    total_start_time = time.time()
    
    for page_idx, image in enumerate(images, 1):
        print(f"📄 正在识别第 {page_idx}/{total_pages} 页...")
        page_start_time = time.time()
        
        try:
            # 执行 OCR
            # PaddleOCR 需要 numpy.ndarray 或文件路径，不能直接使用 PIL.Image
            # 将 PIL.Image 转换为 numpy.ndarray
            import numpy as np
            img_array = np.array(image)
            
            # 新版本推荐使用 predict 方法
            try:
                result = ocr.predict(img_array)
            except (AttributeError, TypeError):
                # 如果不支持 predict，使用 ocr 方法
                try:
                    result = ocr.ocr(img_array, cls=True)
                except TypeError:
                    result = ocr.ocr(img_array)
            page_elapsed = time.time() - page_start_time
            print(f"   ⏱️  耗时: {page_elapsed:.2f} 秒")
            
            # 处理不同版本的返回格式
            # 新版本的 predict 方法可能返回不同的格式
            ocr_lines = []
            
            if isinstance(result, list):
                # 如果是列表，检查第一个元素
                if len(result) > 0:
                    if isinstance(result[0], list):
                        # 旧格式：[[[box], (text, conf)], ...]
                        ocr_lines = result[0]
                    else:
                        # 可能是新格式或其他格式
                        ocr_lines = result
                else:
                    ocr_lines = []
            elif hasattr(result, 'rec_text'):
                # 新版本返回对象，有 rec_text 属性
                rec_texts = result.rec_text
                if isinstance(rec_texts, list) and len(rec_texts) > 0:
                    ocr_lines = [[None, (text, 1.0)] for text in rec_texts]
                else:
                    ocr_lines = []
            elif isinstance(result, dict):
                # 如果是字典，尝试提取文本
                if 'rec_text' in result:
                    rec_texts = result['rec_text']
                    if isinstance(rec_texts, list) and len(rec_texts) > 0:
                        ocr_lines = [[None, (text, 1.0)] for text in rec_texts]
                    else:
                        ocr_lines = []
                elif 'text' in result:
                    text = result['text']
                    if isinstance(text, list):
                        ocr_lines = [[None, (t, 1.0)] for t in text]
                    else:
                        ocr_lines = [[None, (text, 1.0)]]
                else:
                    ocr_lines = []
            else:
                # 其他格式，尝试转换
                ocr_lines = [result] if result else []
            
            if not ocr_lines:
                print(f"   ⚠️  第 {page_idx} 页未识别到内容\n")
                all_pages_text.append("")
                all_pages_results.append({
                    "page": page_idx,
                    "lines": 0,
                    "texts": []
                })
                continue
            
            # 提取文本
            page_texts = []
            page_results = []
            
            for line in ocr_lines:
                if not line:
                    continue
                
                # 处理不同的行格式
                try:
                    if isinstance(line, list) and len(line) >= 2:
                        # 标准格式：[[box], (text, conf)]
                        box = line[0] if line[0] is not None else None
                        text_info = line[1]
                        if isinstance(text_info, tuple) and len(text_info) >= 1:
                            text = text_info[0]
                            confidence = text_info[1] if len(text_info) > 1 else 1.0
                        else:
                            text = str(text_info)
                            confidence = 1.0
                    elif isinstance(line, str):
                        # 如果直接是字符串
                        text = line
                        confidence = 1.0
                        box = None
                    else:
                        # 其他格式，尝试提取
                        text = str(line)
                        confidence = 1.0
                        box = None
                    
                    if text and text.strip():
                        page_texts.append(text)
                        page_results.append({
                            "text": text,
                            "confidence": confidence,
                            "box": box
                        })
                except Exception as line_error:
                    # 如果某一行处理失败，跳过并记录
                    print(f"   ⚠️  跳过无法解析的行: {line_error}")
                    continue
            
            all_pages_text.append("\n".join(page_texts))
            all_pages_results.append({
                "page": page_idx,
                "lines": len(page_texts),
                "texts": page_results
            })
            
            print(f"   ✅ 识别到 {len(page_texts)} 行文本\n")
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"   ❌ 第 {page_idx} 页识别失败: {str(e)}")
            print(f"   错误详情: {error_detail.split(chr(10))[-2] if error_detail else '未知错误'}\n")
            all_pages_text.append("")
            all_pages_results.append({
                "page": page_idx,
                "error": str(e),
                "error_detail": error_detail,
                "texts": []
            })
    
    total_elapsed = time.time() - total_start_time
    print(f"{'='*80}")
    print(f"✅ 全部完成，总耗时: {total_elapsed:.2f} 秒")
    print(f"{'='*80}\n")
    
    # 保存结果
    txt_file = os.path.join(output_dir, f"{input_filename}_ocr.txt")
    json_file = os.path.join(output_dir, f"{input_filename}_ocr.json")
    
    # 保存合并的文本
    with open(txt_file, "w", encoding="utf-8") as f:
        for page_idx, page_text in enumerate(all_pages_text, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"第 {page_idx} 页\n")
            f.write(f"{'='*80}\n\n")
            f.write(page_text)
            f.write("\n\n")
    
    print(f"💾 文本结果已保存: {txt_file}")
    
    # 保存 JSON
    import json
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "pdf_path": pdf_path,
            "total_pages": total_pages,
            "pages": all_pages_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"💾 JSON结果已保存: {json_file}")


if __name__ == "__main__":
    print("=" * 80)
    print("🔍 PaddleOCR 本地测试工具")
    print("=" * 80)
    print("1. 测试图片 OCR")
    print("2. 测试 PDF OCR")
    print("=" * 80)
    
    choice = input("\n请选择测试类型 [1/2]: ").strip()
    
    if choice == "1":
        image_path = input("请输入图片文件路径: ").strip().strip('"').strip("'")
        if not image_path:
            print("❌ 请提供图片文件路径")
            sys.exit(1)
        test_image_ocr(image_path)
    elif choice == "2":
        pdf_path = input("请输入PDF文件路径: ").strip().strip('"').strip("'")
        if not pdf_path:
            print("❌ 请提供PDF文件路径")
            sys.exit(1)
        test_pdf_ocr(pdf_path)
    else:
        print("❌ 无效的选择")
        sys.exit(1)
