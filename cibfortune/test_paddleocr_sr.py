#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试双三次插值图像超分辨率功能
"""

import sys
import os

# 导入image模块
try:
    from image import paddleocr_super_resolution
    print("成功导入image模块")
except ImportError as e:
    print(f"导入失败: {e}")
    print("请确保image.py在当前目录")
    sys.exit(1)

# 测试PaddleOCR超分
if __name__ == "__main__":
    image_path = "complexbg.png"
    
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("测试双三次插值图像超分辨率功能")
    print("=" * 60)
    
    # 示例1: 使用默认设置（推荐，自动限制最大尺寸为2500x2500）
    result = paddleocr_super_resolution(
        image_path,
        output_path="paddleocr_sr_output.jpg"
    )
    
    # 示例2: 使用放大倍数（仍会受最大尺寸限制）
    # result = paddleocr_super_resolution(
    #     image_path,
    #     output_path="paddleocr_sr_output.jpg",
    #     scale=4  # 放大4倍，但不超过2500x2500
    # )
    
    # 示例3: 自定义最大尺寸限制
    # result = paddleocr_super_resolution(
    #     image_path,
    #     output_path="paddleocr_sr_output.jpg",
    #     max_width=3000,  # 自定义最大宽度3000像素
    #     max_height=3000  # 自定义最大高度3000像素
    # )
    
    if result is not None:
        print("\n测试成功！输出文件: paddleocr_sr_output.jpg")
    else:
        print("\n测试失败，请检查错误信息")



