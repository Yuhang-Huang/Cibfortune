#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 PaddleOCR 是否已安装"""

try:
    from paddleocr import PaddleOCR
    print("✅ PaddleOCR 已安装")
    try:
        import paddleocr
        if hasattr(paddleocr, '__version__'):
            print(f"   版本: {paddleocr.__version__}")
    except:
        pass
except ImportError:
    print("❌ PaddleOCR 未安装")
    print("\n💡 安装方法:")
    print("   pip install paddleocr")
    print("   或者:")
    print("   pip install paddlepaddle paddleocr")

