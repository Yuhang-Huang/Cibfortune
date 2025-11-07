#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容层：高级界面已合并进 gradio_unified 模块。
保留此文件以兼容旧的导入路径。
"""

from gradio_unified import AdvancedQwen3VLApp

# 为保持旧代码可用，继续暴露同名实例
app = AdvancedQwen3VLApp()
