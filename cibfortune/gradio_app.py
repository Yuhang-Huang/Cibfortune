#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B-Instruct Gradio Web界面
提供友好的Web界面来使用Qwen3-VL模型
"""

import os
import gradio as gr
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import time
import json

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class Qwen3VLGradioApp:
    """Qwen3-VL Gradio应用类"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_path = "/data/storage1/wulin/models/qwen3-vl-8b-instruct"
        self.is_loaded = False
        self.chat_history = []
        
    def load_model(self, progress=gr.Progress()):
        """加载模型"""
        if self.is_loaded:
            return "✅ 模型已经加载完成！"
        
        try:
            progress(0.1, desc="检查模型路径...")
            if not os.path.exists(self.model_path):
                return f"❌ 模型路径不存在: {self.model_path}"
            
            progress(0.3, desc="加载模型...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype="auto",
                device_map="auto"
            )
            
            progress(0.7, desc="加载处理器...")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            progress(1.0, desc="完成！")
            self.is_loaded = True
            
            return "✅ 模型加载成功！可以开始使用了。"
            
        except Exception as e:
            return f"❌ 模型加载失败: {str(e)}"
    
    def chat_with_image(self, image, text, history, max_tokens, temperature):
        """与图像对话"""
        if not self.is_loaded:
            return history, "❌ 请先加载模型！"
        
        if image is None:
            return history, "❌ 请上传图像！"
        
        if not text.strip():
            return history, "❌ 请输入问题！"
        
        try:
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            
            # 准备输入
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # 生成回答
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False
                )
            
            # 处理输出
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            response = output_text[0]
            
            # 更新历史记录
            history.append([f"👤 {text}", f"🤖 {response}"])
            
            return history, ""
            
        except Exception as e:
            error_msg = f"❌ 生成失败: {str(e)}"
            history.append([f"👤 {text}", error_msg])
            return history, ""
    
    def ocr_analysis(self, image):
        """OCR文字识别"""
        if not self.is_loaded:
            return "❌ 请先加载模型！"
        
        if image is None:
            return "❌ 请上传图像！"
        
        try:
            prompt = "请识别并提取这张图片中的所有文字内容。如果图片中有多种语言，请分别标注语言类型。"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return f"📝 OCR识别结果:\n\n{output_text[0]}"
            
        except Exception as e:
            return f"❌ OCR识别失败: {str(e)}"
    
    def spatial_analysis(self, image):
        """空间感知分析"""
        if not self.is_loaded:
            return "❌ 请先加载模型！"
        
        if image is None:
            return "❌ 请上传图像！"
        
        try:
            prompt = """请分析这张图片中的空间关系，包括：
            1. 物体的相对位置关系
            2. 视角和观察角度
            3. 物体的遮挡关系
            4. 深度和距离感
            5. 空间布局的整体描述"""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return f"🔍 空间分析结果:\n\n{output_text[0]}"
            
        except Exception as e:
            return f"❌ 空间分析失败: {str(e)}"
    
    def visual_coding(self, image, output_format):
        """视觉编程"""
        if not self.is_loaded:
            return "❌ 请先加载模型！"
        
        if image is None:
            return "❌ 请上传图像！"
        
        try:
            format_prompts = {
                "HTML": "请根据这张图片生成对应的HTML代码，包括结构、样式和布局。",
                "CSS": "请根据这张图片生成对应的CSS样式代码。",
                "JavaScript": "请根据这张图片生成对应的JavaScript代码。",
                "Python": "请根据这张图片生成对应的Python代码。"
            }
            
            prompt = format_prompts.get(output_format, format_prompts["HTML"])
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return f"💻 {output_format}代码:\n\n```{output_format.lower()}\n{output_text[0]}\n```"
            
        except Exception as e:
            return f"❌ 代码生成失败: {str(e)}"
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
        return []

# 创建应用实例
app = Qwen3VLGradioApp()

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(
        title="Qwen3-VL-8B-Instruct Web界面",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🤖 Qwen3-VL-8B-Instruct Web界面
        
        欢迎使用Qwen3-VL-8B-Instruct多模态大语言模型！这个界面提供了友好的Web交互方式。
        
        **主要功能：**
        - 🖼️ 图像理解和对话
        - 📝 OCR文字识别
        - 🔍 空间感知分析
        - 💻 视觉编程（生成代码）
        """)
        
        with gr.Tab("🚀 模型管理"):
            gr.Markdown("### 模型加载")
            with gr.Row():
                load_btn = gr.Button("🔄 加载模型", variant="primary", size="lg")
                status_text = gr.Textbox(
                    label="状态", 
                    value="⏳ 模型未加载，请点击加载模型按钮",
                    interactive=False
                )
            
            load_btn.click(
                app.load_model,
                outputs=[status_text]
            )
        
        with gr.Tab("💬 图像对话"):
            gr.Markdown("### 与图像进行对话")
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="上传图像",
                        type="pil",
                        height=400
                    )
                    
                    with gr.Row():
                        max_tokens = gr.Slider(
                            minimum=50, maximum=1024, value=256,
                            label="最大生成长度"
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.7,
                            label="创造性 (Temperature)"
                        )
                
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        height=400,
                        show_label=True
                    )
                    
                    with gr.Row():
                        text_input = gr.Textbox(
                            label="输入问题",
                            placeholder="请描述这张图片...",
                            lines=2
                        )
                        send_btn = gr.Button("发送", variant="primary")
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ 清空历史")
            
            # 事件绑定
            send_btn.click(
                app.chat_with_image,
                inputs=[image_input, text_input, chatbot, max_tokens, temperature],
                outputs=[chatbot, text_input]
            )
            
            text_input.submit(
                app.chat_with_image,
                inputs=[image_input, text_input, chatbot, max_tokens, temperature],
                outputs=[chatbot, text_input]
            )
            
            clear_btn.click(
                app.clear_history,
                outputs=[chatbot]
            )
        
        with gr.Tab("📝 OCR识别"):
            gr.Markdown("### 文字识别")
            
            with gr.Row():
                with gr.Column():
                    ocr_image = gr.Image(
                        label="上传图像进行OCR识别",
                        type="pil",
                        height=300
                    )
                    ocr_btn = gr.Button("🔍 开始识别", variant="primary")
                
                with gr.Column():
                    ocr_result = gr.Textbox(
                        label="识别结果",
                        lines=15,
                        max_lines=20
                    )
            
            ocr_btn.click(
                app.ocr_analysis,
                inputs=[ocr_image],
                outputs=[ocr_result]
            )
        
        with gr.Tab("🔍 空间分析"):
            gr.Markdown("### 空间感知分析")
            
            with gr.Row():
                with gr.Column():
                    spatial_image = gr.Image(
                        label="上传图像进行空间分析",
                        type="pil",
                        height=300
                    )
                    spatial_btn = gr.Button("🔍 开始分析", variant="primary")
                
                with gr.Column():
                    spatial_result = gr.Textbox(
                        label="分析结果",
                        lines=15,
                        max_lines=20
                    )
            
            spatial_btn.click(
                app.spatial_analysis,
                inputs=[spatial_image],
                outputs=[spatial_result]
            )
        
        with gr.Tab("💻 视觉编程"):
            gr.Markdown("### 从图像生成代码")
            
            with gr.Row():
                with gr.Column():
                    code_image = gr.Image(
                        label="上传图像生成代码",
                        type="pil",
                        height=300
                    )
                    
                    code_format = gr.Dropdown(
                        choices=["HTML", "CSS", "JavaScript", "Python"],
                        value="HTML",
                        label="选择代码类型"
                    )
                    
                    code_btn = gr.Button("💻 生成代码", variant="primary")
                
                with gr.Column():
                    code_result = gr.Textbox(
                        label="生成的代码",
                        lines=15,
                        max_lines=20
                    )
            
            code_btn.click(
                app.visual_coding,
                inputs=[code_image, code_format],
                outputs=[code_result]
            )
        
        with gr.Tab("ℹ️ 使用说明"):
            gr.Markdown("""
            ## 使用说明
            
            ### 1. 模型管理
            - 首次使用需要点击"加载模型"按钮
            - 模型加载可能需要几分钟时间
            - 加载完成后可以开始使用所有功能
            
            ### 2. 图像对话
            - 上传图像后可以与其进行对话
            - 支持多轮对话，保持上下文
            - 可以调整生成参数（长度、创造性）
            
            ### 3. OCR识别
            - 上传包含文字的图像
            - 自动识别并提取所有文字内容
            - 支持32种语言识别
            
            ### 4. 空间分析
            - 分析图像中的空间关系
            - 包括物体位置、视角、遮挡关系等
            - 适用于3D场景理解
            
            ### 5. 视觉编程
            - 从图像生成各种类型的代码
            - 支持HTML、CSS、JavaScript、Python
            - 适用于UI设计和原型开发
            
            ### 注意事项
            - 确保模型路径正确：`/data/storage1/wulin/models/qwen3-vl-8b-instruct`
            - 需要足够的内存（建议16GB+）
            - 支持GPU加速（自动检测）
            """)
    
    return interface

def main():
    """主函数"""
    print("🚀 启动Qwen3-VL-8B-Instruct Web界面...")
    
    # 创建界面
    interface = create_interface()
    
    # 启动服务
    interface.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口
        share=False,            # 不创建公共链接
        debug=True,             # 调试模式
        show_error=True         # 显示错误
    )

if __name__ == "__main__":
    main()
