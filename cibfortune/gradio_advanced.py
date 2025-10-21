#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B-Instruct 高级Gradio界面
包含更多高级功能和更好的用户体验
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
import base64
from datetime import datetime

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class AdvancedQwen3VLApp:
    """高级Qwen3-VL应用类"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_path = "/data/storage1/wulin/models/qwen3-vl-8b-instruct"
        self.is_loaded = False
        self.chat_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_model(self, progress=gr.Progress()):
        """加载模型"""
        if self.is_loaded:
            return "✅ 模型已经加载完成！", gr.update(interactive=True)
        
        try:
            progress(0.1, desc="检查模型路径...")
            if not os.path.exists(self.model_path):
                return f"❌ 模型路径不存在: {self.model_path}", gr.update(interactive=False)
            
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
            
            return "✅ 模型加载成功！可以开始使用了。", gr.update(interactive=True)
            
        except Exception as e:
            return f"❌ 模型加载失败: {str(e)}", gr.update(interactive=False)
    
    def chat_with_image(self, image, text, history, max_tokens, temperature, top_p, top_k, repetition_penalty: float = 1.0, presence_penalty: float = 1.5):
        """与图像对话"""
        if not self.is_loaded:
            return history, "❌ 请先加载模型！", ""
        
        if image is None:
            return history, "❌ 请上传图像！", ""
        
        if not text.strip():
            return history, "❌ 请输入问题！", ""
        
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
            
            # 生成参数
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": True if temperature > 0 else False,
                "repetition_penalty": repetition_penalty
                # presence_penalty 参数为 OpenAI 风格，Transformers 不原生支持，此处保留占位
            }
            
            # 生成回答
            start_time = time.time()
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_kwargs)
            
            generation_time = time.time() - start_time
            
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
            
            # 生成统计信息
            stats = f"⏱️ 生成时间: {generation_time:.2f}秒 | 📝 生成长度: {len(response)}字符"
            
            return history, "", stats
            
        except Exception as e:
            error_msg = f"❌ 生成失败: {str(e)}"
            history.append([f"👤 {text}", error_msg])
            return history, "", f"❌ 错误: {str(e)}"
    
    def batch_analysis(self, images, analysis_type):
        """批量分析"""
        if not self.is_loaded:
            return "❌ 请先加载模型！"
        
        if not images:
            return "❌ 请上传图像！"
        
        results = []
        
        for i, image in enumerate(images):
            try:
                if analysis_type == "描述":
                    prompt = "请真实、详细、客观地描述这张图片的内容。"
                elif analysis_type == "OCR":
                    prompt = "请识别并提取这张图片中的所有文字内容，尽量还原原本样式，并标注语言类型。"
                elif analysis_type == "空间分析":
                    prompt = "请分析这张图片中的空间关系和物体位置，包括相对位置、视角、遮挡、深度与距离感，并给出整体布局描述。"
                elif analysis_type == "情感分析":
                    prompt = "请分析这张图片传达的情感或氛围。"
                else:
                    prompt = "请分析这张图片。"
                
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
                    generated_ids = self.model.generate(**inputs, max_new_tokens=512)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                results.append(f"📷 图像 {i+1}:\n{output_text[0]}\n" + "="*50 + "\n")
                
            except Exception as e:
                results.append(f"📷 图像 {i+1}: ❌ 分析失败 - {str(e)}\n" + "="*50 + "\n")
        
        return "".join(results)
    
    def compare_images(self, image1, image2, comparison_type):
        """图像对比"""
        if not self.is_loaded:
            return "❌ 请先加载模型！"
        
        if image1 is None or image2 is None:
            return "❌ 请上传两张图像进行对比！"
        
        try:
            if comparison_type == "相似性":
                prompt = "请对比这两张图片，分析它们的相似之处和不同之处。"
            elif comparison_type == "风格":
                prompt = "请对比这两张图片的艺术风格、色彩搭配和构图特点。"
            elif comparison_type == "内容":
                prompt = "请对比这两张图片的内容，分析它们描述的场景或主题。"
            else:
                prompt = "请对比这两张图片，提供详细的对比分析。"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image1},
                        {"type": "image", "image": image2},
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
            
            return f"🔍 对比分析结果:\n\n{output_text[0]}"
            
        except Exception as e:
            return f"❌ 对比分析失败: {str(e)}"
    
    def export_chat_history(self):
        """导出对话历史"""
        if not self.chat_history:
            return "❌ 没有对话历史可导出！"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
            
            # 保存为JSON格式
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            
            return f"✅ 对话历史已导出到: {filename}"
            
        except Exception as e:
            return f"❌ 导出失败: {str(e)}"
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
        return []

# 创建应用实例
app = AdvancedQwen3VLApp()

def create_advanced_interface():
    """创建高级Gradio界面"""
    
    with gr.Blocks(
        title="Qwen3-VL-8B-Instruct 高级界面",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
        }
        .stats-box {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🤖 多模态大语言模型智能分析助手
        
        **功能特色：**
        - 🖼️ 智能图像理解和对话 \t 📝 多语言OCR识别
        - 🔍 空间感知和情感分析 \t 💻 视觉编程代码生成
        - 📊 批量图像处理 \t 🔄 图像对比分析
        - 💾 对话历史导出 \t 📖 使用说明
        """)
        
        with gr.Tab("🚀 模型管理"):
            gr.Markdown("### 模型加载与管理")
            with gr.Row():
                with gr.Column():
                    load_btn = gr.Button("🔄 加载模型", variant="primary", size="lg")
                    status_text = gr.Textbox(
                        label="状态", 
                        value="⏳ 模型未加载，请点击加载模型按钮",
                        interactive=False
                    )
                with gr.Column():
                    model_info = gr.Textbox(
                        label="模型信息",
                        value=f"模型路径: {app.model_path}",
                        interactive=False
                    )
            
            load_btn.click(
                app.load_model,
                outputs=[status_text, load_btn]
            )
        
        with gr.Tab("💬 智能对话"):
            gr.Markdown("### 与图像进行智能对话")
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="上传图像",
                        type="pil",
                        height=400
                    )
                    
                    with gr.Accordion("🎛️ 生成参数", open=False):
                        max_tokens = gr.Slider(
                            minimum=50, maximum=2048, value=256,
                            label="最大生成长度"
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.7,
                            label="创造性 (Temperature)"
                        )
                        top_p = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.8,
                            label="Top-p"
                        )
                        top_k = gr.Slider(
                            minimum=1, maximum=100, value=20,
                            label="Top-k"
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
                        export_btn = gr.Button("💾 导出历史")
                    
                    stats_output = gr.Textbox(
                        label="生成统计",
                        interactive=False
                    )
            
            # 事件绑定
            send_btn.click(
                app.chat_with_image,
                inputs=[image_input, text_input, chatbot, max_tokens, temperature, top_p, top_k],
                outputs=[chatbot, text_input, stats_output]
            )
            
            text_input.submit(
                app.chat_with_image,
                inputs=[image_input, text_input, chatbot, max_tokens, temperature, top_p, top_k],
                outputs=[chatbot, text_input, stats_output]
            )
            
            clear_btn.click(
                app.clear_history,
                outputs=[chatbot]
            )
            
            export_btn.click(
                app.export_chat_history,
                outputs=[stats_output]
            )
        
        with gr.Tab("📊 批量分析"):
            gr.Markdown("### 批量图像分析")
            
            with gr.Row():
                with gr.Column():
                    batch_images = gr.File(
                        label="上传多个图像",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    
                    analysis_type = gr.Dropdown(
                        choices=["描述", "OCR", "空间分析", "情感分析"],
                        value="描述",
                        label="分析类型"
                    )
                    
                    batch_btn = gr.Button("🔍 开始批量分析", variant="primary")
                
                with gr.Column():
                    batch_result = gr.Textbox(
                        label="批量分析结果",
                        lines=20,
                        max_lines=30
                    )
            
            batch_btn.click(
                app.batch_analysis,
                inputs=[batch_images, analysis_type],
                outputs=[batch_result]
            )
        
        with gr.Tab("🔄 图像对比"):
            gr.Markdown("### 图像对比分析")
            
            with gr.Row():
                with gr.Column():
                    compare_image1 = gr.Image(
                        label="图像1",
                        type="pil",
                        height=200
                    )
                    compare_image2 = gr.Image(
                        label="图像2", 
                        type="pil",
                        height=200
                    )
                    
                    comparison_type = gr.Dropdown(
                        choices=["相似性", "风格", "内容", "综合"],
                        value="相似性",
                        label="对比类型"
                    )
                    
                    compare_btn = gr.Button("🔄 开始对比", variant="primary")
                
                with gr.Column():
                    compare_result = gr.Textbox(
                        label="对比结果",
                        lines=20,
                        max_lines=25
                    )
            
            compare_btn.click(
                app.compare_images,
                inputs=[compare_image1, compare_image2, comparison_type],
                outputs=[compare_result]
            )
        
        with gr.Tab("ℹ️ 使用说明"):
            gr.Markdown("""
            ## 📖 详细使用说明
            
            ### 🚀 模型管理
            - **加载模型**: 首次使用必须点击"加载模型"按钮
            - **模型路径**: `/data/storage1/wulin/models/qwen3-vl-8b-instruct`
            - **加载时间**: 通常需要10秒，请耐心等待
            
            ### 💬 智能对话
            - **图像上传**: 支持JPG、PNG等常见格式
            - **参数调节**: 
              - 最大生成长度: 控制回答的详细程度
              - 创造性: 数值越高回答越有创意
              - Top-p/Top-k: 控制生成的随机性
            - **多轮对话**: 支持基于图像的连续对话
            - **历史管理**: 可清空或导出对话历史
            
            ### 📊 批量分析
            - **多图像上传**: 一次可上传多张图像
            - **分析类型**:
              - 描述: 详细描述图像内容
              - OCR: 提取图像中的文字
              - 空间分析: 分析空间关系
              - 情感分析: 分析图像情感氛围
            
            ### 🔄 图像对比
            - **对比类型**:
              - 相似性: 分析图像的相似和差异
              - 风格: 对比艺术风格和色彩
              - 内容: 对比场景和主题
              - 综合: 全面的对比分析
            
            ### ⚠️ 注意事项
            - 确保有足够的内存（建议16GB+）
            - 支持GPU加速（自动检测）
            - 大图像可能需要更长的处理时间
            - 建议一次处理不超过10张图像
            """)
    
    return interface

def main():
    """主函数"""
    print("🚀 启动Qwen3-VL-8B-Instruct 高级Web界面...")
    
    # 创建界面
    interface = create_advanced_interface()
    
    # 启动服务
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # 使用不同端口避免冲突
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
