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
import csv
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
        self.chat_messages = []
        self.last_image = None
        self.last_saved_image_path = None
        self.last_image_digest = None
        self.last_ocr_markdown = None
        
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
    
    def _prepare_user_message(self, image, prompt):
        prompt_clean = (prompt or "").strip()
        resolved_image = image if image is not None else self.last_image
        if resolved_image is None:
            raise ValueError("❌ 请上传图像！")
        if not prompt_clean:
            raise ValueError("❌ 请输入问题！")
        if image is not None:
            self.last_image = image
        content = [
            {"type": "image", "image": resolved_image},
            {"type": "text", "text": prompt_clean},
        ]
        return prompt_clean, {"role": "user", "content": content}

    def _run_inference(self,
                       image,
                       prompt,
                       max_tokens,
                       temperature,
                       top_p,
                       top_k,
                       repetition_penalty,
                       prepared=None):
        if prepared is None:
            prompt_clean, user_message = self._prepare_user_message(image, prompt)
        else:
            prompt_clean, user_message = prepared
        messages = self.chat_messages + [user_message]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": True if temperature > 0 else False,
            "repetition_penalty": repetition_penalty
        }

        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generation_time = time.time() - start_time

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]

        assistant_message = {"role": "assistant", "content": [{"type": "text", "text": response}]}
        self.chat_messages.extend([user_message, assistant_message])
        return prompt_clean, response, generation_time

    def _clone_history(self, history):
        return [[turn[0], turn[1]] for turn in history]

    def _chunk_response(self, text, chunk_size=80):
        if not text:
            return []
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _parse_markdown_sections(self, markdown_text):
        """
        将 Markdown 文本拆分为 table/text 段，支持：
        - 管道表格（| a | b |）
        - HTML <table>（若存在）
        并在解析前对围栏代码块进行去围栏清洗，确保导出与渲染一致。
        """
        sections = []
        if not markdown_text:
            return sections
        
        # 1) 先去掉围栏，使得“代码块中的表格”也能被识别为可导出的内容
        cleaned_md = self._sanitize_markdown(markdown_text)
        
        # 2) 先尝试解析 HTML 表格（若模型输出了 <table>）
        html_tables = []
        try:
            from bs4 import BeautifulSoup  # 可选依赖
            soup = BeautifulSoup(cleaned_md, "html.parser")
            for t in soup.find_all("table"):
                headers = []
                header_row = t.find("tr")
                if header_row:
                    # 如果有 <th> 用 th；否则用首行的 td 作为 header
                    ths = header_row.find_all("th")
                    if ths:
                        headers = [th.get_text(strip=True) for th in ths]
                        data_rows = header_row.find_next_siblings("tr")
                    else:
                        tds = header_row.find_all("td")
                        headers = [td.get_text(strip=True) for td in tds]
                        data_rows = header_row.find_next_siblings("tr")
                rows = []
                for r in (data_rows or []):
                    cols = r.find_all(["td", "th"])
                    rows.append([c.get_text(strip=True) for c in cols])
                if headers or rows:
                    html_tables.append({"type": "table", "header": headers, "rows": rows})
        except Exception:
            # 如果 bs4 不在环境中，则略过 HTML 解析
            pass
        
        # 3) 解析管道表格
        lines = cleaned_md.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 管道表格判定：当前行和下一行构成 header + 分隔
            is_table = (
                stripped.startswith("|")
                and stripped.count("|") >= 2
                and i + 1 < len(lines)
                and set(lines[i + 1].replace("|", "").strip()) <= set("-: ")
                and lines[i + 1].strip().startswith("|")
            )
            
            if is_table:
                header = [cell.strip() for cell in stripped.strip("|").split("|")]
                i += 2  # 跳过 header 与分隔线
                rows = []
                while i < len(lines):
                    row_line = lines[i].strip()
                    if not (row_line.startswith("|") and row_line.count("|") >= 2):
                        break
                    row = [cell.strip() for cell in row_line.strip("|").split("|")]
                    rows.append(row)
                    i += 1
                sections.append({"type": "table", "header": header, "rows": rows})
                continue
            
            # 普通文本块（直到遇到下一个表格或文件结束）
            text_block = []
            while i < len(lines):
                current = lines[i]
                stripped_current = current.strip()
                next_is_table = (
                    stripped_current.startswith("|")
                    and stripped_current.count("|") >= 2
                    and i + 1 < len(lines)
                    and set(lines[i + 1].replace("|", "").strip()) <= set("-: ")
                    and lines[i + 1].strip().startswith("|")
                )
                if next_is_table:
                    break
                text_block.append(current)
                i += 1
                # 保留空行，改善段落分隔的可读性
                if i < len(lines) and lines[i] == "":
                    text_block.append(lines[i])
            
            text_content = "\n".join(text_block).strip("\n")
            if text_content:
                sections.append({"type": "text", "text": text_content})
        
        # 4) 若存在 HTML 表，优先把 HTML 表也加入（放在解析结果前面，避免遗漏）
        if html_tables:
            # 将 HTML 表插在最前面（也可根据需要合并/去重）
            sections = html_tables + sections
        
        return sections

    def chat_with_image(self, image, text, history, max_tokens, temperature, top_p, top_k, repetition_penalty: float = 1.0, presence_penalty: float = 1.5):
        """与图像对话（流式反馈）"""
        original_text = text

        if not self.is_loaded:
            yield history, original_text, gr.update(value="❌ 请先加载模型！", visible=True)
            return

        try:
            prepared = self._prepare_user_message(image, text)
        except ValueError as exc:
            yield history, original_text, gr.update(value=str(exc), visible=True)
            return

        prompt_clean, _ = prepared
        history_copy = self._clone_history(history)
        history_copy.append([f"👤 {prompt_clean}", "🤖 正在思考..."])
        yield self._clone_history(history_copy), original_text, gr.update(value="🤖 正在思考...", visible=True)

        try:
            _, response, generation_time = self._run_inference(
                image,
                text,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                prepared=prepared
            )
        except Exception as e:
            history_copy[-1][1] = f"❌ 生成失败: {str(e)}"
            self.chat_history = self._clone_history(history_copy)
            yield self._clone_history(history_copy), original_text, gr.update(value=f"❌ 错误: {str(e)}", visible=True)
            return

        assembled = ""
        chunks = self._chunk_response(response)
        if not chunks:
            chunks = [""]
        for chunk in chunks:
            assembled += chunk
            history_copy[-1][1] = f"🤖 {assembled}▌"
            yield self._clone_history(history_copy), original_text, gr.update(value=f"🤖 {assembled}▌", visible=True)

        stats = (
            f"⏱️ 生成时间: {generation_time:.2f}秒 | 📝 生成长度: {len(response)}字符"
            f" | ⚙️ 最大长度: {max_tokens}"
        )
        if max_tokens > 1024:
            stats += " | ⏳ 提示: 较大的最大长度可能延长生成时间"
        history_copy[-1][1] = f"🤖 {response}"
        self.chat_history = self._clone_history(history_copy)
        yield self._clone_history(history_copy), original_text, gr.update(value=stats, visible=True)

    def _sanitize_markdown(self, text: str) -> str:
        if not text:
            return ""
        s = text.strip()
        lines = s.splitlines()
        out = []
        in_fence = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence:
                out.append(line)
        cleaned = "\n".join(out).strip()
        return cleaned if cleaned else s

    def ocr_analysis(self, image, prompt: str = None):
        """OCR文字识别，可选自定义提示词"""
        if not self.is_loaded:
            return "❌ 请先加载模型！"
        default_prompt = (
            "请识别并提取这张图片中的所有文字内容，尽量还原原本样式，并标注语言类型。"
            " 请确保所有带样式或表格内容使用Markdown表格表示。"
        )
        effective_prompt = (prompt or "").strip() or default_prompt
        try:
            prompt_clean, response, _ = self._run_inference(
                image,
                effective_prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.8,
                top_k=50,
                repetition_penalty=1.0
            )
            cleaned = self._sanitize_markdown(response)
            self.chat_history.append([f"👤 {prompt_clean}", f"🤖 {cleaned}"])
            self.last_ocr_markdown = f"## OCR识别结果\n\n{cleaned}"
            return f"📝 OCR识别结果:\n\n{cleaned}"
        except ValueError as exc:
            return str(exc)
        except Exception as e:
            return f"❌ OCR识别失败: {str(e)}"

    def spatial_analysis(self, image, prompt: str = None):
        """空间感知分析，可选自定义提示词"""
        if not self.is_loaded:
            return "❌ 请先加载模型！"
        default_prompt = (
            "请分析这张图片中的空间关系，包括相对位置、视角、遮挡、深度与距离感，并给出整体布局描述。"
        )
        effective_prompt = (prompt or "").strip() or default_prompt
        try:
            prompt_clean, response, _ = self._run_inference(
                image,
                effective_prompt,
                max_tokens=768,
                temperature=0.7,
                top_p=0.8,
                top_k=50,
                repetition_penalty=1.0
            )
            self.chat_history.append([f"👤 {prompt_clean}", f"🤖 {response}"])
            return f"📐 空间分析结果:\n\n{response}"
        except ValueError as exc:
            return str(exc)
        except Exception as e:
            return f"❌ 空间分析失败: {str(e)}"

    def visual_coding(self, image, output_format: str = "HTML", prompt: str = None):
        """视觉编程生成代码，可选自定义提示词"""
        if not self.is_loaded:
            return "❌ 请先加载模型！"
        base_prompts = {
            "HTML": "请根据图片生成对应的HTML结构代码，包含必要的语义标签。",
            "CSS": "请为该图片对应的界面生成合理的CSS样式代码，包括布局与颜色。",
            "JavaScript": "请根据图片交互生成JavaScript代码示例，包含必要的事件与逻辑。",
            "Python": "请生成能复现该界面/布局的Python示例代码（如使用streamlit或flask的伪代码）。",
        }
        default_prompt = base_prompts.get(output_format, base_prompts["HTML"]) + " 请只输出代码，不要额外说明。"
        effective_prompt = (prompt or "").strip() or default_prompt
        try:
            prompt_clean, response, _ = self._run_inference(
                image,
                effective_prompt,
                max_tokens=1024,
                temperature=0.4,
                top_p=0.8,
                top_k=50,
                repetition_penalty=1.0
            )
            self.chat_history.append([f"👤 {prompt_clean}", f"🤖 {response}"])
            return response
        except ValueError as exc:
            return str(exc)
        except Exception as e:
            return f"❌ 视觉编程失败: {str(e)}"
    
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
        self.chat_messages = []
        self.last_image = None
        self.last_saved_image_path = None
        self.last_image_digest = None
        self.last_ocr_markdown = None
        if hasattr(self, "session_turn_image_paths"):
            self.session_turn_image_paths.clear()
        return []

    def export_last_ocr(self):
        if not self.last_ocr_markdown:
            return "❌ 没有可保存的文本样式，请先执行一次OCR识别！"

        export_dir = os.path.join("ocr_exports")
        os.makedirs(export_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sections = self._parse_markdown_sections(self.last_ocr_markdown)

        excel_path = os.path.join(export_dir, f"ocr_{timestamp}.xlsx")
        excel_note = ""
        try:
            from openpyxl import Workbook

            wb = Workbook()
            ws = wb.active
            ws.title = "表格1" if sections else "OCR文本"
            table_idx = 0
            for section in sections:
                if section["type"] == "table":
                    table_idx += 1
                    if table_idx > 1:
                        ws = wb.create_sheet(title=f"表格{table_idx}")
                    ws.append(section["header"])
                    for row in section["rows"]:
                        ws.append(row)
                elif section["type"] == "text" and section["text"]:
                    if table_idx > 0:
                        ws = wb.create_sheet(title=f"文本{table_idx}")
                    for line in section["text"].splitlines():
                        ws.append([line])
            if not sections:
                for line in self.last_ocr_markdown.splitlines():
                    ws.append([line])
            wb.save(excel_path)
        except Exception as exc:
            excel_path = os.path.join(export_dir, f"ocr_{timestamp}.csv")
            with open(excel_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["OCR Result"])
                for line in self.last_ocr_markdown.splitlines():
                    writer.writerow([line])
            excel_note = f"⚠️ Excel导出失败({exc})，已保存为CSV"

        json_path = os.path.join(export_dir, f"ocr_{timestamp}.json")
        json_content = {
            "markdown": self.last_ocr_markdown,
            "sections": sections,
        }
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(json_content, f, ensure_ascii=False, indent=2)

        message_lines = [
            "✅ 文本样式已保存：",
            f"- Excel: {excel_path}" + (f" ({excel_note})" if excel_note else ""),
            f"- JSON: {json_path}",
        ]
        return "\n".join(message_lines)

# 创建应用实例
app = AdvancedQwen3VLApp()

def create_advanced_interface():
    """创建高级Gradio界面"""
    
    with gr.Blocks(
        title="Qwen3-VL-8B-Instruct 高级界面",
        theme=gr.themes.Soft(),
        css="""
        :root {
            --radius-lg: 18px;
            --radius-md: 12px;
            --surface: #ffffff;
            --surface-muted: #f4f6fb;
            --surface-border: #e2e8f0;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
            --accent: #2563eb;
            --accent-soft: rgba(37, 99, 235, 0.12);
        }
        body {
            background: linear-gradient(140deg, #eef2ff 0%, #f8fafc 45%, #ffffff 100%);
            color: var(--text-primary);
        }
        .gradio-container {
            max-width: 1600px !important;
            margin: 0 auto;
            padding: 18px 22px 48px;
            color: var(--text-primary);
        }
        #advanced-header {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(96, 165, 250, 0.1));
            border: 1px solid rgba(37, 99, 235, 0.18);
            padding: 22px 26px;
            border-radius: 24px;
            box-shadow: 0 16px 32px rgba(15, 23, 42, 0.08);
            margin-bottom: 20px;
        }
        #advanced-header h1 {
            margin: 0 0 6px;
            font-size: 26px;
            font-weight: 600;
            letter-spacing: 0.2px;
            color: var(--text-primary);
        }
        #advanced-header p {
            margin: 0;
            font-size: 15px;
            color: var(--text-secondary);
        }
        .gradio-container .tabs {
            background: transparent;
            border: none;
        }
        .gradio-container .tabitem {
            border-radius: var(--radius-md);
            background: #f8fafc;
            border: 1px solid transparent;
            color: var(--text-secondary);
        }
        .gradio-container .tabitem.selected {
            border-color: rgba(37, 99, 235, 0.25);
            color: var(--text-primary);
            background: #ffffff;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.08);
        }
        #advanced-input-panel, #advanced-chat-panel, #advanced-secondary-panel {
            background: var(--surface);
            border-radius: 22px;
            padding: 20px 22px;
            border: 1px solid var(--surface-border);
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.06);
        }
        #advanced-chat-panel {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        #advanced-chatbot > .wrap {
            background: #f8fafc;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            padding: 8px 10px;
        }
        #advanced-chatbot .message {
            border-radius: 14px !important;
            padding: 12px 14px !important;
            font-size: 15px;
            line-height: 1.6;
            color: var(--text-primary);
        }
        #advanced-chatbot .message.user {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.18), rgba(59, 130, 246, 0.12));
            border: 1px solid rgba(37, 99, 235, 0.25);
            color: var(--text-primary);
            align-self: flex-end;
        }
        #advanced-chatbot .message.bot {
            background: #ffffff;
            border: 1px solid rgba(203, 213, 225, 0.9);
            color: var(--text-primary);
            align-self: flex-start;
        }
        #advanced-chatbot .message.bot .markdown ul {
            padding-left: 22px;
        }
        #advanced-query textarea {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: var(--surface);
            color: var(--text-primary);
            box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.05);
        }
        #advanced-query textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
        }
        #advanced-params .slider {
            padding: 6px 0;
        }
        #advanced-params .slider input[type="range"]::-webkit-slider-thumb {
            background: var(--accent);
        }
        #advanced-params .slider input[type="range"]::-moz-range-thumb {
            background: var(--accent);
        }
        #advanced-stats textarea {
            background: var(--accent-soft);
            border: 1px solid rgba(37, 99, 235, 0.2);
            border-radius: 14px;
            color: var(--text-primary);
            font-weight: 500;
        }
        .gradio-container .gradio-button.primary {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            border: none;
            color: #ffffff;
            font-weight: 600;
            box-shadow: 0 16px 28px rgba(37, 99, 235, 0.22);
        }
        .gradio-container .gradio-button.primary:hover {
            filter: brightness(1.03);
        }
        .gradio-container .gradio-button.secondary {
            background: rgba(37, 99, 235, 0.1);
            border: 1px solid rgba(37, 99, 235, 0.18);
            color: var(--text-primary);
        }
        .gradio-container textarea,
        .gradio-container input[type="text"],
        .gradio-container input[type="number"] {
            background: var(--surface);
            border: 1px solid rgba(148, 163, 184, 0.35);
            color: var(--text-primary);
            border-radius: 14px;
        }
        .gradio-container textarea:focus,
        .gradio-container input[type="text"]:focus,
        .gradio-container input[type="number"]:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
        }
        .gradio-container .slider > label,
        .gradio-container .checkbox-group > label,
        .gradio-container .radio-group > label {
            color: var(--text-secondary);
        }
        #ocr-md {
            max-height: 560px;
            overflow: auto;
            border: 1px solid rgba(148, 163, 184, 0.35);
            padding: 12px;
            border-radius: 14px;
            background: #ffffff;
        }
        """
    ) as interface:
        
        gr.HTML("""
        <section id="advanced-header">
            <h1>🤖 多模态大语言模型智能分析助手</h1>
            <p>升级后的页面布局与对话框样式，让图像问答与高级分析体验更沉浸、更高效。</p>
        </section>
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
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Group(elem_id="advanced-input-panel"):
                        gr.Markdown("### 图像与生成设置")
                        image_input = gr.Image(
                            label="上传图像",
                            type="pil",
                            height=390
                        )

                        with gr.Accordion("🎛️ 生成参数", open=False, elem_id="advanced-params"):
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
                    with gr.Group(elem_id="advanced-chat-panel"):
                        gr.Markdown("### 对话与输出")
                        chatbot = gr.Chatbot(
                            label=None,
                            height=600,
                            show_label=False,
                            type="tuples",
                            elem_id="advanced-chatbot",
                            render_markdown=True
                        )
                        text_input = gr.Textbox(
                            label=None,
                            placeholder="输入想了解的内容，按 Enter 或点击发送。",
                            lines=3,
                            elem_id="advanced-query"
                        )
                        send_btn = gr.Button("发送", variant="primary")

                        stats_output = gr.Markdown(
                            value="",
                            elem_id="advanced-stats"
                        )

                        with gr.Row():
                            clear_btn = gr.Button("🗑️ 清空历史", variant="secondary")
                            export_btn = gr.Button("📁 导出对话", variant="secondary")

        with gr.Tab("📝 OCR识别"):
            gr.Markdown("### 文字识别")
            with gr.Row():
                with gr.Column(scale=1):
                    ocr_image = gr.Image(
                        label="上传图像进行OCR识别",
                        type="pil",
                        height=320
                    )
                    ocr_btn = gr.Button("🔍 开始识别", variant="primary")
                with gr.Column(scale=1):
                    with gr.Row():
                        with gr.Column(scale=4):
                            ocr_md = gr.Markdown(
                                value="（识别结果会以 Markdown 渲染，包括表格）",
                                elem_id="ocr-md"
                            )
                        with gr.Column(scale=1):
                            save_style_btn = gr.Button("💾 导出样式", variant="secondary", interactive=False)
                            ocr_export_status = gr.Textbox(
                                label="导出状态",
                                interactive=False,
                                lines=4
                            )

        # 事件绑定
        def _run_ocr(image):
            result = app.ocr_analysis(image)
            can_save = bool(app.last_ocr_markdown)
            # 对显示内容：如果成功，展示渲染后的 Markdown；如果失败，将错误提示放到右侧状态框
            display_md = app.last_ocr_markdown if can_save else ""
            status = "" if can_save else result
            return display_md, gr.update(interactive=can_save), status

        def _clear_all():
            app.clear_history()
            return (
                [],
                "",
                gr.update(value="", visible=False),
                gr.update(value="（识别结果会以 Markdown 渲染，包括表格）", visible=True),
                gr.update(interactive=False),
                "",
            )

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
            _clear_all,
            outputs=[chatbot, text_input, stats_output, ocr_md, save_style_btn, ocr_export_status]
        )

        export_btn.click(
            app.export_chat_history,
            outputs=[stats_output]
        )

        ocr_btn.click(
            _run_ocr,
            inputs=[ocr_image],
            outputs=[ocr_md, save_style_btn, ocr_export_status]
        )

        save_style_btn.click(
            app.export_last_ocr,
            outputs=[ocr_export_status]
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
                    batch_result = gr.Markdown()
            
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
                    compare_result = gr.Markdown()
            
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
    
    interface.queue()

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
