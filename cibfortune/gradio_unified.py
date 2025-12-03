#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B-Instruct 统一Gradio界面
支持在同一界面内切换「通用版」与「专业版」，并提供触屏友好样式
"""

import os
import json
import inspect
import io
import hashlib
import time
import csv
import html
import numpy as np
from datetime import datetime
import shutil
import atexit
import gc

import gradio as gr
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from ocr_card_rag_api import CardOCRWithRAG

try:
    import torch
except Exception:
    torch = None

# 统一环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class AdvancedQwen3VLApp:
    """高级Qwen3-VL应用类"""

    def __init__(self):
        self.model = None
        self.processor = None
        """D:\cibfortune\Cibfortune\cibfortune\models\qwen3-vl-2b-instruct"""
        self.model_path = "/data/storage1/wulin/models/qwen3-vl-8b-instruct"
        self.is_loaded = False
        self.chat_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.chat_messages = []
        self.last_image = None
        self.last_saved_image_path = None
        self.last_image_digest = None
        self.last_ocr_markdown = None
        self.last_ocr_html = None
        # 卡证OCR多模态RAG组件
        self.card_rag_store = None
        self.card_rag_ready = False
        self.card_rag_dir = "rag_cards"
        # API 卡证OCR（RAG + Qwen API）
        self.card_api = None
<<<<<<< Updated upstream
        # 字段模板目录
        self.field_templates_dir = "card_field_templates"
=======
        self.card_api_feature_mode = "clip"
>>>>>>> Stashed changes

    def _ensure_card_rag_loaded(self):
        """懒加载卡证RAG图片库（若存在 rag_cards 目录），支持多种RAG实现方式。"""
        if self.card_rag_ready:
            return
        try:
            if not os.path.isdir(self.card_rag_dir):
                self.card_rag_ready = True  # 标记为已尝试，避免重复检查
                return
            
            # 优先尝试使用 multimodal_rag 模块
            try:
                from multimodal_rag import MultiModalDocumentLoader, MultiModalVectorStore
                loader = MultiModalDocumentLoader()
                docs = loader.load_images_from_folder(self.card_rag_dir)
                if not docs:
                    self.card_rag_ready = True
                    return
                store = MultiModalVectorStore(persist_directory="./multimodal_chroma_card")
                store.create_vector_store(docs)
                self.card_rag_store = store
                self.card_rag_ready = True
                print("✅ 使用multimodal_rag加载RAG图片库成功")
                return
            except Exception as e:
                print(f"⚠️ 使用multimodal_rag加载失败: {e}，尝试使用简化版RAG")
            
            # 如果multimodal_rag不可用，尝试使用SimpleRAGStore（从ocr_card_rag_api导入）
            try:
                from ocr_card_rag_api import SimpleRAGStore
                print("使用简化版RAG功能（基于卡面样式特征）...")
                store = SimpleRAGStore(use_style_features=True)
                store.load_images_from_folder(self.card_rag_dir)
                
                if not store.image_embeddings:
                    print("⚠️ RAG图片库为空")
                    self.card_rag_ready = True
                    return False
                
                self.card_rag_store = store
                self.card_rag_ready = True
                print(f"✅ 使用简化版RAG加载成功，共 {len(store.image_embeddings)} 张图片")
                return
            except Exception as e:
                print(f"⚠️ 使用简化版RAG加载失败: {e}")
            
            # 如果都失败了，标记为已尝试
            self.card_rag_ready = True
        except Exception as e:
            print(f"加载RAG图片库失败: {e}")
            self.card_rag_ready = True

    def _ensure_card_api_loaded(self):
        """懒加载卡证OCR（支持 在线API模式 + 离线RAG模式）"""
        if self.card_api is not None:
            return

        try:
            # 自动判断是否可用 API：环境变量中找 key
            env_key = os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
            has_api_key = bool(env_key)

            # 判断当前是否是本地模型路径（本地路径无需调用 API）
            is_local_model = isinstance(self.model_path, str) and os.path.isdir(self.model_path)

            # 决策：只要本地模型 or 无 key → 强制离线模式
            use_api = has_api_key and (not is_local_model)

            api = CardOCRWithRAG(
                api_key=env_key if use_api else None,
                model="qwen-vl-plus" if use_api else "local-offline",
                rag_image_dir=self.card_rag_dir,
                persist_directory="./multimodal_chroma_card",
                rag_feature_mode=self.card_api_feature_mode,
                use_api=use_api,   # ⭐ 决定是否调用 API
            )

            # 加载模型（离线模式不会初始化 OpenAI client）
            api.load_model()

            # 加载 RAG 图片库
            api.load_rag_library()

            self.card_api = api

            mode_str = "在线API模式" if use_api else "离线RAG模式"
            print(f"🟩 卡证OCR 已初始化（{mode_str}）")

        except Exception as e:
            print(f"❌ 卡证OCR初始化失败: {e}")
            self.card_api = None

    def set_card_api_feature_mode(self, selection: str):
        """更新API版卡证OCR所使用的RAG特征模式。"""
        normalized = "clip" if selection and "clip" in selection.lower() else "style"
        if normalized != self.card_api_feature_mode:
            self.card_api_feature_mode = normalized
            # 重新初始化客户端，使新设置生效
            self.card_api = None

    def _rag_search_card(self, image, top_k: int = 3):
        """
        对输入图片进行RAG检索，返回相似图片信息（与ocr_card_rag_api.py中的逻辑一致）
        
        Args:
            image: 输入图片（PIL Image）
            top_k: 返回最相似的k张图片
            
        Returns:
            相似图片列表，每个元素包含 {filename, similarity, metadata}
        """
        if not self.card_rag_store or not hasattr(self.card_rag_store, "image_embeddings"):
            return []
            
        try:
            # 生成查询图片的嵌入向量
            # 兼容两种实现：MultiModalVectorStore 使用 .embeddings.embed_image，SimpleRAGStore 直接使用 .embed_image
            if hasattr(self.card_rag_store, "embeddings") and hasattr(self.card_rag_store.embeddings, "embed_image"):
                # 使用 MultiModalVectorStore
                query_emb = self.card_rag_store.embeddings.embed_image(image)
            elif hasattr(self.card_rag_store, "embed_image"):
                # 使用 SimpleRAGStore
                query_emb = self.card_rag_store.embed_image(image)
            else:
                print("⚠️ RAG存储不支持embed_image方法")
                return []
            
            # 计算与图片库中所有图片的相似度
            similarities = []
            # 如果SimpleRAGStore有compute_similarity方法，使用它（支持样式相似度）
            use_compute_similarity = hasattr(self.card_rag_store, "compute_similarity")
            
            # 确保查询向量的维度
            query_dim = len(query_emb) if hasattr(query_emb, '__len__') else query_emb.shape[0] if hasattr(query_emb, 'shape') else 0
            
            for idx, emb in enumerate(self.card_rag_store.image_embeddings):
                try:
                    # 检查维度是否匹配
                    emb_dim = len(emb) if hasattr(emb, '__len__') else emb.shape[0] if hasattr(emb, 'shape') else 0
                    
                    if query_dim != emb_dim:
                        # 维度不匹配，跳过或使用默认相似度
                        print(f"⚠️ 特征维度不匹配: 查询向量={query_dim}, 图片库向量={emb_dim}，跳过该图片")
                        continue
                    
                    if use_compute_similarity:
                        # 使用样式相似度或CLIP相似度（根据SimpleRAGStore的配置）
                        similarity = self.card_rag_store.compute_similarity(query_emb, emb)
                    else:
                        # 使用余弦相似度（MultiModalVectorStore）
                        dot_product = np.dot(query_emb, emb)
                        norm_query = np.linalg.norm(query_emb)
                        norm_emb = np.linalg.norm(emb)
                        denom = norm_query * norm_emb + 1e-8
                        similarity = float(dot_product / denom) if denom > 0 else 0.0
                    similarities.append((similarity, idx))
                except Exception as e:
                    # 如果计算相似度时出错，跳过该图片
                    print(f"⚠️ 计算相似度失败（图片{idx}）: {str(e)}")
                    continue
            
            # 排序并取Top-K
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_results = []
            
            for sim, idx in similarities[:top_k]:
                if idx < len(self.card_rag_store.image_metadatas):
                    meta = self.card_rag_store.image_metadatas[idx]
                    filename = meta.get("filename") or os.path.basename(meta.get("source", "")) or f"图片{idx+1}"
                    top_results.append({
                        "filename": filename,
                        "similarity": sim,
                        "metadata": meta
                    })
                    
            return top_results
            
        except Exception as e:
            print(f"⚠️ RAG检索失败: {str(e)}")
            return []

    def _build_enhanced_prompt_card(self, base_prompt: str, rag_results: list, custom_prompt: str = None):
        """
        构建增强后的提示词（包含RAG检索结果，与ocr_card_rag_api.py中的逻辑一致）
        
        Args:
            base_prompt: 基础提示词
            rag_results: RAG检索结果
            custom_prompt: 用户自定义提示词
            
        Returns:
            增强后的完整提示词
        """
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = base_prompt
            
        # 如果有RAG检索结果，添加到提示词中
        if rag_results:
            rag_context = "\n基于图片库检索到的相似卡证：\n"
            for rank, result in enumerate(rag_results, 1):
                filename = result["filename"]
                similarity = result["similarity"]
                rag_context += f"- 卡面{rank}: {filename} | 相似度={similarity:.3f}\n"
            rag_context += "\n"
            filenames = [result["filename"].split(".")[0] for result in rag_results]
            banks = [filename.split("_")[0] for filename in filenames]
            prompt = rag_context + prompt
            prompt = prompt + (
                f"6. 如果是银行卡且字段列表包含'卡面类型'，则按照以下规则填充：\n"
                f"  - 基于图片库检索到的相似卡证结果{filenames}，填充\"卡面类型\"字段。字段值规则如下：\n"
                f"       -**禁止**自定义、生成、猜测或编造新的卡面类型值。\n"
                f"       -当出现任何不确定、模糊或不匹配情况时，\"卡面类型\"字段的值**必须且只能为\"其他\"**。\n"
                f"       -若识别出的\"发卡行\"字段的值存在与{banks}中银行名称相同的情况，"
                f"则\"卡面类型\"字段的值只能从{filenames}中**严格选择一个**。\n"
            )
            
        return prompt

    def load_model(self, progress=gr.Progress()):
        """加载模型"""
        if self.is_loaded:
            return "✅ 模型已经加载完成！", gr.update(interactive=True)

        if torch is None:
            return "❌ 模型加载失败: 未检测到PyTorch，请先安装。", gr.update(interactive=False)

        try:
            progress(0.1, desc="检查模型路径...")
            if not os.path.exists(self.model_path):
                return f"❌ 模型路径不存在: {self.model_path}", gr.update(interactive=False)

            progress(0.3, desc="加载模型...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype="auto",
                device_map="cuda",
                load_in_4bit=False,
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

    @staticmethod
    def _text_to_html_block(text: str) -> str:
        if not text:
            return ""
        escaped = html.escape(text)
        replaced = escaped.replace("\n", "<br>")
        return f'<div class="ocr-text">{replaced}</div>'

    @staticmethod
    def _table_to_html_block(header, rows) -> str:
        header = header or []
        rows = rows or []
        thead = ""
        if header:
            header_cells = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in header)
            thead = f"<thead><tr>{header_cells}</tr></thead>"
        body_rows = []
        for row in rows:
            row_cells = "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
            body_rows.append(f"<tr>{row_cells}</tr>")
        tbody = f"<tbody>{''.join(body_rows)}</tbody>" if body_rows else "<tbody></tbody>"
        return f'<table class="ocr-table">{thead}{tbody}</table>'

    def _render_sections_as_html(self, markdown_text: str) -> str:
        if not markdown_text:
            return ""
        sections = self._parse_markdown_sections(markdown_text)
        if not sections:
            escaped = html.escape(markdown_text.strip())
            return f"<pre>{escaped}</pre>" if escaped else ""
        blocks = []
        for section in sections:
            if section.get("type") == "table":
                blocks.append(self._table_to_html_block(section.get("header"), section.get("rows")))
            elif section.get("type") == "text":
                blocks.append(self._text_to_html_block(section.get("text", "")))
        return '<div class="ocr-preview">' + "".join(blocks) + "</div>"

    def chat_with_image(self, image, text, history, max_tokens, temperature, top_p, top_k, repetition_penalty: float = 1.0, presence_penalty: float = 1.5):
        """与图像对话（流式反馈）"""
        original_text = text

        if not self.is_loaded:
            yield history, original_text, "❌ 请先加载模型！"
            return

        try:
            prepared = self._prepare_user_message(image, text)
        except ValueError as exc:
            yield history, original_text, str(exc)
            return

        prompt_clean, _ = prepared
        history_copy = self._clone_history(history)
        history_copy.append([f"👤 {prompt_clean}", "🤖 正在思考..."])
        yield self._clone_history(history_copy), original_text, "🤖 正在思考..."

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
            yield self._clone_history(history_copy), original_text, f"❌ 错误: {str(e)}"
            return

        assembled = ""
        chunks = self._chunk_response(response)
        if not chunks:
            chunks = [""]
        for chunk in chunks:
            assembled += chunk
            history_copy[-1][1] = f"🤖 {assembled}▌"
            yield self._clone_history(history_copy), original_text, f"🤖 {assembled}▌"

        stats = (
            f"⏱️ 生成时间: {generation_time:.2f}秒 | 📝 生成长度: {len(response)}字符"
            f" | ⚙️ 最大长度: {max_tokens}"
        )
        if max_tokens > 1024:
            stats += " | ⏳ 提示: 较大的最大长度可能延长生成时间"
        history_copy[-1][1] = f"🤖 {response}"
        self.chat_history = self._clone_history(history_copy)
        yield self._clone_history(history_copy), original_text, stats

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
            self.last_ocr_html = "<h2>OCR识别结果</h2>" + self._render_sections_as_html(cleaned)
            return f"📝 OCR识别结果:\n\n{cleaned}"
        except ValueError as exc:
            return str(exc)
        except Exception as e:
            return f"❌ OCR识别失败: {str(e)}"

    def ocr_card(self, image, prompt: str = None):
        """卡证OCR识别：身份证/银行卡/驾驶证等结构化提取（使用本地模型，流程与API版本一致）"""
        if not self.is_loaded:
            return "❌ 请先加载模型！"
        
        # 使用与ocr_card_api相同的默认提示词
        default_prompt = (
            "你是专业的卡证OCR引擎，请对输入图片进行结构化识别，并仅输出Markdown表格。\n"
            "\n"
            "任务要求如下：\n"
            "\n"
            "1. 识别卡证类型：只允许从以下类别中选择一种：\n"
            "   - 身份证 / 银行卡 / 驾驶证 / 护照 / 工牌 / 其他。\n"
            "   Markdown表格中添加\"卡证类型\"字段，并用类别选择赋值。\n"
            "   **重要**：如果识别为银行卡，必须严格遵守第3条银行卡特殊要求！\n"
            "\n"
            "2. 输出格式：\n"
            "   - 以Markdown表格形式输出所有识别出的关键字段及其对应的值。\n"
            "   - 若字段中包含\"卡号\"，请确保该字段的值仅包含数字。\n"
            "   - 不要使用代码块标记符号（例如 ``` ）。\n"
            "\n"
            "3. 银行卡特殊要求（必须严格遵守）：\n"
            "   如果识别的卡证类型是银行卡，必须在Markdown表格的最后额外添加一个字段：\n"
            "   - 字段名：卡面类型（必须添加，不可省略）。\n"
            "   - 基于图片库检索到的相似卡证结果，填充\"卡面类型\"字段。字段值规则如下：\n"
            "       ① 当出现任何不确定、模糊或不匹配情况时，\"卡面类型\"字段的值**必须且只能为\"其他\"**，不得填写相似图片名或其他文本。\n"
            "       ② 若识别出的\"发卡行\"字段的值与这些相似卡证文件名中`_`前面的银行名称相同，"
            "则\"卡面类型\"字段的值只能从相似卡证文件名中**严格选择一个**，格式为`银行名称_卡面类型`，去掉文件后缀名，如`中国银行_visa卡`。\n"
            "       ③ 禁止自定义、生成、猜测或编造新的卡面类型值。任何不存在基于图片库检索到的相似卡证文件名的值都视为错误。\n"
            "   **重要提醒**：银行卡的Markdown表格必须包含\"卡面类型\"字段，这是强制要求，不能省略！\n"
            "   - 如果不是银行卡，则不添加\"卡面类型\"字段。\n"
            "\n"
            "4. 输出限制：\n"
            "   - 最终输出只包含Markdown表格。\n"
            "   - 禁止输出任何其他文字或解释性内容。\n"
            "   - 如果是银行卡，表格中必须包含\"卡面类型\"字段，否则输出不完整。\n"
        )

        effective_prompt = (prompt or "").strip() or default_prompt

        # RAG检索（使用与API版本相同的逻辑）
        rag_results = []
        try:
            self._ensure_card_rag_loaded()
            if self.card_rag_store and getattr(self.card_rag_store, "image_embeddings", None):
                rag_results = self._rag_search_card(image, top_k=3)
        except Exception as e:
            print(f"⚠️ RAG检索失败: {str(e)}")
            rag_results = []

        # 在终端输出RAG相似度匹配结果（与API版本一致）
        if rag_results:
            print("\n" + "=" * 60)
            print("📊 RAG相似度匹配结果")
            print("=" * 60)
            print(f"找到 {len(rag_results)} 张相似图片：\n")
            for i, r in enumerate(rag_results, 1):
                filename = r.get("filename", "未知")
                similarity = r.get("similarity", 0.0)
                print(f"  {i}. {filename}")
                print(f"     相似度: {similarity:.4f} ({similarity*100:.2f}%)")
            print("=" * 60 + "\n")
        else:
            print("\n⚠️ 未找到相似图片\n")

        # 构建增强提示词（使用与API版本相同的逻辑）
        enhanced_prompt = self._build_enhanced_prompt_card(
            base_prompt=default_prompt,
            rag_results=rag_results,
            custom_prompt=effective_prompt if (prompt or "").strip() else None
        )

        # 在终端输出发送给模型的完整prompt（与API版本一致）
        print("\n" + "=" * 80)
        print("📝 发送给模型的完整Prompt")
        print("=" * 80)
        print(enhanced_prompt)
        print("=" * 80 + "\n")

        try:
            # 使用本地模型进行推理
            prompt_clean, response, _ = self._run_inference(
                image,
                enhanced_prompt,
                max_tokens=1024,
                temperature=0.3,
                top_p=0.8,
                top_k=40,
                repetition_penalty=1.05
            )
            cleaned = self._sanitize_markdown(response)
            self.chat_history.append([f"👤 {prompt_clean}", f"🤖 {cleaned}"])
            self.last_ocr_markdown = f"## 卡证OCR识别结果\n\n{cleaned}"
            self.last_ocr_html = "<h2>卡证OCR识别结果</h2>" + self._render_sections_as_html(cleaned)
            return f"🪪 卡证OCR识别结果:\n\n{cleaned}"
        except ValueError as exc:
            return str(exc)
        except Exception as e:
            return f"❌ 卡证OCR识别失败: {str(e)}"

    def ocr_card_api(self, image, prompt: str = None):
        """卡证OCR识别（API调用 + RAG增强）"""
        # 注：如无需强制本地模型加载，可移除此判断
        try:
            self._ensure_card_api_loaded()
            if self.card_api is None:
                return "�?卡证OCR API初始化失败"
            default_prompt = (
                "你是专业的卡证OCR引擎，请对输入图片进行结构化识别，并仅输出Markdown表格。\n"
                "\n"
                "任务要求如下：\n"
                "\n"
            "1. 识别卡证类型：只允许从以下类别中选择一种：\n"
            "   - 身份证 / 银行卡 / 驾驶证 / 护照 / 工牌 / 其他。\n"
            "   Markdown表格中添加“卡证类型”字段，并用类别选择赋值。\n"
            "   **重要**：如果识别为银行卡，必须严格遵守第3条银行卡特殊要求！\n"
            "\n"
            "2. 输出格式：\n"
            "   - 以Markdown表格形式输出所有识别出的关键字段及其对应的值。\n"
            "   - 若字段中包含“卡号”，请确保该字段的值仅包含数字。\n"
            "   - 不要使用代码块标记符号（例如 ``` ）。\n"
            "\n"
            "3. 银行卡特殊要求（必须严格遵守）：\n"
            "   如果识别的卡证类型是银行卡，必须在Markdown表格的最后额外添加一个字段：\n"
            "   - 字段名：卡面类型（必须添加，不可省略）。\n"
            "   - 基于图片库检索到的相似卡证结果，填充“卡面类型”字段。字段值规则如下：\n"
            "       ① 当出现任何不确定、模糊或不匹配情况时，“卡面类型”字段的值**必须且只能为“其他”**，不得填写相似图片名或其他文本。\n"
            "       ② 若识别出的“发卡行”字段的值与这些相似卡证文件名中`_`前面的银行名称相同，"
            "则“卡面类型”字段的值只能从相似卡证文件名中**严格选择一个**，格式为`银行名称_卡面类型`，去掉文件后缀名，如`中国银行_visa卡`。\n"
            "       ③ 禁止自定义、生成、猜测或编造新的卡面类型值。任何不存在基于图片库检索到的相似卡证文件名的值都视为错误。\n"
            "   **重要提醒**：银行卡的Markdown表格必须包含“卡面类型”字段，这是强制要求，不能省略！\n"
            "   - 如果不是银行卡，则不添加“卡面类型”字段。\n"
            "\n"
            "4. 输出限制：\n"
            "   - 最终输出只包含Markdown表格。\n"
            "   - 禁止输出任何其他文字或解释性内容。\n"
            "   - 如果是银行卡，表格中必须包含“卡面类型”字段，否则输出不完整。\n"
            )

            effective_prompt = (prompt or "").strip() or default_prompt
            result = self.card_api.recognize_card(
                image,
                custom_prompt=effective_prompt,
                use_rag=True,
            )
            if not result.get("success"):
                return f"�?卡证OCR API调用失败: {result.get('error') or '未知错误'}"

            # 在终端输出RAG相似度匹配结果
            rag_info = result.get("rag_info")
            if rag_info and rag_info.get("enabled") and rag_info.get("results"):
                print("\n" + "=" * 60)
                print("📊 RAG相似度匹配结果")
                print("=" * 60)
                print(f"找到 {len(rag_info['results'])} 张相似图片：\n")
                for i, r in enumerate(rag_info["results"], 1):
                    filename = r.get("filename", "未知")
                    similarity = r.get("similarity", 0.0)
                    print(f"  {i}. {filename}")
                    print(f"     相似度: {similarity:.4f} ({similarity*100:.2f}%)")
                print("=" * 60 + "\n")
            elif rag_info and not rag_info.get("enabled"):
                print(f"\n⚠️ RAG未启用: {rag_info.get('reason', '未知原因')}\n")
            else:
                print("\n⚠️ 未找到相似图片\n")

            cleaned = self._sanitize_markdown(result.get("result") or "")
            self.last_ocr_markdown = f"## 卡证OCR识别（API）结果\n\n{cleaned}"
            self.last_ocr_html = "<h2>卡证OCR识别（API）结果</h2>" + self._render_sections_as_html(cleaned)
            return f"🪪 卡证OCR识别（API）结果:\n\n{cleaned}"
        except Exception as e:
            return f"�?卡证OCR API识别失败: {str(e)}"

    def ocr_receipt(self, image, prompt: str = None):
        """票据OCR识别：发票/小票等表格与关键项解析"""
        if not self.is_loaded:
            return "❌ 请先加载模型！"
        default_prompt = (
            "你是发票/小票OCR专家。请解析图片中的票据并输出：\n"
            "- 以Markdown表格给出关键信息：票据类型、开票日期、发票代码、发票号码、校验码、购买方、销售方、税号、项目、数量、单价、金额、税率、税额、合计金额(含税/不含税)；\n"
            "- 若检测到多行项目，请以表格形式逐行列出；\n"
            "- 表格下方给出识别置信度与可疑项提示；\n"
            "- 不要使用围栏代码块，保持Markdown可渲染。"
        )
        effective_prompt = (prompt or "").strip() or default_prompt
        try:
            prompt_clean, response, _ = self._run_inference(
                image,
                effective_prompt,
                max_tokens=1536,
                temperature=0.2,
                top_p=0.8,
                top_k=40,
                repetition_penalty=1.05
            )
            cleaned = self._sanitize_markdown(response)
            self.chat_history.append([f"👤 {prompt_clean}", f"🤖 {cleaned}"])
            self.last_ocr_markdown = f"## 票据OCR识别结果\n\n{cleaned}"
            self.last_ocr_html = "<h2>票据OCR识别结果</h2>" + self._render_sections_as_html(cleaned)
            return f"🧾 票据OCR识别结果:\n\n{cleaned}"
        except ValueError as exc:
            return str(exc)
        except Exception as e:
            return f"❌ 票据OCR识别失败: {str(e)}"

    def ocr_agreement(self, image, prompt: str = None):
        """协议OCR识别：合同/协议段落与条款解析"""
        if not self.is_loaded:
            return "❌ 请先加载模型！"
        default_prompt = (
            "你是合同/协议OCR与条款解析助手。请完成：\n"
            "1) 识别全文，保持段落结构；\n"
            "2) 以Markdown表格提炼关键信息：合同名称、甲方、乙方、签署日期、生效日期、终止日期、金额/币种、违约条款、争议解决、签章情况；\n"
            "3) 如有编号的条款，保留编号并逐条列出；\n"
            "4) 在末尾给出“风险提示”列表（如空白处、涂改处、关键要素缺失等）；\n"
            "5) 不要输出围栏代码块。"
        )
        effective_prompt = (prompt or "").strip() or default_prompt
        try:
            prompt_clean, response, _ = self._run_inference(
                image,
                effective_prompt,
                max_tokens=2048,
                temperature=0.3,
                top_p=0.8,
                top_k=40,
                repetition_penalty=1.05
            )
            cleaned = self._sanitize_markdown(response)
            self.chat_history.append([f"👤 {prompt_clean}", f"🤖 {cleaned}"])
            self.last_ocr_markdown = f"## 协议OCR识别结果\n\n{cleaned}"
            self.last_ocr_html = "<h2>协议OCR识别结果</h2>" + self._render_sections_as_html(cleaned)
            return f"📄 协议OCR识别结果:\n\n{cleaned}"
        except ValueError as exc:
            return str(exc)
        except Exception as e:
            return f"❌ 协议OCR识别失败: {str(e)}"

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
        self.last_ocr_html = None
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


DEFAULT_TASK_PROMPTS = {
    "任务问答": "请根据图片完成指定任务。",
    "OCR识别": "请识别并提取这张图片中的所有文字内容，并标注语言类型。请确保所有带样式或表格内容使用Markdown表格表示。",
    "卡证OCR识别": "请进行卡证类识别并以Markdown表格输出关键字段（如姓名、证件号、有效期、卡号等）",
    "票据OCR识别": "请解析发票/小票等票据，输出关键信息和多行项目表格，并在下方给出置信度与可疑项。",
    "协议OCR识别": "请提取合同/协议关键信息（甲乙方、日期、金额、条款等），保留段落与条款编号，并在末尾给出风险提示。",
    "空间分析": "请分析这张图片中的空间关系，包括相对位置、视角、遮挡、深度与距离感，并给出整体布局描述。",
    "情感分析": "请分析这张图片传达的情感或氛围，并说明理由。",
}

VISUAL_CODING_PROMPTS = {
    "HTML": "请根据图片生成对应的HTML结构代码，包含必要的语义标签。请只输出代码，不要额外说明。",
    "CSS": "请为该图片对应的界面生成合理的CSS样式代码，包括布局与颜色。请只输出代码，不要额外说明。",
    "JavaScript": "请根据图片交互生成JavaScript代码示例，包含必要的事件与逻辑。请只输出代码，不要额外说明。",
    "Python": "请生成能复现该界面/布局的Python示例代码（如使用streamlit或flask的伪代码）。请只输出代码，不要额外说明。",
}


def _plain_text_to_html(text: str) -> str:
    if not text:
        return ""
    escaped = html.escape(str(text))
    replaced = escaped.replace("\n", "<br>")
    return f'<div class="stats-text">{replaced}</div>'


def _get_default_prompt(task: str, code_format: str = None) -> str:
    if task == "视觉编程":
        fmt = code_format or "HTML"
        return VISUAL_CODING_PROMPTS.get(fmt, VISUAL_CODING_PROMPTS["HTML"])
    return DEFAULT_TASK_PROMPTS.get(task, DEFAULT_TASK_PROMPTS["任务问答"])


# 单例应用
app = AdvancedQwen3VLApp()

# 会话级图片保存目录与轨迹
IMAGE_SAVE_ROOT = "chat_history/images"
SESSION_IMAGE_DIR = os.path.join(IMAGE_SAVE_ROOT, getattr(app, "session_id", datetime.now().strftime("%Y%m%d_%H%M%S")))
os.makedirs(SESSION_IMAGE_DIR, exist_ok=True)
app.session_turn_image_paths = []  # 与对话轮次对齐的图片路径（无图则为 None）


def _toggle_mode(mode, current_task, current_code_format):
    """根据模式切换组件可见性，并预填充默认提示。"""
    is_pro = (mode == "专业版")
    task_value = current_task if is_pro else "任务问答"
    code_visible = is_pro and task_value == "视觉编程"
    text_value = _get_default_prompt(task_value, current_code_format) if is_pro else ""
    rag_visible = is_pro and task_value == "卡证OCR识别（API）"
    return (
        gr.update(visible=is_pro),                       # adv_params_box
        gr.update(visible=is_pro),                       # stats_output
        gr.update(visible=is_pro),                       # tab_batch
        gr.update(visible=is_pro),                       # tab_compare
        gr.update(visible=is_pro, value=task_value),     # pro_task dropdown
        gr.update(visible=code_visible),                 # code_format dropdown
        gr.update(visible=rag_visible),                  # rag_feature_selector
        gr.update(value=text_value),                     # text_input prompt
    )


def _toggle_task(task, code_format):
    """任务切换时调整代码下拉可见性并预填提示。"""
    is_visual = (task == "视觉编程")
    prompt = _get_default_prompt(task, code_format)
    code_kwargs = {"visible": is_visual}
    if is_visual and not code_format:
        code_kwargs["value"] = "HTML"
    rag_visible = (task == "卡证OCR识别（API）")
    return gr.update(**code_kwargs), gr.update(value=prompt), gr.update(visible=rag_visible)


def _update_code_prompt(task, code_format):
    if task != "视觉编程":
        return gr.update()
    return gr.update(value=_get_default_prompt(task, code_format))


def handle_unified_chat(image,
                        text,
                        history,
                        max_tokens,
                        temperature,
                        top_p,
                        top_k,
                        mode,
                        pro_task,
                        rag_feature_mode,
                        code_format,
                        repetition_penalty,
                        presence_penalty):
    """统一的发送处理：
    - 通用版：普通问答
    - 专业版：按任务分派到不同方法
    返回: history, cleared_text, stats
    """
    user_text = (text or "").strip()
    saved_image_path = None
    image_digest = None
    if image is not None:
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_digest = hashlib.md5(buffer.getvalue()).hexdigest()
        except Exception:
            image_digest = None

        should_save = image_digest is None or image_digest != getattr(app, "last_image_digest", None)
        if should_save:
            try:
                ts = datetime.now().strftime("%H%M%S%f")
                saved_image_path = os.path.join(SESSION_IMAGE_DIR, f"img_{ts}.png")
                image.save(saved_image_path)
                if image_digest is not None:
                    app.last_image_digest = image_digest
            except Exception:
                saved_image_path = None

    prev_turns = len(history)
    image_recorded = False

    def record_image_path():
        nonlocal image_recorded
        if image_recorded:
            return
        if saved_image_path and saved_image_path != getattr(app, "last_saved_image_path", None):
            app.session_turn_image_paths.append(saved_image_path)
            app.last_saved_image_path = saved_image_path
            if image_digest is not None:
                app.last_image_digest = image_digest
        else:
            existing = getattr(app, "last_saved_image_path", None)
            app.session_turn_image_paths.append(existing)
        image_recorded = True

    try:
        if mode == "通用版":
            effective_prompt = user_text
            chat_result = app.chat_with_image(
                image,
                effective_prompt,
                history,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                presence_penalty
            )

            if inspect.isgenerator(chat_result):
                for out_history, cleared, stats in chat_result:
                    if not image_recorded and len(out_history) > prev_turns:
                        record_image_path()
                    app.chat_history = out_history
                    button_update = gr.update(interactive=bool(app.last_ocr_markdown))
                    stats_update = gr.update(value=_plain_text_to_html(stats), visible=True)
                    yield out_history, cleared, stats_update, button_update, gr.update(value="", visible=True)
            else:
                out_history, cleared, stats = chat_result
                if not image_recorded and len(out_history) > prev_turns:
                    record_image_path()
                app.chat_history = out_history
                button_update = gr.update(interactive=bool(app.last_ocr_markdown))
                stats_update = gr.update(value=_plain_text_to_html(stats), visible=True)
                yield out_history, cleared, stats_update, button_update, gr.update(value="", visible=True)

        else:
            task = pro_task or "任务问答"
            if task == "OCR识别":
                if image is None:
                    stats_update = gr.update(value=_plain_text_to_html("❌ 请上传图像！"), visible=True)
                    yield history, text, stats_update, gr.update(interactive=False), "❌ 请上传图像！"
                    return

                result = app.ocr_analysis(image)

                if result.startswith("❌"):
                    stats_update = gr.update(value="", visible=True)
                    yield history, text, stats_update, gr.update(interactive=False), result
                    return

                prompt_text = user_text if user_text else _get_default_prompt(task, code_format)
                updated_history = history + [[f"👤 {prompt_text}", result]]
                app.chat_history = updated_history
                if not image_recorded:
                    record_image_path()
                ocr_preview = app.last_ocr_html or _plain_text_to_html(app.last_ocr_markdown or "")
                stats_update = gr.update(value=ocr_preview, visible=True)
                status_update = "✅ OCR识别完成，可导出样式"
                yield updated_history, "", stats_update, gr.update(interactive=bool(app.last_ocr_markdown)), status_update
                return

            if task == "卡证OCR识别（API）":
                app.set_card_api_feature_mode(rag_feature_mode)
                if image is None:
                    stats_update = gr.update(value=_plain_text_to_html("❌ 请上传图像！"), visible=True)
                    yield history, text, stats_update, gr.update(interactive=False), "❌ 请上传图像！"
                    return
                result = app.ocr_card_api(image)
                if result.startswith("❌"):
                    stats_update = gr.update(value="", visible=True)
                    yield history, text, stats_update, gr.update(interactive=False), result
                    return
                prompt_text = user_text if user_text else _get_default_prompt("卡证OCR识别", code_format)
                updated_history = history + [[f"👤 {prompt_text}", result]]
                app.chat_history = updated_history
                if not image_recorded:
                    record_image_path()
                ocr_preview = app.last_ocr_html or _plain_text_to_html(app.last_ocr_markdown or "")
                stats_update = gr.update(value=ocr_preview, visible=True)
                yield updated_history, "", stats_update, gr.update(interactive=bool(app.last_ocr_markdown)), "✅ 卡证OCR识别(API)完成，可导出样式"
                return

            if task == "卡证OCR识别":
                if image is None:
                    stats_update = gr.update(value=_plain_text_to_html("❌ 请上传图像！"), visible=True)
                    yield history, text, stats_update, gr.update(interactive=False), "❌ 请上传图像！"
                    return
                result = app.ocr_card(image)
                if result.startswith("❌"):
                    stats_update = gr.update(value="", visible=True)
                    yield history, text, stats_update, gr.update(interactive=False), result
                    return
                prompt_text = user_text if user_text else _get_default_prompt(task, code_format)
                updated_history = history + [[f"👤 {prompt_text}", result]]
                app.chat_history = updated_history
                if not image_recorded:
                    record_image_path()
                ocr_preview = app.last_ocr_html or _plain_text_to_html(app.last_ocr_markdown or "")
                stats_update = gr.update(value=ocr_preview, visible=True)
                yield updated_history, "", stats_update, gr.update(interactive=bool(app.last_ocr_markdown)), "✅ 卡证OCR识别完成，可导出样式"
                return

            if task == "票据OCR识别":
                if image is None:
                    stats_update = gr.update(value=_plain_text_to_html("❌ 请上传图像！"), visible=True)
                    yield history, text, stats_update, gr.update(interactive=False), "❌ 请上传图像！"
                    return
                result = app.ocr_receipt(image)
                if result.startswith("❌"):
                    stats_update = gr.update(value="", visible=True)
                    yield history, text, stats_update, gr.update(interactive=False), result
                    return
                prompt_text = user_text if user_text else _get_default_prompt(task, code_format)
                updated_history = history + [[f"👤 {prompt_text}", result]]
                app.chat_history = updated_history
                if not image_recorded:
                    record_image_path()
                ocr_preview = app.last_ocr_html or _plain_text_to_html(app.last_ocr_markdown or "")
                stats_update = gr.update(value=ocr_preview, visible=True)
                yield updated_history, "", stats_update, gr.update(interactive=bool(app.last_ocr_markdown)), "✅ 票据OCR识别完成，可导出样式"
                return

            if task == "协议OCR识别":
                if image is None:
                    stats_update = gr.update(value=_plain_text_to_html("❌ 请上传图像！"), visible=True)
                    yield history, text, stats_update, gr.update(interactive=False), "❌ 请上传图像！"
                    return
                result = app.ocr_agreement(image)
                if result.startswith("❌"):
                    stats_update = gr.update(value="", visible=True)
                    yield history, text, stats_update, gr.update(interactive=False), result
                    return
                prompt_text = user_text if user_text else _get_default_prompt(task, code_format)
                updated_history = history + [[f"👤 {prompt_text}", result]]
                app.chat_history = updated_history
                if not image_recorded:
                    record_image_path()
                ocr_preview = app.last_ocr_html or _plain_text_to_html(app.last_ocr_markdown or "")
                stats_update = gr.update(value=ocr_preview, visible=True)
                yield updated_history, "", stats_update, gr.update(interactive=bool(app.last_ocr_markdown)), "✅ 协议OCR识别完成，可导出样式"
                return

            effective_prompt = user_text if user_text else _get_default_prompt(task, code_format)
            chat_result = app.chat_with_image(
                image,
                effective_prompt,
                history,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                presence_penalty
            )

            if inspect.isgenerator(chat_result):
                for out_history, cleared, stats in chat_result:
                    if not image_recorded and len(out_history) > prev_turns:
                        record_image_path()
                    app.chat_history = out_history
                    button_update = gr.update(interactive=bool(app.last_ocr_markdown))
                    stats_update = gr.update(value=_plain_text_to_html(stats), visible=True)
                    yield out_history, cleared, stats_update, button_update, gr.update()
            else:
                out_history, cleared, stats = chat_result
                if not image_recorded and len(out_history) > prev_turns:
                    record_image_path()
                app.chat_history = out_history
                button_update = gr.update(interactive=bool(app.last_ocr_markdown))
                stats_update = gr.update(value=_plain_text_to_html(stats), visible=True)
                yield out_history, cleared, stats_update, button_update, gr.update()

        if not image_recorded and len(app.chat_history) > prev_turns:
            record_image_path()

    except Exception as e:
        history.append(["👤", f"❌ 错误: {str(e)}"])
        app.chat_history = history
        if not image_recorded and len(history) > prev_turns:
            record_image_path()
        button_update = gr.update(interactive=bool(app.last_ocr_markdown))
        stats_update = gr.update(value=_plain_text_to_html(f"❌ 错误: {str(e)}"), visible=True)
        yield history, text, stats_update, button_update, f"❌ 错误: {str(e)}"


def save_chat_to_folder(save_dir, history):
    """将当前聊天历史保存到指定文件夹（JSON）。"""
    try:
        if not save_dir:
            return "❌ 保存失败：未指定保存目录"
        os.makedirs(save_dir, exist_ok=True)
        # 每次保存使用独立导出子目录，避免图片累积到同一目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(save_dir, timestamp)
        os.makedirs(export_dir, exist_ok=True)
        images_target_dir = os.path.join(export_dir, "images")
        os.makedirs(images_target_dir, exist_ok=True)

        image_paths = getattr(app, "session_turn_image_paths", [])
        copied_rel_paths = []
        seen_images = set()
        for p in image_paths:
            abs_path = os.path.abspath(p) if p else None
            if not p or not os.path.exists(p) or abs_path in seen_images:
                copied_rel_paths.append(None)
                continue
            try:
                basename = os.path.basename(p)
                target = os.path.join(images_target_dir, basename)
                if os.path.abspath(p) != os.path.abspath(target):
                    shutil.copy2(p, target)
                seen_images.add(abs_path)
                copied_rel_paths.append(os.path.join("images", basename))
            except Exception:
                copied_rel_paths.append(None)
        filename = os.path.join(export_dir, f"chat_history_{timestamp}.json")
        # history 是 [(user, bot), ...]
        data = []
        turns = history or []
        for idx, pair in enumerate(turns):
            try:
                u, b = pair
            except Exception:
                u, b = pair, ""
            img_rel = copied_rel_paths[idx] if idx < len(copied_rel_paths) else None
            data.append({"user": u, "assistant": b, "image_path": img_rel})
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return f"✅ 已保存到: {filename}"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"


def create_unified_interface():
    """创建统一Gradio界面。"""

    touch_css = """
    :root {
        --radius-lg: 22px;
        --radius-md: 14px;
        --surface: #ffffff;
        --surface-muted: #f5f7fb;
        --surface-border: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #64748b;
        --accent: #2563eb;
        --accent-soft: rgba(37, 99, 235, 0.12);
    }
    body {
        background: linear-gradient(135deg, #eef2ff 0%, #f9fafc 55%, #ffffff 100%);
        color: var(--text-primary);
    }
    .gradio-container {
        max-width: 1650px !important;
        margin: 0 auto;
        padding: 20px 24px 48px;
        font-size: 16px;
        color: var(--text-primary);
    }
    .gradio-container .gr-markdown {
        color: var(--text-primary);
    }
    #unified-header {
        background: linear-gradient(130deg, rgba(37, 99, 235, 0.12), rgba(59, 130, 246, 0.1));
        border: 1px solid rgba(37, 99, 235, 0.18);
        padding: 24px 28px;
        border-radius: 28px;
        box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
        margin-bottom: 22px;
    }
    #unified-header h1 {
        margin: 0 0 6px;
        font-size: 26px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    #unified-header p {
        margin: 0;
        color: var(--text-secondary);
    }
    #unified-mode-bar {
        background: var(--surface);
        border-radius: 24px;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.06);
        padding: 20px 22px;
        gap: 18px;
        margin-bottom: 20px;
        border: 1px solid var(--surface-border);
    }
    #unified-mode-bar .gradio-button,
    #unified-mode-bar button {
        font-size: 16px !important;
        padding: 12px 18px !important;
        border-radius: 14px !important;
    }
    #unified-mode-bar textarea,
    #unified-mode-bar input[type="text"] {
        background: var(--surface);
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 14px;
        color: var(--text-primary);
        box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    #unified-mode-bar textarea:focus,
    #unified-mode-bar input[type="text"]:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
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
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.08);
    }
    #unified-input-panel,
    #unified-chat-panel,
    #unified-batch-panel,
    #unified-compare-panel {
        background: var(--surface);
        border-radius: 24px;
        padding: 22px 24px;
        box-shadow: 0 22px 44px rgba(15, 23, 42, 0.06);
        border: 1px solid var(--surface-border);
    }
    #unified-input-panel .gradio-slider > label,
    #unified-input-panel .gradio-dropdown > label {
        color: var(--text-secondary);
    }
    #unified-chat-panel {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }
    #unified-chatbot > .wrap {
        background: #f8fafc;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 8px 10px;
    }
    #unified-chatbot .message {
        border-radius: 16px !important;
        padding: 12px 14px !important;
        line-height: 1.6;
        font-size: 15px;
        color: var(--text-primary);
    }
    #unified-chatbot .message.user {
        background: linear-gradient(138deg, rgba(37, 99, 235, 0.16), rgba(96, 165, 250, 0.12));
        border: 1px solid rgba(37, 99, 235, 0.22);
        color: var(--text-primary);
        align-self: flex-end;
    }
    #unified-chatbot .message.bot {
        background: #ffffff;
        border: 1px solid rgba(203, 213, 225, 0.9);
        color: var(--text-primary);
        align-self: flex-start;
    }
    #unified-query textarea {
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background: var(--surface);
        color: var(--text-primary);
        box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.05);
    }
    #unified-query textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }
    #unified-stats .stats-text {
        background: var(--accent-soft);
        border-radius: 16px;
        border: 1px solid rgba(37, 99, 235, 0.2);
        color: var(--text-primary);
        font-weight: 500;
        padding: 12px 14px;
        line-height: 1.6;
        margin-bottom: 12px;
        word-break: break-word;
    }
    .gradio-container .gradio-button.primary {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        border: none;
        color: #ffffff;
        font-weight: 600;
        box-shadow: 0 18px 30px rgba(37, 99, 235, 0.22);
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
        border-radius: 16px;
    }
    .gradio-container textarea:focus,
    .gradio-container input[type="text"]:focus,
    .gradio-container input[type="number"]:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }
    #unified-batch-panel textarea,
    #unified-compare-panel textarea {
        min-height: 320px;
    }
    .gradio-container .dropdown span.label,
    .gradio-container .slider > label,
    .gradio-container .dropdown label {
        color: var(--text-secondary);
    }
    .gradio-container .gradio-dropdown .wrap select,
    .gradio-container .gradio-dropdown .wrap button {
        background: var(--surface);
        color: var(--text-primary);
        border-color: rgba(148, 163, 184, 0.4);
    }
    .gradio-container .gradio-dropdown .wrap select:focus,
    .gradio-container .gradio-dropdown .wrap button:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }
    /*
    Bigger markdown preview area for unified stats (OCR/table preview)
    */
    #unified-stats {
        max-height: 560px;
        overflow: auto;
        border: 1px solid rgba(148, 163, 184, 0.35);
        padding: 12px 14px;
        border-radius: 14px;
        background: #ffffff;
    }
    #unified-stats table {
        width: 100%;
        border-collapse: collapse;
        margin: 8px 0 14px;
    }
    #unified-stats th,
    #unified-stats td {
        border: 1px solid #e5e7eb;
        padding: 8px 10px;
        text-align: left;
        vertical-align: top;
        font-size: 14px;
        line-height: 1.55;
    }
    #unified-stats thead th {
        background: #f8fafc;
        font-weight: 600;
    }
    #unified-stats code {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 1px 4px;
        border-radius: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    """

    with gr.Blocks(
        title="多模态大模型智能助手",
        theme=gr.themes.Soft(),
        css=touch_css
    ) as interface:

        gr.HTML("""
        <section id="unified-header">
          <h1>🤖 多模态大模型智能助手</h1>
          <p>全新布局与对话样式，通用 / 专业双模式随心切换，支持任务分派与对话保存。</p>
        </section>
        """)

        with gr.Row(elem_id="unified-mode-bar"):
            with gr.Column(scale=1, min_width=240):
                mode = gr.Radio(
                    choices=["通用版", "专业版"], value="通用版", label="界面模式"
                )
                pro_task = gr.Dropdown(
                    choices=["任务问答", "OCR识别", "卡证OCR识别", "卡证OCR识别（API）", "票据OCR识别", "协议OCR识别", "空间分析", "视觉编程", "情感分析"],
                    value="任务问答",
                    label="专业任务",
                    visible=False,
                )
                rag_feature_selector = gr.Radio(
                    choices=["样式特征RAG", "CLIP图像特征"],
                    value="样式特征RAG",
                    label="卡证RAG特征模式",
                    visible=False,
                )
            with gr.Column(scale=1, min_width=240):
                load_btn = gr.Button("🔄 加载模型", variant="primary")
                status_text = gr.Textbox(
                    label="运行状态",
                    value="⏳ 模型未加载，请点击加载模型按钮",
                    interactive=False,
                    lines=3,
                )
            with gr.Column(scale=1, min_width=240):
                save_btn = gr.Button("💾 保存当前对话", variant="secondary")
                save_dir = gr.Textbox(value="chat_history", label="保存目录", interactive=False)

        load_btn.click(app.load_model, outputs=[status_text, load_btn])

        # 样式在 Blocks 实例化时应用，无需运行时切换

        with gr.Tab("💬 图像对话"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Group(elem_id="unified-input-panel"):
                        gr.Markdown("### 图像与参数设置")
                        image_input = gr.Image(label="上传图像", type="pil", height=400)

                        # 通用参数
                        with gr.Row(equal_height=True):
                            max_tokens = gr.Slider(
                                minimum=512, maximum=16384, value=8192, label="最大生成长度 (out_seq_length)"
                            )
                            temperature = gr.Slider(
                                minimum=0.0, maximum=2.0, value=0.7, label="创造性 (temperature)"
                            )

                        # 专业参数容器（默认隐藏）
                        with gr.Accordion("🎛️ 高级参数", open=False, visible=False) as adv_params_box:
                            top_p = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.8, label="top_p"
                            )
                            top_k = gr.Slider(
                                minimum=0, maximum=100, value=20, label="top_k"
                            )
                            repetition_penalty = gr.Slider(
                                minimum=0.8, maximum=2.0, value=1.0, step=0.05, label="repetition_penalty"
                            )
                            presence_penalty = gr.Slider(
                                minimum=0.0, maximum=3.0, value=1.5, step=0.1, label="presence_penalty (占位)"
                            )

                with gr.Column(scale=2):
                    with gr.Group(elem_id="unified-chat-panel"):
                        gr.Markdown("### 对话与输出")
                        chatbot = gr.Chatbot(
                            label=None,
                            height=500,
                            show_label=False,
                            type="tuples",
                            elem_id="unified-chatbot",
                            render_markdown=True
                        )
                        text_input = gr.Textbox(
                            label=None,
                            placeholder="输入想了解的内容，按 Enter 或点击发送。",
                            lines=3,
                            elem_id="unified-query"
                        )
                        with gr.Row():
                            send_btn = gr.Button("发送", variant="primary", scale=1)
                            clear_btn = gr.Button("🗑️ 清空历史", variant="secondary", scale=1)
                        with gr.Row():
                            with gr.Column(scale=4):
                                stats_output = gr.HTML(
                                    value="",
                                    visible=False,
                                    elem_id="unified-stats"
                                )
                            with gr.Column(scale=1, min_width=220):
                                ocr_export_btn = gr.Button("💾 导出样式", variant="secondary", interactive=False)
                                ocr_export_status = gr.Textbox(
                                    label="导出状态",
                                    interactive=False,
                                    lines=4
                                )
                code_format = gr.Dropdown(
                    choices=["HTML", "CSS", "JavaScript", "Python"],
                    value="HTML",
                    label="代码类型",
                    visible=False,
                )

            # 通用/专业两种调用路径（利用同一高级应用，专业多两个参数与统计输出）
            send_btn.click(
                handle_unified_chat,
                inputs=[image_input, text_input, chatbot, max_tokens, temperature, top_p, top_k, mode, pro_task, rag_feature_selector, code_format, repetition_penalty, presence_penalty],
                outputs=[chatbot, text_input, stats_output, ocr_export_btn, ocr_export_status],
            )
            text_input.submit(
                handle_unified_chat,
                inputs=[image_input, text_input, chatbot, max_tokens, temperature, top_p, top_k, mode, pro_task, rag_feature_selector, code_format, repetition_penalty, presence_penalty],
                outputs=[chatbot, text_input, stats_output, ocr_export_btn, ocr_export_status],
            )
            def _clear_session():
                app.clear_history()
                return [], "", gr.update(value="", visible=False), gr.update(interactive=False), ""

            clear_btn.click(
                _clear_session,
                outputs=[chatbot, text_input, stats_output, ocr_export_btn, ocr_export_status],
            )
        save_btn.click(save_chat_to_folder, inputs=[save_dir, chatbot], outputs=[status_text])

        ocr_export_btn.click(
            app.export_last_ocr,
            outputs=[ocr_export_status],
        )

        # 高级功能Tab（默认隐藏，通过模式切换显示）
        with gr.Tab("📊 批量分析", visible=False) as tab_batch:
            with gr.Group(elem_id="unified-batch-panel"):
                with gr.Row():
                    with gr.Column():
                        batch_images = gr.File(
                            label="上传多个图像", file_count="multiple", file_types=["image"]
                        )
                        analysis_type = gr.Dropdown(
                            choices=["描述", "OCR", "空间分析", "情感分析"], value="描述", label="分析类型"
                        )
                        batch_btn = gr.Button("🔍 开始批量分析", variant="primary")
                    with gr.Column():
                        batch_result = gr.Markdown()
            batch_btn.click(app.batch_analysis, inputs=[batch_images, analysis_type], outputs=[batch_result])

        with gr.Tab("🔄 图像对比", visible=False) as tab_compare:
            with gr.Group(elem_id="unified-compare-panel"):
                with gr.Row():
                    with gr.Column():
                        compare_image1 = gr.Image(label="图像1", type="pil", height=220)
                        compare_image2 = gr.Image(label="图像2", type="pil", height=220)
                        comparison_type = gr.Dropdown(
                            choices=["相似性", "风格", "内容", "综合"], value="相似性", label="对比类型"
                        )
                        compare_btn = gr.Button("🔄 开始对比", variant="primary")
                    with gr.Column():
                        compare_result = gr.Markdown()
            compare_btn.click(
                app.compare_images,
                inputs=[compare_image1, compare_image2, comparison_type],
                outputs=[compare_result],
            )

        with gr.Tab("ℹ️ 使用说明"):
            gr.Markdown(
                """
                - 先点击「加载模型」后再使用各项功能。
                - 「通用版」适合快速上手与日常使用；「专业版」提供更细粒度的生成参数与高级功能（批量分析、图像对比）。
                - 已默认优化为更易触摸点击的界面尺寸。
                """
            )

        # 绑定模式切换：控制高级组件可见性
        mode.change(
            _toggle_mode,
            inputs=[mode, pro_task, code_format],
            outputs=[adv_params_box, stats_output, tab_batch, tab_compare, pro_task, code_format, rag_feature_selector, text_input],
        )

        pro_task.change(
            _toggle_task,
            inputs=[pro_task, code_format],
            outputs=[code_format, text_input, rag_feature_selector],
        )

        code_format.change(
            _update_code_prompt,
            inputs=[pro_task, code_format],
            outputs=[text_input],
        )

    return interface


def main():
    print("🚀 启动Qwen3-VL-8B-Instruct 统一Web界面...")
    interface = create_unified_interface()

    def _cleanup():
        # 清理模型与显存
        try:
            app.model = None
            app.processor = None
            app.is_loaded = False
        except Exception:
            pass
        try:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass

    # 注册进程退出清理
    atexit.register(_cleanup)
    interface.queue()
    interface.launch(
        server_name="127.0.0.1",
        server_port=None,  # 自动选择可用端口，避免端口占用错误
        share=False,
        debug=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
