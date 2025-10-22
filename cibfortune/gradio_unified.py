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
from datetime import datetime
import gradio as gr
import shutil
import atexit
import gc
try:
    import torch
except Exception:
    torch = None

# 统一环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 直接复用高级版应用的能力（包含基础能力的超集）
from gradio_advanced import AdvancedQwen3VLApp


DEFAULT_TASK_PROMPTS = {
    "任务问答": "请根据图片完成指定任务，并给出详细的分析与结论。",
    "OCR识别": "请识别并提取这张图片中的所有文字内容，并标注语言类型。请确保所有带样式或表格内容使用Markdown表格表示。",
    "空间分析": "请分析这张图片中的空间关系，包括相对位置、视角、遮挡、深度与距离感，并给出整体布局描述。",
    "情感分析": "请分析这张图片传达的情感或氛围，并说明理由。",
}

VISUAL_CODING_PROMPTS = {
    "HTML": "请根据图片生成对应的HTML结构代码，包含必要的语义标签。请只输出代码，不要额外说明。",
    "CSS": "请为该图片对应的界面生成合理的CSS样式代码，包括布局与颜色。请只输出代码，不要额外说明。",
    "JavaScript": "请根据图片交互生成JavaScript代码示例，包含必要的事件与逻辑。请只输出代码，不要额外说明。",
    "Python": "请生成能复现该界面/布局的Python示例代码（如使用streamlit或flask的伪代码）。请只输出代码，不要额外说明。",
}


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
    return (
        gr.update(visible=is_pro),                       # adv_params_box
        gr.update(visible=is_pro),                       # stats_output
        gr.update(visible=is_pro),                       # tab_batch
        gr.update(visible=is_pro),                       # tab_compare
        gr.update(visible=is_pro, value=task_value),     # pro_task dropdown
        gr.update(visible=code_visible),                 # code_format dropdown
        gr.update(value=text_value),                     # text_input prompt
    )


def _toggle_task(task, code_format):
    """任务切换时调整代码下拉可见性并预填提示。"""
    is_visual = (task == "视觉编程")
    prompt = _get_default_prompt(task, code_format)
    code_kwargs = {"visible": is_visual}
    if is_visual and not code_format:
        code_kwargs["value"] = "HTML"
    return gr.update(**code_kwargs), gr.update(value=prompt)


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
        else:
            task = pro_task or "任务问答"
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
                yield out_history, cleared, stats, button_update
        else:
            out_history, cleared, stats = chat_result
            if not image_recorded and len(out_history) > prev_turns:
                record_image_path()
            app.chat_history = out_history
            button_update = gr.update(interactive=bool(app.last_ocr_markdown))
            yield out_history, cleared, stats, button_update

        if not image_recorded and len(app.chat_history) > prev_turns:
            record_image_path()

    except Exception as e:
        history.append(["👤", f"❌ 错误: {str(e)}"])
        app.chat_history = history
        if not image_recorded and len(history) > prev_turns:
            record_image_path()
        button_update = gr.update(interactive=bool(app.last_ocr_markdown))
        yield history, text, f"❌ 错误: {str(e)}", button_update


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
    #unified-stats textarea {
        background: var(--accent-soft);
        border-radius: 16px;
        border: 1px solid rgba(37, 99, 235, 0.2);
        color: var(--text-primary);
        font-weight: 500;
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
                    choices=["任务问答", "OCR识别", "空间分析", "视觉编程", "情感分析"],
                    value="任务问答",
                    label="专业任务",
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
                save_dir = gr.Textbox(value="chat_history", label="保存目录", interactive=False)
                save_btn = gr.Button("💾 保存当前对话", variant="secondary")
            with gr.Column(scale=1, min_width=240):
                ocr_export_btn = gr.Button("💾 保存文本样式", variant="secondary", interactive=False)
                ocr_export_status = gr.Textbox(
                    label="保存状态",
                    interactive=False,
                    lines=2
                )

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
                            height=420,
                            show_label=False,
                            type="tuples",
                            elem_id="unified-chatbot"
                        )
                        with gr.Row():
                            text_input = gr.Textbox(
                                label=None,
                                placeholder="输入想了解的内容，按 Enter 或点击发送。",
                                lines=2,
                                elem_id="unified-query"
                            )
                            send_btn = gr.Button("发送", variant="primary")

                        with gr.Row():
                            clear_btn = gr.Button("🗑️ 清空历史", variant="secondary")
                        stats_output = gr.Textbox(
                            label=None,
                            placeholder="生成速度与长度统计会显示在这里。",
                            interactive=False,
                            visible=False,
                            elem_id="unified-stats"
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
                inputs=[image_input, text_input, chatbot, max_tokens, temperature, top_p, top_k, mode, pro_task, code_format, repetition_penalty, presence_penalty],
                outputs=[chatbot, text_input, stats_output, ocr_export_btn],
            )
            text_input.submit(
                handle_unified_chat,
                inputs=[image_input, text_input, chatbot, max_tokens, temperature, top_p, top_k, mode, pro_task, code_format, repetition_penalty, presence_penalty],
                outputs=[chatbot, text_input, stats_output, ocr_export_btn],
            )
            clear_btn.click(app.clear_history, outputs=[chatbot])
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
                        batch_result = gr.Textbox(label="批量分析结果", lines=20, max_lines=30)
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
                        compare_result = gr.Textbox(label="对比结果", lines=20, max_lines=25)
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
            outputs=[adv_params_box, stats_output, tab_batch, tab_compare, pro_task, code_format, text_input],
        )

        pro_task.change(
            _toggle_task,
            inputs=[pro_task, code_format],
            outputs=[code_format, text_input],
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
        server_name="0.0.0.0",
        server_port=None,  # 自动选择可用端口，避免端口占用错误
        share=False,
        debug=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
