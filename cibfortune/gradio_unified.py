#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B-Instruct 统一Gradio界面
支持在同一界面内切换「通用版」与「专业版」，并提供触屏友好样式
"""

import os
import json
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


# 单例应用
app = AdvancedQwen3VLApp()

# 会话级图片保存目录与轨迹
IMAGE_SAVE_ROOT = "chat_history/images"
SESSION_IMAGE_DIR = os.path.join(IMAGE_SAVE_ROOT, getattr(app, "session_id", datetime.now().strftime("%Y%m%d_%H%M%S")))
os.makedirs(SESSION_IMAGE_DIR, exist_ok=True)
app.session_turn_image_paths = []  # 与对话轮次对齐的图片路径（无图则为 None）


def _toggle_mode(mode):
    """根据模式切换组件可见性。
    通用版隐藏高级参数/统计/高级功能Tab；专业版全部显示。
    """
    is_pro = (mode == "专业版")
    return (
        gr.update(visible=is_pro),   # adv_params_box
        gr.update(visible=is_pro),   # stats_output
        gr.update(visible=is_pro),   # tab_batch
        gr.update(visible=is_pro),   # tab_compare
        gr.update(visible=is_pro),   # pro_task dropdown
        gr.update(visible=False),    # code_format dropdown (重置隐藏，按任务再控制)
    )


def _toggle_task(task):
    return gr.update(visible=(task == "视觉编程"))


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
    try:
        # 若传入了图片，先将图片保存到会话目录，记录该轮图片路径
        saved_image_path = None
        if image is not None:
            try:
                ts = datetime.now().strftime("%H%M%S%f")
                saved_image_path = os.path.join(SESSION_IMAGE_DIR, f"img_{ts}.png")
                image.save(saved_image_path)
            except Exception:
                saved_image_path = None
        if mode == "通用版":
            # 普通问答（使用高级接口以获得一致的返回结构）
            out_history, cleared, stats = app.chat_with_image(image, text, history, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty)
            app.chat_history = out_history
            # 只有在历史有新增时记录图片路径
            app.session_turn_image_paths.append(saved_image_path)
            return out_history, cleared, stats

        # 专业版任务分派
        task = pro_task or "任务问答"
        if task == "任务问答":
            out_history, cleared, stats = app.chat_with_image(image, text, history, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty)
            app.chat_history = out_history
            app.session_turn_image_paths.append(saved_image_path)
            return out_history, cleared, stats
        
        if image is None:
            # 与高级接口保持一致的输出结构
            return history, "❌ 请上传图像！", ""

        if task == "OCR识别":
            if hasattr(app, "ocr_analysis"):
                result = app.ocr_analysis(image)
            else:
                # 回退：用问答接口模拟 OCR
                prompt = "请识别并提取这张图片中的所有文字内容，尽量还原原本样式，并标注语言类型。"
                out_history, cleared, stats = app.chat_with_image(image, prompt, history, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty)
                app.chat_history = out_history
                app.session_turn_image_paths.append(saved_image_path)
                return out_history, cleared, stats
            history.append(["👤 [OCR识别]", result])
            app.chat_history = history
            app.session_turn_image_paths.append(saved_image_path)
            return history, "", ""

        if task == "空间分析":
            if hasattr(app, "spatial_analysis"):
                result = app.spatial_analysis(image)
            else:
                # 回退：用问答接口模拟空间分析
                prompt = "请分析这张图片中的空间关系，包括相对位置、视角、遮挡、深度与距离感，并给出整体布局描述。"
                out_history, cleared, stats = app.chat_with_image(image, prompt, history, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty)
                app.chat_history = out_history
                app.session_turn_image_paths.append(saved_image_path)
                return out_history, cleared, stats
            history.append(["👤 [空间分析]", result])
            app.chat_history = history
            app.session_turn_image_paths.append(saved_image_path)
            return history, "", ""

        if task == "视觉编程":
            fmt = code_format or "HTML"
            if hasattr(app, "visual_coding"):
                result = app.visual_coding(image, fmt)
            else:
                # 回退：用问答接口提示生成对应代码
                prompts = {
                    "HTML": "请根据图片生成对应的HTML结构代码，包含必要的语义标签。",
                    "CSS": "请为该图片对应的界面生成合理的CSS样式代码，包括布局与颜色。",
                    "JavaScript": "请根据图片交互生成JavaScript代码示例，包含必要的事件与逻辑。",
                    "Python": "请生成能复现该界面/布局的Python示例代码（如使用streamlit或flask的伪代码）。",
                }
                prompt = prompts.get(fmt, prompts["HTML"]) + " 请只输出代码，不要额外说明。"
                out_history, cleared, stats = app.chat_with_image(image, prompt, history, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty)
                app.chat_history = out_history
                app.session_turn_image_paths.append(saved_image_path)
                return out_history, cleared, stats
            history.append([f"👤 [视觉编程:{fmt}]", result])
            app.chat_history = history
            app.session_turn_image_paths.append(saved_image_path)
            return history, "", ""

        if task == "情感分析":
            # 复用批量接口的提示风格或直接用问答提示
            prompt = (text or "").strip() or "请分析这张图片传达的情感或氛围，并给出理由。"
            # 走问答路径以节省实现，给定清晰任务提示
            composed = f"[情感分析] {prompt}"
            out_history, cleared, stats = app.chat_with_image(image, composed, history, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty)
            app.chat_history = out_history
            app.session_turn_image_paths.append(saved_image_path)
            return out_history, cleared, stats

        # 兜底走问答
        out_history, cleared, stats = app.chat_with_image(image, text, history, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty)
        app.chat_history = out_history
        app.session_turn_image_paths.append(saved_image_path)
        return out_history, cleared, stats

    except Exception as e:
        history.append(["👤", f"❌ 错误: {str(e)}"])
        app.chat_history = history
        app.session_turn_image_paths.append(None)
        return history, "", f"❌ 错误: {str(e)}"


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
        for p in image_paths:
            if not p:
                copied_rel_paths.append(None)
                continue
            try:
                basename = os.path.basename(p)
                target = os.path.join(images_target_dir, basename)
                if os.path.abspath(p) != os.path.abspath(target):
                    shutil.copy2(p, target)
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
    :root { --radius-xxl: 14px; }
    .gradio-container { max-width: 1400px !important; font-size: 16px; }
    /* 顶部横幅 */
    .app-hero { 
        background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
        color: #fff; padding: 18px 16px; border-radius: 14px; margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }
    .app-hero h1 { font-size: 22px; margin: 0 0 6px 0; }
    .app-hero p { margin: 0; opacity: .95; }
    /* 通用控件优化 */
    .gradio-container .btn, .gradio-container button, .gradio-container .gr-button { 
        font-size: 16px !important; padding: 12px 18px !important; border-radius: 10px !important;
    }
    .gradio-container input[type="text"],
    .gradio-container textarea { font-size: 16px !important; padding: 10px 12px !important; }
    .gradio-container .gr-box { border-radius: 12px !important; }
    .gradio-container .tabitem, .gradio-container .tabs { gap: 8px; }
    .gradio-container .image-container { touch-action: manipulation; }
    .toolbar { display: flex; align-items: center; gap: 8px; }
    """

    with gr.Blocks(
        title="多模态大模型智能助手",
        theme=gr.themes.Soft(),
        css=touch_css
    ) as interface:

        gr.HTML("""
        <div class="app-hero">
          <h1>🤖 多模态大模型智能助手</h1>
          <p>在「通用版」与「专业版」间一键切换，支持任务分派与本地保存。</p>
        </div>
        """)

        with gr.Row():
            mode = gr.Radio(
                choices=["通用版", "专业版"], value="通用版", label="界面模式"
            )
            load_btn = gr.Button("🔄 加载模型", variant="primary")
            status_text = gr.Textbox(
                label="状态",
                value="⏳ 模型未加载，请点击加载模型按钮",
                interactive=False,
            )
            pro_task = gr.Dropdown(
                choices=["任务问答", "OCR识别", "空间分析", "视觉编程", "情感分析"],
                value="任务问答",
                label="任务类型",
                visible=False,
            )
            save_dir = gr.Textbox(value="chat_history", label="保存目录", interactive=False)
            save_btn = gr.Button("💾 保存对话", variant="secondary")

        load_btn.click(app.load_model, outputs=[status_text, load_btn])

        # 样式在 Blocks 实例化时应用，无需运行时切换

        with gr.Tab("💬 图像对话"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="上传图像", type="pil", height=420)

                    # 通用参数
                    with gr.Row(equal_height=True):
                        max_tokens = gr.Slider(
                            minimum=1024, maximum=16384, value=8192, label="最大生成长度 (out_seq_length)"
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
                    chatbot = gr.Chatbot(label="对话历史", height=420, show_label=True, type="tuples")
                    with gr.Row():
                        text_input = gr.Textbox(
                            label="输入问题", placeholder="请描述这张图片...", lines=2
                        )
                        send_btn = gr.Button("发送", variant="primary")

                    with gr.Row():
                        clear_btn = gr.Button("🗑️ 清空历史")
                        stats_output = gr.Textbox(
                            label="生成统计", interactive=False, visible=False
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
                outputs=[chatbot, text_input, stats_output],
            )
            text_input.submit(
                handle_unified_chat,
                inputs=[image_input, text_input, chatbot, max_tokens, temperature, top_p, top_k, mode, pro_task, code_format, repetition_penalty, presence_penalty],
                outputs=[chatbot, text_input, stats_output],
            )
            clear_btn.click(app.clear_history, outputs=[chatbot])
            save_btn.click(save_chat_to_folder, inputs=[save_dir, chatbot], outputs=[status_text])

        # 高级功能Tab（默认隐藏，通过模式切换显示）
        with gr.Tab("📊 批量分析", visible=False) as tab_batch:
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
            inputs=[mode],
            outputs=[adv_params_box, stats_output, tab_batch, tab_compare, pro_task, code_format],
        )

        pro_task.change(
            _toggle_task,
            inputs=[pro_task],
            outputs=[code_format],
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
    interface.launch(
        server_name="0.0.0.0",
        server_port=None,  # 自动选择可用端口，避免端口占用错误
        share=False,
        debug=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()


