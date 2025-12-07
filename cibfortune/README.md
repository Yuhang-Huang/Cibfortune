# Cibfortune Qwen3-VL-8B-Instruct 工具集

这个子目录包含一个围绕 **Qwen3-VL-8B-Instruct** 构建的多模态应用集合：提供图文聊天、OCR、空间理解、视觉编程和基于 RAG 的文档/卡片识别能力，同时内置了带菜单的 Web 启动器与导出工具。

## 目录结构概览

| 区块 | 说明 |
| --- | --- |
| `launch_web.py` | 交互式启动器，提供界面选择、依赖检查、批量卡片/单据识别等命令菜单。 |
| `gradio_unified.py` | 单一 Gradio 界面，集成图文问答、OCR、空间分析、视觉编程、文档 RAG、卡片识别等多个标签页；也提供 `batch_output()` 和 `AdvancedQwen3VLApp` 供脚本式批处理。 |
| `gradio_app.py` | 轻量版 Gradio 聊天/OCR/视觉编程界面，适合快速演示；自带模型加载与导出 Markdown/CSV/JSON 的逻辑。 |
| `models/download_model.py` | 通过 `huggingface-cli` 拉取 Qwen3 模型的辅助脚本，包含 `HF_ENDPOINT` 镜像与 token 设定；可根据需求修改 `--local-dir`。 |
| `ocr_card_rag_api.py` | 卡片 OCR + RAG 增强的核心功能，支持多种 RAG 实现（multimodal_rag、简化样式特征、CLIP）、OpenAI/Qwen API 调用、样式相似度提示。 |
| `doc_knowledge_base.py` | 文档知识库封装，支持 SentenceTransformers/Transformers embedding、Chroma/FAISS 向量存储，可与 `gradio_unified` 的文档标签页联动。 |
| `image.py` / `seal_removal.py` | 提供超分辨率预处理和水印消除辅助函数，提升 OCR 识别质量。 |
| `styles.py` | Gradio UI 样式片段。 |
| `card_field_templates.md` | 卡片/证件字段模板，供卡片识别模块生成 Markdown 表格时参考。 |
| `rag_cards/` | 可以放置样本卡片图，用于 `SimpleRAGStore`/`multimodal_rag` 搜索更贴近的结果。 |
| `ocr_exports/` | Gradio 界面 OCR 导出时写入的 Markdown/Excel/CSV/JSON 文件。 |
| `tests/` | `baseline_paddleocr.py` 与 `baseline_qwenvl.py` 可作为最小依赖运行示例/回归验证。 |

## 先决条件与环境准备

1. 安装 Python 3.8+（推荐 3.9/3.10）和 CUDA 11.8+ 的 PyTorch 2.0+。GPU 内存 6GB 以上可获得流畅体验，CPU 也能运行但速度较慢。
2. 通过 `pip install -r requirements.txt` 或 `launch_web.py` 菜单中的“安装依赖”选项装好 `gradio`, `torch`, `transformers`, `sentencepiece`, `Pillow`, `requests` 等基础库。
3. 默认 `HF_ENDPOINT` 已在脚本中设置为 `https://hf-mirror.com`，若需切换请在环境中重新导出，下载模型时还需配置 `HUGGINGFACE_TOKEN`。
4. 模型文件放在 `models/qwen3-vl-8b-instruct`（或修改脚本中的 `model_path`），可运行 `python models/download_model.py` 直接从 Hugging Face 获取（记得改到实际落地目录）。

## 快速启动步骤

1. **启动菜单**：`python launch_web.py` 会打印横幅与菜单项，按提示可以：
   - 选择 “智能助手” 进入 `gradio_unified.py`；
   - 检查系统依赖/模型；
   - 安装依赖；
   - 运行 `batch_output("card")` / `batch_output("bill")` 批量识别 `tests/dataset` 下的图像。
2. **直连 Web 界面**：
   - `python gradio_unified.py`：强烈推荐，加载模型后点击各标签即可切换聊天、OCR、空间分析、视觉编程、文档识别等能力（“文档”页可加载 `doc/knowledge base`，预览 OCR 分页、导出 Markdown/Excel/JSON）。
   - `python gradio_app.py`：用于展示聊天、OCR 和视觉编程的轻量界面，内置 OCR 导出按钮、历史清除、模型加载、参数调节（max tokens/temperature）功能。
3. **批量/脚本使用**：
   - 用 `from gradio_unified import AdvancedQwen3VLApp` 在脚本中直接调用 `load_model()`、`ocr_card_with_fields()`、`spatial_analysis()` 等方法。
   - `ocr_card_rag_api.CardOCRWithRAG` 可独立封装成命令行识别器，内置 `recognize_card()`、`general_prompt()`，同时支持 RAG 搜索与 Qwen API。
   - `doc_knowledge_base.DocumentKnowledgeBase` 支持添加文档、向量化、保存 Chroma/FAISS 索引，并可被 `gradio_unified` 的文档标签页调用以生成结构化字段。

## Web 界面功能细节（`gradio_unified.py`）

- **Chat（图文问答）**：上传图片 + 自定义问题，支持上下文保持；默认使用 `processor.apply_chat_template` 的多轮消息格式，并可调 `max_tokens / temperature`。
- **OCR（卡片/文档）**：一键 OCR、支持自动导出 Markdown/CSV/Excel/JSON（结果写入 `ocr_exports/`）。`Card OCR` 会自动调用 `card_field_templates.md` 生成字段表格并可输出 HTML/Markdown。
- **Document / 文档识别**：支持 PDF（依赖 PyMuPDF 或 pdf2image）、勾选去印章、分页导航，支持 Field extraction（按关键字段列表提取表格），导出多格式。
- **空间分析**：描述物体位置、视角、遮挡等；默认 prompt 依赖 `AdvancedQwen3VLApp.spatial_analysis`。
- **视觉编程**：支持生成 HTML/CSS/JavaScript/Python 代码片段；`visual_coding()` 会根据选择的格式自动切换 Prompt。
- **RAG 增强**：默认在需要的接口里调用 `SimpleRAGStore` 或 `multimodal_rag`，`rag_cards/` 放置样本图促使识别结果更贴近样式；还可以向 `CardOCRWithRAG` 注入 `rag_image_dir`。
- **文档知识库（可选）**：若安装 Chromadb/FAISS + SentenceTransformers，可自动向量化文档并在 UI 中调用检索结果；`launch_web` 会尝试 `from doc_knowledge_base import DocumentKnowledgeBase`，失败则提示不可用。

## 辅助工具与导出

- `image.py` 提供坐标超分函数 `paddleocr_super_resolution()`，`AdvancedQwen3VLApp` 中 `_super_resolve_image_for_ocr()` 会在 OCR 前调用以提升识别率。
- `seal_removal.py` 实现 HSV 版印章消除，可在 OCR 前级联调用，提升卡片/证件清晰度。
- `styles.py` 定义 Gradio 样式与字体，现有界面已直接导入。
- `ocr_exports/` 会按时间戳保存 OCR 结果，无需手动创建。导出函数会尝试 Excel，如失败回退到 CSV + JSON。
- `rag_cards/` 目录用于存放 RAG 检索的卡片样本；`gradio_unified` 运行时会自动加载（多种方式尝试 `multimodal_rag`, `SimpleRAGStore`）。
- `tests/baseline_paddleocr.py` 与 `tests/baseline_qwenvl.py` 可跑通最简单的 PaddleOCR/Qwen 测试，验证依赖是否完整。

## 调试与故障排查建议

1. 如果模型加载失败，确认 `models/qwen3-vl-8b-instruct` 中含 `config.json`、`pytorch_model.bin` 等文件，并可在脚本中把 `model_path` 指向本地目录。
2. 依赖缺失时先运行 `python launch_web.py` → “安装依赖”，或手动 `pip install -r requirements.txt`。
3. GPU 内存不足：可在 `gradio_app.py`/`gradio_unified.py` 中把 `dtype="auto"` 改为 `torch.float16`，或者在加载时加 `torch_dtype=torch.float32` 及 `device_map="auto"`。
4. 若需要 OpenAI/Qwen API，置入密钥后 `CardOCRWithRAG` 会检测 `HUGGINGFACE_TOKEN` / `openai`，可在 `ocr_card_rag_api.py` 中设置 `api_key`。

## 参考资料

- [Qwen3-VL-8B-Instruct 模型页面](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- `models/download_model.py` 中使用的 `huggingface-cli download` 可用于其他权重的管理。
