# Qwen3-VL-8B-Instruct 使用指南

本项目提供了使用Qwen3-VL-8B-Instruct多模态大语言模型的完整代码示例，根据[官方文档](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)要求编写。

## 功能特性

Qwen3-VL-8B-Instruct是Qwen系列中最强大的视觉语言模型，具有以下特性：

- **视觉智能体**: 操作PC/移动端GUI界面
- **视觉编程增强**: 从图像/视频生成Draw.io/HTML/CSS/JS代码
- **高级空间感知**: 判断物体位置、视角和遮挡关系
- **长上下文和视频理解**: 原生256K上下文，可扩展至1M
- **增强多模态推理**: 在STEM/数学方面表现出色
- **升级视觉识别**: 能够"识别一切"
- **扩展OCR**: 支持32种语言

## 文件说明

### 核心文件（推荐使用）
- `test_local_model.py` - 本地模型测试脚本
- `quick_demo.py` - 快速演示脚本
- `qwen3_vl_basic_usage.py` - 基础使用示例
- `qwen3_vl_advanced_usage.py` - 高级功能示例

### Web界面（推荐）
- `launch_web.py` - Web界面启动器（推荐）
- `gradio_unified.py` - 统一Web界面（通用版 / 专业版一站式体验）
- `gradio_app.py` - 基础Web界面
- `start_web.py` - 简单启动脚本

### 辅助文件
- `quick_start.py` - 一键启动脚本（首次使用）
- `install_dependencies.py` - 依赖安装脚本
- `test_installation.py` - 安装测试脚本
- `requirements.txt` - 依赖包列表
- `README.md` - 使用说明文档

## 快速开始

### Web界面使用（最推荐）

启动友好的Web界面：

```bash
# 启动统一界面（推荐）
python launch_web.py

# 或者直接启动基础界面
python gradio_app.py
# 或者直接启动统一界面
python gradio_unified.py
```

### 命令行使用

```bash
# 1. 测试本地模型加载
python test_local_model.py

# 2. 快速演示
python quick_demo.py

# 3. 基础功能示例
python qwen3_vl_basic_usage.py

# 4. 高级功能示例  
python qwen3_vl_advanced_usage.py
```

### 一键启动（首次使用）

```bash
# 运行快速启动脚本，包含所有设置步骤
python quick_start.py
```

### 手动安装步骤

#### 1. 安装依赖

```bash
# 方法1: 使用安装脚本（推荐）
python install_dependencies.py

# 方法2: 手动安装
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers
```

#### 2. 测试安装

```bash
# 验证所有依赖是否正确安装
python test_installation.py
```

#### 3. 下载模型

```bash
# 使用现有脚本下载模型
python download_model.py
```

或者手动下载：
```bash
huggingface-cli download --resume-download Qwen/Qwen3-VL-8B-Instruct --local-dir ./models/qwen3-vl-8b-instruct
```

## 使用方法

### 基础使用

```python
from qwen3_vl_basic_usage import load_model_and_processor, chat_with_image

# 加载模型
model, processor = load_model_and_processor()

# 与图像对话
response = chat_with_image(
    model, processor, 
    image, 
    "请描述这张图片"
)
print(response)
```

### 高级功能

```python
from qwen3_vl_advanced_usage import Qwen3VLAdvanced

# 初始化高级功能类
qwen_vl = Qwen3VLAdvanced()

# OCR文字识别
ocr_result = qwen_vl.ocr_analysis("image.jpg")

# 空间感知分析
spatial_result = qwen_vl.spatial_analysis("image.jpg")

# 视觉编程
html_code = qwen_vl.visual_coding("image.jpg", "html")

# 多图像分析
multi_result = qwen_vl.multi_image_analysis(
    ["image1.jpg", "image2.jpg"], 
    "对比这两张图片"
)
```

## 运行示例

### 基础功能测试
```bash
python qwen3_vl_basic_usage.py
```

### 高级功能测试
```bash
python qwen3_vl_advanced_usage.py
```

## 配置说明

### 环境变量
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像加速
```

### 模型配置
```python
# 使用Flash Attention加速（推荐）
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
```

### 生成参数
```python
# 视觉任务推荐参数
generation_params = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5
}

# 纯文本任务推荐参数
text_params = {
    "max_new_tokens": 2048,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 40,
    "repetition_penalty": 1.0,
    "presence_penalty": 2.0
}
```

## 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (可选，用于GPU加速)
- 至少16GB内存（推荐32GB+）
- 至少20GB存储空间

## 注意事项

1. **内存要求**: 模型较大，建议使用GPU或至少16GB内存
2. **网络连接**: 首次运行需要下载模型，确保网络连接稳定
3. **依赖版本**: 建议使用最新版本的transformers库
4. **Flash Attention**: 推荐启用以获得更好的性能和内存效率

## 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 使用CPU模式
   model = Qwen3VLForConditionalGeneration.from_pretrained(
       "Qwen/Qwen3-VL-8B-Instruct",
       torch_dtype=torch.float32,
       device_map="cpu"
   )
   ```

2. **下载失败**
   ```bash
   # 使用镜像
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **Flash Attention安装失败**
   ```bash
   # 跳过Flash Attention
   pip install transformers --no-deps
   ```

## 许可证

本项目遵循Apache 2.0许可证。模型使用请参考[Qwen3-VL官方许可证](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。

## 参考链接

- [Qwen3-VL-8B-Instruct模型页面](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen3技术报告](https://arxiv.org/abs/2505.09388)
- [Transformers文档](https://huggingface.co/docs/transformers)
