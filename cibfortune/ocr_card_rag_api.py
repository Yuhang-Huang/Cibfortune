#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卡证OCR识别 - RAG增强 + Qwen3-VL API调用
先进行多模态RAG检索增强，再调用Qwen3-VL大模型API获取识别结果
"""

import os
import time
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, Dict, List, Tuple

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("请安装openai: pip install openai")
    OPENAI_AVAILABLE = False

# 尝试导入RAG相关模块（支持多种导入方式）
RAG_AVAILABLE = False
MultiModalDocumentLoader = None
MultiModalVectorStore = None

# 方式1: 从 multimodal_rag 导入
try:
    from multimodal_rag import MultiModalDocumentLoader, MultiModalVectorStore
    RAG_AVAILABLE = True
except ImportError:
    # 方式2: 从 api 导入（如果 api.py 包含 multimodal_rag 的内容）
    try:
        import api
        MultiModalDocumentLoader = api.MultiModalDocumentLoader
        MultiModalVectorStore = api.MultiModalVectorStore
        RAG_AVAILABLE = True
    except (ImportError, AttributeError):
        # 方式3: 使用样式特征RAG（推荐，基于颜色、布局、边缘，无需torch）
        # 只要numpy和PIL可用即可，opencv可选
        try:
            import numpy as np
            from PIL import Image
            RAG_AVAILABLE = True
            print("使用样式特征RAG功能（基于颜色、布局、边缘，无需torch）")
        except ImportError:
            # 方式4: 使用CLIP模型（需要transformers和torch）
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch
                RAG_AVAILABLE = True
                print("使用简化版RAG功能（CLIP模型）")
            except ImportError:
                print("警告: RAG功能将不可用（需要安装numpy、PIL或transformers模块）")
                RAG_AVAILABLE = False


# 样式特征提取器（基于颜色、布局、边缘等）
class StyleFeatureExtractor:
    """提取卡证的样式特征（颜色、布局、边缘等）"""
    
    def __init__(self):
        try:
            import cv2
            self.cv2 = cv2
            self.use_cv2 = True
        except ImportError:
            self.use_cv2 = False
            print("⚠️ opencv-python未安装，样式特征提取功能受限")
    
    def extract_style_features(self, image: Image.Image) -> np.ndarray:
        """
        提取图片的样式特征
        
        Args:
            image: PIL Image对象
            
        Returns:
            样式特征向量（numpy数组）
        """
        features = []
        
        # 转换为numpy数组
        if self.use_cv2:
            img_array = np.array(image.convert('RGB'))
            img_bgr = img_array[:, :, ::-1]  # RGB to BGR for OpenCV
        else:
            img_array = np.array(image.convert('RGB'))
        
        # 1. 颜色直方图特征（HSV色彩空间，更能反映卡面颜色风格）
        color_feature_size = 150  # 50*3 = 150
        try:
            if self.use_cv2:
                hsv = self.cv2.cvtColor(img_bgr, self.cv2.COLOR_BGR2HSV)
                # H(色调), S(饱和度), V(明度) 直方图
                hist_h = self.cv2.calcHist([hsv], [0], None, [50], [0, 180]).flatten()
                hist_s = self.cv2.calcHist([hsv], [1], None, [50], [0, 256]).flatten()
                hist_v = self.cv2.calcHist([hsv], [2], None, [50], [0, 256]).flatten()
                # 归一化
                hist_h = hist_h / (hist_h.sum() + 1e-8)
                hist_s = hist_s / (hist_s.sum() + 1e-8)
                hist_v = hist_v / (hist_v.sum() + 1e-8)
                features.extend(hist_h)
                features.extend(hist_s)
                features.extend(hist_v)
            else:
                # 使用PIL计算RGB直方图
                hist_r = np.histogram(img_array[:, :, 0], bins=50, range=(0, 256))[0]
                hist_g = np.histogram(img_array[:, :, 1], bins=50, range=(0, 256))[0]
                hist_b = np.histogram(img_array[:, :, 2], bins=50, range=(0, 256))[0]
                # 归一化
                hist_r = hist_r / (hist_r.sum() + 1e-8)
                hist_g = hist_g / (hist_g.sum() + 1e-8)
                hist_b = hist_b / (hist_b.sum() + 1e-8)
                features.extend(hist_r)
                features.extend(hist_g)
                features.extend(hist_b)
        except Exception as e:
            print(f"⚠️ 颜色特征提取失败: {e}")
            # 使用默认值确保维度一致
            features.extend([0.0] * color_feature_size)
        
        # 2. 边缘特征（反映卡面边框和布局）
        edge_feature_size = 9  # 3x3 = 9
        try:
            if self.use_cv2:
                gray = self.cv2.cvtColor(img_bgr, self.cv2.COLOR_BGR2GRAY)
                edges = self.cv2.Canny(gray, 50, 150)
                # 边缘密度（分成9个区域）
                h, w = edges.shape
                h_step, w_step = h // 3, w // 3
                edge_densities = []
                for i in range(3):
                    for j in range(3):
                        region = edges[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                        density = np.sum(region > 0) / (region.size + 1e-8)
                        edge_densities.append(density)
                features.extend(edge_densities)
            else:
                # 使用PIL的简单边缘检测
                from PIL import ImageFilter
                edges = image.convert('L').filter(ImageFilter.FIND_EDGES)
                edge_array = np.array(edges)
                # 简化版边缘密度
                h, w = edge_array.shape
                h_step, w_step = h // 3, w // 3
                edge_densities = []
                for i in range(3):
                    for j in range(3):
                        region = edge_array[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                        density = np.sum(region > 128) / (region.size + 1e-8)
                        edge_densities.append(density)
                features.extend(edge_densities)
        except Exception as e:
            print(f"⚠️ 边缘特征提取失败: {e}")
            # 使用默认值确保维度一致
            features.extend([0.0] * edge_feature_size)
        
        # 3. 主要颜色特征（提取卡面主色调）
        try:
            # 使用K-means提取主要颜色（简化版：直接采样）
            # 确保图片是RGB格式
            img_rgb = image.convert('RGB')
            img_resized = img_rgb.resize((100, 100))
            img_array = np.array(img_resized)
            
            # 检查数组形状，确保是 (height, width, 3) 格式
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                pixels = img_array.reshape(-1, 3)
            elif len(img_array.shape) == 2:
                # 如果是灰度图，转换为RGB
                pixels = np.stack([img_array, img_array, img_array], axis=-1).reshape(-1, 3)
            else:
                # 其他情况，尝试直接使用
                pixels = img_array.reshape(-1, img_array.shape[-1] if len(img_array.shape) > 2 else 1)
                if pixels.shape[1] != 3:
                    # 如果无法转换为3通道，使用默认值
                    pixels = np.array([[128, 128, 128]] * 10000)  # 使用灰色作为默认值
            
            # 采样部分像素
            sample_size = min(1000, len(pixels))
            if len(pixels) > sample_size:
                indices = np.random.choice(len(pixels), sample_size, replace=False)
                pixels = pixels[indices]
            
            # 计算主要颜色（RGB均值）
            if pixels.shape[1] == 3:
                main_colors = np.mean(pixels, axis=0)
                features.extend(main_colors / 255.0)  # 归一化到0-1
            else:
                # 如果维度不对，使用默认值
                features.extend([0.5, 0.5, 0.5])  # 灰色
        except Exception as e:
            print(f"⚠️ 主色特征提取失败: {e}")
            # 使用默认值避免特征维度不一致
            features.extend([0.5, 0.5, 0.5])  # 灰色
        
        # 4. 图像尺寸和宽高比（反映卡面比例）
        w, h = image.size
        aspect_ratio = h / (w + 1e-8)
        features.append(aspect_ratio)
        # 归一化的尺寸
        total_pixels = w * h
        features.append(np.log(total_pixels / 1000000.0))  # 对数归一化
        
        return np.array(features, dtype=np.float32)
    
    def compute_style_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        计算两个样式特征向量的相似度
        
        Args:
            features1: 第一个图片的样式特征
            features2: 第二个图片的样式特征
            
        Returns:
            相似度分数（0-1之间）
        """
        # 使用余弦相似度
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        denom = norm1 * norm2 + 1e-8
        similarity = float(dot_product / denom) if denom > 0 else 0.0
        # 将余弦相似度从[-1, 1]映射到[0, 1]
        similarity = (similarity + 1.0) / 2.0
        return similarity


# 简化版RAG实现（基于样式特征而非CLIP）
class SimpleRAGStore:
    """简化版RAG存储，基于卡面样式特征计算相似度"""
    
    def __init__(self, use_style_features=True):
        """
        初始化RAG存储
        
        Args:
            use_style_features: 是否使用样式特征（True）或CLIP特征（False）
        """
        self.use_style_features = use_style_features
        self.style_extractor = StyleFeatureExtractor() if use_style_features else None
        
        if not use_style_features:
            # 使用CLIP模型（原有方式）
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch
                self.torch = torch
                
                # 检查torch版本
                torch_version = torch.__version__
                print(f"检测到torch版本: {torch_version}")
                
                # 检查torch版本是否满足要求（>=2.6）
                try:
                    from packaging import version
                    if version.parse(torch_version) < version.parse("2.6.0"):
                        print(f"⚠️ 警告: torch版本 {torch_version} 低于2.6，可能存在安全漏洞")
                        print("建议升级: pip install --upgrade torch>=2.6")
                except ImportError:
                    # 如果没有packaging库，使用简单字符串比较
                    try:
                        major, minor = map(int, torch_version.split('.')[:2])
                        if major < 2 or (major == 2 and minor < 6):
                            print(f"⚠️ 警告: torch版本 {torch_version} 可能低于2.6，建议升级")
                    except:
                        pass
                
                # 尝试加载CLIP模型（transformers会自动尝试使用safetensors如果可用）
                try:
                    print(f"正在加载CLIP模型")
                    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    # 尝试使用safetensors格式加载（如果模型支持）
                    try:
                        # 使用use_safetensors参数（如果transformers版本支持）
                        self.model = CLIPModel.from_pretrained(
                            "openai/clip-vit-base-patch32",
                            use_safetensors=True,
                            low_cpu_mem_usage=True
                        )
                    except TypeError:
                        # 如果use_safetensors参数不支持，使用默认方式
                        # transformers会自动选择safetensors如果可用
                        self.model = CLIPModel.from_pretrained(
                            "openai/clip-vit-base-patch32",
                            low_cpu_mem_usage=True
                        )
                    self.model.eval()
                    
                except Exception as load_error:
                    error_str = str(load_error)
                    # 检查是否是torch版本问题
                    if "torch.load" in error_str or "CVE-2025-32434" in error_str or "requires users" in error_str.lower():
                        raise ImportError(
                            f"❌ torch版本过低，存在安全漏洞！\n"
                            f"当前版本: {torch_version}\n"
                            f"请升级torch到至少v2.6:\n"
                            f"  pip install --upgrade torch>=2.6\n"
                            f"或者使用safetensors格式的模型（如果可用）。\n"
                            f"详细错误: {error_str}"
                        )
                    else:
                        raise ImportError(f"无法加载CLIP模型: {load_error}")
                
                print("✅ CLIP模型加载成功")
            except ImportError as e:
                raise ImportError(f"无法加载CLIP模型: {e}")
        else:
            print("✅ 使用样式特征提取（基于颜色、布局、边缘）")
        
        self.image_embeddings = []  # 存储样式特征或CLIP嵌入
        self.image_metadatas = []
    
    def load_images_from_folder(self, folder_path):
        """从文件夹加载图片并生成样式特征或嵌入向量"""
        self.image_embeddings = []
        self.image_metadatas = []
        
        if not os.path.isdir(folder_path):
            print(f"⚠️ 文件夹不存在: {folder_path}")
            return
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and any(filename.lower().endswith(fmt) for fmt in supported_formats):
                image_files.append((file_path, filename))
        
        print(f"找到 {len(image_files)} 张图片，正在生成{'样式特征' if self.use_style_features else '嵌入向量'}...")
        
        for file_path, filename in image_files:
            try:
                image = Image.open(file_path)
                
                if self.use_style_features:
                    # 使用样式特征提取
                    embedding = self.style_extractor.extract_style_features(image)
                else:
                    # 使用CLIP模型
                    inputs = self.processor(images=image, return_tensors="pt")
                    with self.torch.no_grad():
                        image_features = self.model.get_image_features(**inputs)
                    embedding = image_features.numpy().flatten()
                
                self.image_embeddings.append(embedding)
                self.image_metadatas.append({
                    "filename": filename,
                    "source": file_path,
                    "type": "image"
                })
            except Exception as e:
                print(f"⚠️ 处理图片 {filename} 时出错: {e}")
                continue
        
        print(f"✅ 成功加载 {len(self.image_embeddings)} 张图片的{'样式特征' if self.use_style_features else '嵌入向量'}")
    
    def embed_image(self, image):
        """生成图片的样式特征或嵌入向量"""
        if self.use_style_features:
            # 使用样式特征提取
            return self.style_extractor.extract_style_features(image)
        else:
            # 使用CLIP模型
            inputs = self.processor(images=image, return_tensors="pt")
            with self.torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            return image_features.numpy().flatten()
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """计算两个特征向量的相似度"""
        if self.use_style_features:
            # 使用样式相似度计算
            return self.style_extractor.compute_style_similarity(features1, features2)
        else:
            # 使用余弦相似度（CLIP特征）
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            denom = norm1 * norm2 + 1e-8
            similarity = float(dot_product / denom) if denom > 0 else 0.0
            return similarity


class CardOCRWithRAG:
    """卡证OCR识别 - 带RAG增强的Qwen3-VL API调用类"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-plus",
        base_url: Optional[str] = None,
        rag_image_dir: str = "rag_cards",
        persist_directory: str = "./multimodal_chroma_card"
    ):
        """
        初始化卡证OCR识别器
        
        Args:
            api_key: Qwen API密钥，如果为None则从环境变量QWEN_API_KEY或OPENAI_API_KEY读取
            model: Qwen模型名称，默认使用qwen-vl-plus（支持视觉），可选qwen-vl-max, qwen-vl-plus等
            base_url: Qwen API基础URL，如果为None则使用默认的兼容模式端点
            rag_image_dir: RAG图片库目录路径
            persist_directory: RAG向量存储持久化目录
        """
        # Qwen API配置
        # 优先使用传入的api_key，然后环境变量，最后使用默认key
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-c59d629c4b324848a9252e996437666b"
        self.model = model
        # Qwen API 默认使用兼容OpenAI格式的端点
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = None
        self.is_loaded = False
        
        # RAG相关
        self.rag_image_dir = rag_image_dir
        self.persist_directory = persist_directory
        self.card_rag_store = None
        self.card_rag_ready = False
        
    def load_model(self):
        """初始化Qwen API客户端"""
        if self.is_loaded:
            print("✅ Qwen API客户端已经初始化")
            return True
            
        if not OPENAI_AVAILABLE:
            print("❌ openai库未安装，无法使用API")
            print("请安装: pip install openai")
            return False
            
        if not self.api_key:
            print("❌ Qwen API密钥未设置，请设置api_key参数或环境变量QWEN_API_KEY")
            return False
            
        try:
            print(f"正在初始化Qwen API客户端（模型: {self.model}）...")
            print(f"API端点: {self.base_url}")
            
            # 创建OpenAI兼容的客户端（Qwen API使用OpenAI兼容格式）
            client_kwargs = {
                "api_key": self.api_key,
                "base_url": self.base_url
            }
                
            self.client = openai.OpenAI(**client_kwargs)
            self.is_loaded = True
            print("✅ Qwen API客户端初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ Qwen API客户端初始化失败: {str(e)}")
            return False
    
    def load_rag_library(self):
        """加载RAG图片库"""
        if self.card_rag_ready:
            return self.card_rag_store is not None
            
        if not RAG_AVAILABLE:
            print("⚠️ RAG功能不可用（需要安装transformers或multimodal_rag模块）")
            self.card_rag_ready = True
            return False
            
        try:
            if not os.path.isdir(self.rag_image_dir):
                print(f"⚠️ RAG图片库目录不存在: {self.rag_image_dir}")
                self.card_rag_ready = True
                return False
                
            print(f"正在加载RAG图片库: {self.rag_image_dir}")
            
            # 优先使用 multimodal_rag 模块
            if MultiModalDocumentLoader and MultiModalVectorStore:
                try:
                    loader = MultiModalDocumentLoader()
                    docs = loader.load_images_from_folder(self.rag_image_dir)
                    
                    if not docs:
                        print("⚠️ RAG图片库为空")
                        self.card_rag_ready = True
                        return False
                        
                    print(f"找到 {len(docs)} 张图片，正在建立向量索引...")
                    store = MultiModalVectorStore(persist_directory=self.persist_directory)
                    store.create_vector_store(docs)
                    self.card_rag_store = store
                    self.card_rag_ready = True
                    print(f"✅ RAG图片库加载成功（使用multimodal_rag），共 {len(store.image_embeddings)} 张图片")
                    return True
                except Exception as e:
                    print(f"⚠️ 使用multimodal_rag加载失败，尝试使用简化版: {e}")
            
            # 使用简化版RAG（默认使用样式特征，更适用于卡面样式匹配）
            try:
                print("使用简化版RAG功能（基于卡面样式特征）...")
                store = SimpleRAGStore(use_style_features=True)  # 使用样式特征而非CLIP
                store.load_images_from_folder(self.rag_image_dir)
                
                if not store.image_embeddings:
                    print("⚠️ RAG图片库为空")
                    self.card_rag_ready = True
                    return False
                
                self.card_rag_store = store
                self.card_rag_ready = True
                print(f"✅ RAG图片库加载成功（使用简化版），共 {len(store.image_embeddings)} 张图片")
                return True
            except ImportError as e:
                # ImportError通常表示torch版本问题或依赖缺失
                error_str = str(e)
                if "torch" in error_str.lower() or "CVE" in error_str or "version" in error_str.lower():
                    print(f"⚠️ 简化版RAG加载失败（torch版本问题）:")
                    print(f"   {error_str}")
                    print("\n💡 解决方案:")
                    print("   1. 升级torch: pip install --upgrade torch>=2.6")
                    print("   2. 或者暂时禁用RAG功能，直接使用OCR识别")
                    print("   3. 或者使用gradio_unified.py中的RAG功能（如果可用）")
                else:
                    print(f"⚠️ 简化版RAG加载失败: {error_str}")
                self.card_rag_store = None
                self.card_rag_ready = True
                return False
            except Exception as e:
                print(f"⚠️ 简化版RAG加载失败: {str(e)}")
                self.card_rag_store = None
                self.card_rag_ready = True
                return False
            
        except Exception as e:
            print(f"⚠️ RAG图片库加载失败: {str(e)}")
            self.card_rag_store = None
            self.card_rag_ready = True
            return False
    
    def _rag_search(self, image: Image.Image, top_k: int = 3) -> List[Dict]:
        """
        对输入图片进行RAG检索，返回相似图片信息
        
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
    
    def _image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """
        将PIL Image转换为base64编码的字符串
        
        Args:
            image: PIL Image对象
            format: 图片格式，默认PNG
            
        Returns:
            base64编码的图片字符串（data URI格式）
        """
        buffer = BytesIO()
        image.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # 根据格式确定MIME类型
        mime_types = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "WEBP": "image/webp"
        }
        mime_type = mime_types.get(format.upper(), "image/png")
        
        return f"data:{mime_type};base64,{img_base64}"
    
    def _build_enhanced_prompt(
        self,
        base_prompt: str,
        rag_results: List[Dict],
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        构建增强后的提示词（包含RAG检索结果）
        
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
            prompt = prompt+ (
                f"6. 如果是银行卡且字段列表包含'卡面类型'，则按照以下规则填充：\n"
                f"  - 基于图片库检索到的相似卡证结果{filenames}，填充“卡面类型”字段。字段值规则如下：\n"
                f"       -**禁止**自定义、生成、猜测或编造新的卡面类型值。\n"
                f"       -当出现任何不确定、模糊或不匹配情况时，“卡面类型”字段的值**必须且只能为“其他”**。\n"
                f"       -若识别出的“发卡行”字段的值存在与{banks}中银行名称相同的情况，"
                f"则“卡面类型”字段的值只能从{filenames}中**严格选择一个**。\n"

            )
            
        return prompt
    
    def recognize_card(
        self,
        image: Image.Image,
        custom_prompt: Optional[str],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.8,
        use_rag: bool = True,
        top_k_rag: int = 3
    ) -> Dict:
        """
        识别卡证图片
        
        Args:
            image: 输入图片（PIL Image）
            custom_prompt: 自定义提示词，如果为None则使用默认提示词
            max_tokens: 最大生成长度
            temperature: 温度参数（0.0-2.0）
            top_p: top_p采样参数（0.0-1.0）
            use_rag: 是否使用RAG增强
            top_k_rag: RAG检索返回的相似图片数量
            
        Returns:
            识别结果字典，包含：
            - result: OCR识别结果文本
            - rag_info: RAG检索信息（如果有）
            - generation_time: 生成耗时
            - success: 是否成功
            - error: 错误信息（如果有）
        """
        if not self.is_loaded:
            return {
                "success": False,
                "error": "模型未加载，请先调用load_model()",
                "result": None,
                "rag_info": None,
                "generation_time": 0
            }
        
        # 默认提示词
        # default_prompt = (
        #     "你是专业的卡证OCR引擎。请对图片进行结构化识别：\n"
        #     "1) 判断卡证类型（身份证/银行卡/驾驶证/护照/工牌/其他）；\n"
        #     "2) 以Markdown表格输出关键字段和值；字段示例：姓名/姓名(EN)、性别、民族、生日、住址、公民身份号码、签发机关、有效期限、卡号、有效期、发卡行等，卡号中只能包含数字；\n"
        #     "3) 若有头像或水印信息，请在表格下方以文本补充说明；\n"
        #     "4) 保持原图文字内容尽量完整，不要输出围栏代码块；\n"
        #     "5) 如果和给定的卡证图片库中的图片相似，请在表格下方给出相似度，并给出相似卡证的图片名称。"
        # )
        default_prompt = None

        # RAG检索
        rag_results = []
        if use_rag and self.card_rag_store:
            rag_results = self._rag_search(image, top_k=top_k_rag)
        
        # 构建增强提示词
        enhanced_prompt = self._build_enhanced_prompt(
            custom_prompt,
            rag_results,
            default_prompt
        )
        
        # 将图片转换为base64
        image_base64 = self._image_to_base64(image)
        
        # 在终端输出发送给API的完整prompt
        print("\n" + "=" * 80)
        print("📝 发送给API的完整Prompt")
        print("=" * 80)
        print(enhanced_prompt)
        print("=" * 80 + "\n")
        
        # 准备Qwen API消息格式（兼容OpenAI格式）
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": enhanced_prompt
                    }
                ]
            }
        ]
        
        # 调用Qwen API
        try:
            start_time = time.time()
            
            # 准备API参数
            api_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # top_p参数：如果小于1.0则添加，否则不传（使用默认值）
            if top_p < 1.0:
                api_params["top_p"] = top_p
            
            response = self.client.chat.completions.create(**api_params)
            
            generation_time = time.time() - start_time
            
            # 提取响应文本
            result_text = response.choices[0].message.content
            
            # 构建RAG信息
            rag_info = None
            if rag_results:
                rag_info = {
                    "enabled": True,
                    "top_k": len(rag_results),
                    "results": rag_results
                }
            else:
                rag_info = {"enabled": False, "reason": "RAG未启用或图片库为空"}
            
            return {
                "success": True,
                "result": result_text,
                "rag_info": rag_info,
                "generation_time": generation_time,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": None,
                "rag_info": None,
                "generation_time": 0
            }
    
    def recognize_from_file(
        self,
        image_path: str,
        **kwargs
    ) -> Dict:
        """
        从文件路径识别卡证
        
        Args:
            image_path: 图片文件路径
            **kwargs: 传递给recognize_card的其他参数
            
        Returns:
            识别结果字典
        """
        try:
            image = Image.open(image_path)
            return self.recognize_card(image, **kwargs)
        except Exception as e:
            return {
                "success": False,
                "error": f"图片加载失败: {str(e)}",
                "result": None,
                "rag_info": None,
                "generation_time": 0
            }


def main():
    """示例使用"""
    print("=" * 60)
    print("卡证OCR识别 - RAG增强 + Qwen3-VL API调用")
    print("=" * 60)
    
    # 创建识别器实例（会自动使用环境变量或默认API key）
    ocr = CardOCRWithRAG(
        api_key=None,  # 如果为None，会使用环境变量或默认key
        model="qwen-vl-plus",  # 或使用 "qwen-vl-max", "qwen-vl-max-longcontext"
        rag_image_dir="rag_cards",
        persist_directory="./multimodal_chroma_card"
    )
    
    print(f"使用API密钥: {ocr.api_key[:10]}...")
    
    # 初始化Qwen API客户端
    print("\n1. 初始化Qwen API客户端...")
    if not ocr.load_model():
        print("❌ Qwen API客户端初始化失败，退出")
        return
    
    # 加载RAG图片库（可选）
    print("\n2. 加载RAG图片库...")
    ocr.load_rag_library()
    
    # 示例：识别卡证图片
    print("\n3. 开始识别...")
    test_image_path = input("请输入卡证图片路径（或按Enter跳过）: ").strip()
    
    if not test_image_path:
        print("跳过测试")
        return
        
    if not os.path.exists(test_image_path):
        print(f"❌ 文件不存在: {test_image_path}")
        return
    
    # 执行识别
    result = ocr.recognize_from_file(test_image_path, use_rag=True)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("识别结果")
    print("=" * 60)
    
    if result["success"]:
        print(f"\n✅ 识别成功（耗时: {result['generation_time']:.2f}秒）")
        print(f"\n识别结果:\n{result['result']}")
        
        if result["rag_info"] and result["rag_info"]["enabled"]:
            print(f"\n📊 RAG检索信息:")
            print(f"  找到 {result['rag_info']['top_k']} 张相似图片")
            for i, r in enumerate(result["rag_info"]["results"], 1):
                print(f"  {i}. {r['filename']} (相似度: {r['similarity']:.3f})")
    else:
        print(f"\n❌ 识别失败: {result['error']}")


if __name__ == "__main__":
    main()

