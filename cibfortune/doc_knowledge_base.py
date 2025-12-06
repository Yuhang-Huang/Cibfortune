#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档知识库模块
支持文本向量化、存储和检索，用于RAG增强问答
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np

# 尝试导入向量数据库和embedding库
VECTOR_STORE_AVAILABLE = False
EMBEDDING_AVAILABLE = False

# 方式1: 使用Chroma（推荐，轻量级）
try:
    import chromadb
    from chromadb.config import Settings
    VECTOR_STORE_AVAILABLE = True
    VECTOR_STORE_TYPE = "chroma"
except ImportError:
    pass

# 方式2: 使用FAISS（备选）
if not VECTOR_STORE_AVAILABLE:
    try:
        import faiss
        VECTOR_STORE_AVAILABLE = True
        VECTOR_STORE_TYPE = "faiss"
    except ImportError:
        pass

# 尝试导入embedding模型
# 方式1: sentence-transformers（推荐）
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
    EMBEDDING_TYPE = "sentence_transformers"
except ImportError:
    pass

# 方式2: 使用transformers直接加载模型（备选）
if not EMBEDDING_AVAILABLE:
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        EMBEDDING_AVAILABLE = True
        EMBEDDING_TYPE = "transformers"
    except ImportError:
        pass


class DocumentKnowledgeBase:
    """文档知识库类，支持文本向量化、存储和检索"""
    
    def __init__(
        self,
        persist_directory: str = "./doc_knowledge_base",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "documents",
        qwen3vl_model = None,
        qwen3vl_processor = None
    ):
        """
        初始化文档知识库
        
        Args:
            persist_directory: 持久化目录路径
            embedding_model: embedding模型名称或路径（当使用qwen3-vl时可以为None）
            collection_name: 向量数据库集合名称
            qwen3vl_model: Qwen3-VL模型实例（如果提供，将使用其文本编码器）
            qwen3vl_processor: Qwen3-VL处理器实例（如果提供，将使用其tokenizer）
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # 创建持久化目录
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化embedding模型
        self.embedding_model = None
        self.embedding_dim = None
        self.use_qwen3vl = qwen3vl_model is not None and qwen3vl_processor is not None
        self.qwen3vl_model = qwen3vl_model
        self.qwen3vl_processor = qwen3vl_processor
        
        if self.use_qwen3vl:
            self._load_qwen3vl_embedding()
        else:
            self._load_embedding_model()
        
        # 初始化向量存储
        self.vector_store = None
        self.doc_metadata = {}  # 文档元数据 {doc_id: {filename, timestamp, chunk_count, ...}}
        self._load_vector_store()
        
        # 加载文档元数据
        self._load_metadata()

    def _get_embed_tokens_module(self):
        """
        兼容不同版本Qwen3-VL的embedding获取方式。
        优先返回nn.Embedding模块或可调用的输入嵌入函数。
        """
        # 直接属性
        candidates = [
            getattr(self.qwen3vl_model, "embed_tokens", None),
            getattr(getattr(self.qwen3vl_model, "model", None), "embed_tokens", None),
            getattr(getattr(self.qwen3vl_model, "language_model", None), "embed_tokens", None),
            getattr(getattr(self.qwen3vl_model, "transformer", None), "embed_tokens", None),
        ]
        for cand in candidates:
            if cand is not None:
                return cand
        # get_input_embeddings 返回的通常是nn.Embedding
        if hasattr(self.qwen3vl_model, "get_input_embeddings"):
            try:
                return self.qwen3vl_model.get_input_embeddings()
            except Exception:
                pass
        return None
    
    def _load_qwen3vl_embedding(self):
        """使用Qwen3-VL模型的文本编码器"""
        try:
            import torch
            # 获取tokenizer
            if hasattr(self.qwen3vl_processor, 'tokenizer'):
                self.tokenizer = self.qwen3vl_processor.tokenizer
            else:
                self.tokenizer = self.qwen3vl_processor
            
            # 获取文本编码器 - 尝试多种方式
            self.text_model = None
            if hasattr(self.qwen3vl_model, 'model') and hasattr(self.qwen3vl_model.model, 'text_model'):
                self.text_model = self.qwen3vl_model.model.text_model
            elif hasattr(self.qwen3vl_model, 'text_model'):
                self.text_model = self.qwen3vl_model.text_model
            
            # 获取embedding维度 - 从配置中获取
            if hasattr(self.qwen3vl_model, 'config'):
                if hasattr(self.qwen3vl_model.config, 'text_config') and hasattr(self.qwen3vl_model.config.text_config, 'hidden_size'):
                    self.embedding_dim = self.qwen3vl_model.config.text_config.hidden_size
                elif hasattr(self.qwen3vl_model.config, 'hidden_size'):
                    self.embedding_dim = self.qwen3vl_model.config.hidden_size
                else:
                    # 默认值，通过测试获取
                    try:
                        test_emb = self._encode_text_qwen3vl(["test"])
                        self.embedding_dim = test_emb.shape[1]
                    except:
                        self.embedding_dim = 4096  # qwen3-vl-8b的默认维度
            else:
                # 如果没有config，通过测试获取
                try:
                    test_emb = self._encode_text_qwen3vl(["test"])
                    self.embedding_dim = test_emb.shape[1]
                except:
                    self.embedding_dim = 4096  # qwen3-vl-8b的默认维度
            
            print(f"✅ 使用Qwen3-VL文本编码器，维度: {self.embedding_dim}")
            return True
        except Exception as e:
            print(f"❌ 加载Qwen3-VL文本编码器失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_embedding_model(self):
        """加载独立embedding模型"""
        if not EMBEDDING_AVAILABLE:
            print("⚠️ 未安装embedding库，知识库功能将不可用")
            print("请安装: pip install sentence-transformers 或 pip install transformers torch")
            return False
        
        try:
            if EMBEDDING_TYPE == "sentence_transformers":
                print(f"正在加载embedding模型: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                # 获取embedding维度
                test_embedding = self.embedding_model.encode(["test"])
                self.embedding_dim = test_embedding.shape[1]
                print(f"✅ Embedding模型加载成功，维度: {self.embedding_dim}")
                return True
            elif EMBEDDING_TYPE == "transformers":
                print(f"正在加载embedding模型: {self.embedding_model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
                self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
                self.embedding_model.eval()
                # 获取embedding维度
                self.embedding_dim = self.embedding_model.config.hidden_size
                print(f"✅ Embedding模型加载成功，维度: {self.embedding_dim}")
                return True
        except Exception as e:
            print(f"❌ Embedding模型加载失败: {e}")
            return False
    
    def _encode_text_qwen3vl(self, texts: List[str]) -> np.ndarray:
        """
        使用Qwen3-VL模型的文本编码器将文本列表编码为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量数组 (n_texts, embedding_dim)
        """
        import torch
        if not hasattr(self, 'tokenizer') or not self.tokenizer:
            raise ValueError("Qwen3-VL tokenizer未加载")
        
        with torch.no_grad():
            # 使用tokenizer对文本进行编码
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=2048,  # qwen3-vl支持更长上下文
                return_tensors="pt"
            )
            
            # 移动到模型所在的设备
            device = next(self.qwen3vl_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 使用text_model获取文本embedding
            if self.text_model:
                try:
                    outputs = self.text_model(**inputs)
                    # 使用最后一层的隐藏状态的均值作为embedding
                    if hasattr(outputs, 'last_hidden_state'):
                        hidden_states = outputs.last_hidden_state
                    elif hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
                        hidden_states = outputs.hidden_states[-1]
                    elif isinstance(outputs, tuple) and len(outputs) > 0:
                        hidden_states = outputs[0]
                    else:
                        raise ValueError("无法从text_model输出中提取hidden_states")
                    
                    # 使用attention_mask进行加权平均（如果有）
                    attention_mask = inputs.get('attention_mask', None)
                    if attention_mask is not None:
                        # 扩展attention_mask维度以匹配hidden_states
                        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                        # 对有效token求加权平均
                        sum_embeddings = (hidden_states * attention_mask_expanded).sum(dim=1)
                        sum_mask = attention_mask_expanded.sum(dim=1)
                        embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                    else:
                        embeddings = hidden_states.mean(dim=1).cpu().numpy()
                except Exception as e:
                    print(f"⚠️ 使用text_model编码失败: {e}，尝试使用embed_tokens")
                    # 回退到使用embed_tokens
                    input_ids = inputs['input_ids']
                    embed_tokens_module = self._get_embed_tokens_module()
                    if embed_tokens_module is not None:
                        embeddings_tensor = embed_tokens_module(input_ids)
                    else:
                        raise ValueError("无法找到embed_tokens层")
                    
                    attention_mask = inputs.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                        sum_embeddings = (embeddings_tensor * attention_mask_expanded).sum(dim=1)
                        sum_mask = attention_mask_expanded.sum(dim=1)
                        embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                else:
                    embeddings = embeddings_tensor.mean(dim=1).cpu().numpy()
            else:
                # 如果没有text_model，使用embed_tokens
                input_ids = inputs['input_ids']
                embed_tokens_module = self._get_embed_tokens_module()
                if embed_tokens_module is None:
                    raise ValueError("无法找到text_model或embed_tokens层")
                embeddings_tensor = embed_tokens_module(input_ids)
                
                # 使用attention_mask进行加权平均
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                    sum_embeddings = (embeddings_tensor * attention_mask_expanded).sum(dim=1)
                    sum_mask = attention_mask_expanded.sum(dim=1)
                    embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                else:
                    embeddings = embeddings_tensor.mean(dim=1).cpu().numpy()
            
            return np.array(embeddings)
    
    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表编码为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量数组 (n_texts, embedding_dim)
        """
        if self.use_qwen3vl:
            return self._encode_text_qwen3vl(texts)
        
        if not self.embedding_model:
            raise ValueError("Embedding模型未加载")
        
        if EMBEDDING_TYPE == "sentence_transformers":
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return np.array(embeddings)
        elif EMBEDDING_TYPE == "transformers":
            import torch
            with torch.no_grad():
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                outputs = self.embedding_model(**inputs)
                # 使用[CLS] token的embedding或平均pooling
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            return embeddings
        else:
            raise ValueError("不支持的embedding类型")
    
    def _load_vector_store(self):
        """加载向量存储"""
        if not VECTOR_STORE_AVAILABLE:
            print("⚠️ 未安装向量数据库库，使用内存存储")
            print("请安装: pip install chromadb 或 pip install faiss-cpu")
            self.vector_store = {"embeddings": [], "texts": [], "ids": [], "metadatas": []}
            return
        
        try:
            if VECTOR_STORE_TYPE == "chroma":
                # 使用Chroma
                chroma_client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
                # 获取或创建集合
                try:
                    self.collection = chroma_client.get_collection(name=self.collection_name)
                    print(f"✅ 加载已有向量数据库，包含 {self.collection.count()} 条记录")
                except:
                    self.collection = chroma_client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    print("✅ 创建新的向量数据库")
                self.vector_store = "chroma"
                return True
            elif VECTOR_STORE_TYPE == "faiss":
                # 使用FAISS
                index_path = os.path.join(self.persist_directory, "faiss.index")
                metadata_path = os.path.join(self.persist_directory, "faiss_metadata.json")
                
                if os.path.exists(index_path) and self.embedding_dim:
                    self.index = faiss.read_index(index_path)
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        self.faiss_metadata = json.load(f)
                    print(f"✅ 加载FAISS索引，包含 {self.index.ntotal} 条记录")
                else:
                    if not self.embedding_dim:
                        self.embedding_dim = 384  # 默认维度
                    self.index = faiss.IndexFlatIP(self.embedding_dim)  # 内积索引（用于余弦相似度）
                    self.faiss_metadata = {"texts": [], "ids": [], "metadatas": []}
                    print("✅ 创建新的FAISS索引")
                self.vector_store = "faiss"
                return True
        except Exception as e:
            print(f"⚠️ 向量存储加载失败: {e}，使用内存存储")
            self.vector_store = {"embeddings": [], "texts": [], "ids": [], "metadatas": []}
            return False
    
    def _save_vector_store(self):
        """保存向量存储"""
        if self.vector_store == "faiss":
            index_path = os.path.join(self.persist_directory, "faiss.index")
            metadata_path = os.path.join(self.persist_directory, "faiss_metadata.json")
            faiss.write_index(self.index, index_path)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.faiss_metadata, f, ensure_ascii=False, indent=2)
        # Chroma会自动持久化，无需手动保存
    
    def _load_metadata(self):
        """加载文档元数据"""
        metadata_path = os.path.join(self.persist_directory, "doc_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.doc_metadata = json.load(f)
                print(f"✅ 加载文档元数据，共 {len(self.doc_metadata)} 个文档")
            except Exception as e:
                print(f"⚠️ 加载文档元数据失败: {e}")
                self.doc_metadata = {}
        else:
            self.doc_metadata = {}
    
    def _save_metadata(self):
        """保存文档元数据"""
        metadata_path = os.path.join(self.persist_directory, "doc_metadata.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.doc_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存文档元数据失败: {e}")
    
    def add_document(
        self,
        text: str,
        filename: str,
        chunks: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        添加文档到知识库
        
        Args:
            text: 文档完整文本
            filename: 文档文件名
            chunks: 文本切片列表（如果为None，会自动切片）
            metadata: 额外元数据
            
        Returns:
            文档ID
        """
        if not self.use_qwen3vl and not self.embedding_model:
            raise ValueError("Embedding模型未加载，无法添加文档")
        
        # 生成文档ID
        doc_id = hashlib.md5(f"{filename}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        # 如果没有提供切片，自动切片
        if chunks is None:
            chunks = self._chunk_text(text)
        
        if not chunks:
            raise ValueError("文档文本为空或切片失败")
        
        # 生成embedding
        print(f"正在为文档 '{filename}' 生成 {len(chunks)} 个切片的向量...")
        embeddings = self._encode_text(chunks)
        
        # 添加到向量存储
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        chunk_metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "chunk_count": len(chunks),
                **(metadata or {})
            }
            for i in range(len(chunks))
        ]
        
        if self.vector_store == "chroma":
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=chunk_metadatas
            )
        elif self.vector_store == "faiss":
            # 归一化向量（用于余弦相似度）
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            self.faiss_metadata["texts"].extend(chunks)
            self.faiss_metadata["ids"].extend(chunk_ids)
            self.faiss_metadata["metadatas"].extend(chunk_metadatas)
            self._save_vector_store()
        else:
            # 内存存储
            self.vector_store["embeddings"].extend(embeddings.tolist())
            self.vector_store["texts"].extend(chunks)
            self.vector_store["ids"].extend(chunk_ids)
            self.vector_store["metadatas"].extend(chunk_metadatas)
        
        # 保存文档元数据
        self.doc_metadata[doc_id] = {
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            "text_length": len(text),
            **(metadata or {})
        }
        self._save_metadata()
        
        print(f"✅ 文档 '{filename}' 已添加到知识库，共 {len(chunks)} 个切片")
        return doc_id
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        对文本进行切片
        
        Args:
            text: 要切片的文本
            chunk_size: 每个切片的最大字符数
            overlap: 切片之间的重叠字符数
            
        Returns:
            文本切片列表
        """
        if not text:
            return []
        
        if overlap >= chunk_size:
            overlap = max(1, chunk_size // 10)
        
        chunks = []
        start = 0
        text_length = len(text)
        
        # 限制最大文本长度
        if text_length > 1000000:
            text = text[:1000000]
            text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
            if start <= start - chunk_size:  # 防止无限循环
                start = end
        
        return chunks
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        搜索知识库
        
        Args:
            query: 查询文本
            top_k: 返回最相似的k个结果
            doc_ids: 限制搜索的文档ID列表（可选）
            
        Returns:
            搜索结果列表，每个元素包含 {text, score, metadata}
        """
        if not self.use_qwen3vl and not self.embedding_model:
            return []
        
        # 清洗 doc_ids，避免空字符串/None 导致 where 过滤异常
        if doc_ids:
            doc_ids = [d for d in doc_ids if d]
            if not doc_ids:
                doc_ids = None
        
        # 生成查询向量
        query_embedding = self._encode_text([query])[0]
        
        if self.vector_store == "chroma":
            # 使用Chroma搜索
            where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where
            )
            
            # 格式化结果，兼容空结果或不同返回形态
            ids_list = results.get("ids") or []
            docs_list = results.get("documents") or []
            metas_list = results.get("metadatas") or []
            dists_list = results.get("distances") or []

            # Chroma通常返回二维列表，这里统一成二维结构后再取首行
            def _first_row(lst):
                if not lst:
                    return []
                return lst[0] if isinstance(lst[0], (list, tuple)) else lst

            ids_row = _first_row(ids_list)
            docs_row = _first_row(docs_list)
            metas_row = _first_row(metas_list)
            dists_row = _first_row(dists_list)

            if not ids_row:
                return []

            search_results = []
            for i, chunk_id in enumerate(ids_row):
                try:
                    text_val = docs_row[i] if i < len(docs_row) else ""
                    meta_val = metas_row[i] if i < len(metas_row) else {}
                    dist_val = dists_row[i] if i < len(dists_row) else 0.0
                    search_results.append({
                        "text": text_val,
                        "score": 1 - dist_val if dists_row else 0.0,  # 转换为相似度
                        "metadata": meta_val,
                        "id": chunk_id
                    })
                except Exception:
                    # 防御式处理，避免单条异常导致整体失败
                    continue
            return search_results
        
        elif self.vector_store == "faiss":
            # 使用FAISS搜索
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
            
            search_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.faiss_metadata["texts"]):
                    metadata = self.faiss_metadata["metadatas"][idx]
                    # 如果指定了doc_ids，进行过滤
                    if doc_ids and metadata.get("doc_id") not in doc_ids:
                        continue
                    search_results.append({
                        "text": self.faiss_metadata["texts"][idx],
                        "score": float(score),
                        "metadata": metadata
                    })
            return search_results
        
        else:
            # 内存存储搜索
            if not self.vector_store["embeddings"]:
                return []
            
            # 计算余弦相似度
            embeddings = np.array(self.vector_store["embeddings"])
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return []
            
            # 归一化查询向量
            query_embedding = query_embedding / query_norm
            
            # 计算相似度
            similarities = np.dot(embeddings, query_embedding)
            
            # 获取top_k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            search_results = []
            for idx in top_indices:
                metadata = self.vector_store["metadatas"][idx]
                # 如果指定了doc_ids，进行过滤
                if doc_ids and metadata.get("doc_id") not in doc_ids:
                    continue
                search_results.append({
                    "text": self.vector_store["texts"][idx],
                    "score": float(similarities[idx]),
                    "metadata": metadata
                })
            return search_results
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            是否成功删除
        """
        if doc_id not in self.doc_metadata:
            return False
        
        # 从向量存储中删除
        if self.vector_store == "chroma":
            # 查找所有相关的chunk IDs
            results = self.collection.get(where={"doc_id": doc_id})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
        elif self.vector_store == "faiss":
            # FAISS不支持直接删除，需要重建索引
            # 这里简化处理：标记为删除，在下次添加时重建
            # 实际应用中可以考虑定期重建索引
            pass
        
        # 删除元数据
        del self.doc_metadata[doc_id]
        self._save_metadata()
        
        print(f"✅ 文档 {doc_id} 已删除")
        return True
    
    def list_documents(self) -> List[Dict]:
        """
        列出所有文档
        
        Returns:
            文档列表，每个元素包含 {doc_id, filename, timestamp, chunk_count, ...}
        """
        return list(self.doc_metadata.values())
    
    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.doc_metadata)
    
    def get_chunk_count(self) -> int:
        """获取切片总数"""
        if self.vector_store == "chroma":
            return self.collection.count()
        elif self.vector_store == "faiss":
            return self.index.ntotal
        else:
            return len(self.vector_store.get("texts", []))


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("文档知识库测试")
    print("=" * 60)
    
    kb = DocumentKnowledgeBase()
    
    # 添加测试文档
    test_text = """
    这是一份测试合同文档。
    合同编号：TEST-2024-001
    甲方：测试公司A
    乙方：测试公司B
    合同内容：双方就某项服务达成协议。
    合同金额：10000元
    签署日期：2024年1月1日
    """
    
    doc_id = kb.add_document(test_text, "test_contract.txt")
    print(f"\n添加文档，ID: {doc_id}")
    
    # 搜索测试
    results = kb.search("合同金额是多少", top_k=3)
    print(f"\n搜索结果（共 {len(results)} 条）:")
    for i, result in enumerate(results, 1):
        print(f"{i}. 相似度: {result['score']:.3f}")
        print(f"   文本: {result['text'][:100]}...")
        print(f"   文档: {result['metadata'].get('filename', 'N/A')}")
    
    # 列出文档
    docs = kb.list_documents()
    print(f"\n知识库中共有 {len(docs)} 个文档:")
    for doc in docs:
        print(f"  - {doc['filename']} (ID: {doc.get('doc_id', 'N/A')})")
