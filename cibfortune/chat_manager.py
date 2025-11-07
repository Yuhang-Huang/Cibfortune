#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话记录管理器
处理对话历史的保存、加载和管理
"""

import os
import json
import sqlite3
import base64
from datetime import datetime
from PIL import Image
import io
import hashlib

class ChatManager:
    """对话记录管理器"""
    
    def __init__(self, data_dir="chat_data"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "chat_history.db")
        self.images_dir = os.path.join(data_dir, "images")
        
        # 创建目录
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建会话表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建对话表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                image_path TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        # 创建标签表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                tag TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, name=None):
        """创建新会话"""
        if name is None:
            name = f"会话_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("INSERT INTO sessions (name) VALUES (?)", (name,))
        session_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def save_message(self, session_id, role, content, image=None):
        """保存消息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 保存图像
        image_path = None
        if image:
            image_path = self._save_image(image, session_id)
        
        # 保存消息
        cursor.execute("""
            INSERT INTO conversations (session_id, role, content, image_path)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, content, image_path))
        
        # 更新会话时间
        cursor.execute("""
            UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
        
        return cursor.lastrowid
    
    def _save_image(self, image, session_id):
        """保存图像到本地"""
        # 生成图像文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"session_{session_id}_{timestamp}.jpg"
        filepath = os.path.join(self.images_dir, filename)
        
        # 保存图像
        if isinstance(image, Image.Image):
            image.save(filepath, "JPEG", quality=95)
        else:
            # 如果是base64或其他格式，需要转换
            image.save(filepath, "JPEG", quality=95)
        
        return filepath
    
    def get_sessions(self):
        """获取所有会话列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.id, s.name, s.created_at, s.updated_at,
                   COUNT(c.id) as message_count
            FROM sessions s
            LEFT JOIN conversations c ON s.id = c.session_id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
        """)
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'id': row[0],
                'name': row[1],
                'created_at': row[2],
                'updated_at': row[3],
                'message_count': row[4]
            })
        
        conn.close()
        return sessions
    
    def get_conversation(self, session_id):
        """获取指定会话的对话记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, content, image_path, timestamp
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        conversations = []
        for row in cursor.fetchall():
            image = None
            if row[2]:  # image_path
                try:
                    image = Image.open(row[2])
                except:
                    image = None
            
            conversations.append({
                'role': row[0],
                'content': row[1],
                'image': image,
                'image_path': row[2],
                'timestamp': row[3]
            })
        
        conn.close()
        return conversations
    
    def update_session_name(self, session_id, new_name):
        """更新会话名称"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE sessions SET name = ? WHERE id = ?", (new_name, session_id))
        
        conn.commit()
        conn.close()
    
    def delete_session(self, session_id):
        """删除会话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 删除图像文件
        cursor.execute("SELECT image_path FROM conversations WHERE session_id = ?", (session_id,))
        for row in cursor.fetchall():
            if row[0] and os.path.exists(row[0]):
                try:
                    os.remove(row[0])
                except:
                    pass
        
        # 删除对话记录
        cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        
        # 删除标签
        cursor.execute("DELETE FROM tags WHERE session_id = ?", (session_id,))
        
        # 删除会话
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        
        conn.commit()
        conn.close()
    
    def export_session(self, session_id, format="json"):
        """导出会话"""
        session_info = None
        conversations = self.get_conversation(session_id)
        
        # 获取会话信息
        sessions = self.get_sessions()
        for session in sessions:
            if session['id'] == session_id:
                session_info = session
                break
        
        if format == "json":
            data = {
                'session_info': session_info,
                'conversations': []
            }
            
            for conv in conversations:
                conv_data = {
                    'role': conv['role'],
                    'content': conv['content'],
                    'timestamp': conv['timestamp']
                }
                
                # 处理图像
                if conv['image']:
                    # 将图像转换为base64
                    img_buffer = io.BytesIO()
                    conv['image'].save(img_buffer, format='JPEG')
                    img_str = base64.b64encode(img_buffer.getvalue()).decode()
                    conv_data['image_base64'] = img_str
                
                data['conversations'].append(conv_data)
            
            # 保存到文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"session_{session_id}_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return filepath
        
        elif format == "txt":
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"session_{session_id}_{timestamp}.txt"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"会话: {session_info['name']}\n")
                f.write(f"创建时间: {session_info['created_at']}\n")
                f.write(f"更新时间: {session_info['updated_at']}\n")
                f.write("="*50 + "\n\n")
                
                for conv in conversations:
                    role_icon = "👤" if conv['role'] == "user" else "🤖"
                    f.write(f"{role_icon} {conv['role']} ({conv['timestamp']}):\n")
                    f.write(f"{conv['content']}\n")
                    if conv['image_path']:
                        f.write(f"[包含图像: {conv['image_path']}]\n")
                    f.write("\n" + "-"*30 + "\n\n")
            
            return filepath
    
    def search_conversations(self, query):
        """搜索对话内容"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.session_id, s.name, c.role, c.content, c.timestamp
            FROM conversations c
            JOIN sessions s ON c.session_id = s.id
            WHERE c.content LIKE ?
            ORDER BY c.timestamp DESC
        """, (f"%{query}%",))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'session_id': row[0],
                'session_name': row[1],
                'role': row[2],
                'content': row[3],
                'timestamp': row[4]
            })
        
        conn.close()
        return results
    
    def get_statistics(self):
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总会话数
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]
        
        # 总消息数
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_messages = cursor.fetchone()[0]
        
        # 最近活跃会话
        cursor.execute("""
            SELECT name, updated_at
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT 5
        """)
        recent_sessions = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_sessions': total_sessions,
            'total_messages': total_messages,
            'recent_sessions': recent_sessions
        }

# 全局实例
chat_manager = ChatManager()

