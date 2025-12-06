import base64
import requests
import time
import json
import pandas as pd
from pathlib import Path
import openai
import os
from PIL import Image
from io import BytesIO

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
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

def recognize_card(
    image_base64,
    api_url,
    token,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    top_p: float = 0.8,
) :
    prompt = f"你是专业的票据OCR引擎。请仔细阅读并识别输入图片中的所有内容，并生成相应的HTML表格，需使用rowspan和colspan维持空间结构，禁止输出其他任何内容。\n"
    
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
                    "text": prompt
                }
            ]
        }
    ]
    
    
    # 调用Qwen API（在线模式）
    try:
        client_kwargs = {
            "api_key": token,
            "base_url": api_url
        }
        client = openai.OpenAI(**client_kwargs)
        start_time = time.time()
        
        # 准备API参数
        api_params = {
            "model": "qwen3-vl-plus",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # top_p参数：如果小于1.0则添加，否则不传（使用默认值）
        if top_p < 1.0:
            api_params["top_p"] = top_p
        
        response = client.chat.completions.create(**api_params)
        
        generation_time = time.time() - start_time
        
        # 提取响应文本
        result_text = response.choices[0].message.content
        
        print(result_text)
        return {
            "success": True,
            "result": result_text,
            "generation_time": generation_time,
            "error": None
        }
        
    except Exception as e:
        print(e)



def ocr_via_api(folder_path, output_json_path, api_url, token):
    """
    遍历文件夹，通过API进行识别，并将结果保存为 JSON 文件
    """
    input_dir = Path(folder_path)
    if not input_dir.exists():
        print(f"错误: 文件夹 {folder_path} 不存在")
        return

    # 支持的图片格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_extensions]
    total_files = len(files)
    
    # 用于存储所有结果的列表
    data_list = []

    print(f"连接 API 地址: {api_url}")
    print(f"开始处理 {total_files} 个文件...")

    for index, file_path in enumerate(files, 1):
        file_name = file_path.name
        print(f"[{index}/{total_files}] 发送请求: {file_name} ...", end="", flush=True)
        
        try:
            # 1. 准备 Base64 数据
            img_b64 = image_to_base64(Image.open(file_path))
            
            res = recognize_card(img_b64, api_url, token)

            # 4. 【修改点】收集成功的数据
            result_item = {
                "filename": file_name,
                "result": res["result"],         # 识别结果列表
                "time": res["generation_time"]   # 耗时
            }
            data_list.append(result_item)
            
            print(f" 完成 (耗时 {res["generation_time"]:.2f}s)")

        except requests.exceptions.ConnectionError:
            print(" 失败! 无法连接到服务器。")
            data_list.append({
                "filename": file_name,
                "result": None,
                "error": "连接被拒绝",
                "time": 0
            })
        except Exception as e:
            print(f" 失败! 错误: {e}")
            data_list.append({
                "filename": file_name,
                "result": None,
                "error": str(e),
                "time": 0
            })

    # 5. 【修改点】将结果写入 JSON 文件
    if data_list:
        print(f"\n正在保存结果到 {output_json_path} ...")
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                # ensure_ascii=False 保证输出的是中文而不是 Unicode 编码
                # indent=4 保证输出格式化，易于阅读
                json.dump(data_list, f, ensure_ascii=False, indent=4)
            print("保存完成！")
        except Exception as e:
            print(f"保存文件失败: {e}")
    else:
        print("没有产生任何数据。")


if __name__ == "__main__":
    # --- 配置区域 ---
    # 你的 PaddleHub 服务地址
    # 如果是本地默认启动，通常是 http://127.0.0.1:8866/predict/ch_pp-ocrv3
    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    TOKEN = "sk-7236eb30f8c94bfdb7113c89f907b490"
    DATASET_PATH = "./dataset"
    RESULT_PATH = "./res"
    
    base_folder = Path(DATASET_PATH)

    type = None
    subdirs = [item.name for item in base_folder.iterdir() if item.is_dir()]

    #test all subdir in dataset
    if not type:
        for subdir in subdirs:
            INPUT_FOLDER = f"{DATASET_PATH}/{subdir}"
            OUTPUT_FILE = f"{RESULT_PATH}/{subdir}_baseline_qwenvl.json"

            # 自动创建测试文件夹
            Path(INPUT_FOLDER).mkdir(exist_ok=True)
            
            ocr_via_api(INPUT_FOLDER, OUTPUT_FILE, API_URL, TOKEN)
    #test specified type
    else:
        INPUT_FOLDER = f"{DATASET_PATH}/{type}"
        OUTPUT_FILE = f"{RESULT_PATH}/{type}_baseline_qwenvl.json"

        # 自动创建测试文件夹
        Path(INPUT_FOLDER).mkdir(exist_ok=True)
        
        ocr_via_api(INPUT_FOLDER, OUTPUT_FILE, API_URL, TOKEN)