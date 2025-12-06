import base64
import requests
import time
import json
import pandas as pd
from pathlib import Path

def get_image_base64(image_path):
    """
    读取图片并转换为Base64字符串
    """
    with open(image_path, 'rb') as f:
        # 读取文件内容
        image_data = f.read()
        # 转换为base64并解码为utf-8字符串
        base64_str = base64.b64encode(image_data).decode('utf8')
    return base64_str

def ocr_via_api(folder_path, output_excel_path, api_url, token):
    """
    遍历文件夹，通过API进行识别
    """
    input_dir = Path(folder_path)
    if not input_dir.exists():
        print(f"错误: 文件夹 {folder_path} 不存在")
        return

    # 支持的图片格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_extensions]
    total_files = len(files)
    
    data_list = []
    
    # 设置请求头
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json"
    }

    print(f"连接 API 地址: {api_url}")
    print(f"开始处理 {total_files} 个文件...")

    for index, file_path in enumerate(files, 1):
        file_name = file_path.name
        print(f"[{index}/{total_files}] 发送请求: {file_name} ...", end="", flush=True)
        
        try:
            # 1. 准备 Base64 数据
            img_b64 = get_image_base64(file_path)
            
            payload = {
                "file": img_b64,
                "fileType": 1,
                "useDocOrientationClassify": False,
                "useDocUnwarping": False,
                "useTextlineOrientation": False,
            }
            
            # 2. 发送请求并计时
            start_time = time.time()
            response = requests.post(url=api_url, headers=headers, data=json.dumps(payload))
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            
            # 3. 解析响应
            # 期望格式: {'msg': '', 'results': [{'data': [{'text': '...', ...}]}], 'status': '0'}
            resp_json = response.json()
            #print(resp_json)
            ocr_result = resp_json["result"]
            ocr_results = ocr_result.get("ocrResults", [])
            print(ocr_results)
            # 4. 记录数据
            # data_list.append({
            #     "文件名": file_name,
            #     "API耗时(秒)": round(elapsed_time, 4),
            #     "识别内容": extracted_text,
            #     "文件路径": str(file_path),
            #     "HTTP状态码": response.status_code
            # })
            
            print(f" 完成 (耗时 {elapsed_time:.2f}s)")

        except requests.exceptions.ConnectionError:
            print(" 失败! 无法连接到服务器，请检查 API 是否启动。")
            # 如果连接不上，可能服务挂了，直接退出或记录错误
            data_list.append({"文件名": file_name, "API耗时(秒)": 0, "识别内容": "连接被拒绝", "HTTP状态码": 0})
        except Exception as e:
            print(f" 失败! 错误: {e}")
            data_list.append({"文件名": file_name, "API耗时(秒)": 0, "识别内容": str(e), "HTTP状态码": 0})

    # 保存 Excel
    # if data_list:
    #     df = pd.DataFrame(data_list)
    #     df.to_excel(output_excel_path, index=False)
    #     print(f"\n处理结束，结果已保存至: {output_excel_path}")

if __name__ == "__main__":
    # --- 配置区域 ---
    # 你的 PaddleHub 服务地址
    # 如果是本地默认启动，通常是 http://127.0.0.1:8866/predict/ch_pp-ocrv3
    API_URL = "https://g8e2x4l851qer5i3.aistudio-app.com/ocr"
    TOKEN = "0143529bb9cf856a58a342760f5bdc39a85e13b4"
    
    INPUT_FOLDER = r"./dataset/本票"
    OUTPUT_FILE = r"./api_ocr_results.xlsx"
    
    # 自动创建测试文件夹
    Path(INPUT_FOLDER).mkdir(exist_ok=True)
    
    ocr_via_api(INPUT_FOLDER, OUTPUT_FILE, API_URL, TOKEN)