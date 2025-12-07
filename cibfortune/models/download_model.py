import os
# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_TOKEN'] = '' #需要自行去huggingface官网申请token
os.system(
    'huggingface-cli download Qwen/Qwen3-VL-8B-Instruct '
    '--local-dir /home/wulin/Huangyh/models/Qwen3-VL-8B-Instruct '
    '--local-dir-use-symlinks False '
    '--resume-download '
    '--include "*"'
)