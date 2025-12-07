# Cibfortune
兴火燎原创新大赛

模型下载：
```
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_TOKEN'] = 'YOUR_TOKEN'
os.system(
    'huggingface-cli download Qwen/Qwen3-VL-8B-Instruct '
    '--local-dir xxxx/models/Qwen3-VL-8B-Instruct '
    '--local-dir-use-symlinks False '
    '--resume-download '
    '--include "*"'
)
```

数据集链接：https://github.com/Yuhang-Huang/Cibfortune/tree/main/cibfortune/datasets
