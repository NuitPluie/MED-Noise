import re
import json
import torch
import string
import numpy as np
from tqdm import tqdm
from transformers import LlavaProcessor, LlavaForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer
from PIL import Image
import os

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色

# 模型路径
model_path = "/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/shared/mllm_ckpts/MedM-VL-2D-3B-en"

# 加载模型和处理器
try:
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:3",  # 根据实际GPU编号修改
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("✅ 模型和处理器加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

# 数据路径
noise_name = 'origin'  # 修改为所需的噪声类型，例如 'BC+other'
input_data_path = f'/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/data/MAT-Benchmark/all_noise_100_test/{noise_name}/data.json'

# 提取文件夹名作为输出文件名
folder_name = os.path.basename(os.path.dirname(input_data_path))  # 提取 'BC'
output_filename = f"MedM-VL-2D-3B-en_result/{folder_name}_result.json"

# 确保输出目录存在
os.makedirs("MedM-VL-2D-3B-en_result", exist_ok=True)

# 加载数据
if not os.path.exists(input_data_path):
    print(f"❌ 数据文件不存在: {input_data_path}")
    exit(1)

try:
    with open(input_data_path, 'r') as file:
        wikimultihopqa = json.load(file)
    print(f"数据样本数: {len(wikimultihopqa)}")
except Exception as e:
    print(f"❌ 加载数据失败: {e}")
    exit(1)

combine_results = []
for item in tqdm(wikimultihopqa[:]):
    print("########################################")
    try:
        # 获取图像路径
        if item['type'][0] == 'crop':
            input_image_path = item['ori_image_path']
        else:
            input_image_path = item['processed_image_path']
        input_image_path = f'/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/data/MAT-Benchmark/all_noise_100_test/{noise_name}/images/' + input_image_path
        
        # 检查图像文件是否存在
        if not os.path.exists(input_image_path):
            print(f"❌ 图像文件不存在: {input_image_path}")
            pred_answer = "Image not found"
            combine_results.append({'pred_answer': pred_answer, 'gt': item['answer'], 'query': item['question']})
            continue

        # 加载图像
        image = Image.open(input_image_path).convert("RGB")
        
        # 构造输入文本
        query = item['question']
        input_text = query + '\n' + "Answer the question directly. The answer should be very brief."
        print(RED + input_text + RESET)
        print(GREEN + str(item['answer']) + RESET)
        
        # 构造prompt - 使用正确的格式
        prompt = f"USER: <image>\n{input_text}\nASSISTANT:"
        
        # 🔧 修复：正确的参数顺序 - images 在前，text 在后
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        # 推理
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,  # 降低生成长度
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                use_cache=True,
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.eos_token_id  # 添加 pad_token_id
            )
        
        # 🔧 修复：使用 tokenizer 而不是 processor 进行解码
        result = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # 提取回答部分
        if "ASSISTANT:" in result:
            pred_answer = result.split("ASSISTANT:")[-1].strip()
        elif "USER:" in result:
            # 如果包含完整对话，提取最后的回答
            parts = result.split("ASSISTANT:")
            if len(parts) > 1:
                pred_answer = parts[-1].strip()
            else:
                pred_answer = result.split("USER:")[-1].strip()
        else:
            pred_answer = result.strip()
        
        print(YELLOW + pred_answer + RESET)

    except Exception as e:
        print(f"❌ 推理失败: {e}")
        print(f"错误详情: {type(e).__name__}: {str(e)}")
        pred_answer = "Error in processing"
    
    combine_results.append(
        {'pred_answer': pred_answer, 'gt': item['answer'], 'query': item['question']}
    )

# 保存结果
try:
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(combine_results, f, ensure_ascii=False, indent=4)
    print(f"✅ 结果已保存到: {output_filename}")
    print(f"📊 总共处理了 {len(combine_results)} 个样本")
except Exception as e:
    print(f"❌ 保存结果失败: {e}")