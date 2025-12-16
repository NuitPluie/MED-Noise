import re
import json
import torch
import string
import numpy as np
import traceback  # 添加traceback模块
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import os

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

# 模型路径
model_path = "/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/shared/mllm_ckpts/llava-med-v1.5-mistral-7b-hf"

# 加载模型和处理器
print("加载模型...")
try:
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:3",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("✅ 本地模型加载成功")
except Exception as e:
    print(f"本地加载失败: {e}")
    print("尝试从HuggingFace加载...")
    model_path = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:3",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("✅ HuggingFace模型加载成功")

# 数据路径
noise_name = 'BC'
input_data_path = f'/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/data/MAT-Benchmark/all_noise_100_test/{noise_name}/data.json'
folder_name = os.path.basename(os.path.dirname(input_data_path))
output_filename = f"llava-med-v1.5-mistral-7b-hf_result/{folder_name}_result.json"
os.makedirs("llava-med-v1.5-mistral-7b-hf_result", exist_ok=True)

# 加载数据
with open(input_data_path, 'r') as file:
    wikimultihopqa = json.load(file)
print(f"数据样本数: {len(wikimultihopqa)}")

combine_results = []
for item in tqdm(wikimultihopqa[:]):
    try:
        print(f"🔍 处理数据: {item}")  # 打印完整的item数据
        
        # 🔧 修复：安全地获取图像路径
        print(f"🔍 item['type']: {item['type']}, type: {type(item['type'])}")
        print(f"🔍 item['type'][0]: {item['type'][0]}")
        
        if item['type'][0] == 'crop':
            image_path = item['ori_image_path']
            print(f"🔍 使用 ori_image_path: {image_path}, type: {type(image_path)}")
        else:
            image_path = item['processed_image_path']
            print(f"🔍 使用 processed_image_path: {image_path}, type: {type(image_path)}")
        
        # 🔧 处理路径可能是列表的情况
        if isinstance(image_path, list):
            print(f"🔍 image_path 是列表，取第一个元素")
            image_path = image_path[0] if image_path else ""
        image_path = str(image_path)  # 确保是字符串
        print(f"🔍 最终 image_path: {image_path}")
        
        base_path = f'/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/data/MAT-Benchmark/all_noise_100_test/{noise_name}/images/'
        print(f"🔍 base_path: {base_path}")
        print(f"🔍 准备拼接路径...")
        
        input_image_path = base_path + image_path
        print(f"🔍 拼接后的路径: {input_image_path}")
        
        if not os.path.exists(input_image_path):
            pred_answer = "Image not found"
            combine_results.append({'pred_answer': pred_answer, 'gt': item['answer'], 'query': item['question']})
            continue

        # 加载图像
        image = Image.open(input_image_path).convert("RGB")
        
        # 🔧 处理问题可能是列表的情况
        query = item['question']
        print(f"🔍 query: {query}, type: {type(query)}")
        if isinstance(query, list):
            query = query[0] if query else ""
        query = str(query)
        
        input_text = query + '\n' + "Answer the question directly. The answer should be very brief."
        print(RED + input_text + RESET)
        
        # 🔧 修复：处理答案可能是列表的情况
        answer = item['answer']
        print(f"🔍 answer: {answer}, type: {type(answer)}")
        if isinstance(answer, list):
            answer_str = ', '.join(str(ans) for ans in answer)  # 将列表转换为字符串
        else:
            answer_str = str(answer)  # 确保是字符串
        print(GREEN + answer_str + RESET)
        
        # 🔧 使用官方示例的格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": input_text}
                ]
            }
        ]
        
        # 🔧 使用官方的apply_chat_template方法
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 🔧 按照官方示例处理输入
        inputs = processor(
            images=[image], text=prompt, return_tensors="pt"
        ).to(model.device, torch.bfloat16)
        
        # 🔧 使用官方的推理方式
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        # 🔧 使用官方的解码方式
        result = processor.decode(output_ids[0], skip_special_tokens=True)
        
        # 提取回答部分 - 移除输入的prompt部分
        if prompt in result:
            pred_answer = result.replace(prompt, "").strip()
        elif "assistant" in result.lower():
            # 查找assistant后的内容
            parts = result.lower().split("assistant")
            if len(parts) > 1:
                pred_answer = result[result.lower().find("assistant") + len("assistant"):].strip()
            else:
                pred_answer = result.strip()
        else:
            pred_answer = result.strip()
        
        # 清理回答开头的冒号或其他符号
        if pred_answer.startswith(":") or pred_answer.startswith("："):
            pred_answer = pred_answer[1:].strip()
        
        print(YELLOW + pred_answer + RESET)

    except Exception as e:
        print(f"❌ 推理失败: {e}")
        print(f"错误详情: {type(e).__name__}: {str(e)}")
        print(f"完整错误追踪:")
        traceback.print_exc()  # 打印完整的错误追踪
        print(f"问题数据: {item}")  # 添加调试信息
        pred_answer = "Error in processing"
    
    combine_results.append(
        {'pred_answer': pred_answer, 'gt': item['answer'], 'query': item['question']}
    )

# 保存结果
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(combine_results, f, ensure_ascii=False, indent=4)
print(f"✅ 结果已保存到: {output_filename}")
print(f"📊 总共处理了 {len(combine_results)} 个样本")