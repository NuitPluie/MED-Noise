import re
import json
import torch
import string
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色

# 🔧 根据官方示例修改模型ID和加载方式
model_id = "/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/shared/mllm_ckpts/medgemma-4b-pt"

# 🔧 使用AutoModelForImageTextToText替代Gemma3ForConditionalGeneration
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:3",  # 根据实际GPU编号修改
    trust_remote_code=True
).eval()

processor = AutoProcessor.from_pretrained(model_id)

noise_name = 'SM'  # 修改为所需的噪声类型，例如 'BC+other'

input_data_path = f'/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/data/MAT-Benchmark/all_noise_100_test/{noise_name}/data.json'

# 提取文件夹名作为输出文件名
folder_name = os.path.basename(os.path.dirname(input_data_path))  # 提取文件夹名
output_filename = f"medgemma-4b-pt_result/{folder_name}_result.json"

# 确保输出目录存在
os.makedirs("medgemma-4b-pt_result", exist_ok=True)

with open(input_data_path, 'r') as file:
    wikimultihopqa = json.load(file)
print(f"Loaded {len(wikimultihopqa)} samples")

combine_results = []
for item in tqdm(wikimultihopqa[:]):
    print("########################################")
    
    try:
        # 安全获取图像路径
        if item['type'][0] == 'crop':
            input_image_path = item['ori_image_path']
        else:
            input_image_path = item['processed_image_path']
        
        input_image_path = f'/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/data/MAT-Benchmark/all_noise_100_test/{noise_name}/images/' + input_image_path
        
        query = item['question']
        data_type = item['type']
        item_id = item['id']
        answer = item['answer']

        # 🔧 按照官方示例格式构建prompt
        prompt = f"<start_of_image> {query} Answer the question directly. The answer should be very brief, not to explain."

        print(RED + prompt + RESET)
        print(GREEN + str(answer) + RESET)
        
        # 检查图像文件是否存在
        if not os.path.exists(input_image_path):
            pred_answer = "Image file not found"
            print(f"⚠️ 图像文件不存在: {input_image_path}")
        else:
            # 🔧 按照官方示例加载图像
            image = Image.open(input_image_path).convert("RGB")
            
            # 🔧 按照官方示例处理输入
            inputs = processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            # 🔧 按照官方示例生成回答
            with torch.inference_mode():
                generation = model.generate(
                    **inputs, 
                    max_new_tokens=256,  # 可以根据需要调整
                    do_sample=False,      # 可以改为False使用贪婪解码
                    temperature=0.7,     # 只有do_sample=True时生效
                    top_p=0.9,          # 只有do_sample=True时生效
                    top_k=50            # 只有do_sample=True时生效
                )
                generation = generation[0][input_len:]

            # 🔧 按照官方示例解码输出
            pred_answer = processor.decode(generation, skip_special_tokens=True).strip()
            
        print(YELLOW + pred_answer + RESET)

    except Exception as e:
        print("ERROR OCCURS")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        pred_answer = "Error during processing"
    
    combine_results.append({
        'pred_answer': pred_answer, 
        'gt': answer, 
        'query': query,
        'id': item_id
    })

print(f"Processed {len(combine_results)} samples")
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(combine_results, f, ensure_ascii=False, indent=4)
    
print(f"✅ 结果已保存到: {output_filename}")

# 简单统计
success_count = sum(1 for r in combine_results if r['pred_answer'] not in ["Error during processing", "Image file not found"])
print(f"📊 成功处理样本: {success_count}/{len(combine_results)} ({success_count/len(combine_results)*100:.1f}%)")