import re
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色

# 图像预处理函数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# 🔧 添加安全获取字段的函数
def safe_get_field(d, k):
    v = d.get(k)
    if isinstance(v, list): 
        return v[0] if v else ""
    return v if v is not None else ""

# 设置模型路径
model_path = "/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/shared/mllm_ckpts/Mini-InternVL2-4B-DA-Medical"

# 加载模型和tokenizer
print("Loading model and tokenizer...")
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 🔧 设置tokenizer的pad_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"✅ Set pad_token_id to eos_token_id: {tokenizer.eos_token_id}")

print("Model loaded successfully!")

# 设置生成配置
generation_config = dict(
    max_new_tokens=256,  # 减少token数量提高速度
    do_sample=False,      # 启用采样获得更好结果
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    pad_token_id=tokenizer.pad_token_id,  # 🔧 显式指定pad_token_id
    eos_token_id=tokenizer.eos_token_id
)

noise_name = 'UE+other'  # 修改为所需的噪声类型，例如 'BC+other'
input_data_path = f'/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/data/MAT-Benchmark/all_noise_100_test/{noise_name}/data.json'

# 提取文件夹名作为输出文件名
folder_name = os.path.basename(os.path.dirname(input_data_path))
output_filename = f"Mini-InternVL2-4B-DA-Medical_result/{folder_name}_result.json"

# 确保输出目录存在
os.makedirs("Mini-InternVL2-4B-DA-Medical_result", exist_ok=True)

with open(input_data_path, 'r') as file:
    wikimultihopqa = json.load(file)
print(f"Loaded {len(wikimultihopqa)} samples")

combine_results = []
for item in tqdm(wikimultihopqa):
    print("########################################")
    
    try:
        # 🔧 安全获取字段，防止NoneType错误
        type_field = safe_get_field(item, 'type')
        
        # 决定使用哪个图像路径
        if isinstance(type_field, str) and type_field.startswith('crop'):
            input_image_path = safe_get_field(item, 'ori_image_path')
        else:
            input_image_path = safe_get_field(item, 'processed_image_path')
        
        # 如果还是没有图像路径，尝试其他可能的字段名
        if not input_image_path:
            input_image_path = safe_get_field(item, 'image_path') or safe_get_field(item, 'image')
        
        # 🔧 检查图像路径是否有效
        if not input_image_path or input_image_path == "":
            pred_answer = "No image path provided"
            print(f"⚠️ 没有找到图像路径，item keys: {list(item.keys())}")
        else:
            # 构建完整的图像路径
            full_image_path = f'/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/data/MAT-Benchmark/all_noise_100_test/{noise_name}/images/' + str(input_image_path)
            
            # 获取其他字段
            query = safe_get_field(item, 'question')
            answer = safe_get_field(item, 'answer')
            item_id = safe_get_field(item, 'id')
            
            # 构建输入文本
            input_text = f"<image>\n{query}\nAnswer the question directly. The answer should be very brief."
            
            print(RED + input_text + RESET)
            print(GREEN + str(answer) + RESET)
            
            # 检查图像文件是否存在
            if not os.path.exists(full_image_path):
                pred_answer = "Image file not found"
                print(f"⚠️ 图像文件不存在: {full_image_path}")
            else:
                # 加载并预处理图像
                pixel_values = load_image(full_image_path, max_num=12).to(torch.bfloat16)
                
                # 获取模型设备并移动数据
                model_device = next(model.parameters()).device
                pixel_values = pixel_values.to(model_device)
                
                # 🔧 使用模型进行推理 - Mini-InternVL使用chat接口
                response = model.chat(tokenizer, pixel_values, input_text, generation_config)
                
                # 处理返回值（可能是tuple）
                if isinstance(response, tuple):
                    pred_answer = response[0]
                else:
                    pred_answer = response
                    
                pred_answer = str(pred_answer).strip()
                
                # 移除常见前缀
                for prefix in ("Assistant:", "assistant:", "User:", "user:", "Answer:", "answer:"):
                    if pred_answer.startswith(prefix):
                        pred_answer = pred_answer[len(prefix):].strip()
                        break

        print(YELLOW + pred_answer + RESET)

    except Exception as e:
        print("ERROR OCCURS")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        pred_answer = "Error in processing"
        
        # 设置默认值以防字段获取失败
        query = safe_get_field(item, 'question')
        answer = safe_get_field(item, 'answer')
        item_id = safe_get_field(item, 'id')
    
    combine_results.append({
        'pred_answer': pred_answer, 
        'gt': answer, 
        'query': query, 
        'id': item_id
    })

print(f"Processed {len(combine_results)} samples")
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(combine_results, f, ensure_ascii=False, indent=4)
    
print(f"结果已保存到: {output_filename}")

# 简单统计
success_count = sum(1 for r in combine_results if r['pred_answer'] not in ["Error in processing", "Image file not found", "No image path provided"])
print(f"📊 成功处理样本: {success_count}/{len(combine_results)} ({success_count/len(combine_results)*100:.1f}%)")