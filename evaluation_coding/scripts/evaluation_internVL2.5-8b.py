import re
import json
import torch
import string
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import os
import torchvision.transforms as T

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def is_chinese(text):
    """判断文本是否含中文字符"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

def normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        chinese_punc = "！？｡＂＃＄％＆＇（）＊＋，－．／：；＜＝＞＠［＼］＾＿｀｛｜｝～""''、。：《》【】"
        exclude = set(string.punctuation + chinese_punc)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    s = lower(s)
    s = remove_punc(s)

    if is_chinese(s):
        s = s.replace(" ", "")  # 中文一般去除所有空白
    else:
        s = remove_articles(s)
        s = white_space_fix(s)

    return s

def compute_f1(prediction, ground_truth):
    if prediction is None:
        return 0.0

    norm_pred = normalize(prediction)
    norm_gt = normalize(ground_truth)

    # 中文使用字符级，英文使用词级
    if is_chinese(norm_pred) or is_chinese(norm_gt):
        pred_tokens = list(norm_pred)
        gt_tokens = list(norm_gt)
    else:
        pred_tokens = norm_pred.split()
        gt_tokens = norm_gt.split()

    common = set(pred_tokens) & set(gt_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = num_same / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def exact_match_score(prediction, ground_truth):
    if prediction is None:
        return 0.0
    return int(normalize(prediction) == normalize(ground_truth))

# 模型路径
model_path = "/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/shared/mllm_ckpts/Mini-InternVL2-4B-DA-Medical"

# 加载模型和处理器
try:
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:3",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("✅ 模型和tokenizer加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

# 数据路径
noise_name = 'origin'  # 修改为所需的噪声类型，例如 'BC+other'
input_data_path = f'/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/data/MAT-Benchmark/all_noise_100_test/{noise_name}/data.json'
folder_name = os.path.basename(os.path.dirname(input_data_path))
output_filename = f"Mini-InternVL2-4B-DA-Medical_result/{folder_name}_result.json"
os.makedirs("Mini-InternVL2-4B-DA-Medical_result", exist_ok=True)

# 加载数据
if not os.path.exists(input_data_path):
    print(f"❌ 数据文件不存在: {input_data_path}")
    exit(1)

with open(input_data_path, 'r') as file:
    wikimultihopqa = json.load(file)
print(f"数据样本数: {len(wikimultihopqa)}")

# 🔧 图像预处理函数
def build_transform(input_size):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

# 设置模型为评估模式
model.eval()

combine_results = []
f1_scores = []
em_scores = []

for idx, item in enumerate(tqdm(wikimultihopqa[:])):
    print(f"\n{'='*80}")
    print(f"样本 {idx+1}/{len(wikimultihopqa)}")
    print(f"{'='*80}")
    
    try:
        # 获取图像路径
        if item['type'][0] == 'crop':
            input_image_path = item['ori_image_path']
        else:
            input_image_path = item['processed_image_path']
        input_image_path = f'/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/data/MAT-Benchmark/all_noise_100_test/{noise_name}/images/' + input_image_path
        
        if not os.path.exists(input_image_path):
            pred_answer = "Image not found"
            combine_results.append({'pred_answer': pred_answer, 'gt': item['answer'], 'query': item['question']})
            continue

        # 🔧 正确的图像加载和处理
        image = Image.open(input_image_path).convert("RGB")
        query = item['question']
        input_text = query + '\n' + "Answer the question directly. The answer should be very brief."
        print(RED + f"问题: {input_text}" + RESET)
        print(GREEN + f"标准答案: {str(item['answer'])}" + RESET)
        
        # 🔧 修复InternVL的推理方式
        try:
            generation_config = dict(
                num_beams=1, 
                max_new_tokens=256, 
                do_sample=True,      # 🔧 启用采样
                temperature=0.7,     # 🔧 设置温度参数
                top_p=0.9,           # 🔧 使用top-p采样
                top_k=50,            # 🔧 添加top-k采样
                repetition_penalty=1.1  # 🔧 避免重复
            )
            
            # InternVL2.5需要将图像转换为tensor
            transform = build_transform(input_size=448)
            pixel_values = transform(image).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
            
            # 使用正确的chat方法
            response = model.chat(
                tokenizer=tokenizer, 
                pixel_values=pixel_values,  # 🔧 使用处理后的tensor而不是PIL图像
                question=input_text, 
                generation_config=generation_config
            )
            pred_answer = response
            
        except Exception as inner_e:
            print(f"Chat方法失败: {inner_e}")
            try:
                # 方法2: 使用generate方法
                transform = build_transform(input_size=448)
                pixel_values = transform(image).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
                
                # 构造输入文本
                prompt = f"<image>\nUser: {input_text}\nAssistant:"
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        max_new_tokens=256,
                        num_beams=1,
                        do_sample=False,
                        temperature=0.2,
                        pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.pad_token_id
                    )
                
                # 只解码新生成的token
                new_tokens = output_ids[0][input_ids.shape[1]:]
                pred_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
            except Exception as inner_e2:
                print(f"Generate方法也失败: {inner_e2}")
                pred_answer = "Error in processing"
        
        # 清理回答
        pred_answer = str(pred_answer).strip()
        if pred_answer.startswith("Assistant:"):
            pred_answer = pred_answer[10:].strip()
        if pred_answer.startswith("assistant:"):
            pred_answer = pred_answer[10:].strip()
        if pred_answer.startswith("User:"):
            pred_answer = pred_answer[5:].strip()
        if pred_answer.startswith("user:"):
            pred_answer = pred_answer[5:].strip()
        
        print(YELLOW + f"模型回答: {pred_answer}" + RESET)
        
        # 🔧 计算评估分数
        gts = item['answer']
        if isinstance(gts, str):
            gts = [gts]
        
        # 计算与所有ground truth的最大分数
        f1_score = max([compute_f1(pred_answer, gt) for gt in gts])
        em_score = max([exact_match_score(pred_answer, gt) for gt in gts])
        
        # 如果EM=1，则F1也设为1
        if em_score == 1:
            f1_score = 1.0
        
        f1_scores.append(f1_score)
        em_scores.append(em_score)
        
        print(BLUE + f"F1 分数: {f1_score:.4f}" + RESET)
        print(BLUE + f"EM 分数: {em_score:.4f}" + RESET)

    except Exception as e:
        print(f"❌ 推理失败: {e}")
        print(f"错误详情: {type(e).__name__}: {str(e)}")
        pred_answer = "Error in processing"
        
        # 对于错误的情况也要计算分数
        gts = item['answer']
        if isinstance(gts, str):
            gts = [gts]
        f1_score = 0.0
        em_score = 0.0
        f1_scores.append(f1_score)
        em_scores.append(em_score)
    
    combine_results.append(
        {'pred_answer': pred_answer, 'gt': item['answer'], 'query': item['question']}
    )

# 计算总体统计
avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0

print(f"\n{'='*80}")
print("评估结果汇总:")
print(f"{'='*80}")
print(f"总样本数: {len(wikimultihopqa)}")
print(f"平均 F1 分数: {avg_f1:.4f}")
print(f"平均 EM 分数: {avg_em:.4f}")
print(f"{'='*80}")

# 只保存原始结果
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(combine_results, f, ensure_ascii=False, indent=4)
print(f"✅ 结果已保存到: {output_filename}")
print(f"📊 总共处理了 {len(combine_results)} 个样本")