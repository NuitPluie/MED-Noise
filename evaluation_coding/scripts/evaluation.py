import re
import json
import torch
import string
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def is_chinese(text):
    """判断文本是否含中文字符"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

def normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        chinese_punc = "！？｡＂＃＄％＆＇（）＊＋，－．／：；＜＝＞＠［＼］＾＿｀｛｜｝～“”‘’、。：《》【】"
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

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    if prediction is None:
        return 0.0
    return int(normalize(prediction) == normalize(ground_truth))

def plot_images(image_paths):
    num_images = len(image_paths)
    
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)

        # 如果是灰度图（2D），扩展为 RGB
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        # 如果是带 alpha 通道的 RGBA 图，去掉 alpha
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        ax = axes if num_images == 1 else axes[i]
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def extract_coordinates(text):
    pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    matches = re.findall(pattern, text)
    # 转换为整数列表格式
    coordinates = [list(map(int, match)) for match in matches]
    return coordinates

noise_name = 'OE'  # 修改为所需的噪声类型，例如 'BC+other'

json_path = f"/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/evaluation_coding/llava-1.5-7b-hf_result/{noise_name}_result.json"
with open(json_path, "r") as f:
    data = json.load(f)
print(len(data))

f1 = 0 
em = 0
f1_all = []
em_all = []
count = 0
none_count = 0
for item in data[:]:
    if 'pred_answer' != None:
        count += 1
        # 计算 f1 和 em 
        pred = item['pred_answer']
        gts = item['gt']
        print(pred)
        print(gts)
        # 若gt是str，统一转换为列表处理
        if isinstance(gts, str):
            gts = [gts]
        f1 = max([compute_f1(pred, gt) for gt in gts])
        em = max([exact_match_score(pred, gt) for gt in gts])
        if em == 1:
            f1 =1
        print("F1: " + str(f1))
        # print("EM: " + str(em))
        f1_all.append(f1)
        em_all.append(em)
    else:
        count += 1
        none_count += 1
        f1 = 0.0
        em = 0.0
        # print("F1: " + str(f1))
        # print("EM: " + str(em))
        f1_all.append(f1)
        em_all.append(em)


print(count)
print(none_count)
print('All F1:', sum(f1_all)/100)
print('All EM:', sum(em_all)/100)