import re
import json
import torch
import string
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd

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

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    if prediction is None:
        return 0.0
    return int(normalize(prediction) == normalize(ground_truth))

def evaluate_json_file(json_path):
    """评估单个JSON文件"""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    f1_all = []
    em_all = []
    count = 0
    none_count = 0
    
    for item in data:
        if item.get('pred_answer') is not None:
            count += 1
            # 计算 f1 和 em 
            pred = item['pred_answer']
            gts = item['gt']
            # 若gt是str，统一转换为列表处理
            if isinstance(gts, str):
                gts = [gts]
            f1 = max([compute_f1(pred, gt) for gt in gts])
            print(f"gt:{gts}, pred: {pred}, f1: {f1}")
            em = max([exact_match_score(pred, gt) for gt in gts])
            if em == 1:
                f1 = 1
            f1_all.append(f1)
            em_all.append(em)
        else:
            count += 1
            none_count += 1
            f1 = 0.0
            em = 0.0
            f1_all.append(f1)
            em_all.append(em)
    
    # 计算平均值
    avg_f1 = sum(f1_all) / len(f1_all) if f1_all else 0.0
    avg_em = sum(em_all) / len(em_all) if em_all else 0.0
    
    return {
        'file_name': os.path.basename(json_path),
        'total_count': count,
        'none_count': none_count,
        'valid_count': count - none_count,
        'avg_f1': avg_f1,
        'avg_em': avg_em,
        'f1_scores': f1_all,
        'em_scores': em_all
    }

def main():
    # 设置目录路径
    result_dir = "/cluster/home/user1/YuanWenzhen/workspace/Visual-RFT/Visual-ARFT/evaluation_coding/scripts/Mini-InternVL2-4B-DA-Medical_result"
    
    # 检查目录是否存在
    if not os.path.exists(result_dir):
        print(f"❌ 目录不存在: {result_dir}")
        return
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(result_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"❌ 在目录 {result_dir} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件:")
    for f in json_files:
        print(f"  - {f}")
    
    # 评估所有文件
    all_results = []
    detailed_results = {}
    
    print("\n开始评估...")
    for json_file in tqdm(json_files, desc="评估进度"):
        json_path = os.path.join(result_dir, json_file)
        try:
            result = evaluate_json_file(json_path)
            all_results.append(result)
            detailed_results[json_file] = result
            print(f"✅ {json_file}: F1={result['avg_f1']:.4f}, EM={result['avg_em']:.4f}")
        except Exception as e:
            print(f"❌ 评估 {json_file} 时出错: {e}")
    
    # 创建结果汇总
    summary_results = []
    for result in all_results:
        summary_results.append({
            'File': result['file_name'],
            'Total_Count': result['total_count'],
            'None_Count': result['none_count'],
            'Valid_Count': result['valid_count'],
            'Avg_F1': f"{result['avg_f1']:.4f}",
            'Avg_EM': f"{result['avg_em']:.4f}"
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(summary_results)
    
    # 计算总体平均值
    total_f1_scores = []
    total_em_scores = []
    for result in all_results:
        total_f1_scores.extend(result['f1_scores'])
        total_em_scores.extend(result['em_scores'])
    
    overall_f1 = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0.0
    overall_em = sum(total_em_scores) / len(total_em_scores) if total_em_scores else 0.0
    
    # 添加总体结果行
    overall_row = {
        'File': 'OVERALL',
        'Total_Count': sum(r['total_count'] for r in all_results),
        'None_Count': sum(r['none_count'] for r in all_results),
        'Valid_Count': sum(r['valid_count'] for r in all_results),
        'Avg_F1': f"{overall_f1:.4f}",
        'Avg_EM': f"{overall_em:.4f}"
    }
    df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)
    
    # 保存结果
    output_dir = os.path.join(result_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存CSV格式的汇总结果
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ 汇总结果已保存到: {csv_path}")
    
    # 保存详细的JSON结果
    detailed_json_path = os.path.join(output_dir, "detailed_evaluation_results.json")
    with open(detailed_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    print(f"✅ 详细结果已保存到: {detailed_json_path}")
    
    # 打印结果表格
    print("\n" + "="*80)
    print("评估结果汇总:")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # 保存文本格式的结果
    txt_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("LLaVA-NeXT-Video-7B-hf 模型评估结果\n")
        f.write("="*80 + "\n")
        f.write(f"评估时间: {pd.Timestamp.now()}\n")
        f.write(f"评估目录: {result_dir}\n")
        f.write(f"总文件数: {len(json_files)}\n\n")
        
        f.write("详细结果:\n")
        f.write("-"*80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "-"*80 + "\n")
        
        f.write(f"\n总体性能:\n")
        f.write(f"  整体平均 F1: {overall_f1:.4f}\n")
        f.write(f"  整体平均 EM: {overall_em:.4f}\n")
        f.write(f"  总样本数: {sum(r['total_count'] for r in all_results)}\n")
        f.write(f"  有效样本数: {sum(r['valid_count'] for r in all_results)}\n")
        f.write(f"  无效样本数: {sum(r['none_count'] for r in all_results)}\n")
    
    print(f"✅ 文本结果已保存到: {txt_path}")
    
    print(f"\n🎉 评估完成! 结果保存在: {output_dir}")
    print(f"📊 总体性能: F1={overall_f1:.4f}, EM={overall_em:.4f}")

if __name__ == "__main__":
    # 安装pandas如果没有
    try:
        import pandas as pd
    except ImportError:
        print("安装pandas...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd
    
    main()