import os
import re
import string
import ast
import cv2
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified, Qwen2VLGRPOTrainer_AID
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL

from codebleu import calc_codebleu


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

def normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize(prediction).split()
    ground_truth_tokens = normalize(ground_truth).split()

    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def check_cv2_attributes(code_string):
    """静态检查cv2属性是否存在"""
    
    match = re.search(r"<code>\s*```python(.*?)```.*?</code>", code_string, re.DOTALL)

    if match:
        code_str = match.group(1)
            
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            return False, "无法解析代码进行属性检查"
    
        # 获取cv2的所有真实属性
        cv2_attributes = set(dir(cv2))
    
        # 添加一些常见的正确属性（确保不误判）
        cv2_attributes.update([
            'imread', 'imwrite', 'convertScaleAbs', 'GaussianBlur',
            'filter2D', 'addWeighted', 'bilateralFilter', 'fastNlMeansDenoising',
            'resize', 'cvtColor', 'threshold', 'Canny', 'dilate', 'erode'
        ])
    
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # 检查 cv2.xxx 形式的调用
                if (isinstance(node.value, ast.Name) and 
                    node.value.id == 'cv2' and 
                    node.attr not in cv2_attributes):
                    errors.append(f"cv2 没有属性 '{node.attr}'")
        
            # 检查 from cv2 import xxx 形式
            elif isinstance(node, ast.ImportFrom):
                if node.module == 'cv2' and node.names:
                    for alias in node.names:
                        if alias.name != '*' and alias.name not in cv2_attributes:
                            errors.append(f"cv2 没有属性 '{alias.name}'")
    
        if errors:
            return False, "; ".join(errors)
        return True, "cv2属性检查通过"
    else:
        return False,"匹配失败---+++"


def enhanced_code_validation(student_answer):
    """增强的代码验证，返回分数和详细信息"""
    if not student_answer or not student_answer.strip():
        return 0.0, "代码为空"
    
    # 基础奖励
    base_score = 0.5  # 有代码就给基础分
    
    # CV2属性检查 (30%权重)
    attr_valid, attr_msg = check_cv2_attributes(student_answer)
    if attr_valid:
        attr_score = 0.4
    else:
        return 0.3, f"属性检查失败: {attr_msg}"  # 属性错误扣分
    
    total_score = base_score + attr_score
    return min(1.0, total_score), f"代码验证通过 - {attr_msg}"

def exact_match_score(prediction, ground_truth):
    if prediction is None:
        return 0.0
    return int(normalize(prediction) == normalize(ground_truth))
    
def compute_code_similarity(prediction, ground_truth):
    result = calc_codebleu([prediction], [ground_truth], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    result = result['codebleu']
    return result

def extract_problems(text):
    match = re.search(r"<problem>\s*\{(.*?)\}\s*</problem>", text, re.DOTALL)
    if not match:
        return []

    content = match.group(1)
    # 提取所有用英文单引号包裹的单词
    problems = re.findall(r"'([^']+)'", content)
    return sorted(problems)

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        validation_msg = ""
        
        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                if '<answer>' in sol:
                    if '<code>' in content or '<problem>' in content:
                        reward = 0.0
                        validation_msg = "格式错误：答案类型中包含代码或问题标签"
                    else:
                        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                        
                        # Extract answer from content if it has think/answer tags
                        content_match = re.search(r'<answer>(.*?)</answer>', content)
                        student_answer = content_match.group(1).strip() if content_match else content.strip()

                        # Compare the extracted answers
                        reward = compute_f1(student_answer, ground_truth)
                        validation_msg = f"答案F1分数: {reward:.3f}"
                    
                elif '<code>' in sol:
                    if '<answer>' in content or '<problem>' in content:
                        reward = 0.0
                        validation_msg = "格式错误：代码类型中包含答案或问题标签"
                        with open(log_path, "a") as f:
                            f.write(f"answer: {content}\n")
                    else:
                        sol_match = re.search(r'<code>(.*?)</code>', sol)
                        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                        
                        # Extract code from content
                        content_match = re.search(r'<code>(.*?)</code>', content)
                        student_answer = content_match.group(1).strip() if content_match else content.strip()

                        # 应用增强的代码验证
                        if student_answer:
                            code_score, code_msg = enhanced_code_validation(student_answer)
                            validation_msg = code_msg
                            
                            # 如果代码验证通过，给予奖励
                            if code_score >= 0.5:  # 设置通过阈值
                                # 可以选择基于验证分数给奖励，或者给固定奖励
                                reward = code_score  # 使用验证分数作为奖励
                                # reward = 0.9  # 或者使用固定奖励
                            else:
                                reward = code_score  # 验证失败，给予低分
                        else:
                            reward = 0.0
                            validation_msg = "未提取到代码内容"
                       
                        log_path = os.getenv("LOG_PATH")
                        with open(log_path, "a") as f:
                            f.write(f"answer: {student_answer}\n")
                
                elif '<problem>' in sol:
                    if '<answer>' in content or '<code>' in content:
                        reward = 0.0
                        validation_msg = "格式错误：问题类型中包含答案或代码标签"
                    else:
                        ground_truth = extract_problems(sol)
                        student_answer = extract_problems(content)

                        # Half correct
                        if len(ground_truth) == 2 and 'rotation90' in ground_truth:
                            if len(student_answer) == 1 and 'rotation90' in student_answer:
                                reward = 0.5
                        elif len(ground_truth) == 2 and 'rotation180' in ground_truth:
                            if len(student_answer) == 1 and 'rotation180' in student_answer:
                                reward = 0.5
                        # All correct
                        if reward == 0: 
                            is_equal = ground_truth == student_answer
                            if is_equal:
                                reward = 1.0
                            else:
                                reward = 0.0
                        validation_msg = f"问题识别 - 标准答案: {ground_truth}, 学生答案: {student_answer}"
                    
            except Exception as e:
                reward = 0.0
                validation_msg = f"处理异常: {str(e)}"
                
        rewards.append(reward)
        
        # Debug输出，包含验证信息
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward:.3f} -------------\n")
                f.write(f"Validation: {validation_msg}\n")
    
    return rewards

# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]

def format_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion matches exactly one valid format."""
    pattern_answer = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern_code = r"<think>.*?</think>\s*<code>\s*```python(.*?)```.*?</code>"
    pattern_problem = r"^<think>.*?</think>\s*<problem>\s*\{\s*'[^']+'\s*(?:,\s*'[^']+'\s*)*\}\s*</problem>$"

    completion_contents = [completion[0]["content"] for completion in completions]
    solution_contents = [sol for sol in solution]

    rewards = []
    for content, solution in zip(completion_contents, solution_contents):
        if content.count("<answer>")>=2 or content.count("<code>")>=2 or content.count("<think>")>=2 or content.count("<problem>")>=2:
            rewards.append(0.0)
        elif '<answer>' in solution:
            match_answer = re.fullmatch(pattern_answer, content, re.DOTALL)
            if match_answer:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        elif '<code>' in solution:
            match_code = re.fullmatch(pattern_code, content, re.DOTALL)
            if match_code:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        elif '<problem>' in solution:
            match_problem = re.fullmatch(pattern_problem, content, re.DOTALL)
            if match_problem:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
# )

SYSTEM_PROMPT_AGENT_CODE = """# Role
You are a step-by-step image processing assistant.
Your task is to solve an image-based task by applying OpenCV operations one step at a time, optionally using a reasoning chain.

# Output Format
At each step, output **only one** of the following, preceded by a <think> tag:
1. <problem> Describe the image issue from {'underexposed','overexposure', 'motion_blur', 'brightness_contrast', 'gaussian_noise', 'none','clip_top_left','clip_top_right','clip_bottom_left','clip_bottom_right'} </problem>
2. <code> OpenCV code to process and save the image </code>
3. <answer> Final answer based on the processed image </answer>

# Image Processing Rules
- Always read from `'path_to_input_image.jpg'` and write to `'path_to_output_image.jpg'`.

# Output Format (strict):
Always begin with <think>. Then, depending on current reasoning chain, output one of the following:

## 1. If this is the first step and only the query is given, output in the following format:
<think> Initial analysis of the image issue. </think>
<problem> {'problem1', ...} </problem>

## 2. If <problem> is given, continue with image operations:
<think> Explain what to fix next. </think>
<code>
```python
One Python code block using OpenCV to perform the operation, and save the processed images.
```
</code>

## 3. If ready to conclude:
<think> Summarize the processing steps and provide the result or outcome </think> 
<answer> Final answer, as briefly as possible</answer>

# Current reasoning chain:
"""

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset——method 1
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset——method 2
    # from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)
    # Load the dataset——method 3
    from datasets import Dataset
    dataset = Dataset.from_json(script_args.dataset_name)

    def make_conversation_image(example):
        # 使用相对路径，相对于当前工作目录
        base_image_dir = "data/MAT-Training/surgVQA_images"
        image_filename = example['image_path']
        image_path = os.path.join(base_image_dir, image_filename)
        
        if example['type'] == "pre_problem":
            prompt = example['problem']
            solution = example['solution']
            gt = example['gt']
            formatted_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": SYSTEM_PROMPT_AGENT_CODE + '\n' + prompt},
                    ],
                }
            ]
            # resize image to avoid OOM
            image = PIL.Image.open(image_path).resize((720,720))
            return {"image": image, "prompt": formatted_conversation, 'solution': solution, 'gt': gt}
        elif example['type'] == "pre_code":
            prompt = example['problem']
            context = example['context']
            solution = example['solution']
            gt = example['gt']
            formatted_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": SYSTEM_PROMPT_AGENT_CODE + '\n' + prompt + '\n' + context},
                    ],
                },
            ]
            # resize image to avoid OOM
            image = PIL.Image.open(image_path).resize((720,720))
            return {"image": image, "prompt": formatted_conversation, 'solution': solution, 'gt': gt}
        elif example['type'] == "pre_answer":
            prompt = example['problem']
            context = example['context']
            solution = example['solution']
            gt = example['gt']
            formatted_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": SYSTEM_PROMPT_AGENT_CODE + '\n' + prompt + '\n' + context},
                    ],
                },
            ]
            # resize image to avoid OOM
            image = PIL.Image.open(image_path).resize((720,720))
            return {"image": image, "prompt": formatted_conversation, 'solution': solution, 'gt': gt}

    # if "image" in dataset[script_args.dataset_train_split].features:
    #     print("has image in dataset")
    #     dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    #     # dataset = dataset.remove_columns(["original_question", "original_answer"])

    # else:
    #     print("no image in dataset")
    #     dataset = dataset.map(make_conversation)
    #     dataset = dataset.remove_columns("messages")

    dataset = dataset.map(make_conversation_image)
    dataset = dataset.remove_columns(["image_path", "problem", 'context', 'type'])


    
    trainer_cls = Qwen2VLGRPOTrainer_AID if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        # train_dataset=dataset[script_args.dataset_train_split],
        ### lzy modified
        train_dataset=dataset,
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    print("trainer initialized")
    # Train and push the model to the Hub
    trainer.train()

    print("training finished")
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
