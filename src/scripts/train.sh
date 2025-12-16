
SESSION_NAME="visual_arft_training"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")


echo "🚀 使用tmux启动训练"

echo "📺 会话名称: $SESSION_NAME"

# 创建新的tmux会话并运行训练
tmux new-session -d -s "$SESSION_NAME" bash -c "
cd /MED-Noise

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./qwen2.5-VL-3B-Instruct.txt"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=""
export WANDB_MODE=offline
export WANDB_API_KEY="your_wandb_api_key_here"  # 替换为你的实际API密钥

torchrun --nproc_per_node="3" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="34567" \
    src/visual_arft/src/open_r1/grpo_agent_code.py \
    --output_dir output/Sept10Qwen2.5-VL-3B-Instruct \
    --model_name_or_path output/checkpoint/Qwen2.5-VLcheckpoint-2200 \
    --dataset_name data/MAT-Training/train_data_proc.json \
    --max_prompt_length 2048 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --deepspeed src/visual_arft/local_scripts/zero3.json \
    --report_to wandb \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --save_steps 100 \
    --learning_rate 0.6e-6 \
    --save_total_limit 17 \
    --max_pixels 401408 \
    --num_train_epochs 10 \
    --save_only_model true \
    --run_name qwen2.5vl-3b \
    --num_generations 8  2>&1 | tee training_tmux_${TIMESTAMP}.log

echo '✅ 训练完成 - \$(date)'
bash
"

# 检查会话状态
sleep 3
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "✅ 训练会话正在运行"
    echo "💡 使用 'tmux attach-session -t $SESSION_NAME' 查看训练进度"
else
    echo "❌ 会话创建失败"
fi

# tmux attach-session -t visual_arft_training