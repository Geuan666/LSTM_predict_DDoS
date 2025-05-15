#!/bin/bash
# 训练启动脚本

# 设置CUDA可见设备（如果有多个GPU可以指定）
export CUDA_VISIBLE_DEVICES=0

# 定义基本参数
DATA_PATH="C:\\Users\\17380\\Downloads\\CSV-\\03-11"
OUTPUT_DIR="output"
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=0.001
WEIGHT_DECAY=0.001

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 定义日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/train_$TIMESTAMP.log"

echo "开始训练，日志将保存到 $LOG_FILE"

# 运行训练命令
python main.py \
    --mode train \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --early_stopping \
    --accumulation_steps 4 \
    --input_dim 128 \
    --embedding_dim 32 \
    --hidden_dim 64 \
    --num_layers 2 \
    --dropout 0.3 \
    --attention_heads 4 \
    --threshold_points " 128, 256, 512" \
    --window_size 15 \
    --step_size 1 \
    --n_workers 4 \
    --seed 42 | tee $LOG_FILE

echo "训练完成"