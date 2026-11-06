#!/bin/bash
# 用法: bash scripts/run.sh <task> <config.yaml> [gpu_ids] [master_port]
#   task: classification | text2text | rerank
#   gpu_ids: 形如 0,1,2,3；不传则交互输入，10 秒超时后使用全部 8 卡
export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

TASK="$1"
CONFIG_FILE="$2"
gpuids="$3"
MASTER_PORT="${4:-29500}"

if [ -z "$TASK" ] || [ -z "$CONFIG_FILE" ]; then
    echo "用法: bash scripts/run.sh <task> <config.yaml> [gpu_ids] [master_port]"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在!"
    exit 1
fi

if [ -z "$gpuids" ]; then
    read -p "Please input the GPU ID you want to use，seprate them with en-commas. (for example: 0,1,2,3) >>>" -t 10 gpuids
fi

if [ "$gpuids" != "" ]; then
    gpuids=$(echo ${gpuids} | tr -d '[:space:]')
    export CUDA_VISIBLE_DEVICES="$gpuids"
else
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi
NUM_PROC=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

NNODES=1
MASTER_ADDR="localhost"

echo "分布式训练配置:"
echo "- 任务: $TASK"
echo "- 配置: $CONFIG_FILE"
echo "- CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "- 节点数: $NNODES"
echo "- 每节点进程数: $NUM_PROC"
echo "- Master地址: $MASTER_ADDR"
echo "- Master端口: $MASTER_PORT"
echo ""

echo "开始训练..."
echo "========================"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NUM_PROC \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m clser $TASK $CONFIG_FILE

if [ $? -eq 0 ]; then
    echo ""
    echo "========================"
    echo "训练成功完成!"
else
    echo ""
    echo "========================"
    echo "训练失败，请检查错误信息"
    exit 1
fi
