#!/bin/bash
export OMP_NUM_THREADS=1
echo "muti-gpu distribute setup script"
echo "========================"

read -p "Please input the GPU ID you want to use，seprate them with en-commas. (for example: 0,1,2,3) >>>" -t 10 gpuids

if [ "$gpuids" != "" ]; then
    gpuids=$(echo ${gpuids} | tr -d '[:space:]')
    export CUDA_VISIBLE_DEVICES="$gpuids"
    echo "设置CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    NUM_PROC=$(echo $gpuids | awk -F"," '{print NF}')
    echo "使用 $NUM_PROC 个GPU"
else
    # echo "未提供GPU ID，将使用所有可用GPU"
    # NUM_PROC=$(nvidia-smi --list-gpus | wc -l)
    # echo "检测到 $NUM_PROC 个GPU"
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    # export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
    NUM_PROC=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')
fi

echo $CUDA_VISIBLE_DEVICES

NNODES=1
MASTER_ADDR="localhost"
MASTER_PORT=29500

echo "分布式训练配置:"
echo "- 节点数: $NNODES"
echo "- 每节点进程数: $NUM_PROC"
echo "- Master地址: $MASTER_ADDR"
echo "- Master端口: $MASTER_PORT"
echo ""


PARENT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
PYTHON_SCRIPT="${PARENT_DIR}/src/mideacls/__main__.py"
CONFIG_FILE="${PARENT_DIR}/example/single_args.yaml"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本 $PYTHON_SCRIPT 不存在!"
    exit 1
fi

echo "开始训练..."
echo "========================"


torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NUM_PROC \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $PYTHON_SCRIPT $CONFIG_FILE

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