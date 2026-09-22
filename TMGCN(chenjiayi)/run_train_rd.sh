#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="logs/train_rd_${timestamp}.log"

config_file=$1

if [ -z "$config_file" ]; then
    echo "❌ 错误: 请提供配置文件路径作为参数"
    echo "💡 用法: ./run_train_rd.sh config/config_model5_rd.yaml"
    echo "💡 Full 模型: ./run_train_rd.sh config/config_model5_rd_full.yaml"
    exit 1
fi

if [ ! -f "$config_file" ]; then
    echo "❌ 错误: 配置文件 '$config_file' 不存在"
    exit 1
fi

nohup python main_graph_rd.py ${config_file} > ${logfile} 2>&1 &
train_pid=$!

echo "✅ Reply-Distillation 训练已启动，后台 PID: ${train_pid}"
echo "📄 日志文件路径: ${logfile}"
echo "💡 按 Ctrl+C 停止日志输出，但训练将继续在后台运行。"
echo "💡 使用 'kill -9 ${train_pid}' 可以手动终止训练进程。"

tail -f "${logfile}" &
tail_pid=$!

trap '
  if ps -p ${train_pid} > /dev/null; then
    echo -e "\n⚠️ 日志输出已停止，训练仍在后台运行（PID: ${train_pid}）。"
  else
    echo -e "\n✅ 训练已经完成。"
  fi
  kill ${tail_pid} 2>/dev/null
  exit
' INT

wait ${tail_pid}
