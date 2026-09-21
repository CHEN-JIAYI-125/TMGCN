#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 创建日志目录（如果不存在）
mkdir -p logs

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 定义日志文件路径
logfile="logs/train_${timestamp}.log"

config_file=$1

# 检查是否提供了配置文件参数
if [ -z "$config_file" ]; then
    echo "❌ 错误: 请提供配置文件路径作为参数"
    echo "💡 用法: ./run_train.sh <config_file_path>"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$config_file" ]; then
    echo "❌ 错误: 配置文件 '$config_file' 不存在"
    exit 1
fi

# 启动训练任务并在后台运行，使用 nohup 防止终端退出导致进程终止
nohup python main_graph.py $config_file > ${logfile} 2>&1 &

# 保存后台进程的 PID
train_pid=$!

# 输出信息
echo "✅ 训练已启动，后台 PID: ${train_pid}"
echo "📄 日志文件路径: ${logfile}"
echo "💡 按 Ctrl+C 停止日志输出，但训练将继续在后台运行。"
echo "💡 使用 'kill -9 ${train_pid}' 可以手动终止训练进程。"

# 实时打印日志（直到用户按 Ctrl+C）
tail -f "${logfile}" &
tail_pid=$!

# 捕获 Ctrl+C，停止 tail，但不杀死训练进程
trap '
  # 检查训练进程是否还在运行
  if ps -p ${train_pid} > /dev/null; then
    echo -e "\n⚠️ 日志输出已停止，训练仍在后台运行（PID: ${train_pid}）。"
  else
    echo -e "\n✅ 训练已经完成。"
  fi
  kill ${tail_pid} 2>/dev/null
  exit
' INT

# 等待 tail 进程结束（即用户按 Ctrl+C）
wait ${tail_pid}