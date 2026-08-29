#!/bin/bash
# KinetiQ 启动脚本
# 用法: ./start.sh [端口]  (默认 8501)
set -e
cd "$(dirname "$0")"

PORT="${1:-8501}"
KINETIQ_BIND_ADDRESS="${KINETIQ_ADDRESS:-127.0.0.1}"

if [ ! -d ".venv" ]; then
  echo "未找到虚拟环境，正在创建..."
  /opt/homebrew/bin/python3.12 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip -q
  .venv/bin/python -m pip install -r requirements.txt
fi

if [ ! -f "pose_landmarker.task" ]; then
  echo "未找到 pose_landmarker.task，正在下载模型..."
  curl -L -o pose_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
fi

echo "启动 KinetiQ：http://${KINETIQ_BIND_ADDRESS}:${PORT}"
exec .venv/bin/python -m streamlit run main.py \
  --server.headless true \
  --server.port "$PORT" \
  --server.address "$KINETIQ_BIND_ADDRESS" \
  --browser.gatherUsageStats false
