#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
MODEL=${MODEL:?Set MODEL to your local Qwen embedding directory}
GPU=${GPU:-1}
CONFIG=${CONFIG:-config/config_model5.yaml}
RUN_DIR=${RUN_DIR:-$HOME/tmgcn_runs/qwen_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "Run directory: $RUN_DIR"
echo "Commit: $(git rev-parse HEAD)"
"$PYTHON" -m unittest scripts.test_qwen_graph
"$PYTHON" -m scripts.smoke_qwen_graph
"$PYTHON" -u offline/build_qwen_graph.py --config "$CONFIG" --model "$MODEL" \
  --output "$RUN_DIR/cache" --device cuda:0 --batch-size "${BATCH_SIZE:-4}" \
  --max-length "${MAX_LENGTH:-512}" 2>&1 | tee "$RUN_DIR/embedding.log"
"$PYTHON" -u main_graph.py "$RUN_DIR/cache/train.yaml" 2>&1 | tee "$RUN_DIR/train.log"
echo "TRAINING_COMPLETE commit=$(git rev-parse HEAD)"
tail -n 30 "$RUN_DIR/train.log"
