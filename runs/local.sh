#!/bin/bash

# Exit on error
set -e

# 1. Setup Environment
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# 2. Activate Virtual Environment (ensure you ran 'uv sync --extra gpu')
source .venv/bin/activate

# 3. Prepare Data (if not already done)
# Download just 10 shards (approx 100MB compressed) for a local test
echo "Downloading data..."
python -m nanochat.dataset -n 10

# Train tokenizer on this small subset
echo "Training tokenizer..."
python -m scripts.tok_train --vocab-size=4096 --max-chars=10000000

# 4. Run Training on Single GPU
# We use 'python -m' directly instead of 'torchrun' for a single process.
# Reduced batch size and depth for local consumer GPU (e.g. 24GB VRAM)
echo "Starting training..."
python -m scripts.base_train \
    --device-batch-size=8 \
    --total-batch-size=65536 \
    --depth=8 \
    --max-seq-len=1024 \
    --eval-tokens=102400 \
    --num-iterations=1000 \
    --eval-every=100 \
    --save-every=500 \
    --run=dummy  # Disable wandb logging for local test