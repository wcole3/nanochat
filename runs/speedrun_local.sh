#!/bin/bash

# This script is a local, single-GPU adaptation of the speedrun script.
# It is configured to validte the entire pipeline on a consumer GPU.
# This serves as a functional test of the end-to-end flow.

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash runs/speedrun_local.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download just 10 shards (approx 100MB compressed) for a local test
# This is enough to train a tokenizer and run a small training loop.
echo "Downloading data..."
python -m nanochat.dataset -n 10

# Train tokenizer on this small subset
echo "Training tokenizer..."
python -m scripts.tok_train --vocab-size=4096 --max-chars=10000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Train a small model for local testing
# Reduced depth, batch size, and target data ratio for quick execution
echo "Starting base model training..."
# Using python directly for single GPU
python -m scripts.base_train \
    --depth=8 \
    --target-param-data-ratio=1.0 \
    --device-batch-size=8 \
    --total-batch-size=65536 \
    --max-seq-len=1024 \
    --eval-tokens=102400 \
    --num-iterations=100 \
    --eval-every=20 \
    --save-every=50 \
    --run=$WANDB_RUN

# evaluate the model: CORE metric, BPB on train/val, and draw samples
echo "Evaluating base model..."
python -m scripts.base_eval --device-batch-size=8 --max-seq-len=1024

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
echo "Downloading sft data..."
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
echo "Starting SFT..."
python -m scripts.chat_sft \
    --device-batch-size=8 \
    --run=$WANDB_RUN \
    --learning-rate=1e-4 \
    --num-iterations=50

echo "Evaluating SFT model..."
python -m scripts.chat_eval --device-batch-size=8 -i sft

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
