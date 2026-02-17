#!/bin/bash

# A local, single-GPU version of the scaling laws sweep.
# Adapts the compute budgets and model sizes to fit on a consumer GPU (e.g., RTX 3090/4090).

LABEL="starting_benchmark"
# Much smaller budget than the cluster script (1e18)
# Starting around 1e15 FLOPs (should take a few minutes on a decent GPU)
FLOPS_BUDGETS=(
    1e15
    5e15
    1e16
)
# Smaller, faster models
DEPTHS=(4 6 8 10)

WANDB_RUN="${WANDB_RUN:-dummy}" # Default to no wandb logging for local
EVAL_TOKENS=$((10 * 524288))  # ~5M tokens for final eval (faster)
DEVICE_BATCH_SIZE=8           # Reduced for VRAM
TOTAL_BATCH_SIZE=65536        # Smaller total batch size for quicker steps

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Ensure venv is active
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_laws_results_${LABEL}"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,depth,model_dim,params,num_iterations,tokens_trained,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if a run already exists in results
run_exists() {
    local flops=$1
    local depth=$2
    grep -q "^${flops},${depth}," "$RESULTS_FILE" 2>/dev/null
}

# =============================================================================
# Main Loop
# =============================================================================

for flops in "${FLOPS_BUDGETS[@]}"; do
    log "=============================================="
    log "Compute budget: $flops FLOPs"
    log "=============================================="

    for d in "${DEPTHS[@]}"; do

        # Skip if already completed
        if run_exists "$flops" "$d"; then
            log "Skipping d=$d at $flops FLOPs (already in results)"
            continue
        fi

        log "Training d=$d at $flops FLOPs..."

        TAG="local_scaling_${flops}_d${d}"
        LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

        START_TIME=$(date +%s)

        # Train: using python -m for single process instead of torchrun
        # Explicit batch sizes to prevent OOM
        python -m scripts.base_train \
            --depth=$d \
            --target-flops=$flops \
            --target-param-data-ratio=-1 \
            --device-batch-size=$DEVICE_BATCH_SIZE \
            --total-batch-size=$TOTAL_BATCH_SIZE \
            --eval-tokens=$EVAL_TOKENS \
            --run="${WANDB_RUN}" \
            --model-tag="${TAG}" \
            --core-metric-every=999999 \
            --core-metric-max-per-task=-1 \
            --sample-every=-1 \
            --save-every=-1 \
            2>&1 | tee "$LOG_FILE"

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        # ---------------------------------------------------------------------
        # Extract stats from log (adapted from scaling_laws.sh)
        # Note: Some grep patterns might need adjustment if log format changes

        # Simple total params extraction
        PARAMS=$(grep "total:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        
        # Iterations
        NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
        # If grep failed (e.g. run crashed), skip
        if [ -z "$NUM_ITERS" ]; then
            log "Error: Training may have failed for d=$d. Skipping CSV entry."
            continue
        fi
        
        TOKENS_TRAINED=$((NUM_ITERS * TOTAL_BATCH_SIZE))
        MODEL_DIM=$((d * 64)) # Assuming aspect-ratio 64 default

        # Val BPB
        VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')
        if [ -z "$VAL_BPB" ]; then VAL_BPB="NaN"; fi

        # CORE score
        CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        if [ -z "$CORE_SCORE" ]; then CORE_SCORE="0.0"; fi

        log "  Params: $PARAMS, Iters: $NUM_ITERS, Val BPB: $VAL_BPB, CORE: $CORE_SCORE"

        # Append to CSV
        echo "$flops,$d,$MODEL_DIM,$PARAMS,$NUM_ITERS,$TOKENS_TRAINED,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"
        
        # Optional: clean up log to save space
        # rm "$LOG_FILE"
    done
done

log "=============================================="
log "Local Scaling Laws Sweep Complete"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
if command -v column &> /dev/null; then
    echo ""
    echo "Results:"
    column -t -s',' "$RESULTS_FILE"
fi

# -----------------------------------------------------------------------------
# Generate Plots
python -m scripts.plot_scaling "$RESULTS_FILE"
