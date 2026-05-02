#!/bin/bash
#SBATCH --job-name=aolm_finetune
#SBATCH --output=/scratch/atharv.johar/AOLM/logs/train_%j.log
#SBATCH --error=/scratch/atharv.johar/AOLM/logs/train_%j.err
#SBATCH --partition=u22
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH -w gnode070
#SBATCH -A research
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=atharv.johar@research.iiit.ac.in

# Usage:
#   sbatch training/run_training.sbatch qwen25_math_7b    # primary (~13 hrs)
#   sbatch training/run_training.sbatch qwen25_3b         # small   (~3 hrs)
#   sbatch training/run_training.sbatch llama31_8b        # secondary (~15 hrs)
#
# If no argument given, defaults to qwen25_math_7b.
# Pass the model name via SBATCH --export or as a script argument:
#   sbatch --export=ALL,MODEL=qwen25_3b training/run_training.sbatch

set -euo pipefail

# ------ 0. cd to project root (script may be copied to /var/spool/slurmd by Slurm) ----
# Prefer explicit PROJECT_DIR env var; otherwise use the project's scratch path if present.
# When Slurm runs the job it copies the job script to /var/spool/slurmd, so deriving the
# repo root from the script location is unreliable. Use an explicit, known path here.
if [ -n "${PROJECT_DIR:-}" ]; then
    echo "Using PROJECT_DIR from environment: $PROJECT_DIR"
else
    if [ -d "/scratch/atharv.johar/AOLM-paper" ]; then
        PROJECT_DIR="/scratch/atharv.johar/AOLM-paper"
    else
        # Fallback: try to derive from the script location, otherwise use current dir
        if [ -n "${BASH_SOURCE[0]:-}" ]; then
            SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
            PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
        else
            PROJECT_DIR="$(pwd)"
        fi
    fi
fi
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }
echo "Working directory: $PROJECT_DIR"

# ------ 1. Resolve model ----------------------------------------------------------------------------------------------------------------
MODEL="${MODEL:-${1:-qwen25_7b_4gpu}}"
CONFIG="training/${MODEL}_lora.yaml"
# 4-gpu model has its own config name (qwen25_7b_lora_4gpu.yaml)
if [ "$MODEL" = "qwen25_7b_4gpu" ]; then
    CONFIG="training/qwen25_7b_lora_4gpu.yaml"
fi

# ------ 1. Activate environment --------------------------------------------------------------------------------------------------
ENV_DIR="/scratch/atharv.johar/envs/my_env"

if [ ! -d "$ENV_DIR/bin" ]; then
    echo "ERROR: environment not found at $ENV_DIR"
    exit 1
fi

export PATH="$ENV_DIR/bin:$PATH"
export CONDA_PREFIX="$ENV_DIR"

# ------ 2. Redirect ALL cache / output dirs to /scratch --------------------------------------------------
SCRATCH="/scratch/atharv.johar/AOLM"

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache/datasets"
export HF_HUB_CACHE="$SCRATCH/hf_cache/hub"
export HUGGINGFACE_HUB_CACHE="$SCRATCH/hf_cache/hub"

# Axolotl writes its prepared dataset cache here (can be GBs)
export HF_DATASETS_CACHE="$SCRATCH/hf_cache/datasets"
export AXOLOTL_CACHE_DIR="$SCRATCH/axolotl_cache"

# pip caches
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
export XDG_CACHE_HOME="$SCRATCH/xdg_cache"

# Torch / CUDA scratch
export TORCH_HOME="$SCRATCH/torch_home"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# W&B -- disable if no key set; otherwise redirect run dir
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=disabled
else
    export WANDB_DIR="$SCRATCH/wandb"
    mkdir -p "$WANDB_DIR"
fi

mkdir -p \
    "$SCRATCH/logs" \
    "$SCRATCH/hf_cache/hub" \
    "$SCRATCH/hf_cache/datasets" \
    "$SCRATCH/axolotl_cache" \
    "$SCRATCH/torch_home" \
    "$SCRATCH/pip_cache" \
    "$SCRATCH/xdg_cache" \
    "$SCRATCH/outputs"

# ------ 3. GPU sanity ----------------------------------------------------------------------------------------------------------------------
unset CUDA_VISIBLE_DEVICES
unset SLURM_JOB_GPUS

echo "=========================================="
echo "AOLM Fine-Tuning"
echo "Node     : $(hostname)"
echo "Model    : $MODEL"
echo "Config   : $CONFIG"
echo "Scratch  : $SCRATCH"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ------ 4. HuggingFace login (Llama-3.1 is gated) --------------------------------------------------------------
if [ -n "${HF_TOKEN:-}" ]; then
    echo "[Step 4] Logging into HuggingFace..."
    hf auth login --token "$HF_TOKEN"
fi

# ------ 5. Verify dataset ----------------------------------------------------------------------------------------------------------------
echo "[Step 5] Checking dataset..."
python - <<'EOF'
import json, sys
for path in ["final_ft_dataset/train.jsonl", "final_ft_dataset/val.jsonl", "final_dataset/train.jsonl"]:
    try:
        n = sum(1 for l in open(path, encoding="utf-8") if l.strip())
        print(f"  {path}: {n} entries")
    except FileNotFoundError:
        print(f"  (not found, skipping): {path}")
EOF

# ------ 6. Patch output_dir in config to write under /scratch --------------------------------------
# This makes the adapter land in /scratch/atharv.johar/AOLM/outputs/<model>/
# instead of the project's training/outputs/ (which sits on home2).
ADAPTER_OUT="$SCRATCH/outputs/${MODEL}_lora_r32"
mkdir -p "$ADAPTER_OUT"

# sed replaces the output_dir line in the yaml at runtime (non-destructive copy)
PATCHED_CONFIG="$SCRATCH/axolotl_cache/${MODEL}_lora_patched.yaml"
sed "s|output_dir:.*|output_dir: $ADAPTER_OUT|" "$CONFIG" > "$PATCHED_CONFIG"

echo "[Step 6] Adapter will be saved to: $ADAPTER_OUT"

# ------ 7. Run training ------------------------------------------------------------------------------------------------------------------
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "[Step 7] Starting Axolotl training on $NUM_GPUS GPU(s)..."

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU: DeepSpeed ZeRO-3 handles sharding
    # Requires: pip install deepspeed
    if ! python -c "import deepspeed" 2>/dev/null; then
        echo "[Step 7] Installing deepspeed..."
        pip install deepspeed --quiet
    fi
    deepspeed --num_gpus "$NUM_GPUS" --module axolotl.cli.train "$PATCHED_CONFIG"
else
    python -m axolotl.cli.train "$PATCHED_CONFIG"
fi

echo ""
echo "=========================================="
echo "Training complete."
echo "Adapter : $ADAPTER_OUT"
echo "Logs    : $SCRATCH/logs/train_${SLURM_JOB_ID}.log"
echo "=========================================="
