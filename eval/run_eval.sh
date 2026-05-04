#!/bin/bash
#SBATCH --job-name=aolm-eval
#SBATCH --output=/scratch/atharv.johar/AOLM-paper/eval-results/logs/slurm-%j.out
#SBATCH --error=/scratch/atharv.johar/AOLM-paper/eval-results/logs/slurm-%j.err
#SBATCH --partition=u22
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=36
#SBATCH --mem=32G
#SBATCH -w gnode073
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=atharv.johar@research.iiit.ac.in

# ─────────────────────────────────────────────────────────────────
# AOLM evaluation — works as both:
#   bash evaluation/run_eval.sh [mode] [extra flags]
#   sbatch evaluation/run_eval.sh [mode]
#
# Modes: all | baseline | baseline_rag | finetuned | finetuned_rag
#        no-rag | rag-only | build-index
# ─────────────────────────────────────────────────────────────────

set -e

# ── Locate project root regardless of how the script was invoked ─────
# Under `bash`:  $BASH_SOURCE resolves to the script file
# Under `sbatch`: SLURM_SUBMIT_DIR is set to where sbatch was called from
if [ -n "$SLURM_JOB_ID" ]; then
    # Running as a SLURM job — go to submission directory
    PROJECT_DIR="/scratch/atharv.johar/AOLM-paper"
    echo "================================================================"
    echo "  Job ID   : $SLURM_JOB_ID"
    echo "  Node     : $(hostname)"
    echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
    echo "  Started  : $(date)"
    echo "================================================================"
else
    # Running as plain bash
    PROJECT_DIR="/scratch/atharv.johar/AOLM-paper"
fi

cd "$PROJECT_DIR"
echo "  Project  : $PROJECT_DIR"

# ── Paths ─────────────────────────────────────────────────────────────
export AOLM_SCRATCH="${AOLM_SCRATCH:-/scratch/atharv.johar/AOLM-load}"
export RESULTS_DIR="${RESULTS_DIR:-/scratch/atharv.johar/AOLM-paper/eval-results}"

mkdir -p "$AOLM_SCRATCH/hf_cache"
mkdir -p "$AOLM_SCRATCH/chroma_db"
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/logs"

export HF_HOME="$AOLM_SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME/sentence_transformers"

echo "  Cache    : $AOLM_SCRATCH"
echo "  Results  : $RESULTS_DIR"
echo "================================================================"

# ── Install dependencies if missing ──────────────────────────────────
if ! python -c "import transformers, peft, chromadb, sentence_transformers, bitsandbytes" 2>/dev/null; then
    echo "[SETUP] Installing dependencies..."
    pip install --quiet \
        transformers peft accelerate bitsandbytes \
        sentence-transformers chromadb
fi

# ── HF login (set HF_TOKEN env var before running if needed) ─────────
HF_TOKEN="${HF_TOKEN:-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}"
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

# ── Mode dispatch ─────────────────────────────────────────────────────
MODE="${1:-all}"
EXTRA_ARGS=("${@:2}")

case $MODE in
    all)
        python evaluation/run_all.py --results-dir "$RESULTS_DIR" "${EXTRA_ARGS[@]}"
        ;;
    baseline)
        python evaluation/run_all.py --modes baseline --results-dir "$RESULTS_DIR" "${EXTRA_ARGS[@]}"
        ;;
    baseline_rag)
        python evaluation/run_all.py --modes baseline_rag --results-dir "$RESULTS_DIR" "${EXTRA_ARGS[@]}"
        ;;
    finetuned)
        python evaluation/run_all.py --modes finetuned --results-dir "$RESULTS_DIR" "${EXTRA_ARGS[@]}"
        ;;
    finetuned_rag)
        python evaluation/run_all.py --modes finetuned_rag --results-dir "$RESULTS_DIR" "${EXTRA_ARGS[@]}"
        ;;
    finetuned-all)
        python evaluation/run_all.py --modes finetuned finetuned_rag --results-dir "$RESULTS_DIR" "${EXTRA_ARGS[@]}"
        ;;
    no-rag)
        python evaluation/run_all.py --modes baseline finetuned --results-dir "$RESULTS_DIR" "${EXTRA_ARGS[@]}"
        ;;
    rag-only)
        python evaluation/run_all.py --modes baseline_rag finetuned_rag --results-dir "$RESULTS_DIR" "${EXTRA_ARGS[@]}"
        ;;
    build-index)
        python -c "
import sys, os
sys.path.insert(0, 'evaluation')
from rag_system import RAGSystem
rag = RAGSystem(
    chunks_path='RAG-dataset/concepts.json',
    persist_dir=os.path.join('$AOLM_SCRATCH', 'chroma_db'),
)
print(f'Index ready: {rag.collection.count()} chunks')
"
        ;;
    help|*)
        echo "Usage:"
        echo "  bash  evaluation/run_eval.sh [mode] [extra flags]"
        echo "  sbatch evaluation/run_eval.sh [mode]"
        echo ""
        echo "Modes:"
        echo "  all              All 4 modes x 2 eval files (8 runs total) [default]"
        echo "  baseline         Qwen base model, no RAG"
        echo "  baseline_rag     Qwen base model + Graph RAG"
        echo "  finetuned        AOLM fine-tuned model, no RAG"
        echo "  finetuned_rag    AOLM fine-tuned model + Graph RAG"
        echo "  no-rag           baseline + finetuned (both without RAG)"
        echo "  rag-only         baseline_rag + finetuned_rag"
        echo "  build-index      Build ChromaDB index only (one-time)"
        echo ""
        echo "Extra flags (passed to run_all.py):"
        echo "  --skip-existing       Skip result files that already exist"
        echo "  --temperature 0.0     Set generation temperature"
        echo "  --max-new-tokens 512  Max tokens to generate"
        echo "  --rag-min-score 0.5   Min cross-encoder score for RAG injection"
        echo "  --no-graph            Disable graph expansion"
        echo ""
        echo "Environment variables:"
        echo "  AOLM_SCRATCH  /scratch/atharv.johar/AOLM-load  (HF + Chroma cache)"
        echo "  RESULTS_DIR   /scratch/atharv.johar/AOLM-paper/eval-results"
        echo "  HF_TOKEN      HuggingFace token (if model is private)"
        ;;
esac

if [ -n "$SLURM_JOB_ID" ]; then
    echo "================================================================"
    echo "  Finished : $(date)"
    echo "================================================================"
fi
