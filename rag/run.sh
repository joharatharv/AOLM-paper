#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# CurioChain Math Tutor — Setup & Run Script (SSH / RTX 2080)
# ─────────────────────────────────────────────────────────────────

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ─── 1. Install dependencies ─────────────────────────────────────
echo "========================================="
echo "  Installing dependencies..."
echo "========================================="

pip install --quiet --upgrade pip
pip install --quiet \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    sentence-transformers \
    chromadb \
    peft \
    accelerate \
    bitsandbytes \
    transformers \
    trl

echo "[OK] Dependencies installed"

# ─── 2. Run modes ────────────────────────────────────────────────
MODE=${1:-help}

case $MODE in

  # Fine-tuned model + Graph RAG (interactive)
  ft-rag)
    echo ">>> Fine-tuned model + Graph RAG — Interactive"
    python rag/inference.py --model finetuned --rag --mode interactive \
        --rag-min-score 0.5 --graph-neighbors 2
    ;;

  # Fine-tuned model, no RAG (interactive)
  ft-norag)
    echo ">>> Fine-tuned model, no RAG — Interactive"
    python rag/inference.py --model finetuned --mode interactive
    ;;

  # Baseline model + RAG (interactive)
  base-rag)
    echo ">>> Baseline model + RAG — Interactive"
    python rag/inference.py --model baseline --rag --mode interactive \
        --rag-min-score 0.5 --graph-neighbors 2
    ;;

  # Baseline model, no RAG (interactive)
  base-norag)
    echo ">>> Baseline model, no RAG — Interactive"
    python rag/inference.py --model baseline --mode interactive
    ;;

  # Batch evaluation — all 4 combinations
  eval-all)
    EVAL_FILE=${2:-eval_prompts.json}
    echo ">>> Running all 4 evaluation combinations on: $EVAL_FILE"

    echo "[1/4] Fine-tuned + RAG..."
    python rag/inference.py --model finetuned --rag --mode batch \
        --eval-file "$EVAL_FILE" --output-file results_finetuned_rag.json

    echo "[2/4] Fine-tuned, no RAG..."
    python rag/inference.py --model finetuned --mode batch \
        --eval-file "$EVAL_FILE" --output-file results_finetuned_norag.json

    echo "[3/4] Baseline + RAG..."
    python rag/inference.py --model baseline --rag --mode batch \
        --eval-file "$EVAL_FILE" --output-file results_baseline_rag.json

    echo "[4/4] Baseline, no RAG..."
    python rag/inference.py --model baseline --mode batch \
        --eval-file "$EVAL_FILE" --output-file results_baseline_norag.json

    echo "[DONE] All 4 result files generated."
    ;;

  # Just build the RAG index (no model loading)
  build-index)
    echo ">>> Building ChromaDB index from concepts.json..."
    python -c "
import sys; sys.path.insert(0, '.')
from rag.rag_system import RAGSystem
rag = RAGSystem(chunks_path='RAG-dataset/concepts.json', persist_dir='rag/chroma_db')
print(f'Index built: {rag.collection.count()} chunks')
"
    ;;

  # Test RAG retrieval only (no model)
  test-rag)
    QUERY=${2:-"How to find the general term in binomial expansion?"}
    MIN_SCORE=${3:-0.5}
    echo ">>> Testing Graph RAG retrieval for: $QUERY  (min_score=$MIN_SCORE)"
    python -c "
import sys; sys.path.insert(0, '.')
from rag.rag_system import RAGSystem
rag = RAGSystem(chunks_path='RAG-dataset/concepts.json', persist_dir='rag/chroma_db')
results = rag.retrieve('$QUERY', top_k=10, top_n=3, min_score=$MIN_SCORE, use_graph=True, max_graph_neighbors=2)
if not results:
    print('[Score gate triggered — no context injected]')
else:
    seeds    = [r for r in results if not r.get('graph_source')]
    expanded = [r for r in results if r.get('graph_source')]
    print('Seeds (direct retrieval):')
    for r in seeds:
        print(f\"  {r['chunk_id']:45s}  score={r['rerank_score']:.4f}\")
    if expanded:
        print('Graph-expanded neighbors:')
        for r in expanded:
            print(f\"  {r['chunk_id']:45s}  via '{r['graph_edge']}' from {r['graph_seed']}\")
    print()
    print(rag.format_context(results))
"
    ;;

  help|*)
    echo "========================================="
    echo "  CurioChain Math Tutor — Usage"
    echo "========================================="
    echo ""
    echo "  bash rag/run.sh <mode> [args]"
    echo ""
    echo "  INTERACTIVE MODES:"
    echo "    ft-rag        Fine-tuned model + Graph RAG"
    echo "    ft-norag      Fine-tuned model, no RAG"
    echo "    base-rag      Baseline model + Graph RAG"
    echo "    base-norag    Baseline model, no RAG"
    echo ""
    echo "  BATCH EVALUATION:"
    echo "    eval-all [eval_file.json]"
    echo "                  Run all 4 combinations (ft/base × rag/norag)"
    echo ""
    echo "  UTILITIES:"
    echo "    build-index   Build ChromaDB vector index only"
    echo "    test-rag \"query\" [min_score]"
    echo "                  Test Graph RAG retrieval without loading LLM"
    echo "                  e.g. bash rag/run.sh test-rag \"tangent to circle\" 0.5"
    echo ""
    echo "  GRAPH RAG FLAGS (pass after -- to inference.py directly):"
    echo "    --rag-min-score N   Drop low-score chunks and skip injection if best score < N (default: 0.5)"
    echo "    --no-graph          Disable graph expansion (vector+rerank only)"
    echo "    --graph-neighbors N Max chunks added by graph expansion (default: 2)"
    echo ""
    ;;
esac
