"""
run_all.py — Orchestrates all 8 evaluation runs:
  4 modes × 2 eval files

Modes:
  1. baseline                  (Qwen2.5-Math-7B-Instruct, no RAG)
  2. baseline_rag              (Qwen2.5-Math-7B-Instruct + Graph RAG)
  3. finetuned                 (base + AOLM-qwen-7b adapter, no RAG)
  4. finetuned_rag             (base + AOLM-qwen-7b adapter + Graph RAG)

Eval files:
  - eval_dataset_150.json      (150 entries, general tutor evaluation)
  - eval_trap_prompt.json      (50 entries, IDK escalation protocol traps)

KEY OPTIMISATION: each model is loaded ONLY ONCE.
Workflow:
  1. Load baseline model    → run no-RAG  on both files → run RAG on both files
  2. Free baseline, load FT → run no-RAG  on both files → run RAG on both files

This produces 8 result JSONs in <RESULTS_DIR> with names:
  results_<mode>_<eval_short>.json

Usage:
  python evaluation/run_all.py
  python evaluation/run_all.py --modes baseline finetuned       # subset of modes
  python evaluation/run_all.py --eval-files eval_dataset_150    # subset of files
  python evaluation/run_all.py --skip-existing                  # skip done files
"""

import argparse
import gc
import json
import os
import sys

# Set cache before importing transformers
SCRATCH_ROOT = os.environ.get("AOLM_SCRATCH", "/scratch/atharv.johar/AOLM-load")
HF_CACHE     = os.path.join(SCRATCH_ROOT, "hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"]            = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"]  = os.path.join(HF_CACHE, "datasets")
os.environ["HF_HUB_CACHE"]       = os.path.join(HF_CACHE, "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(HF_CACHE, "sentence_transformers")

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import (
    load_baseline_model, load_finetuned_model,
    batch_inference, BASE_MODEL, FINETUNED_ADAPTER, CHROMA_DB_PATH,
)
from rag_system import RAGSystem


# ─────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR      = os.path.join(PROJECT_ROOT, "eval_data")
RAG_DATA_PATH = os.path.join(PROJECT_ROOT, "RAG-dataset", "concepts.json")

DEFAULT_RESULTS_DIR = "/scratch/atharv.johar/AOLM-paper/eval-results"

EVAL_FILES = {
    "eval_dataset_150":  os.path.join(EVAL_DIR, "eval_dataset_150.json"),
    "eval_trap_prompt":  os.path.join(EVAL_DIR, "eval_trap_prompt.json"),
}

ALL_MODES = ["baseline", "baseline_rag", "finetuned", "finetuned_rag"]


# ─────────────────────────────────────────────────────────────────
# RUN HELPERS
# ─────────────────────────────────────────────────────────────────

def out_path(results_dir: str, mode: str, eval_short: str) -> str:
    return os.path.join(results_dir, f"results_{mode}_{eval_short}.json")


def run_mode(
    model, tokenizer, mode: str,
    eval_files: dict, results_dir: str,
    rag_system: RAGSystem = None,
    skip_existing: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    rag_min_score: float = 0.5,
    rag_use_graph: bool = True,
    rag_max_graph_neighbors: int = 2,
):
    use_rag = mode.endswith("_rag")
    print(f"\n{'='*70}")
    print(f">>> MODE: {mode}   (RAG={'ON' if use_rag else 'OFF'})")
    print(f"{'='*70}")

    for short, eval_path in eval_files.items():
        out = out_path(results_dir, mode, short)
        # Per-item resume is handled inside batch_inference.
        # Only do a full-file skip if --skip-existing is set AND the file is already complete.
        if skip_existing and os.path.exists(out):
            try:
                with open(out, "r", encoding="utf-8") as _f:
                    done = json.load(_f)
                with open(eval_path, "r", encoding="utf-8") as _f:
                    total_items = len(json.load(_f))
                if len(done) >= total_items:
                    print(f"[SKIP] {out} — all {total_items} items already evaluated")
                    continue
            except Exception:
                pass  # fall through to batch_inference which will resume
        print(f"\n--- {mode} on {short} ---")
        batch_inference(
            model, tokenizer,
            eval_file   = eval_path,
            output_file = out,
            rag_system  = rag_system if use_rag else None,
            use_rag     = use_rag,
            rag_min_score = rag_min_score,
            rag_use_graph = rag_use_graph,
            rag_max_graph_neighbors = rag_max_graph_neighbors,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
        )


def free_model(model):
    """Release VRAM before loading the next model."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[CLEANUP] Freed model from VRAM")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes", nargs="+", default=ALL_MODES,
        choices=ALL_MODES,
        help="Which modes to run (default: all 4)",
    )
    parser.add_argument(
        "--eval-files", nargs="+", default=list(EVAL_FILES.keys()),
        choices=list(EVAL_FILES.keys()),
        help="Which eval files to run (default: both)",
    )
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR,
        help=f"Where to save result JSONs (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a (mode, eval_file) combination if its result file already exists",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature",    type=float, default=0.0,
                        help="0.0 = greedy (default; deterministic, paper-reproducible)")
    parser.add_argument("--rag-min-score",  type=float, default=0.5)
    parser.add_argument("--no-graph",       action="store_true")
    parser.add_argument("--graph-neighbors", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"[CONFIG] Cache    : {HF_CACHE}")
    print(f"[CONFIG] ChromaDB : {CHROMA_DB_PATH}")
    print(f"[CONFIG] Results  : {args.results_dir}")
    print(f"[CONFIG] Modes    : {args.modes}")
    print(f"[CONFIG] Files    : {args.eval_files}")

    selected_eval_files = {k: EVAL_FILES[k] for k in args.eval_files}

    # Build RAG system once if any RAG-mode is requested
    rag_system = None
    if any(m.endswith("_rag") for m in args.modes):
        print(f"\n[INIT] Building RAG system (one-time setup)...")
        rag_system = RAGSystem(
            chunks_path=RAG_DATA_PATH,
            persist_dir=CHROMA_DB_PATH,
        )

    rag_kwargs = dict(
        rag_min_score = args.rag_min_score,
        rag_use_graph = not args.no_graph,
        rag_max_graph_neighbors = args.graph_neighbors,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
    )

    # ── Group modes by which model they need so we load each model only once
    baseline_modes  = [m for m in args.modes if m.startswith("baseline")]
    finetuned_modes = [m for m in args.modes if m.startswith("finetuned")]

    if baseline_modes:
        print(f"\n[STAGE 1/2] Loading BASELINE model ({BASE_MODEL})")
        model, tokenizer = load_baseline_model()
        for mode in baseline_modes:
            run_mode(model, tokenizer, mode,
                     selected_eval_files, args.results_dir,
                     rag_system=rag_system,
                     skip_existing=args.skip_existing,
                     **rag_kwargs)
        free_model(model)
        del tokenizer
        gc.collect()

    if finetuned_modes:
        print(f"\n[STAGE 2/2] Loading FINE-TUNED model ({BASE_MODEL} + {FINETUNED_ADAPTER})")
        model, tokenizer = load_finetuned_model()
        for mode in finetuned_modes:
            run_mode(model, tokenizer, mode,
                     selected_eval_files, args.results_dir,
                     rag_system=rag_system,
                     skip_existing=args.skip_existing,
                     **rag_kwargs)
        free_model(model)
        del tokenizer
        gc.collect()

    print(f"\n{'='*70}")
    print(">>> ALL EVALUATIONS COMPLETE")
    print(f"{'='*70}")
    print(f"Result files in: {args.results_dir}")
    for mode in args.modes:
        for short in args.eval_files:
            p = out_path(args.results_dir, mode, short)
            status = "OK" if os.path.exists(p) else "MISSING"
            sz = f"({os.path.getsize(p)//1024} KB)" if os.path.exists(p) else ""
            print(f"  [{status:7s}] {p}  {sz}")


if __name__ == "__main__":
    main()
