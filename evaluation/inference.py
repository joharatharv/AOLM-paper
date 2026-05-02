"""
AOLM Evaluation — Inference module
Supports 4 modes: baseline | finetuned | baseline+rag | finetuned+rag

Base model     : Qwen/Qwen2.5-Math-7B-Instruct
Fine-tuned     : joharatharv/AOLM-qwen-7b   (LoRA adapter on Hub)
GPU            : RTX 2080 (8/11 GB) — 4-bit NF4 + FP16 + grad-ckpt
Cache dir      : /scratch/atharv.johar/AOLM-load/   (set via env vars)
Result dir     : /home2/atharv.johar/AOLM-paper-eval-results/
"""

import argparse
import json
import os
import re
import sys

# ─────────────────────────────────────────────────────────────────
# CACHE REDIRECTION — must be set BEFORE importing transformers
# ─────────────────────────────────────────────────────────────────
SCRATCH_ROOT = os.environ.get("AOLM_SCRATCH", "/scratch/atharv.johar/AOLM-load")
HF_CACHE     = os.path.join(SCRATCH_ROOT, "hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)

os.environ["HF_HOME"]            = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"]  = os.path.join(HF_CACHE, "datasets")
os.environ["HF_HUB_CACHE"]       = os.path.join(HF_CACHE, "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(HF_CACHE, "sentence_transformers")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_system import RAGSystem


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
BASE_MODEL          = "Qwen/Qwen2.5-7B-Instruct"
FINETUNED_ADAPTER   = "joharatharv/AOLM_final_finetuned"
MAX_SEQ_LENGTH      = 1536

CHROMA_DB_PATH      = os.path.join(SCRATCH_ROOT, "chroma_db")
RESULTS_DIR_DEFAULT = "/home2/atharv.johar/AOLM-paper-eval-results"


def _max_memory_map(per_gpu_gb: int = 7) -> dict:
    """Build a max_memory dict for HF device_map='auto'.

    Reserves `per_gpu_gb` per visible GPU (default 7 GiB on an 8 GiB 2080).
    Lets transformers spread the model across all GPUs via pipeline parallelism.
    """
    n = torch.cuda.device_count()
    mem: dict = {i: f"{per_gpu_gb}GiB" for i in range(n)}
    mem["cpu"] = "32GiB"          # fallback offload to CPU if anything overflows
    return mem


# ─────────────────────────────────────────────────────────────────
# MODEL LOADING — plain FP16, NO quantization, multi-GPU pipeline parallel
# ─────────────────────────────────────────────────────────────────

def _disable_flash_sdp():
    """Force SDPA to use the 'math' backend (FP32 softmax) on Turing GPUs.

    Turing (RTX 2080, cc7.5) supports neither FlashAttention nor the SDPA
    memory-efficient kernel. Without this, SDPA throws:
      RuntimeError: FlashAttention only supports Ampere GPUs or newer.
    The math backend upscales all internal ops to FP32, preventing the
    FP16 overflow that causes garbage token repetition.
    """
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)  # also Ampere-only
    torch.backends.cuda.enable_math_sdp(True)            # always available


def load_baseline_model():
    """Base model in plain FP16, sharded across all visible GPUs."""
    _disable_flash_sdp()
    n_gpu = torch.cuda.device_count()
    print(f"[MODEL] Loading baseline: {BASE_MODEL}  (FP16, {n_gpu} GPU(s), SDPA-math)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        max_memory=_max_memory_map(),
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print("[MODEL] Baseline ready")
    return model, tokenizer


def load_finetuned_model(adapter_id=FINETUNED_ADAPTER):
    """Base model (FP16, multi-GPU) + LoRA adapter."""
    _disable_flash_sdp()
    n_gpu = torch.cuda.device_count()
    print(f"[MODEL] Loading fine-tuned: {BASE_MODEL} + {adapter_id}  "
          f"(FP16, {n_gpu} GPU(s), SDPA-math)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        max_memory=_max_memory_map(),
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(base_model, adapter_id)
    model.eval()
    print("[MODEL] Fine-tuned (base + LoRA) ready")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────
# PROMPT CONSTRUCTION
# ─────────────────────────────────────────────────────────────────

def apply_chat_template(tokenizer, system_prompt: str, user_msg: str):
    """Use the model's native chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return input_ids


def inject_rag_into_user_msg(original_user_msg: str, rag_context: str) -> str:
    """Prepend RAG context to the existing user message (preserves training format)."""
    if not rag_context:
        return original_user_msg
    return (
        "RELEVANT REFERENCE MATERIAL (use if helpful):\n"
        f"{rag_context}\n\n"
        f"{original_user_msg}"
    )


# ─────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────

def generate_response(
    model, tokenizer, input_ids,
    max_new_tokens: int = 512,
    temperature: float = 0.0,        # 0.0 → greedy decoding (deterministic, paper-reproducible)
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
) -> str:
    """Generate using greedy by default (temperature=0).

    Greedy avoids the `torch.multinomial` instability seen with Qwen2.5-Math
    + 4-bit NF4 + FP16, where the LM-head logits can briefly contain inf/NaN
    and crash sampling. Greedy never calls multinomial, so it is robust here
    and is also what the Qwen team recommends for math evaluation.
    """
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    target_device = next(model.parameters()).device
    input_ids = input_ids.to(target_device)
    attention_mask = torch.ones_like(input_ids)

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,                      # extra guard against degenerate distributions
            repetition_penalty=repetition_penalty,
        )

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_output(raw: str) -> dict:
    """Extract <thought> and <response> blocks (fall back gracefully)."""
    thought_m  = re.search(r"<thought>(.*?)</thought>",   raw, re.DOTALL)
    response_m = re.search(r"<response>(.*?)</response>", raw, re.DOTALL)
    return {
        "thought":  thought_m.group(1).strip()  if thought_m  else "",
        "response": response_m.group(1).strip() if response_m else raw.strip(),
        "raw":      raw,
    }


# ─────────────────────────────────────────────────────────────────
# BATCH INFERENCE (with resume support)
# ─────────────────────────────────────────────────────────────────

def batch_inference(
    model, tokenizer,
    eval_file: str,
    output_file: str,
    rag_system: RAGSystem = None,
    use_rag: bool = False,
    rag_min_score: float = 0.5,
    rag_use_graph: bool = True,
    rag_max_graph_neighbors: int = 2,
    max_new_tokens: int = 512,
    temperature: float = 0.0,        # 0.0 → greedy
):
    """Run inference across an eval JSON; saves incrementally with resume."""
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    out_dir = os.path.dirname(output_file) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Resume support
    results, done_ids = [], set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_ids = {r["eval_id"] for r in results}
            print(f"[RESUME] Found {len(done_ids)} completed items in {output_file}")
        except Exception as e:
            print(f"[RESUME] Could not load existing results ({e}); starting fresh")
            results, done_ids = [], set()

    total = len(eval_data)
    # Use the same key that we write into the result ("eval_id") so resume works
    remaining = [item for item in eval_data if item.get("id", item.get("eval_id")) not in done_ids]
    print(f"[INFO] Total={total} | Done={len(done_ids)} | Remaining={len(remaining)}")

    for i, item in enumerate(remaining, 1):
        item_id    = item.get("id", item.get("eval_id", f"item_{i}"))
        instruction = item["instruction"]
        user_msg_original = item["input"]
        problem    = item.get("problem", "")
        student    = item.get("student_working", "")

        # ── RAG retrieval
        retrieved_chunks = []
        rag_context = ""
        if use_rag and rag_system is not None:
            query = f"{problem} {student}".strip() or user_msg_original
            retrieved_chunks = rag_system.retrieve(
                query, top_k=10, top_n=3,
                min_score=rag_min_score,
                use_graph=rag_use_graph,
                max_graph_neighbors=rag_max_graph_neighbors,
            )
            rag_context = rag_system.format_context(retrieved_chunks)

        # ── Build user message
        user_msg = inject_rag_into_user_msg(user_msg_original, rag_context)

        # ── Generate
        input_ids = apply_chat_template(tokenizer, instruction, user_msg)
        raw_output = generate_response(
            model, tokenizer, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        parsed = parse_output(raw_output)

        results.append({
            "eval_id":          item_id,
            "base_problem_id":  item.get("base_problem_id", ""),
            "student_state_type": item.get("student_state_type", ""),
            "problem":          problem,
            "student_working":  student,
            "retrieved_chunks": [
                {"chunk_id": c["chunk_id"], "score": c.get("rerank_score", 0)}
                for c in retrieved_chunks
            ],
            "rag_context_used": bool(rag_context),
            "thought":          parsed["thought"],
            "response":         parsed["response"],
            "raw_output":       parsed["raw"],
        })

        # Save after every item
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if i % 5 == 0 or i == len(remaining):
            print(f"  [{len(done_ids) + i}/{total}] saved")

    print(f"[DONE] {len(results)} results -> {output_file}")
    return results


# ─────────────────────────────────────────────────────────────────
# CLI (single-mode runner — used by run_all.py for orchestration)
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "finetuned"], required=True)
    parser.add_argument("--rag",   action="store_true")
    parser.add_argument("--eval-file",   required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--rag-min-score", type=float, default=0.5)
    parser.add_argument("--no-graph",      action="store_true")
    parser.add_argument("--graph-neighbors", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature",    type=float, default=0.0,
                        help="0.0 = greedy decoding (default, recommended for eval)")
    args = parser.parse_args()

    # Load model
    if args.model == "baseline":
        model, tokenizer = load_baseline_model()
    else:
        model, tokenizer = load_finetuned_model()

    # Load RAG if needed
    rag_system = None
    if args.rag:
        rag_system = RAGSystem(
            chunks_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "RAG-dataset", "concepts.json",
            ),
            persist_dir=CHROMA_DB_PATH,
        )

    batch_inference(
        model, tokenizer,
        eval_file   = args.eval_file,
        output_file = args.output_file,
        rag_system  = rag_system,
        use_rag     = args.rag,
        rag_min_score = args.rag_min_score,
        rag_use_graph = not args.no_graph,
        rag_max_graph_neighbors = args.graph_neighbors,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
    )


if __name__ == "__main__":
    main()
