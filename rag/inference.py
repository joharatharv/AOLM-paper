"""
CurioChain Math Tutor — Inference
Supports: fine-tuned / baseline × with / without RAG
Target GPU: RTX 2080 (8GB VRAM, 4-bit quantized)
"""

import argparse
import json
import sys
import os
import re

# Redirect HuggingFace cache to /scratch to avoid home disk quota
_hf_cache = os.environ.get("HF_HOME", "/scratch/lakshmiprajna.p/hf_cache")
os.environ["HF_HOME"] = _hf_cache
os.environ["TRANSFORMERS_CACHE"] = _hf_cache
os.environ["HF_DATASETS_CACHE"] = os.path.join(_hf_cache, "datasets")

import unsloth  # must be before transformers/peft
from unsloth.chat_templates import get_chat_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add parent dir so we can import rag module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.rag_system import RAGSystem


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
BASE_MODEL = "google/gemma-2-9b-it"
FINETUNED_ADAPTER = "joharatharv/AOLM_old_working_ft"
MAX_SEQ_LENGTH = 1536

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

SYSTEM_PROMPT = """You specialize in guided discovery learning for mathematics.

DISCOVERY PROCESS:
<thought> block: Solve the problem completely yourself
Analysis: Determine student's progress from their working
Guidance: Offer a hint that leads to the next step
Engagement: Ask them to show their work

DISCOVERY PRINCIPLES:
• The 'aha moment' should belong to the student
• Hints illuminate the path without walking it
• Validate effort and progress consistently
• Complexity adjusts to student readiness

SPECIAL CASE PROTOCOLS:
• No work + claims stuck → Require any attempt before helping
• Emotionally frustrated → Validate, highlight wins, offer smaller step
• Shortcut-seeking → Redirect focus to learning process
• Knowledge gap → Fill it, then require demonstration

Always request their working at the end."""


# ─────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────

CHATML_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}"
    "{% elif message['role'] == 'assistant' %}"
    "{{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{% else %}"
    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

def load_finetuned_model():
    """Load fine-tuned CurioChain model using the canonical Unsloth inference pattern.

    FastLanguageModel.for_inference() is REQUIRED — it stitches the LoRA weights
    in Unsloth's internal format and switches the model out of training mode.
    Skipping it causes garbled / incoherent generation even when weights are valid.
    """
    print("[MODEL] Loading fine-tuned model via Unsloth...")
    from unsloth import FastLanguageModel

    # Load base + adapter together via Unsloth (adapter_config.json points to base)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINETUNED_ADAPTER,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # CRITICAL: switches model from training mode → inference mode, stitches LoRA
    FastLanguageModel.for_inference(model)

    # Re-apply get_chat_template to fix token ID mappings lost during save.
    # <|im_start|> and <|im_end|> save as token 3 (<unk>) on disk;
    # get_chat_template re-maps them to tokens 106/1 as Unsloth expects.
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    model.eval()
    print("[MODEL] Fine-tuned model ready")
    return model, tokenizer


def load_baseline_model():
    """Load the base Gemma-2-9B-IT without any LoRA adapter."""
    print("[MODEL] Loading baseline model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        dtype=torch.float16,
        device_map="auto",
        max_memory={0: "9GiB", "cpu": "24GiB"},
        attn_implementation="eager",
    )
    model.eval()
    print("[MODEL] Baseline model ready")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────
# PROMPT CONSTRUCTION
# ─────────────────────────────────────────────────────────────────

def build_user_message(
    problem: str,
    student_working: str,
    conversation_history: str = "",
    rag_context: str = "",
) -> str:
    """Build the user-turn content matching the training data format."""
    parts = []

    if rag_context:
        parts.append(
            "RELEVANT REFERENCE MATERIAL (use if helpful):\n"
            f"{rag_context}\n"
        )

    parts.append(f"PROBLEM: {problem}")

    if conversation_history:
        parts.append(f"\n[CONVERSATION HISTORY]\n{conversation_history}")

    parts.append(f"\nSTUDENT'S CURRENT WORKING:\n{student_working}")

    return "\n".join(parts)


def apply_chat_template(tokenizer, system: str, user: str) -> str:
    """Apply chat template and return tokenized input_ids.
    
    If the tokenizer supports system role (ChatML from fine-tuned model), use it.
    Otherwise (baseline Gemma), fold system prompt into the user message.
    """
    # Try with system role first
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return input_ids
    except Exception:
        # Baseline Gemma doesn't support system role — fold into user message
        combined_user = f"{system}\n\n{user}"
        messages = [
            {"role": "user", "content": combined_user},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return input_ids


# ─────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────

def generate_response(
    model,
    tokenizer,
    input_ids,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate a response from the model."""
    # apply_chat_template may return a BatchEncoding; extract the raw tensor
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to(model.device)
    attention_mask = (input_ids != model.config.pad_token_id).long() if hasattr(model, 'config') and model.config.pad_token_id is not None else torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            use_cache=True,
        )

    # Decode only the generated tokens (skip the input)
    generated_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


def parse_response(raw: str) -> dict:
    """Parse the <thought> and <response> blocks from model output."""
    thought_match = re.search(r"<thought>(.*?)</thought>", raw, re.DOTALL)
    response_match = re.search(r"<response>(.*?)</response>", raw, re.DOTALL)

    return {
        "thought": thought_match.group(1).strip() if thought_match else "",
        "response": response_match.group(1).strip() if response_match else raw.strip(),
        "raw": raw,
    }


# ─────────────────────────────────────────────────────────────────
# MAIN INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    problem: str,
    student_working: str,
    conversation_history: str = "",
    rag_system: RAGSystem = None,
    use_rag: bool = True,
    rag_min_score: float = 0.5,
    rag_use_graph: bool = True,
    rag_max_graph_neighbors: int = 2,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt_override: str = None,
):
    """Complete inference pipeline: optional RAG retrieval → prompt → generate → parse.

    Args:
        rag_min_score: Minimum cross-encoder score to inject any context. Chunks
                       scoring below this are discarded (prevents OOD injection).
        rag_use_graph: Whether to expand seed chunks via graph edges (Graph RAG).
        rag_max_graph_neighbors: Max additional chunks added by graph expansion.
    """

    rag_context = ""
    retrieved_chunks = []

    if use_rag and rag_system is not None:
        # Use problem + student working as the retrieval query
        query = f"{problem} {student_working}"
        retrieved_chunks = rag_system.retrieve(
            query,
            top_k=10,
            top_n=3,
            min_score=rag_min_score,
            use_graph=rag_use_graph,
            max_graph_neighbors=rag_max_graph_neighbors,
        )
        rag_context = rag_system.format_context(retrieved_chunks)

    # Build prompt
    user_msg = build_user_message(
        problem=problem,
        student_working=student_working,
        conversation_history=conversation_history,
        rag_context=rag_context,
    )

    sys_prompt = system_prompt_override if system_prompt_override else SYSTEM_PROMPT
    input_ids = apply_chat_template(tokenizer, sys_prompt, user_msg)

    # Generate
    raw_output = generate_response(
        model, tokenizer, input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    parsed = parse_response(raw_output)

    return {
        "problem": problem,
        "student_working": student_working,
        "retrieved_chunks": [
            {"chunk_id": c["chunk_id"], "score": c.get("rerank_score", 0)}
            for c in retrieved_chunks
        ],
        "rag_context_used": bool(rag_context),
        "thought": parsed["thought"],
        "response": parsed["response"],
        "raw_output": parsed["raw"],
    }


# ─────────────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────────

def interactive_mode(
    model,
    tokenizer,
    rag_system=None,
    use_rag=True,
    rag_min_score=0.5,
    rag_use_graph=True,
    rag_max_graph_neighbors=2,
):
    """Interactive tutoring session in the terminal."""
    print("\n" + "=" * 60)
    print("  CurioChain Math Tutor — Interactive Mode")
    rag_status = "OFF"
    if use_rag and rag_system:
        rag_status = f"ON (graph={'ON' if rag_use_graph else 'OFF'}, min_score={rag_min_score})"
    print(f"  RAG: {rag_status}")
    print("  Type 'quit' to exit, 'new' for a new problem")
    print("=" * 60)

    conversation_history = ""
    current_problem = ""

    while True:
        if not current_problem:
            print("\nEnter the math problem:")
            current_problem = input("> ").strip()
            if current_problem.lower() == "quit":
                break
            conversation_history = ""

        print("\nEnter student's working (or 'new' for new problem, 'quit' to exit):")
        student_input = input("> ").strip()

        if student_input.lower() == "quit":
            break
        if student_input.lower() == "new":
            current_problem = ""
            continue

        result = run_inference(
            model=model,
            tokenizer=tokenizer,
            problem=current_problem,
            student_working=student_input,
            conversation_history=conversation_history,
            rag_system=rag_system,
            use_rag=use_rag,
            rag_min_score=rag_min_score,
            rag_use_graph=rag_use_graph,
            rag_max_graph_neighbors=rag_max_graph_neighbors,
        )

        # Display response
        print(f"\n{'─' * 40}")
        if result["retrieved_chunks"]:
            print(f"[RAG] Retrieved: {[c['chunk_id'] for c in result['retrieved_chunks']]}")
        if result["thought"]:
            print(f"\n[Internal Thought]\n{result['thought']}")
        print(f"\n[Tutor Response]\n{result['response']}")
        print(f"{'─' * 40}")

        # Update conversation history
        conversation_history += (
            f"STUDENT: {student_input}\n"
            f"TUTOR: {result['response']}\n\n"
        )


# ─────────────────────────────────────────────────────────────────
# BATCH EVALUATION MODE
# ─────────────────────────────────────────────────────────────────

def parse_eval_input(input_text: str) -> dict:
    """Parse the 'input' field from eval_prompts_full.json.
    
    Format: PROBLEM: ...\n\nSTUDENT: ... (may also contain CONVERSATION HISTORY)
    """
    problem = ""
    student_working = ""
    conversation_history = ""

    # Extract PROBLEM
    prob_match = re.search(r"PROBLEM:\s*(.*?)(?=\n\s*(?:STUDENT|\[CONVERSATION))", input_text, re.DOTALL)
    if prob_match:
        problem = prob_match.group(1).strip()
    else:
        # Fallback: everything before STUDENT is the problem
        parts = re.split(r"\nSTUDENT:", input_text, maxsplit=1)
        problem = parts[0].replace("PROBLEM:", "").strip()

    # Extract CONVERSATION HISTORY if present
    hist_match = re.search(r"\[CONVERSATION HISTORY\]\s*(.*?)(?=\nSTUDENT'S CURRENT|\nSTUDENT:(?!.*TUTOR:))", input_text, re.DOTALL)
    if hist_match:
        conversation_history = hist_match.group(1).strip()

    # Extract STUDENT working (last STUDENT: line)
    student_matches = list(re.finditer(r"STUDENT:\s*(.*?)(?=\n(?:TUTOR:|$)|$)", input_text, re.DOTALL))
    if student_matches:
        student_working = student_matches[-1].group(1).strip()

    return {
        "problem": problem,
        "student_working": student_working,
        "conversation_history": conversation_history,
    }


def batch_inference(
    model,
    tokenizer,
    eval_file: str,
    output_file: str,
    rag_system: RAGSystem = None,
    use_rag: bool = True,
    rag_min_score: float = 0.5,
    rag_use_graph: bool = True,
    rag_max_graph_neighbors: int = 2,
):
    """Run inference on a JSON file of evaluation prompts.

    Saves results incrementally after each item and supports resuming:
    if output_file already exists, items whose eval_id is already present
    are skipped so a timed-out job can be restarted without re-running work.
    """
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    # Resume: load already-completed results if the file exists
    results = []
    done_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_ids = {r["eval_id"] for r in results}
            print(f"[RESUME] Found {len(done_ids)} already-completed items in {output_file}")
        except Exception as e:
            print(f"[RESUME] Could not load existing results ({e}), starting fresh")
            results = []
            done_ids = set()

    total = len(eval_data)
    remaining = [item for item in eval_data if item.get("id", None) not in done_ids]
    print(f"[INFO] Total: {total} | Done: {len(done_ids)} | Remaining: {len(remaining)}")

    for item in remaining:
        item_id = item.get("id", len(results) + 1)
        current = len(results) + 1
        print(f"[{current}/{total}] Processing id={item_id}...")

        # Handle eval_prompts_full.json format (instruction + input)
        if "instruction" in item and "input" in item:
            parsed = parse_eval_input(item["input"])
            problem = parsed["problem"]
            student_working = parsed["student_working"]
            conv_history = parsed["conversation_history"]
            # Use the eval's own instruction as system prompt
            system_prompt = item["instruction"]
        else:
            # Fallback for test_cases.json format
            problem = item.get("problem", "")
            student_working = item.get("student_working", "")
            conv_history = item.get("conversation_history", "")
            system_prompt = None

        result = run_inference(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            student_working=student_working,
            conversation_history=conv_history,
            rag_system=rag_system,
            use_rag=use_rag,
            rag_min_score=rag_min_score,
            rag_use_graph=rag_use_graph,
            rag_max_graph_neighbors=rag_max_graph_neighbors,
            system_prompt_override=system_prompt,
        )

        result["eval_id"] = item_id
        results.append(result)

        # Save after every item so progress survives a timeout
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Saved {len(results)} results to {output_file}")
    return results


# ─────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CurioChain Math Tutor Inference")
    parser.add_argument(
        "--model", choices=["finetuned", "baseline"], default="finetuned",
        help="Which model to use (default: finetuned)"
    )
    parser.add_argument(
        "--rag", action="store_true", default=False,
        help="Enable RAG retrieval"
    )
    parser.add_argument(
        "--mode", choices=["interactive", "batch"], default="interactive",
        help="interactive chat or batch evaluation (default: interactive)"
    )
    parser.add_argument(
        "--eval-file", type=str, default=None,
        help="Path to evaluation JSON file (for batch mode)"
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Path to save batch results (for batch mode)"
    )
    parser.add_argument(
        "--chunks-path", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "RAG-dataset", "concepts.json"),
        help="Path to RAG concepts.json"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
           "--rag-min-score", type=float, default=0.5,
           help="Min cross-encoder score to inject RAG context (default: 0.5). "
               "Chunks below this are dropped individually, and retrieval is skipped "
               "entirely if the best chunk is below threshold."
    )
    parser.add_argument(
        "--no-graph", action="store_true", default=False,
        help="Disable graph expansion (use vector+rerank only)"
    )
    parser.add_argument(
        "--graph-neighbors", type=int, default=2,
        help="Max chunks added by graph expansion (default: 2)"
    )

    args = parser.parse_args()

    # Load model
    if args.model == "finetuned":
        model, tokenizer = load_finetuned_model()
    else:
        model, tokenizer = load_baseline_model()

    # Load RAG if requested
    rag_system = None
    if args.rag:
        chroma_path = os.environ.get(
            "CHROMA_DB_PATH",
            os.path.join(os.path.dirname(__file__), "chroma_db"),
        )
        rag_system = RAGSystem(chunks_path=args.chunks_path, persist_dir=chroma_path)

    # Run
    use_graph = not args.no_graph
    if args.mode == "interactive":
        interactive_mode(
            model, tokenizer, rag_system,
            use_rag=args.rag,
            rag_min_score=args.rag_min_score,
            rag_use_graph=use_graph,
            rag_max_graph_neighbors=args.graph_neighbors,
        )
    else:
        if not args.eval_file:
            print("ERROR: --eval-file required for batch mode")
            sys.exit(1)
        output = args.output_file or f"results_{args.model}_{'rag' if args.rag else 'norag'}.json"
        batch_inference(
            model, tokenizer, args.eval_file, output, rag_system,
            use_rag=args.rag,
            rag_min_score=args.rag_min_score,
            rag_use_graph=use_graph,
            rag_max_graph_neighbors=args.graph_neighbors,
        )


if __name__ == "__main__":
    main()
