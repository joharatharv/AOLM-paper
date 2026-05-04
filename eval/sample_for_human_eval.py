"""
Sample 30 items for human evaluation using stratified random sampling.

Strategy:
- Stratify by student_state_type (5 types × 6 items = 30 total)
- Within each state type, sample randomly from items scored by Llama
  in BOTH baseline and finetuned variants
- Balanced and unbiased — suitable for ACL submission

Output: sampled_items.json — input for generate_assignments.py
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict

EVAL_DIR    = Path(__file__).parent / "eval-results-new"
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_FILE = Path(__file__).parent / "human_eval" / "data" / "sampled_items.json"

LLAMA_FILE  = RESULTS_DIR / "judge_scores_new_llama32_3b.json"
GEMMA_FILE  = RESULTS_DIR / "judge_scores_new_gemma2_2b.json"

SAMPLES_PER_STATE_TYPE = 6  # 5 state types × 6 = 30 total
RANDOM_SEED = 42

VARIANT_FILE_MAP = {
    "baseline_eval_dataset_150":         "results_baseline_eval_dataset_150.json",
    "baseline_rag_eval_dataset_150":     "results_baseline_rag_eval_dataset_150.json",
    "finetuned_eval_dataset_150":        "results_finetuned_eval_dataset_150.json",
    "finetuned_rag_eval_dataset_150":    "results_finetuned_rag_eval_dataset_150.json",
    "baseline_eval_trap_prompt":         "results_baseline_eval_trap_prompt.json",
    "baseline_rag_eval_trap_prompt":     "results_baseline_rag_eval_trap_prompt.json",
    "finetuned_eval_trap_prompt":        "results_finetuned_eval_trap_prompt.json",
    "finetuned_rag_eval_trap_prompt":    "results_finetuned_rag_eval_trap_prompt.json",
}

VARIANT_PAIRS = [
    ("baseline_eval_dataset_150",      "finetuned_eval_dataset_150"),
    ("baseline_rag_eval_dataset_150",  "finetuned_rag_eval_dataset_150"),
    ("baseline_eval_trap_prompt",      "finetuned_eval_trap_prompt"),
    ("baseline_rag_eval_trap_prompt",  "finetuned_rag_eval_trap_prompt"),
]


def load_judge_scores(filepath):
    with open(filepath) as f:
        data = json.load(f)
    scores = {}
    for item in data:
        key = (item["variant"], item["eval_id"])
        scores[key] = item
    return scores


def load_eval_results(variant):
    fname = VARIANT_FILE_MAP.get(variant)
    if not fname:
        return {}
    fpath = EVAL_DIR / fname
    if not fpath.exists():
        return {}
    with open(fpath) as f:
        data = json.load(f)
    return {item["eval_id"]: item for item in data}


def extract_response(output):
    m = re.search(r"<response>(.*?)</response>", output, re.DOTALL)
    return m.group(1).strip() if m else output.strip()


def main():
    random.seed(RANDOM_SEED)

    llama = load_judge_scores(LLAMA_FILE)
    gemma = load_judge_scores(GEMMA_FILE)

    print(f"Llama scores loaded: {len(llama)}")
    print(f"Gemma scores loaded: {len(gemma)}")

    # Build pool of all candidate items grouped by state_type
    # Each candidate: item scored by Llama in both baseline and finetuned
    pool_by_state = defaultdict(list)

    for baseline_var, finetuned_var in VARIANT_PAIRS:
        baseline_results = load_eval_results(baseline_var)
        finetuned_results = load_eval_results(finetuned_var)

        for eval_id, b_item in baseline_results.items():
            f_item = finetuned_results.get(eval_id)
            if not f_item:
                continue

            b_llama = llama.get((baseline_var, eval_id))
            f_llama = llama.get((finetuned_var, eval_id))

            if not b_llama or not f_llama:
                continue

            state_type = b_item.get("student_state_type", "unknown")

            pool_by_state[state_type].append({
                "item_id":                f"{baseline_var}__{eval_id}",
                "eval_id":                eval_id,
                "variant_pair":           f"{baseline_var} vs {finetuned_var}",
                "student_state_type":     state_type,
                "problem":                b_item.get("problem", ""),
                "student_working":        b_item.get("student_working", ""),
                "baseline_response":      extract_response(b_item.get("output", b_item.get("raw_output", ""))),
                "finetuned_response":     extract_response(f_item.get("output", f_item.get("raw_output", ""))),
                "baseline_variant":       baseline_var,
                "finetuned_variant":      finetuned_var,
                "llama_baseline_scores":  b_llama,
                "llama_finetuned_scores": f_llama,
            })

    print("\nPool size by state type:")
    for st, items in sorted(pool_by_state.items()):
        print(f"  {st}: {len(items)} candidates")

    # Stratified random sample: 6 per state type, capped at 30 total
    # Only use the 5 standard state types for base variants
    STANDARD_STATE_TYPES = [
        "cold_start", "partial_correct", "wrong_confident",
        "almost_done", "arithmetic_error"
    ]
    # For trap variants, group all trap state types into one stratum
    trap_pool = []
    for state_type, candidates in pool_by_state.items():
        if state_type not in STANDARD_STATE_TYPES:
            trap_pool.extend(candidates)

    # Build final pool: 5 standard + 1 trap stratum = 6 strata × 5 items = 30
    strata = {st: pool_by_state.get(st, []) for st in STANDARD_STATE_TYPES}
    strata["trap"] = trap_pool

    ITEMS_PER_STRATUM = 5  # 6 strata × 5 = 30

    sampled = []
    for state_type, candidates in strata.items():
        n = min(ITEMS_PER_STRATUM, len(candidates))
        selected = random.sample(candidates, n)
        sampled.extend(selected)
        print(f"  [{state_type}] sampled {n}/{len(candidates)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    from collections import Counter
    print(f"\nTotal sampled: {len(sampled)} items")
    print(f"Output: {OUTPUT_FILE}")
    print("\nBreakdown by state type:")
    for st, count in Counter(s["student_state_type"] for s in sampled).items():
        print(f"  {st}: {count}")
    print("\nBreakdown by variant pair:")
    for pair, count in Counter(s["variant_pair"] for s in sampled).items():
        print(f"  {pair}: {count}")


if __name__ == "__main__":
    main()
