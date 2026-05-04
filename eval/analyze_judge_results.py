"""
LLM-as-a-Judge Analysis
- Per-variant averages for D1-D6
- Baseline vs Finetuned comparison
- RAG effect
- Trap prompt effect
- Inter-judge agreement (Llama vs Gemma)
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"
LLAMA_FILE  = RESULTS_DIR / "judge_scores_new_llama32_3b.json"
GEMMA_FILE  = RESULTS_DIR / "judge_scores_new_gemma2_2b.json"

DIMS = ["D1_answer_leakage", "D2_socratic_quality", "D3_thought_correctness",
        "D4_hint_calibration", "D5_format_compliance", "D6_engagement_request"]

VARIANT_LABELS = {
    "baseline_eval_dataset_150":        "Baseline",
    "finetuned_eval_dataset_150":       "Finetuned",
    "baseline_rag_eval_dataset_150":    "Baseline+RAG",
    "finetuned_rag_eval_dataset_150":   "Finetuned+RAG",
    "baseline_eval_trap_prompt":        "Baseline (Trap)",
    "finetuned_eval_trap_prompt":       "Finetuned (Trap)",
    "baseline_rag_eval_trap_prompt":    "Baseline+RAG (Trap)",
    "finetuned_rag_eval_trap_prompt":   "Finetuned+RAG (Trap)",
}


def load_scores(filepath):
    with open(filepath) as f:
        return json.load(f)


def get_numeric(item, dim):
    v = item.get(dim)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def variant_stats(data, judge_name):
    print(f"\n{'='*70}")
    print(f"  JUDGE: {judge_name}")
    print(f"{'='*70}")

    by_variant = defaultdict(list)
    for item in data:
        by_variant[item["variant"]].append(item)

    header = f"{'Variant':<28} {'N':>4} {'D1':>6} {'D2':>6} {'D3':>6} {'D4':>6} {'D5':>6} {'D6':>6}"
    print(f"\n{header}")
    print("-" * 70)

    results = {}
    for variant in sorted(by_variant.keys()):
        items = by_variant[variant]
        label = VARIANT_LABELS.get(variant, variant)
        row = {"n": len(items)}
        scores_str = f"{label:<28} {len(items):>4}"
        for dim in DIMS:
            vals = [get_numeric(i, dim) for i in items if get_numeric(i, dim) is not None]
            if vals:
                mean = np.mean(vals)
                row[dim] = mean
                scores_str += f" {mean:>6.2f}"
            else:
                row[dim] = None
                scores_str += f" {'N/A':>6}"
        print(scores_str)
        results[variant] = row

    return results


def compare_baseline_finetuned(llama_results, gemma_results):
    print(f"\n{'='*70}")
    print("  BASELINE vs FINETUNED COMPARISON")
    print(f"{'='*70}")

    pairs = [
        ("baseline_eval_dataset_150",      "finetuned_eval_dataset_150",      "Base"),
        ("baseline_rag_eval_dataset_150",  "finetuned_rag_eval_dataset_150",  "RAG"),
        ("baseline_eval_trap_prompt",      "finetuned_eval_trap_prompt",      "Trap"),
        ("baseline_rag_eval_trap_prompt",  "finetuned_rag_eval_trap_prompt",  "Trap+RAG"),
    ]

    print(f"\n{'Condition':<12} {'Dim':<8} {'Base(L)':>8} {'Fine(L)':>8} {'Δ(L)':>7} {'Base(G)':>8} {'Fine(G)':>8} {'Δ(G)':>7}")
    print("-" * 70)

    for base_var, fine_var, label in pairs:
        for dim in ["D1_answer_leakage", "D2_socratic_quality", "D4_hint_calibration"]:
            b_l = llama_results.get(base_var, {}).get(dim)
            f_l = llama_results.get(fine_var, {}).get(dim)
            b_g = gemma_results.get(base_var, {}).get(dim)
            f_g = gemma_results.get(fine_var, {}).get(dim)

            def fmt(v): return f"{v:.2f}" if v is not None else " N/A"
            def delta(a, b):
                if a is not None and b is not None:
                    d = b - a
                    return f"{d:+.2f}"
                return "  N/A"

            print(f"{label:<12} {dim:<8} {fmt(b_l):>8} {fmt(f_l):>8} {delta(b_l,f_l):>7} {fmt(b_g):>8} {fmt(f_g):>8} {delta(b_g,f_g):>7}")
        print()


def inter_judge_agreement(llama_data, gemma_data):
    print(f"\n{'='*70}")
    print("  INTER-JUDGE AGREEMENT (Llama vs Gemma)")
    print(f"{'='*70}")

    # Build lookup for gemma by (variant, eval_id)
    gemma_lookup = {(d["variant"], d["eval_id"]): d for d in gemma_data}

    paired = defaultdict(lambda: {"llama": [], "gemma": []})

    for item in llama_data:
        key = (item["variant"], item["eval_id"])
        gemma_item = gemma_lookup.get(key)
        if not gemma_item:
            continue
        for dim in DIMS:
            l_val = get_numeric(item, dim)
            g_val = get_numeric(gemma_item, dim)
            if l_val is not None and g_val is not None:
                paired[dim]["llama"].append(l_val)
                paired[dim]["gemma"].append(g_val)

    print(f"\n{'Dimension':<30} {'N':>5} {'Pearson r':>10} {'p-value':>10} {'Mean Δ':>8}")
    print("-" * 65)

    for dim in DIMS:
        l_vals = paired[dim]["llama"]
        g_vals = paired[dim]["gemma"]
        if len(l_vals) < 10:
            continue
        r, p = stats.pearsonr(l_vals, g_vals)
        mean_delta = np.mean(np.abs(np.array(l_vals) - np.array(g_vals)))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{dim:<30} {len(l_vals):>5} {r:>10.3f} {p:>10.4f}{sig:<3} {mean_delta:>8.3f}")

    # Binary dims: Cohen's kappa
    print(f"\nBinary dimensions (Cohen's kappa):")
    from sklearn.metrics import cohen_kappa_score
    for dim in ["D1_answer_leakage", "D5_format_compliance", "D6_engagement_request"]:
        l_vals = paired[dim]["llama"]
        g_vals = paired[dim]["gemma"]
        if len(l_vals) < 10:
            continue
        kappa = cohen_kappa_score(
            [int(v) for v in l_vals],
            [int(v) for v in g_vals]
        )
        print(f"  {dim:<30} kappa = {kappa:.3f}")


def state_type_breakdown(llama_data):
    print(f"\n{'='*70}")
    print("  SCORES BY STUDENT STATE TYPE (Llama, Finetuned variants)")
    print(f"{'='*70}")

    finetuned_variants = {
        "finetuned_eval_dataset_150", "finetuned_rag_eval_dataset_150",
        "finetuned_eval_trap_prompt", "finetuned_rag_eval_trap_prompt"
    }
    by_state = defaultdict(list)
    for item in llama_data:
        if item["variant"] in finetuned_variants:
            by_state[item["student_state_type"]].append(item)

    print(f"\n{'State Type':<25} {'N':>4} {'D1':>6} {'D2':>6} {'D3':>6} {'D4':>6}")
    print("-" * 50)
    for state in sorted(by_state.keys()):
        items = by_state[state]
        scores = []
        for dim in ["D1_answer_leakage", "D2_socratic_quality",
                    "D3_thought_correctness", "D4_hint_calibration"]:
            vals = [get_numeric(i, dim) for i in items if get_numeric(i, dim) is not None]
            scores.append(f"{np.mean(vals):.2f}" if vals else " N/A")
        print(f"{state:<25} {len(items):>4} {scores[0]:>6} {scores[1]:>6} {scores[2]:>6} {scores[3]:>6}")


def main():
    llama_data = load_scores(LLAMA_FILE)
    gemma_data = load_scores(GEMMA_FILE)
    print(f"Loaded: Llama={len(llama_data)} items, Gemma={len(gemma_data)} items")

    llama_results = variant_stats(llama_data, "Llama 3.2-3B-Instruct")
    gemma_results = variant_stats(gemma_data, "Gemma 2-2B-IT")
    compare_baseline_finetuned(llama_results, gemma_results)
    inter_judge_agreement(llama_data, gemma_data)
    state_type_breakdown(llama_data)


if __name__ == "__main__":
    main()
