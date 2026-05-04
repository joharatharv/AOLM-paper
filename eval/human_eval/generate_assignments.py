"""
Generate balanced student assignments for human evaluation.

Each student rates 6 responses. Each response gets rated 3 times.
No student rates both variants (A/B) of the same item.

Usage:
    python generate_assignments.py \
        --sampled-items sampled_items.json \
        --num-students 30 \
        --student-prefix S \
        --output-assignments data/assignments.json \
        --output-responses data/responses.json

sampled_items.json format:
    [
      {
        "item_id": "item_001",
        "student_state_type": "cold_start",
        "problem": "...",
        "student_working": "...",
        "baseline_response": "...",
        "finetuned_response": "..."
      },
      ...
    ]
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def build_responses(items: list, seed: int = 42) -> list:
    """
    Convert items to a flat list of responses.
    Randomly assign which variant gets label A vs B per item (blinding).
    """
    rng = random.Random(seed)
    responses = []
    for item in items:
        variants = [
            ("baseline",  item["baseline_response"]),
            ("finetuned", item["finetuned_response"]),
        ]
        rng.shuffle(variants)  # random A/B assignment
        for label, (true_variant, model_response) in zip(["A", "B"], variants):
            responses.append({
                "response_id": f"{item['item_id']}_{label}",
                "item_id":     item["item_id"],
                "true_variant": true_variant,          # kept for analysis, not shown to students
                "student_state_type": item["student_state_type"],
                "problem":       item["problem"],
                "student_working": item["student_working"],
                "model_response":  model_response,
            })
    return responses


def generate_assignments(responses: list, num_students: int,
                         responses_per_student: int = 6,
                         raters_per_response: int = 3,
                         seed: int = 42) -> dict:
    """
    Greedy balanced assignment:
    - Each student gets exactly responses_per_student responses
    - Each response gets exactly raters_per_response raters
    - No student sees both A and B of the same item
    """
    rng = random.Random(seed)

    total_needed = len(responses) * raters_per_response
    assert total_needed == num_students * responses_per_student, (
        f"Mismatch: {len(responses)} responses × {raters_per_response} raters "
        f"= {total_needed} ≠ {num_students} students × {responses_per_student} = "
        f"{num_students * responses_per_student}"
    )

    # Expand response pool and shuffle
    pool = responses * raters_per_response
    rng.shuffle(pool)

    student_ids = [f"S{i:02d}" for i in range(1, num_students + 1)]
    assignments = {s: [] for s in student_ids}
    student_items = defaultdict(set)    # s → set of item_ids already assigned
    response_count = defaultdict(int)   # response_id → times assigned

    # Greedy: for each slot in each student, find the best available response
    for student in student_ids:
        attempts = 0
        while len(assignments[student]) < responses_per_student:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError(
                    f"Could not complete assignment for {student} — "
                    "try a different seed or check your item counts."
                )
            candidate = rng.choice(responses)
            rid = candidate["response_id"]
            iid = candidate["item_id"]
            if (iid not in student_items[student]
                    and response_count[rid] < raters_per_response):
                assignments[student].append(rid)
                student_items[student].add(iid)
                response_count[rid] += 1

    return assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampled-items",        required=True)
    parser.add_argument("--num-students",         type=int, default=30)
    parser.add_argument("--responses-per-student",type=int, default=6)
    parser.add_argument("--raters-per-response",  type=int, default=3)
    parser.add_argument("--seed",                 type=int, default=42)
    parser.add_argument("--output-assignments",   default="data/assignments.json")
    parser.add_argument("--output-responses",     default="data/responses.json")
    args = parser.parse_args()

    with open(args.sampled_items) as f:
        items = json.load(f)

    print(f"Loaded {len(items)} items.")

    responses = build_responses(items, seed=args.seed)
    print(f"Built {len(responses)} responses ({len(items)} items × 2 variants).")

    assignments = generate_assignments(
        responses,
        num_students=args.num_students,
        responses_per_student=args.responses_per_student,
        raters_per_response=args.raters_per_response,
        seed=args.seed,
    )

    # Verify
    from collections import Counter
    all_assigned = [rid for rids in assignments.values() for rid in rids]
    counts = Counter(all_assigned)
    assert all(c == args.raters_per_response for c in counts.values()), \
        "Some responses don't have the right number of raters!"
    print(f"Verified: each response rated exactly {args.raters_per_response} times.")

    out_assignments = Path(args.output_assignments)
    out_responses   = Path(args.output_responses)
    out_assignments.parent.mkdir(parents=True, exist_ok=True)
    out_responses.parent.mkdir(parents=True, exist_ok=True)

    with open(out_assignments, "w") as f:
        json.dump(assignments, f, indent=2)
    print(f"Assignments written to {out_assignments}")

    # Save responses without true_variant field (blind version for the app)
    blind_responses = [{k: v for k, v in r.items() if k != "true_variant"}
                       for r in responses]
    with open(out_responses, "w") as f:
        json.dump(blind_responses, f, indent=2, ensure_ascii=False)
    print(f"Responses (blinded) written to {out_responses}")

    # Also save a key file with true_variant for analysis
    key_file = out_responses.parent / "responses_key.json"
    with open(key_file, "w") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    print(f"Response key (with true_variant) written to {key_file}")

    print(f"\nStudent IDs: {list(assignments.keys())[:5]} ... (share these with participants)")


if __name__ == "__main__":
    main()
