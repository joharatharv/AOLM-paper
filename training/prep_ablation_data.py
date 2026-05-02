"""
prep_ablation_data.py
Prepares the data variants needed for ablation experiments.

Run this ONCE before launching ablation training runs:
  python training/prep_ablation_data.py

Creates:
  training/ablation_data/train_500.jsonl     -- 500-entry scale subset
  training/ablation_data/train_1000.jsonl    -- 1000-entry scale subset
  training/ablation_data/train_1500.jsonl    -- 1500-entry scale subset
  training/ablation_data/train_no_thought.jsonl  -- thought blocks stripped
  (train_2313 is just final_dataset/train.jsonl itself)
"""

import json, re, random, os

SEED = 42
random.seed(SEED)

TRAIN_PATH = "final_dataset/train.jsonl"
OUT_DIR    = "training/ablation_data"
os.makedirs(OUT_DIR, exist_ok=True)

train = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8") if l.strip()]
print(f"Loaded {len(train)} training entries")


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):5d} -> {path}")


# ── Data-scale subsets ─────────────────────────────────────────────────────────
# Stratified by phase so every subset has roughly the same phase mix
from collections import defaultdict

def detect_phase(entry):
    meta = entry.get("meta", {}).get("phase", "other")
    return meta if meta else "other"

by_phase = defaultdict(list)
for e in train:
    by_phase[detect_phase(e)].append(e)

for n_target in [500, 1000, 1500]:
    subset = []
    for phase, group in by_phase.items():
        shuffled = list(group)
        random.shuffle(shuffled)
        # take proportional share from each phase
        n_from_phase = max(1, round(len(group) * n_target / len(train)))
        subset.extend(shuffled[:n_from_phase])
    # trim/pad to exact target
    random.shuffle(subset)
    subset = subset[:n_target]
    write_jsonl(f"{OUT_DIR}/train_{n_target}.jsonl", subset)


# ── No-thought-block variant ───────────────────────────────────────────────────
# Strip <thought>...</thought> from every output; keep only <response>...</response>
no_thought = []
for e in train:
    e2 = dict(e)
    out = e2["output"]
    # Extract just the response block content
    m = re.search(r"<response>(.*?)</response>", out, re.DOTALL)
    if m:
        e2["output"] = m.group(1).strip()
    else:
        # fallback: remove thought block
        e2["output"] = re.sub(r"<thought>.*?</thought>\s*", "", out, flags=re.DOTALL).strip()
    no_thought.append(e2)

write_jsonl(f"{OUT_DIR}/train_no_thought.jsonl", no_thought)

print("\nAll ablation datasets prepared.")
