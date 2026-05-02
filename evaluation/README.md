# AOLM Evaluation Suite

Runs the fine-tuned model `joharatharv/AOLM-qwen-7b` (LoRA adapter on
Qwen2.5-Math-7B-Instruct) against the two evaluation files in
`eval_data/`, in **4 modes**:

| # | Mode | Base | Fine-tuned LoRA | Graph RAG |
|---|---|:---:|:---:|:---:|
| 1 | `baseline`        | ✓ | ✗ | ✗ |
| 2 | `baseline_rag`    | ✓ | ✗ | ✓ |
| 3 | `finetuned`       | ✓ | ✓ | ✗ |
| 4 | `finetuned_rag`   | ✓ | ✓ | ✓ |

Each mode runs on **both** eval files → **8 result JSON files** total.

---

## Files in this folder

```
evaluation/
  inference.py        Model loading + batch inference (resume-safe)
  rag_system.py       Vector + cross-encoder + graph expansion RAG
  run_all.py          Orchestrator — loads each model only ONCE
  run.sh              Bash launcher
  run.sbatch          SLURM sbatch script
  README.md           This file
```

## Disk layout (cluster)

```
/scratch/atharv.johar/AOLM-load/        ← caches (not on home quota)
    hf_cache/                              HF model snapshots (Qwen ~14 GB)
    chroma_db/                             ChromaDB vector store (~1 MB)

/home2/atharv.johar/AOLM-paper-eval-results/   ← persistent results
    results_baseline_eval_dataset_150.json
    results_baseline_eval_trap_prompt.json
    results_baseline_rag_eval_dataset_150.json
    results_baseline_rag_eval_trap_prompt.json
    results_finetuned_eval_dataset_150.json
    results_finetuned_eval_trap_prompt.json
    results_finetuned_rag_eval_dataset_150.json
    results_finetuned_rag_eval_trap_prompt.json
    slurm-<JOBID>.out / .err               (when run via sbatch)
```

---

## Quick start

### 1. Upload project to cluster

```bash
# from your local machine
rsync -avz --exclude '.git' --exclude '__pycache__' \
    AOLM-paper/  atharv.johar@<cluster>:~/AOLM-paper/
```

### 2. (One-time) install dependencies + log in to HF

```bash
ssh atharv.johar@<cluster>
cd ~/AOLM-paper

pip install --user transformers peft accelerate bitsandbytes \
                  sentence-transformers chromadb

# If your fine-tuned repo is private, set HF token first:
export HF_TOKEN="hf_xxx..."          # from huggingface.co/settings/tokens
huggingface-cli login --token "$HF_TOKEN"
```

### 3. (One-time) build the ChromaDB index

```bash
bash evaluation/run.sh build-index
# Builds /scratch/atharv.johar/AOLM-load/chroma_db/  (≈ 401 chunks indexed)
```

### 4. Submit the full evaluation as a SLURM job

```bash
sbatch evaluation/run.sbatch
# → 8 result files in /home2/atharv.johar/AOLM-paper-eval-results/
```

Or run a single mode:

```bash
sbatch evaluation/run.sbatch baseline
sbatch evaluation/run.sbatch finetuned_rag
```

Or run interactively (e.g. on `salloc`):

```bash
bash evaluation/run.sh all                      # full sweep
bash evaluation/run.sh finetuned --skip-existing
bash evaluation/run.sh baseline_rag
```

---

## Resume / re-run

- **Within a single result file** — `batch_inference` saves after every item
  and skips IDs that already appear in the output file. So a SLURM timeout
  followed by re-submission picks up exactly where it stopped.
- **Across modes** — pass `--skip-existing` to skip any `(mode, eval_file)`
  combination whose result file already exists and is non-empty.

```bash
sbatch evaluation/run.sbatch all --skip-existing
```

---

## Estimated runtime (RTX 2080, 4-bit Qwen-7B)

| step | items | est. time |
|---|---|---|
| Load baseline model | — | ~2 min |
| Inference: 1 item   | 1  | ~3–5 sec |
| eval_dataset_150    | 150 | ~10–13 min |
| eval_trap_prompt    | 50  | ~3–5 min |
| Per-mode total      | 200 | ~15 min |
| Full sweep (4 modes)| 800 | ~70–90 min + 2× model load |

Total wall-clock: ~**1.5 to 2 hours** for the full sweep.

---

## Output JSON format

Each result file is a list of dicts:

```json
{
  "eval_id": "61_cold_start",
  "base_problem_id": "61",
  "student_state_type": "cold_start",
  "problem":          "Two finite sets have m and n elements...",
  "student_working":  "I think the number of subsets is 2^n, but I'm stuck.",
  "retrieved_chunks": [
    {"chunk_id": "SETS_FORMULA_001", "score": 4.21},
    ...
  ],
  "rag_context_used": true,
  "thought":  "...",            extracted from <thought>...</thought>
  "response": "...",            extracted from <response>...</response>
  "raw_output": "..."           full raw model output
}
```

---

## Common knobs

| Flag | Default | Meaning |
|---|---|---|
| `--temperature`     | 0.7 | sampling temperature |
| `--max-new-tokens`  | 512 | generation length |
| `--rag-min-score`   | 0.5 | drop chunks below this cross-encoder score |
| `--no-graph`        | off | disable graph expansion (vector + rerank only) |
| `--graph-neighbors` | 2   | max chunks added by graph expansion |
| `--skip-existing`   | off | skip already-completed result files |

Pass these after the mode:

```bash
bash evaluation/run.sh all --temperature 0.0 --skip-existing
```

---

## Troubleshooting

**OOM at model load**  
Make sure `bitsandbytes` is installed and CUDA is detected:
```bash
python -c "import torch, bitsandbytes; print(torch.cuda.get_device_name(0))"
```
The configs use `load_in_4bit=True` so the 7B should fit in 8 GB.

**ChromaDB folder corrupted**  
Delete and rebuild:
```bash
rm -rf /scratch/atharv.johar/AOLM-load/chroma_db
bash evaluation/run.sh build-index
```

**HF download fails / quota error**  
Ensure `HF_HOME` points to /scratch (the launcher does this automatically):
```bash
echo $HF_HOME    # should print /scratch/atharv.johar/AOLM-load/hf_cache
```

**Adapter not found**  
The launcher pulls `joharatharv/AOLM-qwen-7b` from the Hub. If the repo is
private, run `huggingface-cli login` first.
