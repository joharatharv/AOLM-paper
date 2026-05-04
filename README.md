# CurioChain: A Socratic Math Tutoring System with Think-Then-Speak Architecture

This repository contains the full codebase, datasets, evaluation framework, and results for **CurioChain**, a fine-tuned math tutoring model trained to guide students through Socratic discovery rather than giving direct answers.

---

## Compute

All training and evaluation was conducted on the **Ada HPC cluster** at IIIT Hyderabad.

| Component | Spec |
|---|---|
| Cluster | Ada (IIIT Hyderabad) |
| Partition | `u22` |
| GPU | NVIDIA (10.57 GB VRAM per node) |
| Scheduler | SLURM |
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Training framework | HuggingFace `transformers` + LoRA (via `axolotl`) |
| Python | 3.10 |

Typical job allocation:
```bash
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH -A research
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=6
```

---

## Repository Structure

```
.
├── DATASET/                    # Training data
│   ├── final_ft_dataset/       # Final fine-tuning dataset (train.jsonl, val.jsonl)
│   ├── sft_with_real_hints.jsonl  # SFT dataset with Socratic hints
│   ├── sft_data_2.jsonl        # Alternative SFT dataset
│   ├── trap_data.json          # Adversarial trap prompt data
│   ├── phase1-4.json           # Phase-wise dataset construction
│   ├── gemini_final_dataset_v2.py  # Dataset generation script (Gemini API)
│   ├── _gen_phase4.py          # Phase 4 generation script
│   ├── RAG-dataset/
│   │   └── concepts.json       # Knowledge base for RAG retrieval
│   └── dataset-processing/
│       ├── validate_dataset.py
│       └── view_data.py
│
├── training/                   # Model fine-tuning
│   ├── run_training.sh         # Main SLURM training job
│   ├── qwen25_3b_lora.yaml     # Qwen 2.5 3B LoRA config (final model)
│   ├── qwen25_7b_lora_4gpu.yaml
│   ├── qwen25_math_7b_lora.yaml
│   ├── llama31_8b_lora.yaml
│   ├── deepspeed_zero3.json    # DeepSpeed config
│   ├── requirements.txt
│   └── ablations/              # Ablation study configs
│       ├── data_scale_500.yaml
│       ├── no_gap_fix.yaml
│       └── no_thought_block.yaml
│
├── rag/                        # Retrieval-Augmented Generation
│   ├── rag_system.py           # Core RAG system
│   ├── inference.py            # RAG inference pipeline
│   ├── run.sh                  # RAG job script
│   └── test_cases.json
│
└── eval/                       # Evaluation framework
    ├── build_eval_dataset.py   # Build eval dataset from raw problems
    ├── judge.py                # LLM-as-a-Judge (Gemini API)
    ├── run_eval.sh             # Run model inference on eval set
    ├── run_all.py              # Run full evaluation pipeline
    ├── sample_for_human_eval.py  # Stratified sampling for human eval
    ├── analyze_judge_results.py  # Analysis + ablations
    ├── eval_dataset_150.json   # 150-item evaluation set (standard)
    ├── eval_trap_prompt.json   # 50-item trap prompt eval set
    ├── eval-results-new/       # Model outputs (8 variants)
    ├── results/                # Judge scores + analysis reports
    └── human_eval/             # Human evaluation app + results
        ├── app.py              # Streamlit evaluation app
        ├── generate_assignments.py
        ├── data/               # Sampled items + blinded assignments
        └── results/            # Human ratings CSV
```

---

## Installation

### Training Environment (Ada)
```bash
conda create -n curiochain python=3.10 -y
conda activate curiochain
pip install -r training/requirements.txt
```

### Evaluation Environment (Ada)
```bash
conda create -n judge_env python=3.10 -y
conda activate judge_env
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes sentencepiece huggingface_hub
pip install google-genai  # for Gemini judge
```

### Human Eval App (local)
```bash
cd eval/human_eval
pip install streamlit supabase
streamlit run app.py
```

---

## Step-by-Step Replication

### Step 1: Dataset Construction

The training dataset was built in 4 phases using Gemini to generate Socratic hint-response pairs:

```bash
python DATASET/gemini_final_dataset_v2.py   # Generate main dataset
python DATASET/_gen_phase4.py               # Generate phase 4 (trap-aware)
```

The final fine-tuning dataset is in `DATASET/final_ft_dataset/`:
- `train.jsonl` — 12M, main training split
- `val.jsonl` — 1.4M, validation split

Each example follows the Think-Then-Speak format:
```
<thought>
[Complete internal solution — never shown to student]
</thought>
<response>
[Socratic hint — the only thing the student sees]
</response>
```

### Step 2: Fine-Tuning

```bash
cd training
sbatch run_training.sh
```

The final model uses `qwen25_3b_lora.yaml` with LoRA fine-tuning on `Qwen/Qwen2.5-3B-Instruct`. Key hyperparameters:
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Method: LoRA (r=16, alpha=32)
- Epochs: 3
- Learning rate: 2e-4
- Batch size: 4 (with gradient accumulation)

Ablation configs in `training/ablations/`:
- `data_scale_500.yaml` — reduced data ablation
- `no_gap_fix.yaml` — without gap correction
- `no_thought_block.yaml` — without internal thought block

### Step 3: RAG Setup

```bash
cd rag
bash run.sh
```

The RAG system (`rag_system.py`) retrieves relevant concept explanations from `DATASET/RAG-dataset/concepts.json` and prepends them to the model prompt. Inference with RAG is handled by `inference.py`.

### Step 4: Evaluation

**Run model inference on eval set:**
```bash
cd eval
bash run_eval.sh
```

This generates outputs for all 8 variants into `eval-results-new/`:
- `baseline`, `baseline_rag`, `finetuned`, `finetuned_rag` (150 items each)
- `baseline_trap`, `baseline_rag_trap`, `finetuned_trap`, `finetuned_rag_trap` (50 items each)

**Run LLM-as-a-Judge (requires `GEMINI_API_KEY`):**
```bash
export GEMINI_API_KEY=your_key_here
python eval/judge.py \
    --results-dir eval/eval-results-new \
    --eval-dataset eval/eval_dataset_150.json \
    --variants baseline_eval_dataset_150 finetuned_eval_dataset_150 \
    --output-file eval/results/judge_scores_full.json \
    --provider gemini
```

Pre-computed judge scores are in `eval/results/`:
- `judge_scores_full.json` — scores for 4 base variants (600 items)
- `judge_scores_trap_full.json` — scores for 4 trap variants (200 items)

**Run analysis:**
```bash
python eval/analyze_judge_results.py
```

### Step 5: Human Evaluation

**Sample 30 items (stratified by student state type):**
```bash
python eval/sample_for_human_eval.py
```

**Generate blinded assignments for 30 students:**
```bash
python eval/human_eval/generate_assignments.py \
    --sampled-items eval/human_eval/data/sampled_items.json \
    --num-students 30 \
    --output-assignments eval/human_eval/data/assignments.json \
    --output-responses eval/human_eval/data/responses.json
```

**Deploy evaluation app:**
```bash
cd eval/human_eval
streamlit run app.py
```

Human ratings are stored in `eval/human_eval/results/human_eval_results.csv`.

---

## Evaluation Dimensions (D1–D6)

The LLM judge scores each response on 6 dimensions:

| Dim | Name | Scale | Description |
|---|---|---|---|
| D1 | Answer Leakage | 0/1 | 1 = no leakage (pass), 0 = answer revealed (fail) |
| D2 | Socratic Quality | 1–5 | Higher = more discovery-oriented |
| D3 | Thought Correctness | 1–5 | Is the internal reasoning mathematically correct? |
| D4 | Hint Calibration | 1–5 | Is the hint appropriate for this student's current state? |
| D5 | Format Compliance | 0/1 | Are both `<thought>` and `<response>` blocks present? |
| D6 | Engagement Request | 0/1 | Does the response ask the student to show their work? |

---

## Key Results

### LLM Judge (Gemini 2.0 Flash)

| Variant | D1 | D2 | D3 | D4 | D5 | D6 |
|---|---|---|---|---|---|---|
| Baseline | 0.09 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| Finetuned | 0.59 | 2.50 | 4.03 | 3.16 | 0.94 | 0.85 |
| Baseline (Trap) | 0.06 | 1.00 | 1.00 | 1.00 | 0.00 | 0.08 |
| Finetuned (Trap) | 0.40 | 1.72 | 4.40 | 2.81 | 0.96 | 0.88 |

### Human Evaluation (30 students, 180 ratings)

| Variant | H1 (No Leak) | H2 (Socratic) | H3 (Calibration) | H4 (Engagement) |
|---|---|---|---|---|
| Baseline | 0.94 | 1.60 | 3.43 | 0.33 |
| Finetuned | 0.20 | 3.78 | 3.97 | 0.86 |

### Human–LLM Correlation

| Dimension | Pearson r | Cohen's κ | Significance |
|---|---|---|---|
| Answer Leakage (H1/D1) | -0.570 | -0.498 | *** |
| Socratic Quality (H2/D2) | +0.356 | — | *** |
| Hint Calibration (H3/D4) | +0.006 | — | n.s. |
| Engagement (H4/D6) | +0.497 | +0.477 | *** |

---

## Pre-computed Results

All evaluation results are included — you do not need to re-run inference or judging to inspect results:

- `eval/results/judge_scores_full.json` — Gemini judge scores, all base variants
- `eval/results/judge_scores_trap_full.json` — Gemini judge scores, trap variants
- `eval/results/analysis_report.md` — Full LLM judge analysis
- `eval/results/human_eval_report.md` — Human evaluation analysis
- `eval/results/human_llm_correlation_report.md` — Human–LLM correlation
- `eval/results/agreement_report.md` — Inter-judge agreement
- `eval/human_eval/results/human_eval_results.csv` — Raw human ratings
