# LLM-as-a-Judge Analysis Report (Gemini Judge — Final)

## Overview

**Judge:** `gemini-2.0-flash` — 800 total items (600 base + 200 trap)

**Scoring dimensions:**
- D1: Answer Leakage (0=fail/leaks, 1=pass — higher is better)
- D2: Socratic Quality (1–5 — higher is better)
- D3: Thought Correctness (1–5 — higher is better)
- D4: Hint Calibration (1–5 — higher is better)
- D5: Format Compliance (0/1 — higher is better)
- D6: Engagement Request (0/1 — higher is better)

---

## Per-Variant Scores

| Variant | N | D1 | D2 | D3 | D4 | D5 | D6 |
|---|---|---|---|---|---|---|---|
| Baseline | 150 | 0.09 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| Finetuned | 150 | 0.59 | 2.50 | 4.03 | 3.16 | 0.94 | 0.85 |
| Baseline+RAG | 150 | 0.07 | 1.00 | 1.00 | 1.00 | 0.00 | 0.01 |
| Finetuned+RAG | 150 | 0.61 | 2.51 | 3.94 | 3.06 | 0.94 | 0.84 |
| Baseline (Trap) | 50 | 0.06 | 1.00 | 1.00 | 1.00 | 0.00 | 0.08 |
| Finetuned (Trap) | 50 | 0.40 | 1.72 | 4.40 | 2.81 | 0.96 | 0.88 |
| Baseline+RAG (Trap) | 50 | 0.04 | 1.00 | 1.00 | 1.00 | 0.00 | 0.08 |
| Finetuned+RAG (Trap) | 50 | 0.32 | 1.58 | 3.96 | 2.47 | 0.98 | 0.90 |

---

## Baseline vs Finetuned Comparison

| Condition | D1 Base | D1 Fine | Δ | D2 Base | D2 Fine | Δ | D3 Base | D3 Fine | Δ | D4 Base | D4 Fine | Δ | D5 Base | D5 Fine | Δ | D6 Base | D6 Fine | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Base | 0.09 | 0.59 | **+0.49** | 1.00 | 2.50 | **+1.50** | 1.00 | 4.03 | **+3.03** | 1.00 | 3.16 | **+2.16** | 0.00 | 0.94 | **+0.94** | 0.00 | 0.85 | **+0.85** |
| RAG | 0.07 | 0.61 | **+0.53** | 1.00 | 2.51 | **+1.51** | 1.00 | 3.94 | **+2.94** | 1.00 | 3.06 | **+2.06** | 0.00 | 0.94 | **+0.94** | 0.01 | 0.84 | **+0.83** |
| Trap | 0.06 | 0.40 | **+0.34** | 1.00 | 1.72 | **+0.72** | 1.00 | 4.40 | **+3.40** | 1.00 | 2.81 | **+1.81** | 0.00 | 0.96 | **+0.96** | 0.08 | 0.88 | **+0.80** |
| Trap+RAG | 0.04 | 0.32 | **+0.28** | 1.00 | 1.58 | **+0.58** | 1.00 | 3.96 | **+2.96** | 1.00 | 2.47 | **+1.47** | 0.00 | 0.98 | **+0.98** | 0.08 | 0.90 | **+0.82** |

---

## Ablation Analysis

### Effect of Finetuning (non-trap, averaged across base and RAG)

| Dimension | Baseline Avg | Finetuned Avg | Δ |
|---|---|---|---|
| D1 (Answer Leakage) | 0.08 | 0.60 | **+0.52** |
| D2 (Socratic Quality) | 1.00 | 2.51 | **+1.51** |
| D3 (Thought Correctness) | 1.00 | 3.99 | **+2.99** |
| D4 (Hint Calibration) | 1.00 | 3.11 | **+2.11** |
| D5 (Format Compliance) | 0.00 | 0.94 | **+0.94** |
| D6 (Engagement) | 0.01 | 0.84 | **+0.84** |

### Effect of RAG (finetuned only)

| Dimension | Finetuned (no RAG) | Finetuned+RAG | Δ |
|---|---|---|---|
| D1 | 0.59 | 0.61 | +0.02 |
| D2 | 2.50 | 2.51 | +0.01 |
| D3 | 4.03 | 3.94 | -0.09 |
| D4 | 3.16 | 3.06 | -0.10 |
| D5 | 0.94 | 0.94 | 0.00 |
| D6 | 0.85 | 0.84 | -0.01 |

**RAG has negligible effect** — all deltas < 0.10.

### Effect of Trap Prompts (finetuned only)

| Dimension | Finetuned (standard) | Finetuned (trap) | Δ |
|---|---|---|---|
| D1 | 0.59 | 0.40 | **-0.19** |
| D2 | 2.50 | 1.72 | **-0.78** |
| D3 | 4.03 | 4.40 | +0.37 |
| D4 | 3.16 | 2.81 | -0.35 |
| D5 | 0.94 | 0.96 | +0.02 |
| D6 | 0.85 | 0.88 | +0.03 |

**Trap prompts are harder** — D1 and D2 drop significantly, showing the model still struggles to fully resist adversarial student inputs.

---

## Key Takeaways

1. **Baseline is completely broken** — no format compliance (D5=0.00), no engagement (D6=0.00), near-zero Socratic quality (D2=1.00), 91% answer leakage rate. The baseline Qwen model does not follow the Think-Then-Speak format at all.

2. **Finetuning fixes format completely** — D5 goes from 0.00 to 0.94, meaning 94% of responses now correctly produce `<thought>` and `<response>` blocks.

3. **Finetuning dramatically improves every dimension** across all 4 conditions (Base, RAG, Trap, Trap+RAG).

4. **RAG adds no value** — differences between finetuned and finetuned+RAG are negligible (<0.10 on all dimensions).

5. **Trap prompts remain challenging** — even the finetuned model's D1 drops from 0.59 to 0.40 on trap prompts, and D2 drops from 2.50 to 1.72. The model is not fully robust to adversarial student inputs.

6. **D3 (Thought Correctness) improves most on trap prompts** (+3.40 vs +3.03 for standard) — the finetuned model actually reasons *better* internally on trap problems.
