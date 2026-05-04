# Inter-Judge Agreement Report

## Overview

Agreement between `meta-llama/Llama-3.2-3B-Instruct` and `google/gemma-2-2b-it` across 783 paired items (items scored by both judges).

---

## Pearson Correlation (Continuous Dimensions)

| Dimension | N | Pearson r | p-value | Significance | Mean |Δ| |
|---|---|---|---|---|---|
| D2_socratic_quality | 783 | 0.219 | <0.001 | *** | 0.315 |
| D4_hint_calibration | 783 | 0.146 | <0.001 | *** | 0.976 |
| D6_engagement_request | 783 | 0.086 | 0.016 | * | 0.402 |
| D3_thought_correctness | 783 | -0.036 | 0.318 | n.s. | 0.499 |
| D1_answer_leakage | 783 | N/A | N/A | — | 0.622 |
| D5_format_compliance | 770 | N/A | N/A | — | 0.000 |

*N/A = one judge produced a constant output (no variance), making correlation undefined.*

---

## Cohen's Kappa (Binary Dimensions)

| Dimension | Kappa | Interpretation |
|---|---|---|
| D1_answer_leakage | 0.000 | No agreement |
| D5_format_compliance | N/A | Perfect agreement (both always 1.00) |
| D6_engagement_request | 0.015 | Near-zero agreement |

---

## Systematic Biases Found in Gemma

Gemma shows clear systematic biases that make it unreliable as a judge:

| Dimension | Gemma Behavior | Llama Behavior |
|---|---|---|
| D1 (Answer Leakage) | Always 0.00 (all responses fail) | 0.60–0.72 (mixed) |
| D6 (Engagement) | Always 1.00 (all responses pass) | 0.20–1.00 (mixed) |
| D3 (Thought Correctness) | Drops to 3.00 for Finetuned Trap | Stays at 5.00 |

**Conclusion:** Gemma 2-2B-IT is too small and undertrained for the complex rubric evaluation required by this task. It defaults to constant outputs on binary dimensions (D1, D6) and shows inconsistent behavior on D3 for finetuned variants.

---

## Agreement Summary

| Dimension | Agreement Level | Reliable? |
|---|---|---|
| D2 (Socratic Quality) | Weak (r=0.219) | Partially |
| D4 (Hint Calibration) | Weak (r=0.146) | Partially |
| D3 (Thought Correctness) | None (r=-0.036) | No |
| D1 (Answer Leakage) | None (kappa=0.000) | No |
| D5 (Format Compliance) | Perfect (both 1.00) | Yes (trivial) |
| D6 (Engagement) | Near-zero (kappa=0.015) | No |

---

## Implications

1. **Low inter-judge agreement strongly motivates human evaluation** — this is a positive finding for the paper, as it demonstrates LLM judges alone are insufficient for this complex pedagogical task.
2. **Llama judge is used as primary** for all reported results, as Gemma's constant outputs on D1/D6 are clearly unreliable.
3. **D2 and D4 have weak but significant agreement** (p<0.001) — these dimensions are the most consistently scored by both judges and are the most meaningful for cross-judge comparison.
4. **Human eval will serve as ground truth** to determine which judge is more accurate on D1, D2, D4, and D6.
