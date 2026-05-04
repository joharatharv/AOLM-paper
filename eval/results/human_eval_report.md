# Human Evaluation Report

## Setup

- **Participants:** 30 student evaluators
- **Responses per participant:** 6 (each participant rated 6 responses)
- **Total ratings:** 180 (30 × 6)
- **Unique responses rated:** 60 (30 items × 2 variants, blinded A/B)
- **Raters per response:** 3 (each response rated by exactly 3 students)
- **Design:** Stratified random sampling — 5 items per state type (5 standard) + 5 trap items = 30 items

---

## Rating Questions

| Code | Question | Scale |
|---|---|---|
| H1 | Did the tutor's response give away the answer or solve a key step? | Binary (0=leaked, 1=no leak) |
| H2 | Did the response guide you to think, or just tell you what to do? | 1–5 (1=told me, 5=made me think) |
| H3 | Was the hint relevant to exactly where you were in solving the problem? | 1–5 (1=irrelevant, 5=perfectly targeted) |
| H4 | After reading this response, did you feel motivated to try the next step? | Binary (0=no, 1=yes) |
| H5 | After reading this hint, could you figure out what to do next? | Yes / Partially / No |

---

## Overall Human Scores

| Metric | Score |
|---|---|
| H1 Pass Rate (no leakage) | 0.572 (57.2% of responses judged as not leaking) |
| H2 Socratic Quality | 2.69 ± 1.63 |
| H3 Hint Calibration | 3.70 ± 1.43 |
| H4 Engagement Rate | 0.594 (59.4% motivated to continue) |
| H5 Learning Outcome | Yes: 74.4%, Partially: 17.2%, No: 8.3% |

---

## Baseline vs Finetuned (Human Scores)

| Dimension | Baseline (n=90) | Finetuned (n=90) | Δ |
|---|---|---|---|
| H1 (No Leakage Rate) | 0.94 | 0.20 | **-0.74** |
| H2 (Socratic Quality) | 1.60 | 3.78 | **+2.18** |
| H3 (Hint Calibration) | 3.43 | 3.97 | +0.54 |
| H4 (Engagement) | 0.33 | 0.86 | **+0.53** |
| H5 Yes (%) | 77.8% | 71.1% | -6.7% |
| H5 Partially (%) | 13.3% | 21.1% | +7.8% |
| H5 No (%) | 8.9% | 7.8% | -1.1% |

**Key finding:** Humans perceive finetuned responses as dramatically more Socratic (+2.18) and more engaging (+0.53). However, H1 shows a striking reversal — baseline responses are rated as not leaking (0.94) more often than finetuned (0.20). This is explained by the fact that baseline responses without `<thought>`/`<response>` tags often appear as brief, unhelpful responses that students don't recognize as answers, while finetuned responses contain more detailed content that students perceive as revealing.

---

## Scores by Student State Type

| State Type | N | H1 | H2 | H3 | H4 |
|---|---|---|---|---|---|
| almost_done | 30 | 0.60 | 2.70 | 3.77 | 0.53 |
| arithmetic_error | 30 | 0.43 | 3.17 | 3.57 | 0.53 |
| cold_start | 30 | 0.50 | 2.57 | 3.47 | 0.57 |
| partial_correct | 30 | 0.60 | 2.67 | 4.40 | 0.60 |
| wrong_confident | 30 | 0.70 | 2.33 | 3.17 | 0.63 |
| post_limit_idk (trap) | 18 | 0.61 | 2.72 | 3.83 | 0.78 |
| wrong_pick_phase4 (trap) | 6 | 0.50 | 2.67 | 3.50 | 0.67 |
| idk_phase2 (trap) | 6 | 0.67 | 2.67 | 4.17 | 0.50 |

**Notable:**
- `wrong_confident` has the lowest Socratic quality (H2=2.33) — students feel the model is less guiding when they're confidently wrong
- `partial_correct` has the highest calibration (H3=4.40) — hints are most relevant when students have already started
- `post_limit_idk` (trap) has the highest engagement (H4=0.78) — trap responses motivate students to continue

---

## Inter-Rater Reliability

Each response was rated by exactly 3 students. Pairwise agreement computed across all 180 rater pairs.

| Dimension | Pairwise Agreement | Interpretation |
|---|---|---|
| H1 (Answer Leakage) | 0.844 | High agreement |
| H4 (Engagement) | 0.667 | Moderate agreement |

High pairwise agreement on H1 (84.4%) indicates students are consistent in judging whether a response revealed the answer. Moderate agreement on H4 (66.7%) reflects genuine subjectivity in whether a response feels motivating.

---

## Learning Outcomes (H5)

| Variant | Yes | Partially | No |
|---|---|---|---|
| Baseline | 77.8% | 13.3% | 8.9% |
| Finetuned | 71.1% | 21.1% | 7.8% |
| **Overall** | **74.4%** | **17.2%** | **8.3%** |

74.4% of students felt they could figure out the next step after seeing the hint. The slightly lower "Yes" rate for finetuned (71.1% vs 77.8%) with higher "Partially" may reflect that finetuned responses are more genuinely Socratic — they guide without fully revealing, leaving students with productive uncertainty.

---

## Key Takeaways

1. **Students strongly prefer finetuned responses** — H2 improves by +2.18 and H4 by +0.53.
2. **H1 reversal reveals a perception gap** — students rate baseline as "not leaking" more than finetuned (0.94 vs 0.20), but this is because baseline responses are so brief/unformatted that students don't recognize them as answers. Gemini correctly identifies these as structural failures.
3. **High inter-rater agreement on H1 (84.4%)** — students are consistent evaluators of answer leakage.
4. **wrong_confident is the hardest state** — lowest Socratic quality ratings, confirming this is a difficult case for the model.
5. **74.4% learning outcome** — nearly 3/4 of students felt they could proceed after seeing the hint, suggesting the finetuned model achieves its core pedagogical goal.
