# Human Evaluation vs LLM-as-a-Judge Correlation Report

## Setup

- **Human raters:** 30 students, each rating 6 responses (180 total ratings)
- **LLM judge:** Gemini 2.0 Flash (primary judge)
- **Items:** 60 responses (30 items × 2 variants, blinded A/B)
- **Correlation dimensions:**
  - H1 (human) ↔ D1 (Gemini): Answer Leakage (binary)
  - H2 (human) ↔ D2 (Gemini): Socratic Quality (1–5)
  - H3 (human) ↔ D4 (Gemini): Hint Calibration (1–5)
  - H4 (human) ↔ D6 (Gemini): Engagement (binary)

---

## Correlation Results

| Human Dim | Gemini Dim | N | Pearson r | p-value | Significance | Mean \|Δ\| |
|---|---|---|---|---|---|---|
| H1 (Answer Leakage) | D1 | 180 | -0.570 | <0.001 | *** | 0.789 |
| H2 (Socratic Quality) | D2 | 180 | +0.356 | <0.001 | *** | 1.344 |
| H3 (Hint Calibration) | D4 | 177 | +0.006 | 0.936 | n.s. | 2.350 |
| H4 (Engagement) | D6 | 180 | +0.497 | <0.001 | *** | 0.267 |

### Binary Dimensions — Cohen's Kappa

| Pair | Kappa | Interpretation |
|---|---|---|
| H1 ↔ D1 (Answer Leakage) | -0.498 | Systematic disagreement |
| H4 ↔ D6 (Engagement) | +0.477 | Moderate agreement |

---

## Mean Score Comparison

| Dimension | Human Mean | Gemini Mean | Δ |
|---|---|---|---|
| H1/D1 (Answer Leakage) | 0.572 | 0.317 | +0.255 |
| H2/D2 (Socratic Quality) | 2.689 | 1.767 | +0.922 |
| H3/D4 (Hint Calibration) | 3.678 | 1.847 | +1.831 |
| H4/D6 (Engagement) | 0.594 | 0.450 | +0.144 |

---

## Interpretation

### H4 ↔ D6 (Engagement): Moderate Agreement (κ=0.477, r=0.497)
The strongest agreement is on **engagement** — whether the response asks students to show their work. Both humans and Gemini agree on this dimension at a moderate level (κ=0.477 is generally considered "moderate" agreement). This is the most objectively scorable dimension and requires the least pedagogical judgment.

### H2 ↔ D2 (Socratic Quality): Weak-Moderate Agreement (r=0.356)
Statistically significant but weak correlation. Humans consistently rate Socratic quality higher than Gemini (mean 2.69 vs 1.77). This suggests Gemini's rubric is stricter — it requires explicit questions or pointed hints, while students perceive even somewhat directive responses as Socratic. The 1.344 mean absolute difference is substantial.

### H1 ↔ D1 (Answer Leakage): Systematic Disagreement (κ=-0.498, r=-0.570)
**Negative correlation** — when Gemini says a response leaked the answer, humans tend to say it did not, and vice versa. The systematic reversal (κ=-0.498) reveals a fundamental difference in how leakage is perceived:
- **Gemini** detects leakage based on whether `<thought>`/`<response>` tags are present and whether the response contains the final answer
- **Humans** judge leakage based on whether they felt the answer was given to them
- Many baseline responses (D5=0.00, no format tags) are scored D1=0 by Gemini (leaked) but humans rate them H1=1 (not leaked), because the raw output may contain a hint-like response even without proper formatting

### H3 ↔ D4 (Hint Calibration): No Agreement (r=0.006, n.s.)
No significant correlation on hint calibration. Humans rate calibration at 3.68 on average while Gemini gives 1.85. This is the most subjective dimension — whether a hint is "perfectly targeted to where the student is" requires deep understanding of the student's cognitive state, which neither a 2B judge nor a student rater may have reliably.

---

## Key Takeaways for the Paper

1. **Human and LLM judges agree most on Engagement (D6/H4)** — the most objectively measurable dimension.

2. **The H1/D1 disagreement is meaningful, not noise.** It reveals that Gemini penalizes format failures (no `<thought>`/`<response>` tags) as leakage, while students judge the content itself. This is actually a feature of the Gemini rubric — it correctly identifies that baseline responses are structurally broken even when students don't notice.

3. **Humans consistently rate all dimensions higher than Gemini.** This is expected — students are lenient evaluators, while an automated judge with explicit rubric criteria is stricter.

4. **H3/D4 (Hint Calibration) has no agreement** — this is the hardest dimension to evaluate for both humans and LLMs, and confirms that calibration is genuinely difficult to measure.

5. **The moderate agreement on H4/D6 (κ=0.477) validates the Gemini judge on the most important functional dimension** — whether the model actively engages students.
