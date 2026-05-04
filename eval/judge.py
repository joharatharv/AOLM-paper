"""
CurioChain — LLM-as-a-Judge scoring script

Usage:
    python eval/judge.py --results-dir /scratch/lakshmiprajna.p/eval_results_v2 \
                    --eval-dataset eval_dataset.json \
                    --output-file judge_scores.json

    # Score a single variant (useful for testing):
    python eval/judge.py --results-dir eval/results \
                    --variants finetuned_rag \
                    --output-file judge_scores.json

Requires (pick one):
    pip install google-genai   &&  export GEMINI_API_KEY=...      (paid tier)
    pip install anthropic      &&  export ANTHROPIC_API_KEY=sk-...

Scores each model response on 7 dimensions (D1–D7).
Saves incrementally — safe to Ctrl+C and resume.
"""

import argparse
import json
import os
import time
import re
import sys
from pathlib import Path

# ── Load .env if present (check eval/ then project root) ─────────────────────
env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# ── Gemini API keys (add your keys here, one per line) ───────────────────────
# ⚠️  WARNING: Do NOT push this file to GitHub with real keys in it.
# ⚠️  Run:  git update-index --assume-unchanged eval/judge.py   to keep it local.
GEMINI_API_KEYS = [
    "AIzaSyBCSoIrn3cFWO-G81CSGBPYxT05X3THlsY",
    "AIzaSyDifDCvX9BkE2Ss1C6F5GjuLlH56_2tglc",
    "AIzaSyDkgj3nAuRoRCNImrX-oF3_HBdzVYQYvCA",
    "AIzaSyA7YGqHC4rZhavpl869nK08huzqYSiIWQ4",
    "AIzaSyCW2RzAofiySsRuJQu-eOe93y7JqBXQ8ZY",
    "AIzaSyAjE6Zx2AuO9V0kB6BmhcnHJng23o6SACA",
    "AIzaSyDFJcCs0hawbHkBIrJnCiZB-NXMKaUYFVg",
    "AIzaSyDetDo7qPoWVc-ElSDMpZXPCfztvtHk-H0",
    "AIzaSyDWl1yXAXnsqyCY9AN9oJtlYzPfvmCKUm8",
    "AIzaSyBJKTy_ku0jYPcgH_7NLhJ9mLKtrWFdLEo"
]

# ── Judge models ──────────────────────────────────────────────────────────────
GEMINI_MODEL        = "gemini-3.1-flash-lite-preview"
ANTHROPIC_MODEL     = "claude-sonnet-4-6"
OLLAMA_MODEL        = "llama3.1:8b"
OLLAMA_URL          = "http://localhost:11434/v1/chat/completions"
TRANSFORMERS_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

# ── Rubric prompt ─────────────────────────────────────────────────────────────
RUBRIC = """You are an evaluator scoring an AI math tutor. The tutor MUST use a "Think-Then-Speak" format:
- A <thought>...</thought> block: internal reasoning, NEVER shown to the student
- A <response>...</response> block: the ONLY thing the student sees. Must be a Socratic hint, never the answer.

CRITICAL RULES — read these before scoring:

1. You will be given THREE clearly-labeled fields:
     - PARSED_THOUGHT: the content of the <thought> block (or "(none)")
     - PARSED_RESPONSE: the content of the <response> block (or "(none — see RAW_OUTPUT)")
     - FORMAT_DETECTED: "BOTH_TAGS_PRESENT" or "MISSING_TAGS"
   When FORMAT_DETECTED = "MISSING_TAGS", the model produced a free-form answer with NO tags.
   The student would see the ENTIRE raw output. Treat the whole raw output as the response.

2. Score based ONLY on what the model actually produced. Do NOT solve the math problem.
   Do NOT describe the problem's math. Quote the model's text as evidence.

3. Common failure pattern (apply these rules strictly):
   - Output has no <thought>/<response> tags AND contains \\boxed{...} or "the answer is X" → it is a worked solution.
       → D1 = 0 (leaked), D2 = 1 (not Socratic), D3 = 1 (no thought block), D4 = 1 (no hint),
          D5 = 0 (no tags), D6 = 0 (no engagement request).
   - Output has both tags but PARSED_RESPONSE contains the final numeric answer or \\boxed{} → D1 = 0.
   - PARSED_THOUGHT is empty/missing → D3 = 1.
   - PARSED_RESPONSE contains no "?" and no imperative ("try", "show", "what is", "can you") → D6 = 0.

4. Before assigning each score, write a one-sentence "evidence" reasoning that QUOTES a short
   snippet from the model's actual output. If you cannot quote evidence, you cannot score high.

Return ONLY valid JSON — no extra text, no markdown fences.

──────────────────────────────────────────────────────────────
SCORING DIMENSIONS
──────────────────────────────────────────────────────────────

[D1] ANSWER LEAKAGE (binary: 0 or 1)
Does the <response> reveal the answer, final result, or complete a key step the student hasn't done?
  1 = No leakage: response withholds the answer and only hints
  0 = FAIL: response gives the answer, final numerical result, or solves a key step for the student
This is a knockout metric.

[D2] SOCRATIC QUALITY (1–5)
Does the response guide discovery rather than instruct?
  5 = Uses a question or pointed hint requiring active thinking; connects to student's prior work
  4 = Good hint that guides without telling; minor passivity (e.g. explains instead of asking)
  3 = Somewhat directive ("try doing X") but student still has to do real work
  2 = Mostly instructional; gives method/formula without asking student to derive it
  1 = Tells student exactly what to do step-by-step; functions as a worked example

[D3] MATHEMATICAL CORRECTNESS OF THOUGHT (1–5)
Is the solution in the <thought> block mathematically correct?
  5 = Complete, correct solution with all steps valid
  4 = Correct approach and final answer; minor notational or intermediate issues
  3 = Correct approach but computational error; final answer may be wrong
  2 = Partially correct setup; key conceptual step missing or wrong
  1 = Incorrect approach or fundamentally wrong solution
  MUST be 1 if PARSED_THOUGHT is "(none)" or empty — there is no thought block to evaluate.
  Verify the actual arithmetic/logic in PARSED_THOUGHT. Do not assume it is correct.

[D4] HINT CALIBRATION (1–5)
Is the hint appropriate for where THIS SPECIFIC student is right now?
  5 = Directly addresses student's exact current state; acknowledges what they did right; points to their precise next step
  4 = Relevant to the problem but slightly misaligned with student's position (hints at already-done step, or skips too far ahead)
  3 = Generic hint applicable to any student at any stage; not personalized
  2 = Addresses wrong part of problem relative to what student showed
  1 = Irrelevant, confusing, or addressed to a completely different problem/step

[D5] FORMAT COMPLIANCE (binary: 0 or 1)
Use FORMAT_DETECTED directly:
  1 only if FORMAT_DETECTED = "BOTH_TAGS_PRESENT" AND both PARSED_THOUGHT and PARSED_RESPONSE are non-empty.
  0 if FORMAT_DETECTED = "MISSING_TAGS", or if either parsed field is empty, or if a tag appears
    inside the wrong field (parsing failure).

[D6] ENGAGEMENT REQUEST (binary: 0 or 1)
Does the response require the student to show their working?
  1 = Explicitly asks student to attempt next step, show work, or demonstrate understanding
  0 = Does not invite participation; could be passively received

[D7] ARITHMETIC/LOGIC SEPARATION (1–5 or "N/A")
ONLY score this if student_state_type is "arithmetic_error" (correct approach, wrong computation).
For all other state types, set this to "N/A".
When the student's reasoning is correct but computation is wrong, does the response:
  5 = Explicitly validates the logical approach AND identifies the arithmetic mistake AND asks student to redo just the calculation
  4 = Catches the arithmetic error and hints at it, but doesn't clearly separate "logic ok" from "calculation wrong"
  3 = Flags something is wrong but treats it as a conceptual mistake rather than computational
  2 = Validates the entire answer (including the wrong computation) without catching the error
  1 = Confused, irrelevant, or ignores the error entirely

──────────────────────────────────────────────────────────────
OUTPUT FORMAT (return ONLY this JSON, nothing else)
──────────────────────────────────────────────────────────────
{
  "D1_answer_leakage": <0 or 1>,
  "D2_socratic_quality": <1-5>,
  "D3_thought_correctness": <1-5>,
  "D4_hint_calibration": <1-5>,
  "D5_format_compliance": <0 or 1>,
  "D6_engagement_request": <0 or 1>,
  "D7_arithmetic_separation": <1-5 or "N/A">,
  "D1_reasoning": "<one sentence>",
  "D2_reasoning": "<one sentence>",
  "D3_reasoning": "<one sentence>",
  "D4_reasoning": "<one sentence>",
  "D7_reasoning": "<one sentence or 'N/A'>"
}"""


_THOUGHT_RE  = re.compile(r"<thought>(.*?)</thought>",   re.DOTALL | re.IGNORECASE)
_RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
_LEAK_RE     = re.compile(
    r"\\boxed\s*\{|"                          # \boxed{...}
    r"\bthe\s+answer\s+is\b|"
    r"\bfinal\s+answer\s*[:=]|"
    r"\btherefore[, ].{0,40}=\s*-?\d",        # "therefore X = 52"
    re.IGNORECASE,
)
_ENGAGE_RE = re.compile(
    r"\?|"
    r"\b(show me|show your|try (?:to|computing|finding|writing)|"
    r"what (?:is|do|would|happens)|how (?:do|would|can|many)|"
    r"can you|could you|your turn|give it a try|attempt)\b",
    re.IGNORECASE,
)


def parse_blocks(raw_output: str):
    """Extract <thought>/<response> contents. Returns (thought, response, both_tags_present)."""
    raw = raw_output or ""
    t_match = _THOUGHT_RE.search(raw)
    r_match = _RESPONSE_RE.search(raw)
    thought  = t_match.group(1).strip() if t_match  else ""
    response = r_match.group(1).strip() if r_match else ""
    both = bool(t_match and r_match and thought and response)
    return thought, response, both


def deterministic_signals(raw_output: str):
    """Programmatic D5/D6 + leakage prefilter. Returns dict with signals."""
    thought, response, both = parse_blocks(raw_output)
    if both:
        student_facing = response
    else:
        student_facing = raw_output or ""
    leakage_detected = bool(_LEAK_RE.search(student_facing))
    engagement       = bool(_ENGAGE_RE.search(student_facing))
    return {
        "thought": thought,
        "response": response,
        "both_tags_present": both,
        "student_facing_text": student_facing,
        "format_compliance": 1 if both else 0,
        "engagement_request": 1 if engagement else 0,
        "leakage_detected": leakage_detected,
    }


def build_judge_prompt(problem: str, student_working: str,
                       student_state_type: str, raw_output: str) -> str:
    sig = deterministic_signals(raw_output)
    parsed_thought   = sig["thought"]   if sig["thought"]   else "(none)"
    parsed_response  = sig["response"]  if sig["response"]  else "(none — see RAW_OUTPUT)"
    fmt_detected     = "BOTH_TAGS_PRESENT" if sig["both_tags_present"] else "MISSING_TAGS"
    leak_hint        = "YES — student-facing text contains \\boxed{} or 'the answer is...'" \
                       if sig["leakage_detected"] else "no obvious final-answer markers detected"
    engage_hint      = "YES" if sig["engagement_request"] else "NO"

    raw_for_prompt = (raw_output or "")[:4000]

    return f"""PROBLEM:
{problem}

STUDENT STATE TYPE: {student_state_type}

STUDENT'S WORKING:
{student_working}

────────────────────────────────────────────────────────────
PRE-COMPUTED SIGNALS (use these — they are deterministic):
  FORMAT_DETECTED: {fmt_detected}
  LEAKAGE_MARKERS_IN_STUDENT_FACING_TEXT: {leak_hint}
  ENGAGEMENT_PHRASING_DETECTED: {engage_hint}

PARSED_THOUGHT (internal reasoning — only used for D3):
{parsed_thought}

PARSED_RESPONSE (this is what the student sees — used for D1/D2/D4/D6):
{parsed_response}

RAW_OUTPUT (full untrimmed, for reference; if FORMAT_DETECTED = MISSING_TAGS, the student sees ALL of this):
{raw_for_prompt}
────────────────────────────────────────────────────────────

Reminders:
- If FORMAT_DETECTED = MISSING_TAGS → D5 = 0. The whole RAW_OUTPUT is what the student sees.
- If LEAKAGE_MARKERS = YES → D1 = 0 (leaked).
- If PARSED_THOUGHT = (none) → D3 = 1.
- Quote the model's actual text in each *_reasoning field. Do NOT paraphrase the math problem."""


# ── Gemini key rotation ──────────────────────────────────────────────────────
class GeminiKeyRotator:
    """Rotates through multiple Gemini API keys on rate-limit errors."""
    def __init__(self, keys: list):
        if not keys:
            raise ValueError("No Gemini API keys provided.")
        self.keys = list(keys)
        self.idx  = 0
        self._make_client()
        print(f"[gemini] Loaded {len(self.keys)} API key(s) for rotation.")

    def _make_client(self):
        from google import genai as google_genai
        self.client = google_genai.Client(api_key=self.keys[self.idx])

    def rotate(self):
        self.idx = (self.idx + 1) % len(self.keys)
        self._make_client()
        print(f"    [gemini] Rotated to key index {self.idx}.")

    def ban_current_key(self):
        """Permanently remove the current key (e.g. 403 access denied)."""
        banned = self.keys.pop(self.idx)
        print(f"    [gemini] Banned key ...{banned[-6:]} (403). {len(self.keys)} key(s) remaining.")
        if not self.keys:
            raise RuntimeError("All Gemini API keys have been banned (403 PERMISSION_DENIED).")
        self.idx = self.idx % len(self.keys)
        self._make_client()


def _extract_scores_regex(text: str) -> dict | None:
    """Fallback: extract integer scores via regex when json.loads fails.
    Returns a dict with at least D1-D4 present, or None if extraction fails."""
    score_keys = [
        "D1_answer_leakage", "D2_socratic_quality", "D3_thought_correctness",
        "D4_hint_calibration", "D5_format_compliance", "D6_engagement_request",
        "D7_arithmetic_separation",
    ]
    result = {}
    for k in score_keys:
        m = re.search(rf'"{k}"\s*:\s*("N/A"|null|\d+)', text)
        if m:
            v = m.group(1)
            result[k] = int(v) if v.isdigit() else (None if v == "null" else "N/A")
    required = {"D1_answer_leakage", "D2_socratic_quality",
                "D3_thought_correctness", "D4_hint_calibration"}
    if required.issubset(result.keys()):
        return result
    return None


def load_gemini_keys(keys_file: str = None) -> list:
    """Load Gemini API keys — priority: keys_file > GEMINI_API_KEYS list > env var."""
    keys = []
    # 1. From file if provided
    if keys_file and os.path.exists(keys_file):
        for line in Path(keys_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keys.append(line)
        if keys:
            return keys
    # 2. From hardcoded list at top of file
    hardcoded = [k.strip() for k in GEMINI_API_KEYS if k.strip()]
    if hardcoded:
        return hardcoded
    # 3. Fallback: single key from environment
    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key:
        keys.append(env_key)
    return keys


def call_judge_gemini(rotator, prompt: str,
                      retries: int = 3, delay: float = 5.0):
    from google.genai import types
    full_prompt = RUBRIC + "\n\n" + prompt
    keys_exhausted_count = 0
    json_fail_count = 0  # consecutive JSON parse failures
    for attempt in range(retries * len(rotator.keys)):
        try:
            # After repeated JSON failures bump temperature so Gemini varies its output
            temperature = 0.0 if json_fail_count == 0 else 0.3
            response = rotator.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=512,
                ),
            )
            keys_exhausted_count = 0  # reset on success
            text = response.text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            # Fix invalid JSON escapes from LaTeX (e.g. \sqrt, \pi, \frac)
            # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
            text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', text)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Escape fix didn't fully repair the JSON — try regex extraction
                fallback = _extract_scores_regex(text)
                if fallback is not None:
                    print(f"    [warn] Used regex fallback to extract scores (attempt {attempt+1})")
                    return fallback
                raise  # let outer except log it and retry
        except json.JSONDecodeError as e:
            json_fail_count += 1
            print(f"    [warn] JSON parse failed (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
        except Exception as e:
            err = str(e).lower()
            print(f"    [gemini error] {e}")
            if "403" in err or "permission_denied" in err:
                # Permanent access denial for this key/project — remove it
                rotator.ban_current_key()
                keys_exhausted_count = 0
            elif "429" in err or "resource_exhausted" in err or "quota" in err or "ratelimit" in err or "rate_limit" in err:
                # Parse retryDelay from error if present (e.g. "retryDelay: '45s'")
                retry_match = re.search(r"retry[_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+)", err)
                suggested_wait = int(retry_match.group(1)) if retry_match else 60
                rotator.rotate()
                keys_exhausted_count += 1
                if keys_exhausted_count >= len(rotator.keys):
                    # All keys hit rate limit — wait for the per-minute window to reset
                    wait = max(suggested_wait, 10)
                    print(f"    [gemini] All keys exhausted, waiting {wait}s for rate limit reset...")
                    time.sleep(wait)
                    keys_exhausted_count = 0
                else:
                    time.sleep(2.0)  # brief pause before trying next key
            else:
                if attempt < retries - 1:
                    time.sleep(delay)
    return None


def call_judge_anthropic(client, prompt: str,
                         retries: int = 3, delay: float = 5.0):
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=512,
                system=RUBRIC,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"    [warn] JSON parse failed (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err or "overloaded" in err:
                wait = delay * (2 ** attempt)
                print(f"    [rate limit] waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [api error] {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
    return None


def call_judge_ollama(prompt: str, retries: int = 3, delay: float = 5.0):
    import urllib.request
    full_prompt = RUBRIC + "\n\n" + prompt
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"    [warn] JSON parse failed (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
        except Exception as e:
            print(f"    [ollama error] {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return None


_hf_pipeline = None

def get_hf_pipeline():
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline, BitsAndBytesConfig
        import torch
        if torch.cuda.is_available():
            print(f"[transformers] Loading {TRANSFORMERS_MODEL} on CUDA (4-bit)...")
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            _hf_pipeline = pipeline(
                "text-generation",
                model=TRANSFORMERS_MODEL,
                model_kwargs={"quantization_config": bnb},
                device_map="auto",
            )
        elif torch.backends.mps.is_available():
            print(f"[transformers] Loading {TRANSFORMERS_MODEL} on MPS...")
            _hf_pipeline = pipeline(
                "text-generation",
                model=TRANSFORMERS_MODEL,
                device="mps",
                dtype=torch.float16,
            )
        else:
            print(f"[transformers] Loading {TRANSFORMERS_MODEL} on CPU...")
            _hf_pipeline = pipeline("text-generation", model=TRANSFORMERS_MODEL)
        print("[transformers] Model ready.")
    return _hf_pipeline


def _repair_json(text: str, error: Exception):
    """Fix JSON with unterminated strings by removing the incomplete trailing field."""
    if "Unterminated string" not in str(error):
        return None
    try:
        # Truncate before the unterminated string's opening quote
        truncated = text[:error.pos - 1]
        # Remove everything back to the last complete field (last comma)
        last_comma = truncated.rfind(',')
        last_brace = truncated.rfind('{')
        if last_comma > last_brace:
            truncated = truncated[:last_comma]
        truncated = truncated.rstrip() + "\n}"
        return json.loads(truncated)
    except Exception:
        return None


def call_judge_transformers(prompt: str, retries: int = 3):
    pipe = get_hf_pipeline()
    full_prompt = RUBRIC + "\n\n" + prompt
    messages = [{"role": "user", "content": full_prompt}]
    for attempt in range(retries):
        try:
            result = pipe(
                messages,
                max_new_tokens=512,
                temperature=0.01,
                do_sample=False,
                return_full_text=False,
            )
            text = result[0]["generated_text"].strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            # Extract first JSON object if model adds extra text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
            # Try direct parse first, then repair if truncated
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                repaired = _repair_json(text, e)
                if repaired is not None:
                    return repaired
                raise
        except json.JSONDecodeError as e:
            print(f"    [warn] JSON parse failed (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"    [transformers error] {e}")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(2)
    return None


def call_judge(client, provider: str, prompt: str):
    if provider == "gemini":
        return call_judge_gemini(client, prompt)  # client is a GeminiKeyRotator
    elif provider == "ollama":
        return call_judge_ollama(prompt)
    elif provider == "transformers":
        return call_judge_transformers(prompt)
    else:
        return call_judge_anthropic(client, prompt)


def load_existing_scores(output_file: str) -> dict:
    """Load already-scored items to allow resuming."""
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)
        # Key: (variant, eval_id)
        done = {(s["variant"], s["eval_id"]): s for s in existing}
        print(f"Resuming: {len(done)} items already scored.")
        return done
    return {}


def save_scores(scores: list, output_file: str):
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing results_*.json files")
    parser.add_argument("--eval-dataset", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dataset_150.json"),
                        help="eval_dataset.json (for student_state_type lookup)")
    parser.add_argument("--output-file", default="judge_scores.json")
    parser.add_argument("--variants", nargs="+",
                        default=["baseline", "finetuned",
                                 "baseline_rag", "finetuned_rag"],
                        help="Which model variants to score")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only first N items per variant (for testing)")
    parser.add_argument("--provider", choices=["gemini", "anthropic", "ollama", "transformers"], default=None,
                        help="Judge model provider (auto-detected from available API keys)")
    parser.add_argument("--gemini-keys-file", default=None,
                        help="Path to file with Gemini API keys, one per line (enables key rotation)")
    parser.add_argument("--judge-model", default=None,
                        help="Override the transformers judge model (e.g. mistralai/Mistral-7B-Instruct-v0.3)")
    args = parser.parse_args()

    if args.judge_model:
        global TRANSFORMERS_MODEL, OLLAMA_MODEL
        TRANSFORMERS_MODEL = args.judge_model
        OLLAMA_MODEL = args.judge_model

    # ── Auto-detect provider from available keys ──────────────────────────────
    provider = args.provider
    if provider is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
        else:
            provider = "ollama"  # fallback to local Ollama

    print(f"Judge provider: {provider}")

    client = None
    if provider == "gemini":
        try:
            from google import genai as google_genai  # noqa: F401 (imported inside rotator too)
        except ImportError:
            print("ERROR: google-genai not installed. Run: pip install google-genai")
            sys.exit(1)
        keys = load_gemini_keys(args.gemini_keys_file)
        if not keys:
            print("ERROR: No Gemini API keys found. Provide --gemini-keys-file or set GEMINI_API_KEY.")
            sys.exit(1)
        client = GeminiKeyRotator(keys)
    elif provider == "anthropic":
        try:
            import anthropic as anthropic_sdk
        except ImportError:
            print("ERROR: anthropic not installed. Run: pip install anthropic")
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        client = anthropic_sdk.Anthropic(api_key=api_key)
    # ollama: no client needed, uses urllib

    # ── Load eval_dataset for state_type metadata ─────────────────────────────
    with open(args.eval_dataset) as f:
        eval_dataset = json.load(f)
    state_map = {item["id"]: item["student_state_type"] for item in eval_dataset}
    print(f"Loaded {len(state_map)} items from {args.eval_dataset}")

    # ── Load existing scores (for resuming) ───────────────────────────────────
    done = load_existing_scores(args.output_file)
    all_scores = list(done.values())

    # ── Score each variant ────────────────────────────────────────────────────
    total_api_calls = 0

    for variant in args.variants:
        # Support both results_{variant}.json and eval_results_{variant}*.json
        result_file = os.path.join(args.results_dir, f"results_{variant}.json")
        if not os.path.exists(result_file):
            import glob
            matches = sorted(glob.glob(
                os.path.join(args.results_dir, f"eval_results_{variant}*.json")
            ))
            if matches:
                result_file = matches[-1]  # use most recent if multiple
            else:
                print(f"\n[skip] no results file found for variant '{variant}'")
                continue

        with open(result_file) as f:
            results = json.load(f)

        if args.limit:
            results = results[:args.limit]

        print(f"\n{'='*55}")
        print(f"  Variant: {variant}  ({len(results)} items)")
        print(f"{'='*55}")

        for i, result in enumerate(results):
            eval_id = str(result.get("eval_id", result.get("id", "")))
            key = (variant, eval_id)

            if key in done:
                print(f"  [{i+1:3d}/{len(results)}] {eval_id:30s} [already scored]")
                continue

            # Get student state type from eval_dataset
            state_type = state_map.get(eval_id, "unknown")

            # Build judge prompt
            prompt = build_judge_prompt(
                problem=result.get("problem", ""),
                student_working=result.get("student_working", ""),
                student_state_type=state_type,
                raw_output=result.get("raw_output", result.get("output", "")),
            )

            print(f"  [{i+1:3d}/{len(results)}] {eval_id:30s} (state: {state_type})", end="", flush=True)

            scores = call_judge(client, provider, prompt)

            if scores is None:
                print(" [FAILED]")
                continue

            # ── Deterministic overrides for objective dimensions ──────────────
            # The small judge mis-scores binary/format dimensions; recompute them.
            sig = deterministic_signals(result.get("raw_output", result.get("output", "")))
            scores["D5_format_compliance"] = sig["format_compliance"]
            scores["D6_engagement_request"] = sig["engagement_request"]
            if sig["leakage_detected"]:
                scores["D1_answer_leakage"] = 0
                scores["D1_reasoning"] = (
                    "Deterministic override: student-facing text contains a "
                    "final-answer marker (\\boxed{}, 'the answer is', etc.)."
                )
            if not sig["thought"]:
                scores["D3_thought_correctness"] = 1
                scores["D3_reasoning"] = "Deterministic override: PARSED_THOUGHT is empty."
            # If no tags at all, the model produced a worked solution — clamp
            # the subjective Socratic-style metrics to floor.
            if not sig["both_tags_present"] and sig["leakage_detected"]:
                d2 = scores.get("D2_socratic_quality", 1)
                d4 = scores.get("D4_hint_calibration", 1)
                scores["D2_socratic_quality"] = min(int(d2) if isinstance(d2, (int, float)) else 1, 1)
                scores["D4_hint_calibration"] = min(int(d4) if isinstance(d4, (int, float)) else 1, 1)

            record = {
                "variant": variant,
                "eval_id": eval_id,
                "base_problem_id": eval_id.split("_")[0] if "_" in eval_id else eval_id,
                "student_state_type": state_type,
                "problem": result.get("problem", ""),
                "student_working": result.get("student_working", ""),
                "rag_context_used": result.get("rag_context_used", False),
                **scores,
            }

            all_scores.append(record)
            done[key] = record
            total_api_calls += 1

            # Save after every item — safe to interrupt
            save_scores(all_scores, args.output_file)

            d1 = scores.get("D1_answer_leakage", "?")
            d2 = scores.get("D2_socratic_quality", "?")
            d3 = scores.get("D3_thought_correctness", "?")
            d4 = scores.get("D4_hint_calibration", "?")
            print(f" D1={d1} D2={d2} D3={d3} D4={d4}")

            # Sleep only needed for API providers, not local inference
            time.sleep(1.0)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Done. {total_api_calls} new API calls made.")
    print(f"  Total scored: {len(all_scores)} items")
    print(f"  Output: {args.output_file}")
    print(f"{'='*55}")

    # Quick aggregate preview
    from collections import defaultdict
    variant_scores = defaultdict(lambda: defaultdict(list))
    for s in all_scores:
        v = s["variant"]
        for dim in ["D1_answer_leakage", "D2_socratic_quality",
                    "D3_thought_correctness", "D4_hint_calibration",
                    "D5_format_compliance", "D6_engagement_request"]:
            val = s.get(dim)
            if isinstance(val, (int, float)):
                variant_scores[v][dim].append(val)

    print(f"\n{'Variant':<22} {'D1 leak%':>8} {'D2 socr':>8} {'D3 math':>8} {'D4 calib':>9}")
    print("-" * 60)
    for v in sorted(variant_scores.keys()):
        if v not in variant_scores:
            continue
        vs = variant_scores[v]
        d1_rate = (1 - (sum(vs['D1_answer_leakage']) / len(vs['D1_answer_leakage']))) * 100 if vs['D1_answer_leakage'] else 0
        d2 = sum(vs['D2_socratic_quality']) / len(vs['D2_socratic_quality']) if vs['D2_socratic_quality'] else 0
        d3 = sum(vs['D3_thought_correctness']) / len(vs['D3_thought_correctness']) if vs['D3_thought_correctness'] else 0
        d4 = sum(vs['D4_hint_calibration']) / len(vs['D4_hint_calibration']) if vs['D4_hint_calibration'] else 0
        print(f"{v:<22} {d1_rate:>7.1f}% {d2:>8.2f} {d3:>8.2f} {d4:>9.2f}")


if __name__ == "__main__":
    main()