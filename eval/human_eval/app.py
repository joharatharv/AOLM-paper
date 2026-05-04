import streamlit as st
import json
import csv
import os
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RESPONSES_FILE = DATA_DIR / "responses.json"
RESPONSES_PLACEHOLDER_FILE = DATA_DIR / "responses_placeholder.json"
ASSIGNMENTS_FILE = DATA_DIR / "assignments.json"
RESULTS_FILE = RESULTS_DIR / "human_eval_results.csv"

FIELDNAMES = [
    "timestamp", "student_id", "response_id", "item_id",
    "student_state_type",
    "h1_answer_leakage", "h2_socratic_quality",
    "h3_hint_calibration", "h4_engagement", "h5_learning_outcome",
]


# ── Supabase ───────────────────────────────────────────────────────────────────

def get_supabase():
    try:
        from supabase import create_client
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["anon_key"]
        return create_client(url, key)
    except Exception:
        return None


def already_submitted(student_id: str) -> set:
    done = set()
    client = get_supabase()
    if client:
        try:
            rows = client.table("human_eval_results") \
                .select("response_id") \
                .eq("student_id", student_id) \
                .execute()
            return {r["response_id"] for r in rows.data}
        except Exception:
            pass
    # Fallback to local CSV
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, newline="") as f:
            for row in csv.DictReader(f):
                if row["student_id"] == student_id:
                    done.add(row["response_id"])
    return done


def save_rating(student_id: str, response: dict, ratings: dict):
    row = {
        "timestamp": datetime.now().isoformat(),
        "student_id": student_id,
        "response_id": response["response_id"],
        "item_id": response["item_id"],
        "student_state_type": response["student_state_type"],
        **ratings,
    }
    # Save to Supabase (primary)
    client = get_supabase()
    if client:
        try:
            client.table("human_eval_results").insert(row).execute()
        except Exception as e:
            st.warning(f"Database save failed, saving locally: {e}")
    # Always save to CSV as backup
    file_exists = RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@st.cache_data
def load_data():
    responses_file = RESPONSES_FILE if RESPONSES_FILE.exists() else RESPONSES_PLACEHOLDER_FILE
    with open(responses_file) as f:
        responses = {r["response_id"]: r for r in json.load(f)}
    with open(ASSIGNMENTS_FILE) as f:
        assignments = json.load(f)
    return responses, assignments


def init_session():
    for key, default in [
        ("app_state", "login"),
        ("student_id", ""),
        ("assigned_responses", []),
        ("current_idx", 0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_login():
    st.title("📐 Math Tutor Evaluation")
    st.markdown(
        """
        Thank you for participating! You'll rate **6 short math tutoring interactions**.

        For each one you'll see:
        - A **math problem**
        - What the **student wrote** so far
        - The **AI tutor's response**

        Then answer 5 quick questions about the response. This takes about **10–15 minutes**.
        """
    )
    st.divider()

    student_id = st.text_input(
        "Enter your Participant ID (given to you by the researcher):",
        placeholder="e.g. S01",
    ).strip()

    if st.button("Start →", type="primary"):
        if not student_id:
            st.error("Please enter your Participant ID.")
            return

        responses, assignments = load_data()

        if student_id not in assignments:
            st.error(f"ID **{student_id}** not found. Please check with the researcher.")
            return

        assigned_ids = assignments[student_id]
        done = already_submitted(student_id)
        remaining = [responses[rid] for rid in assigned_ids if rid in responses and rid not in done]

        if not remaining:
            st.success("You've already completed all your ratings — thank you!")
            return

        st.session_state.student_id = student_id
        st.session_state.assigned_responses = remaining
        st.session_state.current_idx = 0
        st.session_state.app_state = "rating"
        st.rerun()


def page_rating():
    responses = st.session_state.assigned_responses
    idx = st.session_state.current_idx
    total = len(responses)
    r = responses[idx]

    st.progress((idx) / total)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### Response {idx + 1} of {total}")
    with col2:
        state_label = r["student_state_type"].replace("_", " ").title()
        st.caption(f"State: {state_label}")

    st.divider()

    st.markdown("#### 📋 The Problem")
    st.info(r["problem"])

    st.markdown("#### ✏️ What the Student Wrote")
    if r["student_working"]:
        st.warning(r["student_working"])
    else:
        st.warning("_The student hasn't written anything yet._")

    st.markdown("#### 🤖 AI Tutor's Response")
    st.success(r["model_response"])

    st.divider()
    st.markdown("#### 📝 Your Ratings")
    st.caption("Please answer all five questions, then click Submit.")

    with st.form(key=f"form_{idx}_{r['response_id']}"):

        h1 = st.radio(
            "**Q1.  Did the tutor's response give away the answer or solve a key step for the student?**",
            options=["No — it only gave a hint or asked a question",
                     "Yes — it revealed the answer or did a key step"],
            index=None,
            help="If the student could just copy the response and be 'done', choose Yes.",
        )

        st.markdown("")
        h2 = st.select_slider(
            "**Q2.  Did the response guide you to think, or just tell you what to do?**",
            options=[
                "1 — Just told me what to do",
                "2 — Mostly told me",
                "3 — Mix of guiding and telling",
                "4 — Mostly guided me",
                "5 — Made me think for myself",
            ],
        )

        st.markdown("")
        h3 = st.select_slider(
            "**Q3.  Was the hint relevant to exactly where YOU were in solving the problem?**",
            options=[
                "1 — Completely irrelevant to where I was",
                "2 — Mostly irrelevant",
                "3 — Somewhat relevant",
                "4 — Mostly relevant",
                "5 — Perfectly targeted to my situation",
            ],
        )

        st.markdown("")
        h4 = st.radio(
            "**Q4.  After reading this response, did you feel motivated to try the next step yourself?**",
            options=["Yes", "No"],
            index=None,
        )

        st.markdown("")
        h5 = st.radio(
            "**Q5.  After reading this hint, do you think you could figure out what to do next?**",
            options=["Yes — I know what to do next",
                     "Partially — I have a better idea but I'm still unsure",
                     "No — I'm still stuck"],
            index=None,
        )

        submitted = st.form_submit_button("Submit & Continue →", type="primary", use_container_width=True)

        if submitted:
            if h1 is None or h4 is None or h5 is None:
                st.error("Please answer all questions before continuing.")
            else:
                ratings = {
                    "h1_answer_leakage":   1 if h1.startswith("Yes") else 0,
                    "h2_socratic_quality": int(h2[0]),
                    "h3_hint_calibration": int(h3[0]),
                    "h4_engagement":       1 if h4 == "Yes" else 0,
                    "h5_learning_outcome": h5.split(" — ")[0],
                }
                save_rating(st.session_state.student_id, r, ratings)

                if idx + 1 >= total:
                    st.session_state.app_state = "done"
                else:
                    st.session_state.current_idx += 1
                st.rerun()


def page_done():
    st.title("🎉 All done — thank you!")
    st.success("Your ratings have been saved successfully.")
    st.markdown(
        """
        Your feedback helps us understand how well AI math tutors support real students.

        If you have any questions or comments, please reach out to the researcher.
        """
    )
    st.balloons()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Math Tutor Evaluation",
        page_icon="📐",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .stAlert { border-radius: 8px; }
        div[data-testid="stForm"] { border: none; padding: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session()

    if st.session_state.app_state == "login":
        page_login()
    elif st.session_state.app_state == "rating":
        page_rating()
    elif st.session_state.app_state == "done":
        page_done()


if __name__ == "__main__":
    main()
