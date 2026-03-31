"""
Interactive CLI for the mental health intake interview pipeline.

Usage:
    python main.py                  # runs with built-in sample questionnaire
    python main.py --persona 5      # runs with persona test_005 from the sample CSV
    python main.py --max-questions 3

The pipeline:
    1. Registration questionnaire answers are loaded (here: from a sample dict or CSV).
    2. LLM #1 (Interviewer) asks up to k targeted follow-up questions.
    3. LLM #2 (Diagnostician) synthesises everything into a probabilistic assessment.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from pathlib import Path
from typing import Optional

from langgraph.types import Command

from interviewer.graph import app
from interviewer.state import InterviewState

# ── Crisis content ────────────────────────────────────────────────────────────

CRISIS_MESSAGE = """
TODO
"""

# ── Sample questionnaires ─────────────────────────────────────────────────────

SAMPLE_QUESTIONNAIRE = {
    # test_005: concerned, depression profile (female 29, unemployed)
    "q_gender": "female",
    "q_age": 29,
    "q_occupation": "unemployed",
    "q_has_mental_concern": "strong_concern",
    "q_concern_areas": ["depression", "sleep"],
    "q_phq2_1": 3,
    "q_phq2_2": 3,
    "q_gad2_1": 1,
    "q_gad2_2": 0,
    "q_daily_impact": "significantly",
    "q_duration": "more_6m",
    "q_prior_help": "in_past",
    "q_diagnosis_known": "yes",
    "q_app_goal": ["vent", "coping_tips", "find_specialist"],
}


def load_persona(persona_index: int) -> dict:
    """Load a questionnaire from questionnaire_sample_1.csv by row index (1-based)."""
    csv_path = (
        Path(__file__).parent
        / "synthetic_questionnaire_generation"
        / "questionnaire_sample_1.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Sample CSV not found at {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not (1 <= persona_index <= len(rows)):
        raise ValueError(f"Persona index must be 1–{len(rows)}, got {persona_index}")

    row = rows[persona_index - 1]
    print(f"\nLoaded persona: {row['persona_id']}")
    print(f"Description  : {row['persona_description']}")
    print(f"Expected flag: {row['expected_flag']}  (score {row['expected_total_score']})\n")

    return json.loads(row["answers_json"])


# ── Interrupt helpers ─────────────────────────────────────────────────────────

def _get_pending_interrupt(config: dict) -> Optional[dict]:
    """
    Return the value from the first pending interrupt, or None if the graph finished.
    Uses get_state() which works reliably across LangGraph versions.
    """
    snapshot = app.get_state(config)
    if not snapshot.next:
        return None  # graph completed
    for task in snapshot.tasks:
        interrupts = getattr(task, "interrupts", None) or []
        if interrupts:
            return interrupts[0].value
    return None


# ── Core interview runner ─────────────────────────────────────────────────────

def run_interview(
    questionnaire: dict,
    max_questions: int = 5,
    thread_id: Optional[str] = None,
) -> dict:
    """
    Run the full interview pipeline for a given questionnaire.
    Blocks on stdin for each follow-up question.
    Returns the raw diagnosis dict from the diagnostician LLM.
    """
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: InterviewState = {
        "questionnaire": questionnaire,
        "history": [],
        "n_asked": 0,
        "max_questions": max_questions,
        "current_question": None,
        "current_topic": None,
        "sufficient": False,
        "is_urgent": False,
        "diagnosis": None,
    }

    print("\n" + "=" * 62)
    print("  MENTAL HEALTH INTAKE INTERVIEW")
    print("=" * 62)
    print("I'd like to ask you a few questions to better understand")
    print("how you've been feeling. Take your time with each answer.")
    print("Type 'quit' at any time to exit.\n")

    # Kick off the graph (runs check_urgent + possibly first interviewer turn)
    print("Please wait while the agent prepares your first question...")
    app.invoke(initial_state, config)

    while True:
        interrupt_val = _get_pending_interrupt(config)

        if interrupt_val is None:
            # Graph finished — pull final state
            snapshot = app.get_state(config)
            return snapshot.values.get("diagnosis") or {}

        question = interrupt_val["question"]
        progress = interrupt_val.get("progress", "")

        print(f"[{progress}] {question}")
        try:
            answer = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            sys.exit(0)

        if answer.lower() in {"quit", "exit", "q"}:
            print("\nSession ended.")
            sys.exit(0)

        if not answer:
            answer = "(no response)"

        print(f"\nRecorded: \"{answer}\"")
        print("Please wait...")

        # Resume the graph with the user's answer
        app.invoke(Command(resume=answer), config)


# ── Display ───────────────────────────────────────────────────────────────────

def display_diagnosis(diagnosis: dict, is_urgent: bool = False) -> None:
    if is_urgent:
        print(CRISIS_MESSAGE)
        # Diagnostician still ran, show internal result without user-facing block
        if diagnosis:
            _print_internal(diagnosis)
        return

    print("\n" + "=" * 62)
    print("  ASSESSMENT COMPLETE")
    print("=" * 62)

    summary = diagnosis.get("user_facing_summary", "")
    if summary:
        print(f"\n{summary}\n")

    _print_internal(diagnosis)


def _print_internal(diagnosis: dict) -> None:
    print("─" * 62)
    print("SCREENING RESULTS  (internal — not shown to users in prod)")
    print("─" * 62)

    differential = diagnosis.get("differential", {})
    if differential:
        print("\nProbability each condition is present:")
        for condition, prob in sorted(differential.items(), key=lambda x: -x[1]):
            filled = int(prob * 25)
            bar = "█" * filled + "░" * (25 - filled)
            print(f"  {condition:<38} {prob:.2f}  {bar}")

    print(f"\nPrimary hypothesis : {diagnosis.get('primary_hypothesis', 'N/A')}")
    print(f"Confidence         : {diagnosis.get('confidence', 'N/A')}")

    if indicators := diagnosis.get("key_indicators"):
        print("\nKey indicators:")
        for ind in indicators:
            print(f"  • {ind}")

    referral = diagnosis.get("referral_recommended", False)
    urgency = diagnosis.get("referral_urgency")
    referral_str = "Yes" if referral else "No"
    if referral and urgency:
        referral_str += f" — {urgency}"
    print(f"\nReferral recommended: {referral_str}")
    print("=" * 62 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mental health intake interview agent")
    parser.add_argument(
        "--persona", type=int, default=None,
        help="Row index (1–15) from questionnaire_sample_1.csv to use as input",
    )
    parser.add_argument(
        "--max-questions", type=int, default=5,
        help="Maximum follow-up questions the interviewer may ask (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.persona is not None:
        questionnaire = load_persona(args.persona)
    else:
        questionnaire = SAMPLE_QUESTIONNAIRE

    diagnosis = run_interview(questionnaire, max_questions=args.max_questions)

    is_urgent = "self_harm" in questionnaire.get("q_concern_areas", [])
    display_diagnosis(diagnosis, is_urgent=is_urgent)
