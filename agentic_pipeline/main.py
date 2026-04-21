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
from pathlib import Path

from interviewer.runner import run_interview

# ── Crisis content ────────────────────────────────────────────────────────────
#
# CRISIS_MESSAGE is shown immediately to any user whose registration questionnaire
# flagged self-harm ideation (q_concern_areas contains "self_harm").
# It is displayed BEFORE the internal assessment output and is the first thing the
# user sees — so it must prioritise safety resources, not clinical information.
#
# TODO: Review this text with a qualified mental health professional before going live.
# TODO: Localise crisis line numbers to the user's country/region — they vary widely.

CRISIS_MESSAGE = """
Thank you for trusting us with something so personal. What you're feeling matters!
If you are in crisis or feel you may hurt yourself, please reach out to people who will help you immediately.
A real person is ready to listen. Reaching out is an act of courage.
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


# ── Display ───────────────────────────────────────────────────────────────────

def display_diagnosis(diagnosis: dict, is_urgent: bool = False) -> None:
    if is_urgent:
        print(CRISIS_MESSAGE)
        # Diagnostician still ran, show internal result without user-facing block
        if diagnosis:
            _print_internal(diagnosis)
        return

    print("ASSESSMENT COMPLETE")

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
