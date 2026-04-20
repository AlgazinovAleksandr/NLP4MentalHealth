"""
Full end-to-end pipeline:

  1. Load registration questionnaire from JSON (QUESTIONNAIRE_JSON_PATH)
  2. BERT classifier predicts triage class  (BERT_MODEL_PATH)
  3. Interviewer agent gathers follow-up signal
  4. Dynamic system prompt is built from class + interview result
  5. Therapist agent conducts open-ended supportive chat

Usage:
    python run_pipeline.py
    python run_pipeline.py --questionnaire path/to/answers.json
    python run_pipeline.py --max-questions 3

Placeholders to replace before production use:
    QUESTIONNAIRE_JSON_PATH  — path to the user's registration JSON
    BERT_MODEL_PATH          — path to the trained BERT classifier
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from main import run_interview  # reuse the existing interview loop + display logic
from therapist import build_therapist_prompt, run_therapist

# ── Placeholders ──────────────────────────────────────────────────────────────

QUESTIONNAIRE_JSON_PATH = "path/to/questionnaire.json"   # TODO: replace with actual path
BERT_MODEL_PATH = "path/to/bert_classifier"              # TODO: replace with actual path


# ── BERT classifier (placeholder) ────────────────────────────────────────────

def classify_with_bert(questionnaire: dict, model_path: str) -> str:
    """
    Classify the registration questionnaire into a triage class using the BERT model.

    TODO: Load the BERT model from `model_path` and run inference on `questionnaire`.

    Returns:
        One of: "relaxed" | "concerned" | "urgent"

    Raises:
        NotImplementedError: Until the BERT model is integrated.
    """
    raise NotImplementedError(
        "BERT classifier is not yet integrated.\n"
        "Set BERT_MODEL_PATH and implement classify_with_bert(), "
        "or run with --skip-bert to derive the class from the scoring rules instead."
    )


def _fallback_classify(questionnaire: dict) -> str:
    """
    Rule-based fallback used when --skip-bert is set.
    Mirrors the scoring logic in questionnaire_scoring.py.

    Returns:
        "urgent" | "relaxed" | "concerned"
    """
    concern_areas = questionnaire.get("q_concern_areas", [])
    if "self_harm" in concern_areas:
        return "urgent"

    concern_level_score = {
        "fine": 0, "low_concern": 1, "moderate_concern": 2, "strong_concern": 3,
    }.get(questionnaire.get("q_has_mental_concern", "fine"), 0) * 3

    phq2 = questionnaire.get("q_phq2_1", 0) + questionnaire.get("q_phq2_2", 0)
    gad2 = questionnaire.get("q_gad2_1", 0) + questionnaire.get("q_gad2_2", 0)
    phq_score = 3 if phq2 >= 3 else 0
    gad_score = 3 if gad2 >= 3 else 0

    daily_impact_score = {
        "not_at_all": 0, "slightly": 1, "moderately": 2,
        "significantly": 3, "severely": 4,
    }.get(questionnaire.get("q_daily_impact", "not_at_all"), 0)

    prior_help_score = 1 if questionnaire.get("q_prior_help") in {"in_past", "currently"} else 0

    total = concern_level_score + phq_score + gad_score + daily_impact_score + prior_help_score
    return "relaxed" if total <= 2 else "concerned"


# ── Questionnaire loader ───────────────────────────────────────────────────────

def load_questionnaire(json_path: str) -> dict:
    """Load registration questionnaire answers from a JSON file."""
    path = Path(json_path)
    if not path.exists():
        print(f"[ERROR] Questionnaire file not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_full_pipeline(
    questionnaire: dict,
    bert_class: str,
    max_questions: int = 5,
) -> None:
    """
    Execute the full pipeline for a given questionnaire and pre-computed triage class.

    Steps:
      1. Run the interviewer agent (follow-up questions + diagnostician synthesis).
      2. Build the dynamic therapist system prompt.
      3. Start the therapist chat session.
    """
    # ── Step 1: Interviewer + diagnostician ───────────────────────────────────
    print(f"\n[Pipeline] Triage class: {bert_class.upper()}")
    print("[Pipeline] Starting intake interview...\n")

    interview_result = run_interview(questionnaire, max_questions=max_questions)

    # ── Step 2: Build dynamic prompt ──────────────────────────────────────────
    system_prompt = build_therapist_prompt(
        bert_class=bert_class,
        interview_result=interview_result,
        questionnaire=questionnaire,
    )

    # ── Step 3: Therapist chat ────────────────────────────────────────────────
    print("\n[Pipeline] Intake complete. Starting support session...\n")
    run_therapist(system_prompt)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end mental health support pipeline"
    )
    parser.add_argument(
        "--questionnaire",
        type=str,
        default=QUESTIONNAIRE_JSON_PATH,
        help="Path to the registration questionnaire JSON file",
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default=BERT_MODEL_PATH,
        help="Path to the BERT classifier model",
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="Skip BERT and derive triage class from scoring rules (for development)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        help="Maximum follow-up questions the interviewer may ask (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    questionnaire = load_questionnaire(args.questionnaire)

    if args.skip_bert:
        bert_class = _fallback_classify(questionnaire)
        print(f"[Pipeline] BERT skipped — rule-based class: {bert_class}")
    else:
        bert_class = classify_with_bert(questionnaire, args.bert_model)

    run_full_pipeline(
        questionnaire=questionnaire,
        bert_class=bert_class,
        max_questions=args.max_questions,
    )
