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

from interviewer.runner import run_interview
from synthetic_questionnaire_generation import derive_flag
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
    Delegates to derive_flag() in questionnaire_scoring.py — single source of truth.

    Returns:
        "urgent" | "relaxed" | "concerned"
    """
    return derive_flag(questionnaire)


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
