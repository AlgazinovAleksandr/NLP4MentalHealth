"""
Streamlit-compatible adapter for the LangGraph interview pipeline.

Exposes stateless per-turn functions (no stdin/stdout) that store graph state
in a LangGraph thread identified by a UUID kept in Streamlit session_state.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

_AGENTIC = Path(__file__).parent.parent / "agentic_pipeline"
if str(_AGENTIC) not in sys.path:
    sys.path.insert(0, str(_AGENTIC))

from dotenv import load_dotenv

load_dotenv(_AGENTIC / ".env")

from langgraph.types import Command  # noqa: E402

from interviewer.graph import app  # noqa: E402
from interviewer.state import InterviewState  # noqa: E402
from therapist.dynamic_prompt import build_therapist_prompt  # noqa: E402


def start_interview(
    questionnaire: dict,
    max_questions: int = 5,
) -> tuple[str | None, dict, bool, dict]:
    """
    Kick off the LangGraph interview graph.

    Returns:
        (question, config, is_done, diagnosis)
        question  — first interview question, or None if completed immediately
        config    — LangGraph thread config; store in session_state for resume_interview
        is_done   — True when no questions are needed (e.g. urgent flag)
        diagnosis — final diagnosis dict (non-empty only when is_done is True)
    """
    thread_id = str(uuid.uuid4())
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

    app.invoke(initial_state, config)
    question, is_done, diagnosis = _read_state(config)
    return question, config, is_done, diagnosis


def resume_interview(answer: str, config: dict) -> tuple[str | None, bool, dict]:
    """
    Resume the graph with the user's answer.

    Returns:
        (question, is_done, diagnosis)
    """
    app.invoke(Command(resume=answer), config)
    return _read_state(config)


def _read_state(config: dict) -> tuple[str | None, bool, dict]:
    snapshot = app.get_state(config)
    if not snapshot.next:
        diagnosis = snapshot.values.get("diagnosis") or {}
        return None, True, diagnosis
    for task in snapshot.tasks:
        interrupts = getattr(task, "interrupts", None) or []
        if interrupts:
            question = interrupts[0].value["question"]
            return question, False, {}
    return None, True, {}


def build_system_prompt(bert_class: str, interview_result: dict, questionnaire: dict) -> str:
    return build_therapist_prompt(bert_class, interview_result, questionnaire)
