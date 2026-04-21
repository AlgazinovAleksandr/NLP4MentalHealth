"""
Core interview loop — importable by both main.py (CLI) and run_pipeline.py (full pipeline).

Separated from main.py so that run_pipeline.py does not have to import from a CLI
entry point that contains argparse and sys.exit() calls.
"""
from __future__ import annotations

import sys
import uuid
from typing import Optional

from langgraph.types import Command

from .graph import app
from .state import InterviewState


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
