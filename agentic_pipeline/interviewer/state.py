from __future__ import annotations

from typing import Optional, TypedDict


class QAPair(TypedDict):
    question: str
    answer: str
    topic: str


class InterviewState(TypedDict):
    questionnaire: dict            # raw answers from the registration questionnaire
    history: list[QAPair]          # accumulated follow-up Q&A pairs
    n_asked: int                   # number of follow-up questions asked so far
    max_questions: int             # hard upper limit k (default 5)
    current_question: Optional[str]   # question currently pending user input
    current_topic: Optional[str]      # symptom dimension being probed
    sufficient: bool               # interviewer LLM declared enough signal
    is_urgent: bool                # self-harm flag detected — bypasses interview
    diagnosis: Optional[dict]      # final probabilistic assessment output
