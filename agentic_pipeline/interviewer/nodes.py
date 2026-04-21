"""Graph node functions for the interview pipeline."""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from .llm import get_llm
from .prompts import (
    DIAGNOSTICIAN_HUMAN,
    DIAGNOSTICIAN_SYSTEM,
    INTERVIEWER_HUMAN,
    INTERVIEWER_SYSTEM,
)
from .schemas import DiagnosisOutput, InterviewerDecision
from .state import InterviewState, QAPair

_llm = get_llm()

# ── Questionnaire / history formatters ───────────────────────────────────────

_GENDER = {"male": "Male", "female": "Female", "prefer_not": "Prefer not to say"}
_OCCUPATION = {
    "student": "Student", "employed": "Employed",
    "unemployed": "Unemployed", "retired": "Retired", "other": "Other",
}
_CONCERN = {
    "fine": "Fine — just curious",
    "low_concern": "Mild unease (not sure if worth worrying about)",
    "moderate_concern": "Noticeable changes in mood or behaviour",
    "strong_concern": "Significant psychological difficulties",
}
_IMPACT = {
    "not_at_all": "Not at all",
    "slightly": "Slightly",
    "moderately": "Moderately — noticeable but manageable",
    "significantly": "Significantly — affects work / study / relationships",
    "severely": "Severely — struggling with basic tasks",
}
_DURATION = {
    "less_2w": "< 2 weeks", "2w_1m": "2 weeks – 1 month",
    "1m_6m": "1 – 6 months", "more_6m": "> 6 months",
}
_PRIOR = {
    "never": "Never", "in_past": "In the past",
    "currently": "Currently in therapy", "prefer_not": "Prefer not to say",
}


def _format_questionnaire(q: dict) -> str:
    lines: list[str] = []

    lines.append(f"Gender     : {_GENDER.get(q.get('q_gender', ''), 'N/A')}")
    lines.append(f"Age        : {q.get('q_age', 'N/A')}")
    lines.append(f"Occupation : {_OCCUPATION.get(q.get('q_occupation', ''), 'N/A')}")
    lines.append(f"Self-assessed state : {_CONCERN.get(q.get('q_has_mental_concern', ''), 'N/A')}")

    if areas := q.get("q_concern_areas"):
        lines.append(f"Concern areas : {', '.join(areas)}")
        if "self_harm" in areas:
            lines.append("⚠️  URGENT FLAG: self-harm ideation reported")

    phq = q.get("q_phq2_1", 0) + q.get("q_phq2_2", 0)
    gad = q.get("q_gad2_1", 0) + q.get("q_gad2_2", 0)
    lines.append(
        f"PHQ-2 score : {phq}/6  {'[screen positive ≥ 3]' if phq >= 3 else '[below threshold]'}"
    )
    lines.append(
        f"GAD-2 score : {gad}/6  {'[screen positive ≥ 3]' if gad >= 3 else '[below threshold]'}"
    )
    lines.append(f"Daily impact : {_IMPACT.get(q.get('q_daily_impact', ''), 'N/A')}")

    if dur := q.get("q_duration"):
        lines.append(f"Duration : {_DURATION.get(dur, dur)}")

    lines.append(f"Prior help : {_PRIOR.get(q.get('q_prior_help', ''), 'N/A')}")

    if diag := q.get("q_diagnosis_known"):
        lines.append(f"Prior official diagnosis : {diag}")

    if goals := q.get("q_app_goal"):
        lines.append(f"User goals : {', '.join(goals)}")

    return "\n".join(lines)


def _format_history(history: list[QAPair]) -> str:
    if not history:
        return "(No follow-up questions asked yet)"
    lines: list[str] = []
    for i, pair in enumerate(history, 1):
        lines.append(f"Q{i} [{pair['topic']}]: {pair['question']}")
        lines.append(f"A{i}: {pair['answer']}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    """Parse LLM response to dict, stripping markdown fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


def _call_llm(system: str, human: str, schema=None) -> dict:
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    response = _llm.invoke(messages)
    data = _parse_json(response.content)
    if schema is not None:
        # Validate against the Pydantic schema — raises ValidationError on bad LLM output
        data = schema.model_validate(data).model_dump()
    return data


# ── Graph nodes ───────────────────────────────────────────────────────────────

def check_urgent_node(state: InterviewState) -> dict:
    """
    Pre-interview safety check.
    If self-harm is flagged, mark urgent and skip straight to diagnosis.
    """
    concern_areas = state["questionnaire"].get("q_concern_areas", [])
    is_urgent = "self_harm" in concern_areas
    return {
        "is_urgent": is_urgent,
        # Mark sufficient so the router sends us directly to diagnostician
        "sufficient": is_urgent,
    }


def interviewer_node(state: InterviewState) -> dict:
    """
    LLM #1 — Interviewer.
    Reads current state and decides: ask one more question, or declare sufficiency.
    """
    n_asked = state["n_asked"]
    max_q = state["max_questions"]
    remaining = max_q - n_asked

    system = INTERVIEWER_SYSTEM.format(remaining=remaining, max_questions=max_q)
    human = INTERVIEWER_HUMAN.format(
        questionnaire_summary=_format_questionnaire(state["questionnaire"]),
        history_summary=_format_history(state["history"]),
        n_asked=n_asked,
        max_questions=max_q,
    )

    result = _call_llm(system, human, schema=InterviewerDecision)
    sufficient = bool(result.get("sufficient_information", False))

    return {
        # Explicitly null when sufficient so the state is clean
        "current_question": None if sufficient else result.get("next_question"),
        "current_topic": result.get("topic_targeted", "general"),
        "sufficient": sufficient,
    }


def human_input_node(state: InterviewState) -> dict:
    """
    Human-in-the-loop node.
    Pauses graph execution via interrupt() and waits for the user's typed answer.
    On resume, appends the Q&A pair to history and increments n_asked.
    """
    # interrupt() suspends the graph here. The value passed to it is surfaced to
    # the caller via get_state().tasks[i].interrupts[j].value.
    # The graph re-enters this node on resume and interrupt() returns the answer.
    answer: str = interrupt({
        "question": state["current_question"],
        "topic": state["current_topic"],
        "progress": f"{state['n_asked'] + 1}/{state['max_questions']}",
    })

    new_pair: QAPair = {
        "question": state["current_question"],
        "answer": answer,
        "topic": state["current_topic"],
    }
    return {
        "history": state["history"] + [new_pair],
        "n_asked": state["n_asked"] + 1,
    }


def diagnostician_node(state: InterviewState) -> dict:
    """
    LLM #2 — Diagnostician.
    Synthesises questionnaire + full interview transcript into a probabilistic assessment.
    """
    human = DIAGNOSTICIAN_HUMAN.format(
        questionnaire_summary=_format_questionnaire(state["questionnaire"]),
        history_summary=_format_history(state["history"]),
    )
    result = _call_llm(DIAGNOSTICIAN_SYSTEM, human, schema=DiagnosisOutput)
    return {"diagnosis": result}
