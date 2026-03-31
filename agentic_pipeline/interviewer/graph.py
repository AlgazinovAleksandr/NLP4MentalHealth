"""LangGraph definition for the mental health intake interview pipeline."""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import (
    check_urgent_node,
    diagnostician_node,
    human_input_node,
    interviewer_node,
)
from .state import InterviewState


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_after_urgent_check(state: InterviewState) -> str:
    """Skip interview entirely if self-harm was flagged."""
    return "diagnose" if state["is_urgent"] else "interview"


def _route_after_interviewer(state: InterviewState) -> str:
    """Continue interview or move to diagnosis."""
    if state["sufficient"] or state["n_asked"] >= state["max_questions"]:
        return "diagnose"
    return "ask_user"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph():
    """
    Graph topology:

        START
          │
          ▼
    check_urgent ──[urgent]──────────────────────────────────┐
          │ [not urgent]                                      │
          ▼                                                   │
      interviewer ──[sufficient or n >= k]──────────────────►│
          │ [need more]                                       │
          ▼                                                   │
      human_input ──────────────────────────────► (loop)     │
                                                             ▼
                                                       diagnostician
                                                             │
                                                            END
    """
    workflow = StateGraph(InterviewState)

    workflow.add_node("check_urgent", check_urgent_node)
    workflow.add_node("interviewer", interviewer_node)
    workflow.add_node("human_input", human_input_node)
    workflow.add_node("diagnostician", diagnostician_node)

    workflow.set_entry_point("check_urgent")

    workflow.add_conditional_edges(
        "check_urgent",
        _route_after_urgent_check,
        {"interview": "interviewer", "diagnose": "diagnostician"},
    )
    workflow.add_conditional_edges(
        "interviewer",
        _route_after_interviewer,
        {"ask_user": "human_input", "diagnose": "diagnostician"},
    )
    workflow.add_edge("human_input", "interviewer")
    workflow.add_edge("diagnostician", END)

    return workflow.compile(checkpointer=MemorySaver())


app = build_graph()
