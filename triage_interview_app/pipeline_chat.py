from __future__ import annotations

"""
CHAT_PIPELINE — therapist LLM integration.

The system prompt is expected as messages[0] (role="system"), built from the
BERT triage class and interview result by build_system_prompt() in interview_runner.py.
"""

import sys
from pathlib import Path

_AGENTIC = Path(__file__).parent.parent / "agentic_pipeline"
if str(_AGENTIC) not in sys.path:
    sys.path.insert(0, str(_AGENTIC))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_AGENTIC / ".env")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from interviewer.llm import get_llm  # noqa: E402

_SESSION_START_TRIGGER = "[SESSION_START]"
_OPENING_INSTRUCTION = """

SESSION OPENING:
  When the user sends the internal signal [SESSION_START], respond with a warm, \
brief opening message (2–3 sentences). Acknowledge that the user has taken a positive \
step by being here, and invite them to share whatever is on their mind. \
Do NOT summarise the intake findings or reference the questionnaire."""

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(temperature=0.7)
    return _llm


def generate_opening(system_prompt: str) -> str:
    """Generate the therapist's first message using the [SESSION_START] trigger."""
    lc_messages = [
        SystemMessage(content=system_prompt + _OPENING_INSTRUCTION),
        HumanMessage(content=_SESSION_START_TRIGGER),
    ]
    return _get_llm().invoke(lc_messages).content.strip()


def run_pipeline(
    messages: list[dict[str, str]],
    latest_user_message: str,
) -> str:
    """
    Call the therapist LLM for one conversation turn.

    Args:
        messages: Full history including system prompt as messages[0].
        latest_user_message: Newest user turn (last message in history).
    """
    lc_messages = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    return _get_llm().invoke(lc_messages).content.strip()
