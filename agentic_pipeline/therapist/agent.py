"""
Therapist agent — open-ended supportive chat following the intake pipeline.

The therapist sends the first message (an unprompted warm opening).
The session continues until the user types a stop command.
"""
from __future__ import annotations

import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from interviewer.llm import get_llm

STOP_COMMANDS = {"/stop", "quit", "exit", "q"}

# Internal trigger sent to the LLM to produce the opening message.
# It is not shown to the user and is removed from the visible history.
_SESSION_START_TRIGGER = "[SESSION_START]"

# Appended to the system prompt to teach the LLM what [SESSION_START] means.
_OPENING_INSTRUCTION = """\

SESSION OPENING:
  When the user sends the internal signal [SESSION_START], respond with a warm, \
brief opening message (2–3 sentences). Acknowledge that the user has taken a positive \
step by being here, and invite them to share whatever is on their mind. \
Do NOT summarise the intake findings or reference the questionnaire.\
"""


def run_therapist(system_prompt: str) -> None:
    """
    Run the therapist chat session.

    The LLM generates the first message autonomously. The loop continues
    until the user types a stop command, at which point the LLM produces a
    warm closing message before exiting.

    Args:
        system_prompt: Dynamic system prompt from build_therapist_prompt().
    """
    llm = get_llm(temperature=0.7)  # higher temp for natural, varied conversation

    full_system = system_prompt + _OPENING_INSTRUCTION

    # Seed the history: system prompt + internal session-start trigger
    messages: list = [
        SystemMessage(content=full_system),
        HumanMessage(content=_SESSION_START_TRIGGER),
    ]

    print("\n" + "=" * 62)
    print("  YOUR SUPPORT SESSION")
    print("=" * 62)
    print("(Type /stop, quit, or exit at any time to end the session)\n")

    # ── Get opening message ────────────────────────────────────────────────────
    print("Please wait...\n")
    opening_response = llm.invoke(messages)
    opening = opening_response.content.strip()

    # Replace internal trigger with the AI reply so history stays clean
    messages.pop()  # remove HumanMessage([SESSION_START])
    messages.append(AIMessage(content=opening))

    print(f"Therapist: {opening}\n")

    # ── Conversation loop ──────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            sys.exit(0)

        if not user_input:
            continue

        # ── Stop command → warm closing ────────────────────────────────────────
        if user_input.lower() in STOP_COMMANDS:
            messages.append(HumanMessage(content=user_input))
            closing_instruction = (
                "The user has chosen to end the session. "
                "Write a warm, brief closing message (2–3 sentences). "
                "Affirm their courage in reaching out today. "
                "Encourage them to return whenever they need support. "
                "Do not ask any further questions."
            )
            # Append as a system follow-up so the LLM treats it as an instruction
            closing_messages = messages + [SystemMessage(content=closing_instruction)]
            closing = llm.invoke(closing_messages).content.strip()

            print(f"\nTherapist: {closing}\n")
            print("=" * 62)
            print("Session ended. Take care of yourself.")
            print("=" * 62 + "\n")
            return

        # ── Normal turn ────────────────────────────────────────────────────────
        messages.append(HumanMessage(content=user_input))

        print("\nTherapist is thinking...\n")
        response = llm.invoke(messages)
        reply = response.content.strip()
        messages.append(AIMessage(content=reply))

        print(f"Therapist: {reply}\n")
