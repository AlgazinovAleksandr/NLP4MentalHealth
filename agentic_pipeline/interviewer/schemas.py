"""Pydantic schemas for LLM output validation and documentation."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

MENTAL_STATE_CONDITIONS = [
    "Generalized Anxiety Disorder",
    "Major Depressive Disorder",
    "Social Anxiety Disorder",
    "Post-Traumatic Stress Disorder",
    "ADHD",
    "Schizophrenia",
    "Panic Disorder",
    "OCD",
    "Bipolar Disorder",
    "Adjustment Disorder",
    "Healthy",
]


class InterviewerDecision(BaseModel):
    """Expected JSON output from the interviewer LLM each turn."""

    next_question: Optional[str] = Field(
        default=None,
        description=(
            "The single warm, conversational question to ask the user. "
            "null when sufficient_information is true."
        ),
    )
    topic_targeted: str = Field(
        description="Symptom dimension this question probes, e.g. 'sleep', 'anhedonia', 'avoidance'.",
    )
    rationale: str = Field(
        description="1–2 sentences of internal reasoning: why this question at this stage.",
    )
    sufficient_information: bool = Field(
        description="True when enough signal has been collected to proceed to assessment.",
    )


class DiagnosisOutput(BaseModel):
    """Expected JSON output from the diagnostician LLM."""

    differential: dict[str, float] = Field(
        description=(
            "Estimated probability (0.0–1.0) that each condition is present. "
            f"Must include all 11 keys: {MENTAL_STATE_CONDITIONS}. "
            "Conditions are NOT mutually exclusive."
        ),
    )
    primary_hypothesis: str = Field(
        description="Condition with the highest probability, or 'Healthy'.",
    )
    confidence: Literal["low", "moderate", "high"] = Field(
        description="Overall confidence in the assessment given available evidence.",
    )
    key_indicators: list[str] = Field(
        description="3–6 specific signals from the data that drove the assessment.",
    )
    referral_recommended: bool
    referral_urgency: Optional[Literal["routine", "soon", "urgent"]] = Field(
        default=None,
        description="How urgently referral is needed. null when referral_recommended is false.",
    )
    user_facing_summary: str = Field(
        description=(
            "2–3 warm, empathetic sentences summarising what themes emerged. "
            "Never names a specific diagnosis."
        ),
    )
