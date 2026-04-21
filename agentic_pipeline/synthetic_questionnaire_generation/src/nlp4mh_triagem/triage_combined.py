from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

from .questionnaire_validate import validate_answers

Label = Literal["relaxed", "concerned", "urgent"]
TextMode = Literal["off", "lexicon"]

_RANK: dict[str, int] = {"relaxed": 0, "concerned": 1, "urgent": 2}


def _max_label(a: str, b: str) -> str:
    order = ("relaxed", "concerned", "urgent")
    return order[max(_RANK[a], _RANK[b])]


def _normalize_text(s: str) -> str:
    t = unicodedata.normalize("NFKC", s).strip().casefold()
    t = re.sub(r"\s+", " ", t)
    return t


_CRISIS_LEXICON: tuple[str, ...] = (
    "хочу умереть",
    "хочу покончить",
    "покончить с собой",
    "суицид",
    "самоубийств",
    "повешусь",
    "повеситься",
    "зарежу себя",
    "не хочу жить",
    "лучше бы меня не было",
    "самоповрежден",
    "порезал вен",
    "порезала вен",
    "kill myself",
    "want to die",
    "end my life",
    "suicide",
    "self harm",
    "self-harm",
    "hurt myself",
    "hang myself",
)


def lexicon_crisis_hit(free_text: str) -> tuple[bool, list[str]]:
    if not free_text or not free_text.strip():
        return False, []
    n = _normalize_text(free_text)
    hits: list[str] = []
    for needle in _CRISIS_LEXICON:
        if needle in n:
            hits.append(needle)
    return (len(hits) > 0, hits)


def resolve_text_mode(explicit: TextMode | None) -> TextMode:
    if explicit is not None:
        return explicit
    raw = (
        os.environ.get("TRIAGE_FREE_TEXT_MODE") or "lexicon"
    ).strip().lower()
    if raw in ("0", "false", "no", "off", "none"):
        return "off"
    return "lexicon"


@dataclass
class CombinedTriageResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    structural_score: int | None = None
    structural_flag: Label | None = None
    text_mode: TextMode = "lexicon"
    text_crisis_hit: bool = False
    text_matched_patterns: list[str] = field(default_factory=list)
    final_flag: Label | None = None
    notes: str = ""


def combined_triage(
    answers: dict[str, Any],
    free_text: str | None = None,
    *,
    text_mode: TextMode | None = None,
) -> CombinedTriageResult:
    mode = resolve_text_mode(text_mode)
    vr = validate_answers(answers)
    if not vr.ok or vr.computed_flag is None:
        return CombinedTriageResult(
            ok=False,
            errors=list(vr.errors),
            text_mode=mode,
            notes="invalid_answers",
        )

    structural: Label = vr.computed_flag  # type: ignore[assignment]
    text_hit = False
    matched: list[str] = []
    if mode == "lexicon" and free_text:
        text_hit, matched = lexicon_crisis_hit(free_text)

    text_as_flag: Label = "urgent" if text_hit else structural
    final = _max_label(structural, text_as_flag)

    notes_parts = [f"structural={structural}"]
    if mode == "off":
        notes_parts.append("free_text_ignored")
    elif free_text and free_text.strip():
        notes_parts.append(f"free_text_mode={mode}")
    if text_hit:
        notes_parts.append("lexicon_escalated_to_urgent")
    else:
        notes_parts.append("no_text_escalation")

    return CombinedTriageResult(
        ok=True,
        structural_score=vr.computed_score,
        structural_flag=structural,
        text_mode=mode,
        text_crisis_hit=text_hit,
        text_matched_patterns=matched,
        final_flag=final,
        notes="; ".join(notes_parts),
    )


def suggested_actions(final_flag: Label | None) -> list[str]:
    if final_flag is None:
        return ["fix_validation_errors"]
    if final_flag == "urgent":
        return [
            "show_crisis_resources",
            "avoid_deep_therapeutic_chat",
            "prefer_human_escalation_if_configured",
        ]
    if final_flag == "concerned":
        return [
            "gentle_onboarding",
            "suggest_professional_support",
            "monitor_follow_up",
        ]
    return ["standard_flow", "optional_education_content"]


if __name__ == "__main__":
    # Smoke checks (no network)
    minimal = {
        "q_gender": "prefer_not",
        "q_age": 30,
        "q_occupation": "employed",
        "q_has_mental_concern": "fine",
        "q_phq2_1": 0,
        "q_phq2_2": 0,
        "q_gad2_1": 0,
        "q_gad2_2": 0,
        "q_daily_impact": "not_at_all",
        "q_prior_help": "never",
        "q_app_goal": ["just_curious"],
    }
    r0 = combined_triage(minimal, "всё ок", text_mode="lexicon")
    assert r0.ok and r0.final_flag == "relaxed"
    dark_humor = "иногда хочу умереть но шучу"
    r1 = combined_triage(minimal, dark_humor, text_mode="lexicon")
    assert r1.ok and r1.final_flag == "urgent" and r1.text_crisis_hit
    r2 = combined_triage(minimal, dark_humor, text_mode="off")
    assert r2.final_flag == "relaxed"
    print("triage_combined OK", r0.final_flag, r1.final_flag, r2.final_flag)
