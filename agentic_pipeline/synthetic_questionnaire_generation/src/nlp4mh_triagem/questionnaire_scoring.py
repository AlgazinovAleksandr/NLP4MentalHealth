from __future__ import annotations

from typing import Any

CONCERN_SCORE = {"fine": 0, "low_concern": 1, "moderate_concern": 2, "strong_concern": 3}

IMPACT_SCORE = {
    "not_at_all": 0,
    "slightly": 1,
    "moderately": 2,
    "significantly": 3,
    "severely": 4,
}

PRIOR_SCORE = {"never": 0, "in_past": 1, "currently": 1, "prefer_not": 0}


def compute_total_score(answers: dict[str, Any]) -> int:
    h = answers.get("q_has_mental_concern")
    part = CONCERN_SCORE[h] * 3

    phq = int(answers["q_phq2_1"]) + int(answers["q_phq2_2"])
    if phq >= 3:
        part += 3

    gad = int(answers["q_gad2_1"]) + int(answers["q_gad2_2"])
    if gad >= 3:
        part += 3

    di = answers.get("q_daily_impact")
    part += IMPACT_SCORE[di]

    pr = answers.get("q_prior_help")
    part += PRIOR_SCORE[pr]

    return part


def derive_flag(answers: dict[str, Any], total_score: int | None = None) -> str:
    areas = answers.get("q_concern_areas")
    if isinstance(areas, list) and "self_harm" in areas:
        return "urgent"
    if total_score is None:
        total_score = compute_total_score(answers)
    if total_score <= 2:
        return "relaxed"
    return "concerned"
