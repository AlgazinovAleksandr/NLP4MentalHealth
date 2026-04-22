from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .questionnaire_scoring import compute_total_score, derive_flag

GENDER = {"male", "female", "prefer_not"}
OCCUPATION = {"student", "employed", "unemployed", "retired", "other"}
HAS_CONCERN = {"fine", "low_concern", "moderate_concern", "strong_concern"}
CONCERN_AREAS = {
    "anxiety",
    "depression",
    "sleep",
    "concentration",
    "panic",
    "ocd",
    "mood_swings",
    "relationships",
    "self_harm",
    "other",
}
DAILY_IMPACT = {"not_at_all", "slightly", "moderately", "significantly", "severely"}
DURATION = {"less_2w", "2w_1m", "1m_6m", "more_6m"}
PRIOR_HELP = {"never", "in_past", "currently", "prefer_not"}
DIAGNOSIS = {"yes", "no", "prefer_not"}
APP_GOALS = {
    "understand_myself",
    "vent",
    "coping_tips",
    "info",
    "find_specialist",
    "just_curious",
}

NOT_FINE = {"low_concern", "moderate_concern", "strong_concern"}
IMPACT_NEEDS_DURATION = {"moderately", "significantly", "severely"}
PRIOR_NEEDS_DIAGNOSIS = {"in_past", "currently"}

REQUIRED_KEYS = {
    "q_gender",
    "q_age",
    "q_occupation",
    "q_has_mental_concern",
    "q_phq2_1",
    "q_phq2_2",
    "q_gad2_1",
    "q_gad2_2",
    "q_daily_impact",
    "q_prior_help",
    "q_app_goal",
}


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    computed_score: int | None = None
    computed_flag: str | None = None


def _scale(v: Any, qid: str, errors: list[str]) -> bool:
    if not isinstance(v, int) or v < 0 or v > 3:
        errors.append(f"{qid} must be int 0..3, got {v!r}")
        return False
    return True


def validate_answers(answers: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []

    for k in REQUIRED_KEYS:
        if k not in answers:
            errors.append(f"missing required key {k}")

    unknown = set(answers) - REQUIRED_KEYS - {"q_concern_areas", "q_duration", "q_diagnosis_known"}
    for k in sorted(unknown):
        errors.append(f"unexpected key {k}")

    if errors and any("missing" in e for e in errors):
        return ValidationResult(ok=False, errors=errors)

    g = answers.get("q_gender")
    if g not in GENDER:
        errors.append(f"q_gender invalid: {g!r}")

    age = answers.get("q_age")
    if not isinstance(age, int) or age < 13 or age > 100:
        errors.append(f"q_age must be int 13..100, got {age!r}")

    occ = answers.get("q_occupation")
    if occ not in OCCUPATION:
        errors.append(f"q_occupation invalid: {occ!r}")

    h = answers.get("q_has_mental_concern")
    if h not in HAS_CONCERN:
        errors.append(f"q_has_mental_concern invalid: {h!r}")

    for qid in ("q_phq2_1", "q_phq2_2", "q_gad2_1", "q_gad2_2"):
        _scale(answers.get(qid), qid, errors)

    di = answers.get("q_daily_impact")
    if di not in DAILY_IMPACT:
        errors.append(f"q_daily_impact invalid: {di!r}")

    pr = answers.get("q_prior_help")
    if pr not in PRIOR_HELP:
        errors.append(f"q_prior_help invalid: {pr!r}")

    goals = answers.get("q_app_goal")
    if not isinstance(goals, list) or len(goals) < 1:
        errors.append("q_app_goal must be non-empty list")
    elif not all(isinstance(x, str) and x in APP_GOALS for x in goals):
        errors.append(f"q_app_goal invalid codes: {goals!r}")
    elif len(goals) != len(set(goals)):
        errors.append("q_app_goal must not contain duplicates")

    # Skip logic
    has_areas = "q_concern_areas" in answers
    if h == "fine":
        if has_areas:
            errors.append("q_concern_areas must be omitted when q_has_mental_concern is fine")
    elif h in NOT_FINE:
        # Schema marks q_concern_areas as optional; if present, it must be a valid non-empty list
        if has_areas:
            ca = answers["q_concern_areas"]
            if not isinstance(ca, list) or not ca:
                errors.append("q_concern_areas, if present, must be a non-empty list")
            elif not all(x in CONCERN_AREAS for x in ca):
                errors.append(f"q_concern_areas invalid codes: {ca!r}")
            elif len(ca) != len(set(ca)):
                errors.append("q_concern_areas must not contain duplicates")

    has_dur = "q_duration" in answers
    if di in IMPACT_NEEDS_DURATION:
        if not has_dur:
            errors.append("q_duration required when q_daily_impact is moderately/significantly/severely")
        else:
            d = answers["q_duration"]
            if d not in DURATION:
                errors.append(f"q_duration invalid: {d!r}")
    elif has_dur:
        errors.append("q_duration must be omitted when q_daily_impact is not_at_all or slightly")

    has_diag = "q_diagnosis_known" in answers
    if pr in PRIOR_NEEDS_DIAGNOSIS:
        if has_diag:
            dk = answers["q_diagnosis_known"]
            if dk not in DIAGNOSIS:
                errors.append(f"q_diagnosis_known invalid: {dk!r}")
    elif has_diag:
        errors.append("q_diagnosis_known must be omitted when q_prior_help is never or prefer_not")

    computed_score: int | None = None
    computed_flag: str | None = None
    if not errors:
        try:
            computed_score = compute_total_score(answers)
            computed_flag = derive_flag(answers, computed_score)
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"scoring error: {e}")

    return ValidationResult(ok=len(errors) == 0, errors=errors, computed_score=computed_score, computed_flag=computed_flag)


def validate_row(
    row: dict[str, Any],
    *,
    check_expected: bool = True,
) -> ValidationResult:
    base = validate_answers(row["answers"])
    if not base.ok:
        return base
    if not check_expected:
        return base

    errors = list(base.errors)
    exp_f = row.get("expected_flag")
    exp_s = row.get("expected_total_score")

    if base.computed_flag != exp_f:
        errors.append(
            f"expected_flag mismatch: stated {exp_f!r}, computed from answers {base.computed_flag!r}"
        )
    if exp_s is not None and base.computed_score != int(exp_s):
        errors.append(
            f"expected_total_score mismatch: stated {exp_s}, computed {base.computed_score}"
        )

    return ValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        computed_score=base.computed_score,
        computed_flag=base.computed_flag,
    )


def metric_matches_for_row(row: dict[str, Any]) -> tuple[bool | None, bool | None]:
    ans = row.get("answers")
    if not isinstance(ans, dict):
        return None, None
    vr = validate_answers(ans)
    if not vr.ok or vr.computed_score is None or vr.computed_flag is None:
        return None, None
    try:
        exp_s = int(row["expected_total_score"])
    except (KeyError, TypeError, ValueError):
        exp_s = None
    exp_f = row.get("expected_flag")
    score_ok = exp_s is not None and vr.computed_score == exp_s
    flag_ok = isinstance(exp_f, str) and vr.computed_flag == exp_f
    return score_ok, flag_ok
