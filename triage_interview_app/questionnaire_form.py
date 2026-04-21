from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterator

from streamlit.delta_generator import DeltaGenerator


def load_questionnaire(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prune_stale_questionnaire_widget_keys() -> None:
    import streamlit as st

    if st.session_state.get("q_has_mental_concern") == "fine":
        st.session_state.pop("q_concern_areas", None)
    if st.session_state.get("q_daily_impact") not in {
        "moderately",
        "significantly",
        "severely",
    }:
        st.session_state.pop("q_duration", None)
    if st.session_state.get("q_prior_help") not in {"in_past", "currently"}:
        st.session_state.pop("q_diagnosis_known", None)


def iter_questions(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for sec in spec.get("sections", []):
        for q in sec.get("questions", []):
            yield q


def _label(q: dict[str, Any], lang: str) -> str:
    t = q.get("text") or {}
    return str(t.get(lang) or t.get("en") or q.get("id", ""))


def _opt_label(opt: dict[str, Any], lang: str) -> str:
    lab = opt.get("label") or {}
    return str(lab.get(lang) or lab.get("en") or opt.get("code", ""))


def visible(q: dict[str, Any], answers: dict[str, Any]) -> bool:
    di = q.get("display_if")
    if not di:
        return True
    parent = di.get("question_id")
    codes = di.get("answer_codes") or []
    want = set(codes)
    val = answers.get(parent)
    if isinstance(val, list):
        return bool(want.intersection(val))
    return val in want


def gather_answers(
    spec: dict[str, Any], session: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    answers: dict[str, Any] = {}

    for q in iter_questions(spec):
        qid = q.get("id")
        if not isinstance(qid, str):
            continue
        if not visible(q, answers):
            continue
        typ = q.get("type")
        required = bool(q.get("required"))
        val = session.get(qid)

        if typ == "multiple_choice":
            if not isinstance(val, list):
                val = [] if val is None else [val]
            answers[qid] = val
            if required and len(val) < 1:
                errors.append(f"{qid}: select at least one option")
            continue

        if val is None:
            if required:
                errors.append(f"Missing answer: {qid}")
            continue
        if typ == "scale":
            if isinstance(val, str) and val.strip().isdigit():
                val = int(val.strip())
            elif type(val) is not int:
                errors.append(f"{qid}: scale must be int 0..3, got {val!r}")
                continue
        answers[qid] = val

    h = answers.get("q_has_mental_concern")
    if h == "fine" and "q_concern_areas" in answers:
        answers.pop("q_concern_areas", None)

    di = answers.get("q_daily_impact")
    needs_dur = di in {"moderately", "significantly", "severely"}
    if not needs_dur and "q_duration" in answers:
        answers.pop("q_duration", None)

    pr = answers.get("q_prior_help")
    needs_diag = pr in {"in_past", "currently"}
    if not needs_diag and "q_diagnosis_known" in answers:
        answers.pop("q_diagnosis_known", None)

    if errors:
        return None, errors
    return answers, []


def _scale_labels_for_question(
    spec: dict[str, Any], question: dict[str, Any], lang: str
) -> dict[str, str]:
    qid = question.get("id")
    for sec in spec.get("sections", []):
        for qq in sec.get("questions", []):
            if qq.get("id") != qid:
                continue
            sl = sec.get("scale_labels") or {}
            out: dict[str, str] = {}
            for k, v in sl.items():
                if isinstance(v, dict):
                    out[str(k)] = str(v.get(lang) or v.get("en") or k)
            return out
    return {}


def render_questionnaire(
    spec: dict[str, Any],
    *,
    lang: str,
    container: DeltaGenerator | None = None,
) -> None:
    import streamlit as st

    root = container or st
    answers_so_far: dict[str, Any] = {}

    def one(q: dict[str, Any]) -> None:
        nonlocal answers_so_far
        qid = q.get("id")
        if not isinstance(qid, str):
            return
        if not visible(q, answers_so_far):
            return

        typ = q.get("type")
        title = _label(q, lang)

        if typ == "single_choice":
            opts = q.get("options") or []
            codes = [str(o.get("code")) for o in opts if o.get("code") is not None]
            labels = [_opt_label(o, lang) for o in opts]
            if not codes:
                return
            label_by_code = {c: lab for c, lab in zip(codes, labels, strict=True)}
            root.radio(title, codes, format_func=lambda c, lb=label_by_code: lb.get(c, c), key=qid)
            if qid in st.session_state:
                answers_so_far[qid] = st.session_state[qid]

        elif typ == "multiple_choice":
            opts = q.get("options") or []
            codes = [str(o.get("code")) for o in opts if o.get("code") is not None]
            labels = [_opt_label(o, lang) for o in opts]
            if not codes:
                return
            label_by_code = {c: lab for c, lab in zip(codes, labels, strict=True)}
            root.multiselect(
                title,
                codes,
                format_func=lambda c, lb=label_by_code: lb.get(c, c),
                key=qid,
            )
            if qid in st.session_state:
                answers_so_far[qid] = list(st.session_state[qid])

        elif typ == "number":
            vmin = int((q.get("validation") or {}).get("min", 13))
            vmax = int((q.get("validation") or {}).get("max", 100))
            cur = st.session_state.get(qid)
            try:
                default = int(cur)
            except (TypeError, ValueError):
                default = 21
            default = max(vmin, min(vmax, default))
            root.number_input(title, min_value=vmin, max_value=vmax, value=default, key=qid)
            if qid in st.session_state:
                answers_so_far[qid] = int(st.session_state[qid])

        elif typ == "scale":
            smin = int(q.get("scale_min", 0))
            smax = int(q.get("scale_max", 3))
            labels_map = _scale_labels_for_question(spec, q, lang)
            codes_int = list(range(smin, smax + 1))

            def fmt_scale(c: int, lm: dict[str, str] = labels_map) -> str:
                extra = lm.get(str(c))
                return f"{c} — {extra}" if extra else str(c)

            root.select_slider(title, options=codes_int, format_func=fmt_scale, key=qid)
            if qid in st.session_state:
                answers_so_far[qid] = int(st.session_state[qid])

        else:
            root.warning(f"Unsupported question type: {typ!r} ({qid})")

    sections = sorted(spec.get("sections", []), key=lambda s: int(s.get("order", 0)))
    first = True
    for sec in sections:
        if not first:
            root.divider()
        first = False
        sec_title = (sec.get("title") or {}).get(lang) or (sec.get("title") or {}).get("en")
        if sec_title:
            root.markdown(f"#### {html.escape(sec_title)}")
        sub = sec.get("subtitle")
        if isinstance(sub, dict):
            sub_txt = sub.get(lang) or sub.get("en")
            if sub_txt:
                root.caption(sub_txt)
        questions = sorted(sec.get("questions", []), key=lambda q: int(q.get("order", 0)))
        for q in questions:
            one(q)
