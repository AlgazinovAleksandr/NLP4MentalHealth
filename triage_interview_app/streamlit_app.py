from __future__ import annotations


import html
import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
import streamlit.components.v1 as components

from db_log import log_chat_turn, log_triage_session
from interview_runner import build_system_prompt, resume_interview, start_interview
from pipeline_chat import generate_opening, run_pipeline
from questionnaire_form import (
    gather_answers,
    iter_questions,
    load_questionnaire,
    prune_stale_questionnaire_widget_keys,
    render_questionnaire,
)

from backend.config import default_questionnaire_path


def _api_base() -> str:
    return os.environ.get("TRIAGE_API_URL", "http://127.0.0.1:8765").rstrip("/")


def _questionnaire_path() -> Path:
    raw = os.environ.get("QUESTIONNAIRE_JSON", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return default_questionnaire_path()


def _strings(lang: str) -> dict[str, str]:
    if lang == "ru":
        return {
            "page_title": "Первичный опрос",
            "free_text": "Коротко опишите, что вы хотели бы обсудить (по желанию)",
            "free_text_help": "Этот текст учитывается вместе с ответами анкеты. Можно оставить пустым.",
            "continue": "Продолжить",
            "chat_title": "Диалог",
            "chat_placeholder": "Ваше сообщение",
            "restart": "Начать заново",
            "lang_label": "Язык интерфейса",
            "model_probs": "Оценка модели по классам",
            "conn_err": "Сервер недоступен. Запустите triage API и проверьте TRIAGE_API_URL.",
        }
    return {
        "page_title": "Initial questionnaire",
        "free_text": "Anything you would like to share in your own words (optional)",
        "free_text_help": "This text is used together with your answers. You may leave it blank.",
        "continue": "Continue",
        "chat_title": "Conversation",
        "chat_placeholder": "Message",
        "restart": "Start over",
        "lang_label": "Interface language",
        "model_probs": "Model confidence by class",
        "conn_err": "Server unreachable. Start the triage API and check TRIAGE_API_URL.",
    }


def _class_display(name: str, lang: str) -> str:
    if lang == "ru":
        m = {
            "relaxed": "Низкий приоритет внимания",
            "concerned": "Средний приоритет",
            "urgent": "Высокий приоритет",
        }
        return m.get(name, name)
    m = {
        "relaxed": "Lower attention level",
        "concerned": "Moderate attention",
        "urgent": "Higher attention",
    }
    return m.get(name, name)


@st.cache_resource
def _http() -> httpx.Client:
    return httpx.Client(timeout=120.0)


def _session_answer_snapshot(spec: dict[str, Any]) -> dict[str, Any]:
    ids: set[str] = set()
    for q in iter_questions(spec):
        qid = q.get("id")
        if isinstance(qid, str):
            ids.add(qid)
    return {k: st.session_state[k] for k in ids if k in st.session_state}


def _scroll_to_bottom() -> None:
    components.html(
        "<script>const p=window.parent;try{const s=p.document.querySelector("
        "'section.main');if(s)s.scrollTop=s.scrollHeight;}catch(e){}</script>",
        height=0,
        width=0,
    )


def _inject_style() -> None:
    # Avoid overriding sidebar background (dark theme + light bg => invisible labels).
    st.markdown(
        """
        <style>
          div[data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 40rem;
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
          }
          h1 { font-weight: 600; letter-spacing: -0.02em; color: #111827; }
          h4 { color: #374151; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


_SEVERITY_ORDER = ["relaxed", "concerned", "urgent"]

_MAX_Q: dict[str, int] = {"relaxed": 2, "concerned": 4, "urgent": 3}


def _sidebar_probs(ranked: list[dict[str, Any]], lang: str) -> None:
    if not ranked:
        return
    t = _strings(lang)
    st.sidebar.divider()
    st.sidebar.markdown(f"**{t['model_probs']}**")
    sorted_ranked = sorted(
        ranked,
        key=lambda r: _SEVERITY_ORDER.index(r.get("label", ""))
        if r.get("label", "") in _SEVERITY_ORDER
        else 99,
    )
    for r in sorted_ranked:
        lab = str(r.get("label", ""))
        p = float(r.get("prob", 0.0))
        st.sidebar.caption(_class_display(lab, lang))
        st.sidebar.progress(min(1.0, max(0.0, p)))


def main() -> None:
    st.set_page_config(
        page_title="Questionnaire",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_style()

    qpath = _questionnaire_path()
    if not qpath.is_file():
        st.error(f"Questionnaire file not found: {qpath}")
        st.stop()

    spec = load_questionnaire(qpath)

    if "stage" not in st.session_state:
        st.session_state.stage = "questionnaire"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "triage_label" not in st.session_state:
        st.session_state.triage_label = None
    if "answers" not in st.session_state:
        st.session_state.answers = None
    if "triage_ranked" not in st.session_state:
        st.session_state.triage_ranked = None
    if "triage_free_text" not in st.session_state:
        st.session_state.triage_free_text = ""
    if "log_db_session_id" not in st.session_state:
        st.session_state.log_db_session_id = None
    if "interview_config" not in st.session_state:
        st.session_state.interview_config = None
    if "interview_messages" not in st.session_state:
        st.session_state.interview_messages = []

    st.sidebar.markdown("**Settings / Настройки**")
    lang = st.sidebar.selectbox(
        "Language / Язык",
        ["en", "ru"],
        index=0,
        format_func=lambda x: "English" if x == "en" else "Русский",
    )
    t = _strings(lang)

    if st.session_state.stage == "questionnaire":
        title = (spec.get("title") or {}).get(lang) or (spec.get("title") or {}).get("en")
        desc = (spec.get("description") or {}).get(lang) or (spec.get("description") or {}).get(
            "en"
        )
        st.title(title or t["page_title"])
        if desc:
            st.markdown(
                f'<p style="color:#4b5563;font-size:1.05rem;line-height:1.55;">'
                f"{html.escape(desc)}</p>",
                unsafe_allow_html=True,
            )
        pn = spec.get("privacy_notice") or {}
        pn_txt = pn.get(lang) or pn.get("en")
        if pn_txt:
            st.caption(pn_txt)

        render_questionnaire(spec, lang=lang)

        st.markdown("")
        st.markdown(f"**{t['free_text']}**")
        st.caption(t["free_text_help"])
        um = st.text_area(" ", label_visibility="collapsed", height=100, key="triage_free_text")

        if st.button(t["continue"], type="primary", use_container_width=True):
            prune_stale_questionnaire_widget_keys()
            snap = _session_answer_snapshot(spec)
            answers, g_err = gather_answers(spec, snap)
            if answers is None:
                for e in g_err:
                    st.error(e)
                st.stop()

            client = _http()
            with st.spinner("Reviewing your answers..." if lang == "en" else "Обработка анкеты..."):
                try:
                    vr = client.post(f"{_api_base()}/validate", json={"answers": answers})
                except httpx.ConnectError:
                    st.error(t["conn_err"])
                    st.stop()
                if vr.status_code != 200:
                    st.error(vr.text)
                    st.stop()
                vj = vr.json()
                if not vj.get("ok"):
                    for e in vj.get("errors", []):
                        st.error(e)
                    st.stop()

                try:
                    pr = client.post(
                        f"{_api_base()}/predict",
                        json={"answers": answers, "user_message": um.strip()},
                    )
                except httpx.ConnectError:
                    st.error(t["conn_err"])
                    st.stop()
                if pr.status_code != 200:
                    st.error(pr.text)
                    st.stop()
            pj = pr.json()
            bert_label = str(pj.get("label", "")).strip()
            routing_label = str(pj.get("routing_label") or bert_label).strip()
            ranked = pj.get("ranked") or []

            st.session_state.log_db_session_id = log_triage_session(
                answers=answers,
                triage_free_text=(um or "").strip(),
                routing_label=routing_label,
                bert_label=bert_label,
                ranked=ranked if isinstance(ranked, list) else [],
            )

            st.session_state.answers = answers
            st.session_state.triage_label = routing_label
            st.session_state.triage_ranked = ranked
            st.session_state.messages = []
            st.session_state.interview_config = None
            st.session_state.interview_messages = []
            st.session_state.stage = "interview"
            st.rerun()

    elif st.session_state.stage == "interview":
        ranked = st.session_state.triage_ranked or []
        _sidebar_probs(ranked, lang)
        st.sidebar.divider()
        if st.sidebar.button("🔄 " + t["restart"]):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


        def finish_interview(diagnosis: dict) -> None:
            with st.spinner(
                "Подготовка сессии..." if lang == "ru" else "Preparing your session..."
            ):
                system_prompt = build_system_prompt(
                    bert_class=st.session_state.triage_label,
                    interview_result=diagnosis,
                    questionnaire=st.session_state.answers,
                )
                opening = generate_opening(system_prompt)
            st.session_state.messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": opening},
            ]
            st.session_state.stage = "chat"
            st.session_state._scroll_chat_once = True

        for m in st.session_state.interview_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if st.session_state.interview_config is None:
            max_q = _MAX_Q.get(st.session_state.triage_label or "", 4)
            with st.spinner(
                "Подготовка первого вопроса..." if lang == "ru" else "Preparing first question..."
            ):
                question, config, is_done, diagnosis = start_interview(
                    st.session_state.answers, max_questions=max_q
                )
            if is_done:
                finish_interview(diagnosis)
            else:
                st.session_state.interview_config = config
                st.session_state.interview_messages.append(
                    {"role": "assistant", "content": question}
                )
            st.rerun()

        if st.session_state.pop("_pending_interview_reply", False):
            last_answer = st.session_state.interview_messages[-1]["content"]
            with st.spinner("Thinking about your response..." if lang == "en" else "Обдумываю ответ..."):
                question, is_done, diagnosis = resume_interview(
                    last_answer, st.session_state.interview_config
                )
            if is_done:
                finish_interview(diagnosis)
            else:
                st.session_state.interview_messages.append(
                    {"role": "assistant", "content": question}
                )
            st.rerun()

        answer = st.chat_input(
            "Ваш ответ" if lang == "ru" else "Your answer"
        )
        if answer:
            st.session_state.interview_messages.append({"role": "user", "content": answer})
            st.session_state._pending_interview_reply = True
            st.rerun()

    elif st.session_state.stage == "chat":
        if st.session_state.pop("_scroll_chat_once", False):
            _scroll_to_bottom()

        ranked = st.session_state.triage_ranked or []
        _sidebar_probs(ranked, lang)
        st.sidebar.divider()
        if st.sidebar.button("🔄 " + t["restart"]):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        for m in st.session_state.interview_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        st.caption("— " + ("Ваш терапевт" if lang == "ru" else "Your therapist") + " —")

        for m in st.session_state.messages:
            if m["role"] == "system":
                continue
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if st.session_state.pop("_pending_chat_reply", False):
            with st.spinner("Processing your message, please wait..." if lang == "en" else "Обработка сообщения, пожалуйста подождите..."):
                last_user = st.session_state.messages[-1]["content"]
                assistant = run_pipeline(st.session_state.messages, last_user)
            st.session_state.messages.append({"role": "assistant", "content": assistant})
            sid = st.session_state.get("log_db_session_id")
            if isinstance(sid, int) and sid > 0:
                log_chat_turn(sid, role="assistant", content=assistant)
            st.rerun()

        prompt = st.chat_input(t["chat_placeholder"])
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            sid = st.session_state.get("log_db_session_id")
            if isinstance(sid, int) and sid > 0:
                log_chat_turn(sid, role="user", content=prompt)
            st.session_state._pending_chat_reply = True
            st.rerun()


if __name__ == "__main__":
    main()
