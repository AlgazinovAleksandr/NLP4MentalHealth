from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger(__name__)


def default_sqlite_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "triage_sessions.sqlite"


def sqlite_url() -> str:
    raw = os.environ.get("TRIAGE_LOG_SQLITE", "").strip()
    path = Path(raw).expanduser().resolve() if raw else default_sqlite_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


class Base(DeclarativeBase):
    pass


def _utc() -> datetime:
    return datetime.now(timezone.utc)


class TriageSessionRow(Base):
    __tablename__ = "triage_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc,
    )
    answers_json: Mapped[str] = mapped_column(Text)
    triage_free_text: Mapped[str] = mapped_column(Text, default="")
    routing_label: Mapped[str] = mapped_column(String(32), default="")
    bert_label: Mapped[str] = mapped_column(String(32), default="")
    probs_json: Mapped[str] = mapped_column(Text, default="[]")


class ChatTurnRow(Base):
    __tablename__ = "chat_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("triage_sessions.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc,
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)


_engine = None
_schema_done = False


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            sqlite_url(),
            connect_args={"check_same_thread": False},
            echo=False,
        )
    return _engine


def ensure_schema() -> None:
    global _schema_done
    if _schema_done:
        return
    Base.metadata.create_all(get_engine())
    _schema_done = True


def log_triage_session(
    *,
    answers: dict[str, Any],
    triage_free_text: str,
    routing_label: str,
    bert_label: str,
    ranked: list[dict[str, Any]],
) -> int | None:
    ensure_schema()
    try:
        row = TriageSessionRow(
            answers_json=json.dumps(answers, ensure_ascii=False, sort_keys=True),
            triage_free_text=triage_free_text or "",
            routing_label=routing_label or "",
            bert_label=bert_label or "",
            probs_json=json.dumps(ranked, ensure_ascii=False),
        )
        with Session(get_engine()) as s:
            s.add(row)
            s.commit()
            sid = int(row.id)
        return sid
    except Exception as e:
        logger.warning("log_triage_session failed: %s", e)
        return None


def log_chat_turn(session_id: int, *, role: str, content: str) -> None:
    if session_id <= 0:
        return
    ensure_schema()
    try:
        row = ChatTurnRow(session_id=session_id, role=role, content=content or "")
        with Session(get_engine()) as s:
            s.add(row)
            s.commit()
    except Exception as e:
        logger.warning("log_chat_turn failed: %s", e)
