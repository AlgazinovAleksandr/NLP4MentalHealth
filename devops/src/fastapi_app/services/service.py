from __future__ import annotations
import time
from datetime import datetime, timezone
from uuid import UUID, uuid4
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from fastapi_app.services.ml import model
from fastapi_app.db.models.models import PromptRecord
from fastapi_app.schemas.schemas import Diagnosis, RiskLevel, StatsResponse

def _risk_level(confidence: float) -> RiskLevel:
    if confidence >= 0.80:
        return "HIGH"
    if confidence >= 0.50:
        return "MEDIUM"
    return "LOW"


def _token_estimate(text: str) -> int:
    return len([t for t in text.split() if t])


def _pct(sorted_arr: list[float] | list[int], p: float) -> float:
    if not sorted_arr:
        return 0.0
    idx = int(round((p / 100.0) * (len(sorted_arr) - 1)))
    idx = max(0, min(len(sorted_arr) - 1, idx))
    return float(sorted_arr[idx])


def predict_from_message(db: Session, message: str) -> tuple[UUID, Diagnosis]:
    record_id = uuid4()

    start = time.perf_counter()

    pred = model.predict(message)
    diagnosis = Diagnosis(
        label=pred.label,
        confidence=pred.confidence,
        riskLevel=_risk_level(pred.confidence),
        recommendations=None,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    row = PromptRecord(
        id=record_id,
        message=message,
        status="SUCCESS",
        diagnosis=diagnosis.model_dump(),
        latency_ms=float(latency_ms),
        message_length=len(message),
        token_count=_token_estimate(message),
    )
    db.add(row)
    db.commit()

    return record_id, diagnosis


def get_history(db: Session) -> list[PromptRecord]:
    return list(db.scalars(select(PromptRecord).order_by(PromptRecord.created_at.desc())).all())


def delete_history(db: Session) -> None:
    db.execute(delete(PromptRecord))
    db.commit()


def get_stats(db: Session) -> StatsResponse:
    items = list(db.scalars(select(PromptRecord)).all())
    now = datetime.now(timezone.utc)

    if not items:
        return StatsResponse(
            latencyMs={"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0},
            input={"avgLength": 0.0, "avgTokens": 0.0, "p95Tokens": 0.0},
            window={"from": now, "to": now},
            count=0,
        )

    latencies = sorted(float(i.latency_ms) for i in items)
    lengths = [int(i.message_length) for i in items]
    tokens = sorted(int(i.token_count) for i in items)

    mean_latency = float(sum(latencies) / len(latencies))
    avg_len = float(sum(lengths) / len(lengths))
    avg_tokens = float(sum(tokens) / len(tokens))

    window_from = min(i.created_at for i in items)
    window_to = max(i.created_at for i in items)

    return StatsResponse(
        latencyMs={
            "mean": mean_latency,
            "p50": _pct(latencies, 50),
            "p95": _pct(latencies, 95),
            "p99": _pct(latencies, 99),
        },
        input={
            "avgLength": avg_len,
            "avgTokens": avg_tokens,
            "p95Tokens": _pct(tokens, 95),
        },
        window={"from": window_from, "to": window_to},
        count=len(items),
    )
