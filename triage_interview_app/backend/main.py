from __future__ import annotations

import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import default_questionnaire_path, model_dir
from .triage import build_input_text, ensure_triagem_src_on_path, get_triage_model

app = FastAPI(title="BERT triage", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    answers: dict[str, Any]
    user_message: str = ""


class RankedItem(BaseModel):
    label: str
    prob: float


class PredictResponse(BaseModel):
    # Raw BERT argmax (can be wrong on rare JSON + empty-message combos — see dataset coverage).
    label: str
    ranked: list[RankedItem]
    # Rules + optional free-text lexicon (`nlp4mh_triagem.triage_combined`); use for product routing.
    routing_label: str
    structural_flag: str | None = None
    combined_notes: str = ""
    model_input_char_len: int
    # First ~500 chars of the exact string tokenized by BERT (JSON answers + [SEP] + message).
    model_input_preview: str


class ValidateRequest(BaseModel):
    answers: dict[str, Any]


class ValidateResponse(BaseModel):
    ok: bool
    errors: list[str]
    computed_score: int | None = None
    computed_flag: str | None = None


class HealthResponse(BaseModel):
    ok: bool
    model_dir: str
    labels: dict[int, str]
    device: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        m = get_triage_model()
        return HealthResponse(
            ok=True,
            model_dir=str(model_dir()),
            labels=m.id2label,
            device=m.device,
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"error": str(e), "model_dir": str(model_dir())},
        ) from e


@app.post("/validate", response_model=ValidateResponse)
def validate_questionnaire(req: ValidateRequest) -> ValidateResponse:
    ensure_triagem_src_on_path()
    from nlp4mh_triagem.questionnaire_validate import validate_answers as run_validate

    vr = run_validate(req.answers)
    return ValidateResponse(
        ok=vr.ok,
        errors=list(vr.errors),
        computed_score=vr.computed_score,
        computed_flag=vr.computed_flag,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        m = get_triage_model()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"error": str(e), "traceback": traceback.format_exc()},
        ) from e

    text = build_input_text(req.answers, req.user_message)
    try:
        label, ranked = m.predict(text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "traceback": traceback.format_exc()},
        ) from e

    ensure_triagem_src_on_path()
    from nlp4mh_triagem.triage_combined import combined_triage

    cr = combined_triage(req.answers, req.user_message)
    routing = str(cr.final_flag) if cr.ok and cr.final_flag is not None else label
    structural = cr.structural_flag if cr.ok else None
    notes = cr.notes if cr.ok else "combined_triage_not_ok"

    preview = text if len(text) <= 520 else text[:520] + "…"
    return PredictResponse(
        label=label,
        ranked=[RankedItem(label=a, prob=float(b)) for a, b in ranked],
        routing_label=routing,
        structural_flag=structural,
        combined_notes=notes,
        model_input_char_len=len(text),
        model_input_preview=preview,
    )


@app.get("/questionnaire-path")
def questionnaire_path() -> dict[str, str]:
    return {"path": str(default_questionnaire_path())}
