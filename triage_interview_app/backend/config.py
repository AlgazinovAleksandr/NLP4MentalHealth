from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """NLP4MentalHealth/ (parent of triage_interview_app/)."""
    return Path(__file__).resolve().parents[2]


def triagem_src() -> Path:
    return repo_root() / "agentic_pipeline" / "synthetic_questionnaire_generation" / "src"


def synth_runs_root() -> Path:
    return (
        repo_root()
        / "agentic_pipeline"
        / "synthetic_questionnaire_generation"
        / "runs"
    )


def default_model_dir() -> Path:
    """Prefer neutral run if present, else legacy run1 (weights live under synthetic_questionnaire_generation)."""
    base = synth_runs_root()
    for name in ("triage_bert_neutral", "triage_bert_run1"):
        p = base / name / "best_model"
        if p.is_dir():
            return p
    return base / "triage_bert_neutral" / "best_model"


def _remap_env_model_path(p: Path) -> Path:
    """
    Training writes to .../synthetic_questionnaire_generation/runs/<run>/best_model.
    If TRIAGE_MODEL_DIR was set from triage_interview_app (e.g. .../triage_interview_app/runs/...),
    the directory often does not exist — try the same path suffix under synth runs.
    """
    if p.is_dir():
        return p
    parts = p.resolve().parts
    try:
        idx = parts.index("runs")
    except ValueError:
        return p
    tail = Path(*parts[idx + 1 :])
    cand = synth_runs_root() / tail
    return cand if cand.is_dir() else p


def model_dir() -> Path:
    raw = os.environ.get("TRIAGE_MODEL_DIR", "").strip()
    if raw:
        return _remap_env_model_path(Path(raw).expanduser().resolve())
    return default_model_dir()


def default_questionnaire_path() -> Path:
    return (
        repo_root()
        / "agentic_pipeline"
        / "synthetic_questionnaire_generation"
        / "registration_questionnaire_v0.0.1.json"
    )
