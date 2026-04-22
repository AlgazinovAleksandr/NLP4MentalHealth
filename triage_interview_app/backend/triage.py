from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import model_dir, triagem_src


def ensure_triagem_src_on_path() -> None:
    p = str(triagem_src())
    if p not in sys.path:
        sys.path.insert(0, p)


def _norm_id2label(cfg: Any) -> dict[int, str]:
    raw = getattr(cfg, "id2label", None) or {}
    out: dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = str(v)
    return out


class TriageModel:
    def __init__(self, *, model_path: Path, max_length: int = 512) -> None:
        mp = model_path.resolve()
        if not mp.is_dir():
            raise FileNotFoundError(f"TRIAGE_MODEL_DIR is not a directory: {mp}")
        self._tokenizer = AutoTokenizer.from_pretrained(str(mp))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(mp))
        self._model.eval()
        self._id2label = _norm_id2label(self._model.config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._max_length = max_length

    @property
    def id2label(self) -> dict[int, str]:
        return dict(self._id2label)

    @property
    def device(self) -> str:
        return str(self._device)

    @torch.inference_mode()
    def predict(self, text: str) -> tuple[str, list[tuple[str, float]]]:
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        logits = self._model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        pred_i = int(torch.argmax(logits).item())
        ranked = sorted(
            ((self._id2label[i], probs[i]) for i in range(len(probs))),
            key=lambda x: -x[1],
        )
        return self._id2label[pred_i], ranked


_singleton: TriageModel | None = None


def get_triage_model() -> TriageModel:
    global _singleton
    if _singleton is None:
        ensure_triagem_src_on_path()
        _singleton = TriageModel(model_path=model_dir())
    return _singleton


def build_input_text(answers: dict[str, Any], user_message: str) -> str:
    ensure_triagem_src_on_path()
    from nlp4mh_triagem.bert_triage_text import build_bert_triage_text

    return build_bert_triage_text(answers, user_message)
