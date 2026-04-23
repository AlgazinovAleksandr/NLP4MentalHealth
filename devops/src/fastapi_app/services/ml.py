from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pickle

from fastapi_app.config.config import settings
from catboost import CatBoostClassifier

@dataclass
class Prediction:
    label: str
    confidence: float


class MentalHealthModel:
    def __init__(self) -> None:
        self._tfidf: Any | None = None
        self._model: Any | None = None

    def load(self) -> None:
        with open(settings.tfidf_path, "rb") as f:
            self._tfidf = pickle.load(f)

        model = CatBoostClassifier()
        model.load_model(str(settings.catboost_path))
        self._model = model

    def predict(self, message: str) -> Prediction:
        if self._tfidf is None or self._model is None:
            raise RuntimeError("Model is not loaded")

        features = self._tfidf.transform([message])
        pred = self._model.predict(features)
        proba = self._model.predict_proba(features)

        label = str(pred[0])
        confidence = float(max(proba[0]))
        return Prediction(label=label, confidence=confidence)


model = MentalHealthModel()
