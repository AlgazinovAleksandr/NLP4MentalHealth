from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _resources_dir() -> Path:
    return Path(os.getenv("RESOURCES_DIR", str(_repo_root() / "fastapi_app" / "resources")))

@dataclass(frozen=True, slots=True)
class Settings:
    keycloak_internal_url: str = field(default_factory=lambda: _env("KEYCLOAK_INTERNAL_URL", "http://keycloak:8080/auth"))
    keycloak_public_url: str = field(default_factory=lambda: _env("KEYCLOAK_PUBLIC_URL", "http://localhost:8088/auth"))
    keycloak_realm: str = field(default_factory=lambda: _env("KEYCLOAK_REALM", "fastapi_app"))
    keycloak_client_id: str = field(default_factory=lambda: _env("KEYCLOAK_CLIENT_ID", "fastapi_app"))
    keycloak_redirect_uri: str = field(default_factory=lambda: _env("KEYCLOAK_REDIRECT_URI", ""))

    database_url: str = field(default_factory=lambda: _env(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres",
    ))
    db_schema: str = field(default_factory=lambda: _env("APP_DB_SCHEMA", "fastapi_app"))

    repo_root: Path = field(default_factory=_repo_root)
    resources_dir: Path = field(default_factory=_resources_dir)

    tfidf_path: Path = field(default_factory=lambda: Path(_env("TFIDF_PATH", str(_resources_dir() / "tfidf_vectorizer.pkl"))))
    catboost_path: Path = field(default_factory=lambda: Path(_env("CATBOOST_PATH", str(_resources_dir() / "catboost_model.cbm"))))


settings = Settings()
