#!/usr/bin/env sh
set -eu

# checking if importable
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}/app/src"

alembic -c alembic.ini upgrade head

exec uvicorn fastapi_app.main:app --host 0.0.0.0 --port "${APP_PORT:-8080}"
