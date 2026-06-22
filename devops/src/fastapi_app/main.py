from __future__ import annotations

from fastapi import FastAPI, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import PlainTextResponse

from fastapi_app.api.controller import router
from fastapi_app.auth.security import require_admin
from fastapi_app.services.ml import model

app = FastAPI(
    title="FastAPI service for ML-project",
    version="0.0.1",
    openapi_version="3.1.0",
    openapi_url="/openapi.json",
)


@app.exception_handler(RequestValidationError)
def _validation_error(_request: Request, _exc: RequestValidationError):
    return PlainTextResponse("bad request", status_code=400)


@app.on_event("startup")
def _startup() -> None:
    model.load()


@app.get("/docs", include_in_schema=False, dependencies=[Depends(require_admin)])
async def admin_docs() -> PlainTextResponse:
    return get_swagger_ui_html(openapi_url=app.openapi_url, title=f"{app.title} - docs")


@app.get(app.openapi_url, include_in_schema=False, dependencies=[Depends(require_admin)])
async def admin_openapi():
    return app.openapi()


app.include_router(router)
