from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from fastapi_app.db.db import get_db
from fastapi_app.schemas.schemas import (
    ProblemDetails,
    PromptCreateRequest,
    PromptDiagnosisResponse,
    PromptHistoryResponse,
    PromptRecord,
    StatsResponse,
)
from fastapi_app.auth.security import require_admin, require_auth
from fastapi_app.services.service import delete_history, get_history, get_stats, predict_from_message


router = APIRouter(dependencies=[Depends(require_auth)])


@router.post(
    "/forward",
    operation_id="createPrompt",
    response_model=PromptDiagnosisResponse,
    responses={
        400: {"description": "bad request"},
        401: {"description": "Unauthorized"},
        403: {"model": ProblemDetails, "description": "модель не смогла обработать данные"},
        500: {"model": ProblemDetails, "description": "Internal server error"},
    },
)
def forward(
    body: PromptCreateRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> PromptDiagnosisResponse:
    """Create a prompt and return diagnosis (text-only)."""

    try:
        prompt_id, diagnosis = predict_from_message(db, body.message)
        return PromptDiagnosisResponse(id=prompt_id, diagnosis=diagnosis)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=403,
            detail={
                "type": "about:blank",
                "title": "Forbidden",
                "status": 403,
                "detail": "модель не смогла обработать данные",
                "instance": str(request.url),
            },
        )


@router.get("/history", operation_id="getHistory", response_model=PromptHistoryResponse,
            dependencies=[Depends(require_admin)], openapi_extra={"x-roles": ["admin"]})
def history(db: Session = Depends(get_db)) -> PromptHistoryResponse:
    items = []
    for s in get_history(db):
        items.append(
            PromptRecord(
                id=s.id,
                message=s.message,
                createdAt=s.created_at,
                diagnosis=s.diagnosis,
                status=s.status,
            )
        )

    return PromptHistoryResponse(items=items, nextCursor=None)


@router.delete("/history", operation_id="deleteHistory", status_code=204, dependencies=[Depends(require_admin)],
               openapi_extra={"x-roles": ["admin"]})
def history_delete(db: Session = Depends(get_db)) -> None:
    delete_history(db)
    return None


@router.get("/stats", operation_id="getStatistics", response_model=StatsResponse)
def stats(db: Session = Depends(get_db)) -> StatsResponse:
    return StatsResponse.model_validate(get_stats(db).model_dump(by_alias=True))
