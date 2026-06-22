from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PromptCreateRequest(BaseModel):
    message: str = Field(..., min_length=200, max_length=1000)

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]

class Diagnosis(BaseModel):
    label: str
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    riskLevel: RiskLevel
    recommendations: Optional[str] = None

class PromptDiagnosisResponse(BaseModel):
    id: UUID
    diagnosis: Diagnosis

PromptStatus = Literal["PENDING", "SUCCESS", "FAILED"]

class PromptRecord(BaseModel):
    id: UUID
    message: Optional[str] = None
    createdAt: datetime
    diagnosis: Optional[Diagnosis] = None
    status: PromptStatus

class PromptHistoryResponse(BaseModel):
    items: list[PromptRecord]
    nextCursor: Optional[str] = None

class LatencyMs(BaseModel):
    mean: float
    p50: float
    p95: float
    p99: float

class InputStats(BaseModel):
    avgLength: float
    avgTokens: float
    p95Tokens: float

class Window(BaseModel):
    from_: datetime = Field(..., alias="from")
    to: datetime

class StatsResponse(BaseModel):
    latencyMs: LatencyMs
    input: InputStats
    window: Window
    count: int = Field(..., ge=0)

class ProblemDetails(BaseModel):
    type: str
    title: str
    status: int
    detail: Optional[str] = None
    instance: Optional[str] = None

