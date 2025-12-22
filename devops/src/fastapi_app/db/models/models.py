from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Enum, DateTime, Float, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from fastapi_app.db.base import Base

PromptStatusEnum = Enum("PENDING", "SUCCESS", "FAILED", name="prompt_status")

class PromptRecord(Base):
    __tablename__ = "prompt_records"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    status: Mapped[str] = mapped_column(PromptStatusEnum, nullable=False)
    diagnosis: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    message_length: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

