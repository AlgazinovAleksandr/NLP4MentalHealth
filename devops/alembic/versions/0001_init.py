from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enum used by PromptRecord.status
    prompt_status = sa.Enum("PENDING", "SUCCESS", "FAILED", name="prompt_status")

    op.create_table(
        "prompt_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("status", prompt_status, nullable=False),
        sa.Column("diagnosis", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("message_length", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("prompt_records")
    sa.Enum(name="prompt_status").drop(op.get_bind(), checkfirst=True)
