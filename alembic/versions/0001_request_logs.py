"""create request_logs table

Revision ID: 0001_request_logs
Revises: 
Create Date: 2026-06-11

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0001_request_logs"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "request_logs",
        sa.Column("id", sa.Integer(), primary_key=True, index=True, autoincrement=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
        sa.Column("model_requested", sa.String(), nullable=False),
        sa.Column("model_used", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False, server_default="openai"),
        sa.Column("tokens_input", sa.Integer(), server_default="0"),
        sa.Column("tokens_output", sa.Integer(), server_default="0"),
        sa.Column("tokens_saved", sa.Integer(), server_default="0"),
        sa.Column("cost_usd", sa.Float(), server_default="0"),
        sa.Column("savings_usd", sa.Float(), server_default="0"),
        sa.Column("cache_hit", sa.Boolean(), server_default="false"),
        sa.Column("compressed", sa.Boolean(), server_default="false"),
        sa.Column("routed", sa.Boolean(), server_default="false"),
        sa.Column("latency_ms", sa.Integer(), server_default="0"),
        sa.Column("prompt_snippet", sa.String(200), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("request_logs")
