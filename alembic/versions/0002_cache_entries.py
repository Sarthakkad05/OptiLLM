"""create cache_entries table

Revision ID: 0002_cache_entries
Revises: 0001_request_logs
Create Date: 2026-06-11

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0002_cache_entries"
down_revision: Union[str, None] = "0001_request_logs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "cache_entries",
        sa.Column("id", sa.Integer(), primary_key=True, index=True, autoincrement=True),
        sa.Column("faiss_index_id", sa.Integer(), unique=True, nullable=False, index=True),
        sa.Column("response_text", sa.Text(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("tokens_input", sa.Integer(), server_default="0"),
        sa.Column("tokens_output", sa.Integer(), server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("cache_entries")
