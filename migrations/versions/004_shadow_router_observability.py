"""Add shadow-mode router observability columns to request_logs.

The AI router's shadow-mode disagreement signal (rule-based vs AI-predicted
complexity) was being computed and logged (app/engine/router.py route()) but
never persisted anywhere, so it was invisible outside of grepping application
logs. This adds the two columns needed to surface it via analytics/dashboard.

Revision ID: 004_shadow_router_observability
Revises: 003_phase2_tables
Create Date: 2026-09-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '004_shadow_router_observability'
down_revision: Union[str, None] = '003_phase2_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add_column_safe(table: str, column: sa.Column) -> None:
    """Add a column only if it doesn't already exist (idempotent).

    Checked up front rather than by catching the error — see
    _add_column_safe in 002_add_missing_columns.py for why a swallowed
    failure breaks the whole migration run on Postgres.
    """
    insp = sa.inspect(op.get_bind())
    if column.name not in {c['name'] for c in insp.get_columns(table)}:
        op.add_column(table, column)


def upgrade() -> None:
    _add_column_safe('request_logs', sa.Column(
        'shadow_disagreement', sa.Boolean(), nullable=True, server_default=sa.false()))
    _add_column_safe('request_logs', sa.Column(
        'ai_predicted_complexity', sa.String(20), nullable=True))


def downgrade() -> None:
    # Column drops are intentionally omitted — see migrations/versions/002_add_missing_columns.py
    pass
