"""Add Phase 2 tables: router_training_labels, router_training_runs, request_feedback.

Revision ID: 003_phase2_tables
Revises: 002_add_missing_columns
Create Date: 2026-09-21
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '003_phase2_tables'
down_revision: Union[str, None] = '002_add_missing_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table)


def _create_table_safe(table: str, *columns: sa.Column) -> None:
    """Create a table only if it doesn't already exist.

    Checked up front rather than by catching the error — see
    _add_column_safe in 002_add_missing_columns.py for why a swallowed
    failure breaks the whole migration run on Postgres.
    """
    if not _has_table(table):
        op.create_table(table, *columns)


def upgrade() -> None:
    # router_training_labels
    _create_table_safe(
        'router_training_labels',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), index=True),
        sa.Column('request_log_id', sa.Integer(), nullable=True, index=True),
        sa.Column('prompt_snippet', sa.String(500), nullable=True),
        sa.Column('model_requested', sa.String(100), nullable=True),
        sa.Column('complexity_label', sa.String(20), nullable=False, index=True),
        sa.Column('label_source', sa.String(50), server_default='inferred'),
        sa.Column('confidence', sa.Float(), server_default='1.0'),
    )

    # router_training_runs
    _create_table_safe(
        'router_training_runs',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('started_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), index=True),
        sa.Column('samples_trained', sa.Integer(), server_default='0'),
        sa.Column('samples_evaluated', sa.Integer(), server_default='0'),
        sa.Column('current_accuracy', sa.Float(), nullable=True),
        sa.Column('new_accuracy', sa.Float(), nullable=True),
        sa.Column('model_swapped', sa.Boolean(), server_default='0'),
        sa.Column('training_time_ms', sa.Integer(), server_default='0'),
    )

    # request_feedback
    _create_table_safe(
        'request_feedback',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), index=True),
        sa.Column('request_log_id', sa.Integer(), nullable=False, index=True),
        sa.Column('request_id', sa.String(100), nullable=True, index=True),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('issue', sa.String(50), nullable=True),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('user_id', sa.String(100), nullable=True, index=True),
        sa.Column('team_id', sa.String(100), nullable=True, index=True),
    )


def downgrade() -> None:
    for t in ['request_feedback', 'router_training_runs', 'router_training_labels']:
        if _has_table(t):
            op.drop_table(t)
