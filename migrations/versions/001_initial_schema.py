"""Initial database schema migration for OptiLLM.

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-08-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'request_logs',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('model_requested', sa.String(), nullable=False),
        sa.Column('model_used', sa.String(), nullable=False),
        sa.Column('provider', sa.String(), nullable=False, server_default='openai'),
        sa.Column('tag', sa.String(length=100), nullable=True),
        sa.Column('tokens_input', sa.Integer(), server_default='0'),
        sa.Column('tokens_output', sa.Integer(), server_default='0'),
        sa.Column('tokens_saved', sa.Integer(), server_default='0'),
        sa.Column('cost_usd', sa.Float(), server_default='0.0'),
        sa.Column('savings_usd', sa.Float(), server_default='0.0'),
        sa.Column('cache_hit', sa.Boolean(), server_default='0'),
        sa.Column('compressed', sa.Boolean(), server_default='0'),
        sa.Column('routed', sa.Boolean(), server_default='0'),
        sa.Column('latency_ms', sa.Integer(), server_default='0'),
        sa.Column('prompt_snippet', sa.String(length=200), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('correctness_score', sa.Float(), nullable=True),
        sa.Column('relevance_score', sa.Float(), nullable=True),
        sa.Column('completeness_score', sa.Float(), nullable=True),
        sa.Column('hallucination_score', sa.Float(), nullable=True),
        sa.Column('efficiency_score', sa.Float(), nullable=True),
    )

    op.create_table(
        'cache_entries',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('faiss_index_id', sa.Integer(), nullable=False, unique=True),
        sa.Column('prompt_text', sa.Text(), nullable=False, server_default=''),
        sa.Column('response_text', sa.Text(), nullable=False),
        sa.Column('model', sa.String(), nullable=False),
        sa.Column('tokens_input', sa.Integer(), server_default='0'),
        sa.Column('tokens_output', sa.Integer(), server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'key_budgets',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('api_key', sa.String(length=100), nullable=False, unique=True),
        sa.Column('daily_budget_usd', sa.Float(), server_default='10.0'),
        sa.Column('monthly_budget_usd', sa.Float(), server_default='100.0'),
        sa.Column('daily_spent_usd', sa.Float(), server_default='0.0'),
        sa.Column('monthly_spent_usd', sa.Float(), server_default='0.0'),
        sa.Column('last_reset_day', sa.String(length=10), nullable=True),
        sa.Column('last_reset_month', sa.String(length=7), nullable=True),
    )

    op.create_table(
        'tool_audit_logs',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('tool_name', sa.String(length=100), nullable=False),
        sa.Column('arguments', sa.Text(), nullable=True),
        sa.Column('execution_time_ms', sa.Float(), server_default='0.0'),
        sa.Column('success', sa.Boolean(), server_default='1'),
        sa.Column('result_summary', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('tool_audit_logs')
    op.drop_table('key_budgets')
    op.drop_table('cache_entries')
    op.drop_table('request_logs')
