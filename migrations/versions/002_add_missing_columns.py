"""Add all missing columns that were previously patched via raw ALTER TABLE at startup.

Revision ID: 002_add_missing_columns
Revises: 001_initial_schema
Create Date: 2026-09-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002_add_missing_columns'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table)


def _has_column(table: str, column: str) -> bool:
    insp = sa.inspect(op.get_bind())
    return insp.has_table(table) and column in {c['name'] for c in insp.get_columns(table)}


def _add_column_safe(table: str, column: sa.Column) -> None:
    """
    Add a column only if its table exists and the column doesn't (idempotent).

    Existence is checked up front rather than by catching the error: on
    Postgres a failed statement aborts the surrounding transaction, and
    env.py runs every migration in a single transaction, so a swallowed
    failure here would make every later statement — including alembic's own
    version bookkeeping — fail with "current transaction is aborted".
    Rolling back to recover isn't an option either, since that would also
    discard everything earlier migrations (e.g. 001) did in the same
    transaction.
    """
    if _has_table(table) and not _has_column(table, column.name):
        op.add_column(table, column)


def _create_table_safe(table: str, *columns: sa.Column) -> None:
    """Create a table only if it doesn't already exist (see _add_column_safe)."""
    if not _has_table(table):
        op.create_table(table, *columns)


def upgrade() -> None:
    # ── cache_entries ──────────────────────────────────────────────────────────
    _add_column_safe('cache_entries', sa.Column(
        'tenant_id', sa.String(100), nullable=True, server_default='default'))

    # ── request_logs ──────────────────────────────────────────────────────────
    _add_column_safe('request_logs', sa.Column(
        'tag', sa.String(100), nullable=True))
    _add_column_safe('request_logs', sa.Column(
        'quality_score', sa.Float(), nullable=True))
    _add_column_safe('request_logs', sa.Column(
        'correctness_score', sa.Float(), nullable=True))
    _add_column_safe('request_logs', sa.Column(
        'relevance_score', sa.Float(), nullable=True))
    _add_column_safe('request_logs', sa.Column(
        'completeness_score', sa.Float(), nullable=True))
    _add_column_safe('request_logs', sa.Column(
        'hallucination_score', sa.Float(), nullable=True))
    _add_column_safe('request_logs', sa.Column(
        'efficiency_score', sa.Float(), nullable=True))
    _add_column_safe('request_logs', sa.Column(
        'tenant_id', sa.String(100), nullable=True, server_default='default'))
    _add_column_safe('request_logs', sa.Column(
        'team_id', sa.String(100), nullable=True))
    _add_column_safe('request_logs', sa.Column(
        'user_id', sa.String(100), nullable=True))

    # ── key_budgets ───────────────────────────────────────────────────────────
    _add_column_safe('key_budgets', sa.Column(
        'tenant_id', sa.String(100), nullable=True, server_default='default'))

    # ── tool_audit_logs ───────────────────────────────────────────────────────
    _add_column_safe('tool_audit_logs', sa.Column(
        'tenant_id', sa.String(100), nullable=True, server_default='default'))

    # ── New tables added after initial migration ───────────────────────────────
    # tenants
    _create_table_safe(
        'tenants',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('tenant_id', sa.String(100), nullable=False, unique=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('api_key', sa.String(200), nullable=False, unique=True),
        sa.Column('role', sa.String(50), server_default='developer'),
        sa.Column('sla_target_ms', sa.Float(), server_default='500.0'),
        sa.Column('created_at', sa.DateTime(),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # audit_log_entries
    _create_table_safe(
        'audit_log_entries',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('timestamp', sa.DateTime(),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('tenant_id', sa.String(100), server_default='default'),
        sa.Column('actor', sa.String(100), nullable=False),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource', sa.String(200), nullable=False),
        sa.Column('payload_hash', sa.String(64), nullable=False),
        sa.Column('prev_hash', sa.String(64), nullable=False),
        sa.Column('status', sa.String(50), server_default='success'),
    )

    # organizations
    _create_table_safe(
        'organizations',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('org_id', sa.String(100), nullable=False, unique=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('created_at', sa.DateTime(),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # teams
    _create_table_safe(
        'teams',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('team_id', sa.String(100), nullable=False, unique=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('org_id', sa.String(100), nullable=True),
        sa.Column('daily_budget_usd', sa.Float(), server_default='100.0'),
        sa.Column('monthly_budget_usd', sa.Float(), server_default='1000.0'),
        sa.Column('daily_spent_usd', sa.Float(), server_default='0.0'),
        sa.Column('monthly_spent_usd', sa.Float(), server_default='0.0'),
        sa.Column('last_reset_day', sa.String(10), nullable=True),
        sa.Column('last_reset_month', sa.String(7), nullable=True),
        sa.Column('rpm_limit', sa.Integer(), nullable=True),
        sa.Column('tpm_limit', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # users
    _create_table_safe(
        'users',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('user_id', sa.String(100), nullable=False, unique=True),
        sa.Column('email', sa.String(200), nullable=True),
        sa.Column('team_id', sa.String(100), nullable=True),
        sa.Column('role', sa.String(50), server_default='developer'),
        sa.Column('daily_budget_usd', sa.Float(), nullable=True),
        sa.Column('daily_spent_usd', sa.Float(), server_default='0.0'),
        sa.Column('last_reset_day', sa.String(10), nullable=True),
        sa.Column('created_at', sa.DateTime(),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # model_aliases
    _create_table_safe(
        'model_aliases',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('alias', sa.String(200), nullable=False),
        sa.Column('target_model', sa.String(200), nullable=False),
        sa.Column('provider_override', sa.String(100), nullable=True),
        sa.Column('team_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(),
                  server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # api_keys — this table's ORM model (APIKeyRecord) lives in
    # app/api/endpoints/keys.py, not app/db/models.py, and until now no
    # migration ever created it — only a lazy runtime fallback
    # (_ensure_table() in that module) did, the first time a /v1/keys
    # endpoint happened to be hit. On a fresh DB where alembic runs before
    # any request does (e.g. a new Docker deploy), the rpm/tpm columns below
    # could never be added. Create the table here so it always exists.
    _create_table_safe(
        'api_keys',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('key_id', sa.String(length=32), nullable=False, unique=True),
        sa.Column('key_hash', sa.String(length=64), nullable=False, unique=True),
        sa.Column('name', sa.String(length=200), nullable=True),
        sa.Column('project', sa.String(length=200), nullable=True),
        sa.Column('scopes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.Integer(), nullable=False),
        sa.Column('rotated_at', sa.Integer(), nullable=True),
        sa.Column('expires_at', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('last_used_at', sa.Integer(), nullable=True),
    )

    # rpm_limit / tpm_limit columns (added after the table's initial shape above)
    _add_column_safe('api_keys', sa.Column('rpm_limit', sa.Integer(), nullable=True))
    _add_column_safe('api_keys', sa.Column('tpm_limit', sa.Integer(), nullable=True))


def downgrade() -> None:
    # Column drops are intentionally omitted — dropping production columns is
    # a manual, destructive operation. Downgrade only removes new tables.
    for table in ['model_aliases', 'users', 'teams', 'organizations',
                  'audit_log_entries', 'tenants']:
        if _has_table(table):
            op.drop_table(table)
