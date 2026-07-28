"""Existence checks so migrations can run against an already-populated database.

This project's databases were originally created with ``Base.metadata.create_all``
rather than Alembic, so a live database can hold the tables while carrying no
``alembic_version`` row at all. Running ``alembic upgrade head`` there replays
every revision from the baseline, which would collide with the tables that are
already present.

Every migration therefore asks before it acts: create what is missing, leave
what exists untouched. That makes ``upgrade head`` safe to run repeatedly, on a
blank database and on a fully populated one alike.

Import from migrations as ``from src.migration_utils import has_table`` —
alembic.ini sets ``prepend_sys_path = .`` so the repository root is importable.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


def _inspector() -> sa.Inspector:
    # Built per call, never cached: a migration inspects the schema again after
    # its own DDL, and a stale inspector would report the database as it looked
    # before this revision started.
    return sa.inspect(op.get_bind())


def has_table(table: str) -> bool:
    return _inspector().has_table(table)


def has_column(table: str, column: str) -> bool:
    if not has_table(table):
        return False
    return any(existing["name"] == column for existing in _inspector().get_columns(table))


def has_index(table: str, index: str) -> bool:
    if not has_table(table):
        return False
    inspector = _inspector()
    if any(existing["name"] == index for existing in inspector.get_indexes(table)):
        return True
    # A UNIQUE CONSTRAINT is backed by an index of the same name but is not
    # reported by get_indexes, so a unique index created either way is found.
    return any(
        existing["name"] == index for existing in inspector.get_unique_constraints(table)
    )
