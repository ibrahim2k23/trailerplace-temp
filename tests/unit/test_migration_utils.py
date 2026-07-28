"""The existence checks that let `alembic upgrade head` run on a populated database.

These matter because this project's databases were created with create_all()
and carry no alembic_version row, so every revision replays against tables that
already exist. If a check ever returns the wrong answer, a migration either
crashes on a duplicate object or silently rewrites live data.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import sqlalchemy as sa

from src import migration_utils


@pytest.fixture()
def bound_engine():
    """A real database, so the checks run through SQLAlchemy's real inspector."""
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    sa.Table(
        "widgets",
        metadata,
        sa.Column("widget_id", sa.String(32), primary_key=True),
        sa.Column("label", sa.String(64)),
    )
    metadata.create_all(engine)
    with engine.connect() as connection:
        connection.execute(sa.text("CREATE INDEX ix_widgets_label ON widgets (label)"))
        connection.commit()
        with patch.object(migration_utils.op, "get_bind", return_value=connection):
            yield connection


def test_has_table_distinguishes_present_from_absent(bound_engine):
    assert migration_utils.has_table("widgets") is True
    assert migration_utils.has_table("trailer_listings") is False


def test_has_column_reports_columns_of_an_existing_table(bound_engine):
    assert migration_utils.has_column("widgets", "label") is True
    assert migration_utils.has_column("widgets", "not_a_column") is False


def test_has_column_is_false_rather_than_raising_for_a_missing_table(bound_engine):
    # A revision may ask about a column on a table an earlier revision creates.
    # Raising here would abort the upgrade instead of guarding it.
    assert migration_utils.has_column("nonexistent", "anything") is False


def test_has_index_reports_indexes_of_an_existing_table(bound_engine):
    assert migration_utils.has_index("widgets", "ix_widgets_label") is True
    assert migration_utils.has_index("widgets", "ix_widgets_missing") is False


def test_has_index_is_false_rather_than_raising_for_a_missing_table(bound_engine):
    assert migration_utils.has_index("nonexistent", "ix_anything") is False


def test_checks_see_changes_made_after_the_first_call(bound_engine):
    """The inspector must not be cached across calls.

    A migration creates a table and then asks about its indexes in the same
    run; a cached inspector would still report the pre-migration schema.
    """
    assert migration_utils.has_table("late_table") is False
    bound_engine.execute(sa.text("CREATE TABLE late_table (id INTEGER PRIMARY KEY)"))
    assert migration_utils.has_table("late_table") is True
