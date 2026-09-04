"""A process must not answer from a session copy the database has already moved past.

This is the multi-instance case. Behind a load balancer, consecutive turns from one
customer land on different instances, and each keeps its own in-memory _sessions dict.
The old reload condition was "have I never seen this session?", which is only ever right
with a single instance running.

The turns themselves are already serialised - durable_turn holds a Postgres advisory lock
for the whole turn, so two instances cannot run one customer's turns concurrently. What
was NOT guaranteed is that the instance which wins the lock then reads the state the
previous winner wrote.
"""
from src.graph.state import from_snapshot, to_snapshot


class _Row:
    """Stands in for the chatbot_conversations row durable_turn hands back."""

    def __init__(self, snapshot, version):
        self.state_snapshot = snapshot
        self.state_version = version


def _should_reload(sessions, session_id, row) -> bool:
    """The condition from _run_turn, isolated so it can be exercised without a database."""
    stored_version = int(getattr(row, "state_version", 0) or 0) if row else 0
    cached = sessions.get(session_id)
    cached_version = int((cached or {}).get("persisted_state_version") or 0)
    return bool(row and row.state_snapshot and (cached is None or cached_version < stored_version))


def _state_after_turn(version, **fields):
    state = {"session_id": "PSID", "persisted_state_version": version}
    state.update(fields)
    return state


def test_a_cold_process_loads_the_snapshot():
    row = _Row(to_snapshot(_state_after_turn(1)), 1)
    assert _should_reload({}, "PSID", row) is True


def test_a_process_holding_the_current_version_does_not_reload():
    """The single-instance path, and the common one: no needless database round trip."""
    row = _Row(to_snapshot(_state_after_turn(3)), 3)
    assert _should_reload({"PSID": _state_after_turn(3)}, "PSID", row) is False


def test_a_process_left_behind_by_another_instance_reloads():
    """The bug. Instance A cached turn 1; instance B has since committed turn 2."""
    row = _Row(to_snapshot(_state_after_turn(2)), 2)
    stale = {"PSID": _state_after_turn(1)}
    assert _should_reload(stale, "PSID", row) is True


def test_the_reloaded_state_carries_the_other_instance_s_answers():
    """What the customer actually notices: the bot must not forget what it was told."""
    written_by_b = to_snapshot(_state_after_turn(2, category="Utility", slots={"haul_item": "ATV"}))
    row = _Row(written_by_b, 2)
    sessions = {"PSID": _state_after_turn(1, category=None, slots={})}
    assert _should_reload(sessions, "PSID", row)
    sessions["PSID"] = from_snapshot(row.state_snapshot)
    assert sessions["PSID"]["slots"] == {"haul_item": "ATV"}
    assert sessions["PSID"]["category"] == "Utility"


def test_the_restored_version_stops_it_reloading_again():
    """persisted_state_version must survive the snapshot round trip, or every turn reloads."""
    row = _Row(to_snapshot(_state_after_turn(2)), 2)
    restored = from_snapshot(row.state_snapshot)
    assert restored["persisted_state_version"] == 2
    assert _should_reload({"PSID": restored}, "PSID", row) is False


def test_a_brand_new_session_has_no_row_to_load():
    assert _should_reload({}, "PSID", None) is False


def test_a_row_written_before_this_field_existed_still_reloads_a_cold_process():
    """Rows already in Postgres have no persisted_state_version inside their snapshot."""
    legacy = {"session_id": "PSID", "state_schema_version": 1, "slots": {"haul_item": "ATV"}}
    assert _should_reload({}, "PSID", _Row(legacy, 5)) is True
