from __future__ import annotations

import json
import os
from typing import Any


CATEGORY_DEFAULTS: dict[str, dict[str, Any]] = {
    # Flatbeds default to an 8 ft deck width unless the customer supplies a
    # different width. Defaults are stored with source="default", so a width
    # extracted from the same or any later turn replaces this value normally.
    "Flatbed": {"trailer_width_ft": 8.0},
    # Example:
    # "Utility": {"item_or_trailer_width_ft": 6.92, "hitch_type": "Bumper Pull"},
}

# Live-scenario seed hook (M5 `defaults-applied` scenario): set before starting the
# server, e.g. CATEGORY_DEFAULTS_JSON='{"Utility": {"hitch_type": "Bumper Pull"}}'.
# No-op when unset, so it can never affect any other scenario/milestone's behavior.
_env_overrides = os.getenv("CATEGORY_DEFAULTS_JSON")
if _env_overrides:
    CATEGORY_DEFAULTS.update(json.loads(_env_overrides))


def defaults_for(category: str) -> dict[str, Any]:
    return dict(CATEGORY_DEFAULTS.get(category, {}))
