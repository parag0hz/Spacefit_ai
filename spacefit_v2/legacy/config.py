"""Configuration used by the standalone compatibility modules."""
from __future__ import annotations

import os


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_TEMPERATURE = 0.2
OPENAI_MAX_RETRIES = 3

SEARCH_STEP_XY = 0.1
ROTATION_CANDIDATES = [i * 15 for i in range(24)]
WALKWAY_WIDTH = 0.6

WEIGHTS = {
    "collision_free": 10.0,
    "in_boundary": 10.0,
    "door_clearance": 5.0,
    "wall_alignment": 4.0,
    "wall_proximity": 3.5,
    "zone_fit": 3.0,
    "walkway": 4.0,
    "preferred_face": 4.0,
    "semantic_relation": 7.0,
    "centroid_distance": 0.15,
}

