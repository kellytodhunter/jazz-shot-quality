import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_shots(n=200, seed=0) -> pd.DataFrame:
    """Small synthetic frame with the raw schema the pipeline expects."""
    rng = np.random.default_rng(seed)
    x = rng.integers(-240, 240, n)
    y = rng.integers(-40, 420, n)
    dist = np.hypot(x, y) / 10
    three = dist >= 23.75
    return pd.DataFrame({
        "GAME_ID": rng.integers(1, 20, n), "GAME_EVENT_ID": np.arange(n), "PLAYER_ID": rng.integers(1, 6, n),
        "PLAYER_NAME": "P", "TEAM_ID": 1, "TEAM_NAME": "A", "PERIOD": rng.integers(1, 5, n),
        "MINUTES_REMAINING": rng.integers(0, 12, n), "SECONDS_REMAINING": rng.integers(0, 60, n),
        "ACTION_TYPE": "Jump Shot", "SHOT_TYPE": np.where(three, "3PT Field Goal", "2PT Field Goal"),
        "SHOT_ZONE_BASIC": "Mid-Range", "SHOT_ZONE_AREA": "Center(C)", "SHOT_ZONE_RANGE": "8-16 ft.",
        "SHOT_DISTANCE": dist.round().astype(int), "LOC_X": x, "LOC_Y": y,
        "SHOT_MADE_FLAG": rng.integers(0, 2, n), "GAME_DATE": 20250101, "HTM": "AAA", "VTM": "BBB",
    })


@pytest.fixture
def shots():
    return make_shots()
