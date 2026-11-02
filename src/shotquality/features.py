"""Feature engineering. Pure functions of one shot row so the same code scores historical and new data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CORNER_ZONES = {"Left Corner 3", "Right Corner 3"}

NUMERIC = ["x_ft", "y_ft", "dist_ft", "angle_deg", "sec_left", "period_c"]
FLAGS = ["is_three", "is_corner3", "backcourt", "end_of_period"]
CATEGORICAL_GEOM = ["zone_basic", "zone_area"]
CATEGORICAL_ACTION = ["action_type"]

ZONE_GROUPS = {
    "Restricted Area": "Rim",
    "In The Paint (Non-RA)": "Paint (non-RA)",
    "Mid-Range": "Mid-range",
    "Left Corner 3": "Corner 3",
    "Right Corner 3": "Corner 3",
    "Above the Break 3": "Above-break 3",
    "Backcourt": "Backcourt",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with model features added (original columns preserved)."""
    out = df.copy()
    x = out["LOC_X"].to_numpy(dtype=float) / 10.0
    y = out["LOC_Y"].to_numpy(dtype=float) / 10.0

    out["x_ft"] = np.abs(x)  # the court is left/right symmetric for shot quality
    out["y_ft"] = y
    out["dist_ft"] = np.hypot(x, y)
    out["angle_deg"] = np.degrees(np.arctan2(np.abs(x), y))  # >90 deg means behind the basket line
    out["sec_left"] = out["MINUTES_REMAINING"] * 60 + out["SECONDS_REMAINING"]
    out["period_c"] = out["PERIOD"].clip(upper=5)  # all overtimes share a bucket

    out["is_three"] = (out["SHOT_TYPE"] == "3PT Field Goal").astype(int)
    out["is_corner3"] = out["SHOT_ZONE_BASIC"].isin(CORNER_ZONES).astype(int)
    out["backcourt"] = (out["y_ft"] > config.HALF_COURT_FT).astype(int)
    out["end_of_period"] = (out["sec_left"] <= 3).astype(int)

    out["zone_basic"] = out["SHOT_ZONE_BASIC"]
    out["zone_area"] = out["SHOT_ZONE_AREA"]
    out["action_type"] = out["ACTION_TYPE"]
    out["zone_group"] = out["SHOT_ZONE_BASIC"].map(ZONE_GROUPS).fillna("Other")

    out["pt_value"] = np.where(out["is_three"] == 1, 3, 2)
    return out


def point_value(is_three) -> np.ndarray:
    return np.where(np.asarray(is_three) == 1, 3, 2)
