"""Single place for paths, seasons and modelling constants so runs are reproducible."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = ROOT / "models"
SQL_DIR = ROOT / "sql"

DATA_INDEX_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"

# Season key is the calendar year the season started (2024 == 2024-25).
SEASONS = [2021, 2022, 2023, 2024]
TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASON = 2024
TEAM = "Utah Jazz"

SEED = 20260
RIM_Y_OFFSET_FT = 0.0  # shot-chart coordinates are already centred on the basket (tenths of feet)
HALF_COURT_FT = 41.75  # basket is 5.25 ft from the baseline, half court is 47 ft
THREE_POINT_CORNER_FT = 22.0
THREE_POINT_ARC_FT = 23.75

MIN_ACTION_TYPE_SHOTS = 500
MIN_SHOTS_PLAYER = 300
