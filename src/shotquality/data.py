"""Download raw shot-chart files, validate them, and materialise a clean DuckDB/Parquet layer."""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

from . import config

REQUIRED_COLUMNS = [
    "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME", "PERIOD",
    "MINUTES_REMAINING", "SECONDS_REMAINING", "ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "SHOT_DISTANCE", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG",
    "GAME_DATE", "HTM", "VTM",
]


def _index() -> dict:
    with urllib.request.urlopen(config.DATA_INDEX_URL, timeout=30) as r:
        lines = r.read().decode().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def download(seasons=config.SEASONS, raw_dir: Path = config.RAW_DIR) -> list[Path]:
    """Idempotently fetch and unpack one CSV per season."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    index = None
    paths = []
    for season in seasons:
        csv = raw_dir / f"shotdetail_{season}.csv"
        if not csv.exists():
            index = index or _index()
            archive = raw_dir / f"shotdetail_{season}.tar.xz"
            urllib.request.urlretrieve(index[f"shotdetail_{season}"], archive)
            with tarfile.open(archive) as tf:
                tf.extractall(raw_dir)
            archive.unlink()
        paths.append(csv)
    return paths


def validate(df: pd.DataFrame) -> dict:
    """Hard data-quality checks. Raises on anything that would silently bias a model."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    report = {"rows": int(len(df))}
    dupes = int(df.duplicated(["GAME_ID", "GAME_EVENT_ID"]).sum())
    if dupes:
        raise ValueError(f"{dupes} duplicate (GAME_ID, GAME_EVENT_ID) rows")
    if not df["SHOT_MADE_FLAG"].isin([0, 1]).all():
        raise ValueError("SHOT_MADE_FLAG must be binary")
    if not df["SHOT_TYPE"].isin(["2PT Field Goal", "3PT Field Goal"]).all():
        raise ValueError("unexpected SHOT_TYPE values")
    if df[["LOC_X", "LOC_Y", "SHOT_DISTANCE", "PERIOD"]].isna().any().any():
        raise ValueError("nulls in geometry or period columns")

    # The published SHOT_DISTANCE should match the coordinates to within rounding.
    implied = (df["LOC_X"] ** 2 + df["LOC_Y"] ** 2) ** 0.5 / 10
    max_gap = float((implied - df["SHOT_DISTANCE"]).abs().max())
    report["max_distance_gap_ft"] = round(max_gap, 2)
    if max_gap > 1.5:
        raise ValueError(f"SHOT_DISTANCE disagrees with LOC_X/LOC_Y by {max_gap:.1f} ft")

    # A three taken well inside the arc (or a two well outside) means a bad row.
    bad3 = ((df["SHOT_TYPE"] == "3PT Field Goal") & (df["SHOT_DISTANCE"] < 21)).sum()
    bad2 = ((df["SHOT_TYPE"] == "2PT Field Goal") & (df["SHOT_DISTANCE"] > 25)).sum()
    report["inconsistent_shot_type_rows"] = int(bad3 + bad2)
    if (bad3 + bad2) / len(df) > 0.001:
        raise ValueError("too many shot-type/distance contradictions")
    return report


def build_database(seasons=config.SEASONS, raw_dir: Path = config.RAW_DIR,
                   out_dir: Path = config.PROCESSED_DIR) -> tuple[duckdb.DuckDBPyConnection, dict]:
    """Load raw CSVs into one validated `shots` table; persist as Parquet and return a connection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, reports = [], {}
    for season in seasons:
        df = pd.read_csv(raw_dir / f"shotdetail_{season}.csv")
        reports[season] = validate(df)
        df["season"] = season
        frames.append(df)
    shots = pd.concat(frames, ignore_index=True)
    shots["game_date"] = pd.to_datetime(shots["GAME_DATE"].astype(str), format="%Y%m%d")
    shots.drop(columns=["GAME_DATE", "GRID_TYPE", "SHOT_ATTEMPTED_FLAG", "EVENT_TYPE"], errors="ignore") \
         .to_parquet(out_dir / "shots.parquet", index=False)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW shots AS SELECT * FROM read_parquet('{out_dir / 'shots.parquet'}')")
    return con, reports
