import pytest

from conftest import make_shots
from shotquality import data


def test_clean_frame_passes():
    report = data.validate(make_shots())
    assert report["inconsistent_shot_type_rows"] == 0


def test_duplicate_events_rejected():
    df = make_shots()
    df.loc[1, "GAME_EVENT_ID"] = df.loc[0, "GAME_EVENT_ID"]
    df.loc[1, "GAME_ID"] = df.loc[0, "GAME_ID"]
    with pytest.raises(ValueError, match="duplicate"):
        data.validate(df)


def test_non_binary_made_flag_rejected():
    df = make_shots()
    df.loc[0, "SHOT_MADE_FLAG"] = 2
    with pytest.raises(ValueError, match="binary"):
        data.validate(df)


def test_distance_must_match_coordinates():
    df = make_shots()
    df.loc[0, "SHOT_DISTANCE"] = df.loc[0, "SHOT_DISTANCE"] + 10
    with pytest.raises(ValueError, match="disagrees"):
        data.validate(df)


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        data.validate(make_shots().drop(columns=["LOC_X"]))
