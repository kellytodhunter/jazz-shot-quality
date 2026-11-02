"""Check the SQL layer against numbers worked out by hand on a four-shot, two-team game."""
import duckdb
import pandas as pd

from shotquality import config


def test_team_shot_quality_sql_hand_computed():
    scored = pd.DataFrame({
        "season": 2024, "GAME_ID": 1,
        "TEAM_NAME": ["A", "A", "B", "B"],
        "pts": [2, 0, 3, 0], "xpts": [1.0, 1.0, 1.5, 1.5],
    })
    con = duckdb.connect()
    con.register("scored", scored)
    out = con.execute((config.SQL_DIR / "team_shot_quality.sql").read_text(), [2024] * 4).df().set_index("team")
    assert out.loc["A", "pps"] == 1.0 and out.loc["A", "xpps"] == 1.0 and out.loc["A", "shotmaking"] == 0.0
    assert out.loc["B", "shotmaking"] == 0.0
    # A's defence faces B's shots: B scored 1.5 pts/shot against 1.5 expected
    assert out.loc["A", "opp_pps"] == 1.5 and out.loc["A", "opp_xpps"] == 1.5
    assert out.loc["A", "opp_fga"] == 2
