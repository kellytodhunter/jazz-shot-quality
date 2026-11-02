import numpy as np
import pandas as pd

from shotquality import features


def row(**kw):
    base = dict(LOC_X=0, LOC_Y=0, MINUTES_REMAINING=5, SECONDS_REMAINING=30, PERIOD=1,
                SHOT_TYPE="2PT Field Goal", SHOT_ZONE_BASIC="Restricted Area", SHOT_ZONE_AREA="Center(C)",
                ACTION_TYPE="Layup Shot")
    base.update(kw)
    return pd.DataFrame([base])


def test_distance_and_angle_from_coordinates():
    f = features.add_features(row(LOC_X=0, LOC_Y=100)).iloc[0]       # 10 ft straight out
    assert f.dist_ft == 10 and f.angle_deg == 0
    f = features.add_features(row(LOC_X=-100, LOC_Y=0)).iloc[0]      # 10 ft along the baseline line
    assert f.dist_ft == 10 and f.angle_deg == 90


def test_shots_from_behind_the_basket_have_angle_over_90():
    assert features.add_features(row(LOC_X=-30, LOC_Y=-20)).iloc[0].angle_deg > 90


def test_left_right_symmetry():
    a = features.add_features(row(LOC_X=-150, LOC_Y=120)).iloc[0]
    b = features.add_features(row(LOC_X=150, LOC_Y=120)).iloc[0]
    assert a.x_ft == b.x_ft and a.angle_deg == b.angle_deg and a.dist_ft == b.dist_ft


def test_corner_three_and_point_value():
    f = features.add_features(row(SHOT_TYPE="3PT Field Goal", SHOT_ZONE_BASIC="Left Corner 3")).iloc[0]
    assert f.is_corner3 == 1 and f.is_three == 1 and f.pt_value == 3
    assert features.add_features(row()).iloc[0].pt_value == 2


def test_clock_and_period_features():
    f = features.add_features(row(MINUTES_REMAINING=0, SECONDS_REMAINING=2, PERIOD=7)).iloc[0]
    assert f.sec_left == 2 and f.end_of_period == 1 and f.period_c == 5
    assert features.add_features(row()).iloc[0].sec_left == 330


def test_backcourt_flag_and_zone_group():
    assert features.add_features(row(LOC_Y=600)).iloc[0].backcourt == 1
    assert features.add_features(row(LOC_Y=100)).iloc[0].backcourt == 0
    assert features.add_features(row()).iloc[0].zone_group == "Rim"


def test_add_features_does_not_mutate_input(shots):
    before = shots.copy()
    features.add_features(shots)
    pd.testing.assert_frame_equal(before, shots)
