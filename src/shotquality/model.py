"""Make-probability models: a zone-average baseline, a logistic GAM-style model and gradient boosting.

None of these see player or team identity, so (actual - expected) isolates shot-making from shot selection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer

from . import config
from .features import CATEGORICAL_ACTION, CATEGORICAL_GEOM, FLAGS, NUMERIC

EPS = 1e-6


class ZoneBaseline:
    """League make rate by (basic zone, area, range); falls back to the basic zone for thin cells."""

    keys = ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE"]

    def __init__(self, min_cell: int = 100):
        self.min_cell = min_cell

    def fit(self, df: pd.DataFrame, y=None):
        y = df["SHOT_MADE_FLAG"] if y is None else y
        d = df[self.keys].assign(y=np.asarray(y))
        self.global_ = d["y"].mean()
        self.basic_ = d.groupby(self.keys[0])["y"].mean()
        cell = d.groupby(self.keys)["y"].agg(["mean", "size"])
        self.cell_ = cell[cell["size"] >= self.min_cell]["mean"]
        return self

    def predict_proba1(self, df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(df[self.keys])
        p = pd.Series(self.cell_.reindex(idx).to_numpy(), index=df.index)
        fallback = df[self.keys[0]].map(self.basic_).fillna(self.global_)
        return p.fillna(fallback).clip(EPS, 1 - EPS).to_numpy()


def _logit_pipeline(C: float = 1.0) -> Pipeline:
    pre = ColumnTransformer([
        ("dist", SplineTransformer(n_knots=10, degree=3, knots="quantile"), ["dist_ft"]),
        ("angle", SplineTransformer(n_knots=6, degree=3, knots="quantile"), ["angle_deg"]),
        ("flags", "passthrough", FLAGS),
        ("zones", OneHotEncoder(handle_unknown="ignore", min_frequency=200), CATEGORICAL_GEOM),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=500))])


def _hgb_pipeline(categorical: list[str], learning_rate=0.05, max_leaf_nodes=31,
                  l2_regularization=1.0, max_iter=400) -> Pipeline:
    cats = list(categorical)
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC + FLAGS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               min_frequency=config.MIN_ACTION_TYPE_SHOTS), cats),
    ])
    n_num = len(NUMERIC + FLAGS)
    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, l2_regularization=l2_regularization,
        max_iter=max_iter, min_samples_leaf=200,
        categorical_features=list(range(n_num, n_num + len(cats))), random_state=config.SEED,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


FEATURE_SETS = {
    "logit_geometry": NUMERIC + FLAGS + CATEGORICAL_GEOM,
    "gbm_geometry": NUMERIC + FLAGS + CATEGORICAL_GEOM,
    "gbm_full": NUMERIC + FLAGS + CATEGORICAL_GEOM + CATEGORICAL_ACTION,
}


def _build(name: str, **params) -> Pipeline:
    if name == "logit_geometry":
        return _logit_pipeline(**params)
    if name == "gbm_geometry":
        return _hgb_pipeline(CATEGORICAL_GEOM, **params)
    if name == "gbm_full":
        return _hgb_pipeline(CATEGORICAL_GEOM + CATEGORICAL_ACTION, **params)
    raise KeyError(name)


@dataclass
class ShotModel:
    """Thin wrapper so every model exposes `predict(df) -> make probability` on a feature frame."""

    name: str
    params: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "ShotModel":
        if self.name == "zone_baseline":
            self.est_ = ZoneBaseline().fit(df)
        else:
            self.est_ = _build(self.name, **self.params)
            cols = FEATURE_SETS[self.name]
            self.est_.fit(df[cols], df["SHOT_MADE_FLAG"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.name == "zone_baseline":
            return self.est_.predict_proba1(df)
        p = self.est_.predict_proba(df[FEATURE_SETS[self.name]])[:, 1]
        return np.clip(p, EPS, 1 - EPS)


# --- metrics -----------------------------------------------------------------------------------

def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Equal-frequency ECE: weighted mean |observed - predicted| across probability deciles."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(y) * abs(y[c].mean() - p[c].mean()) for c in chunks))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    chunks = np.array_split(np.argsort(p), bins)
    return pd.DataFrame({"predicted": [p[c].mean() for c in chunks],
                         "observed": [y[c].mean() for c in chunks],
                         "n": [len(c) for c in chunks]})


def score(y, p) -> dict:
    return {
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def game_cluster_bootstrap(game_ids, loss_a, loss_b, n_boot: int = 2000, seed: int = config.SEED) -> dict:
    """CI for mean(loss_a - loss_b), resampling whole games (shots within a game are not independent)."""
    games, inv = np.unique(np.asarray(game_ids), return_inverse=True)
    diff = np.asarray(loss_a) - np.asarray(loss_b)
    sums = np.bincount(inv, weights=diff, minlength=len(games))
    counts = np.bincount(inv, minlength=len(games)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(games), np.full(len(games), 1 / len(games)), size=n_boot)
    est = (draws @ sums) / (draws @ counts)
    return {"mean": float(sums.sum() / counts.sum()),
            "lo": float(np.percentile(est, 2.5)), "hi": float(np.percentile(est, 97.5))}


def pointwise_log_loss(y, p) -> np.ndarray:
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# --- tuning and persistence -----------------------------------------------------------------------

GBM_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [8, 15, 31], "l2_regularization": [1.0]}


def tune_gbm(name: str, train: pd.DataFrame, valid: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Pick hyper-parameters on a later validation season (never the test season)."""
    rows = []
    for values in product(*GBM_GRID.values()):
        params = dict(zip(GBM_GRID, values))
        m = ShotModel(name, params).fit(train)
        rows.append({**params, "valid_log_loss": log_loss(valid["SHOT_MADE_FLAG"], m.predict(valid))})
    table = pd.DataFrame(rows).sort_values("valid_log_loss").reset_index(drop=True)
    best = {"learning_rate": float(table.loc[0, "learning_rate"]),
            "max_leaf_nodes": int(table.loc[0, "max_leaf_nodes"]),
            "l2_regularization": float(table.loc[0, "l2_regularization"])}
    return best, table


def save(model: ShotModel, metadata: dict, directory: Path = config.MODEL_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model.name}.joblib"
    joblib.dump(model, path)
    (directory / f"{model.name}.json").write_text(json.dumps(metadata, indent=2, default=str))
    return path


def load(path: Path) -> ShotModel:
    return joblib.load(path)
