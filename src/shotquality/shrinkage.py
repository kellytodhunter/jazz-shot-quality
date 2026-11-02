"""Empirical-Bayes shrinkage for noisy per-player shot-making estimates.

Each player's mean residual (actual points - expected points per shot) is modelled as
    observed_i ~ Normal(true_i, se_i^2),   true_i ~ Normal(mu, tau^2)
with tau^2 estimated from the league by the DerSimonian-Laird moment estimator. The posterior mean pulls
low-volume players toward the league mean in proportion to how noisy their sample is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Prior:
    mu: float
    tau2: float

    @property
    def tau(self) -> float:
        return float(np.sqrt(self.tau2))


def fit_prior(observed, se2) -> Prior:
    """DerSimonian-Laird estimate of the league mean and true-talent variance."""
    y, se2 = np.asarray(observed, float), np.asarray(se2, float)
    if len(y) < 3:
        raise ValueError("need at least 3 players to estimate a prior")
    w = 1.0 / se2
    mu_fixed = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - mu_fixed) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (len(y) - 1)) / c)
    w_star = 1.0 / (se2 + tau2)
    mu = float(np.sum(w_star * y) / np.sum(w_star))
    return Prior(mu=mu, tau2=float(tau2))


def shrink(observed, se2, prior: Prior, level: float = 0.90) -> pd.DataFrame:
    """Posterior mean, sd and central credible interval per player."""
    y, se2 = np.asarray(observed, float), np.asarray(se2, float)
    if prior.tau2 == 0:
        weight = np.zeros_like(y)
    else:
        weight = prior.tau2 / (prior.tau2 + se2)  # share of the player's own signal that is kept
    post = prior.mu + weight * (y - prior.mu)
    sd = np.sqrt(weight * se2)
    z = stats.norm.ppf(0.5 + level / 2)
    return pd.DataFrame({"raw": y, "shrunk": post, "weight": weight, "post_sd": sd,
                         "lo": post - z * sd, "hi": post + z * sd})


def player_table(df: pd.DataFrame, group_cols, min_shots: int = 1) -> pd.DataFrame:
    """Aggregate shot-level `pts`, `xpts`, `var_pts` into per-group estimates of points over expected per shot."""
    g = df.groupby(group_cols, as_index=False).agg(
        shots=("pts", "size"), pts=("pts", "sum"), xpts=("xpts", "sum"), var_sum=("var_pts", "sum"))
    g = g[g["shots"] >= min_shots].copy()
    g["xpps"] = g["xpts"] / g["shots"]
    g["pps"] = g["pts"] / g["shots"]
    g["observed"] = (g["pts"] - g["xpts"]) / g["shots"]
    g["se2"] = g["var_sum"] / g["shots"] ** 2  # Var of a mean of independent Bernoulli*points outcomes
    return g.reset_index(drop=True)


def forecast_skill(history: pd.DataFrame, future: pd.DataFrame) -> dict:
    """Out-of-time check: do shrunk past estimates predict next-season shot-making better than raw ones?

    `history`/`future` come from `player_table`. Errors are weighted by the player's future shot count so the
    comparison reflects how much each player's outcome can be trusted.
    """
    prior = fit_prior(history["observed"], history["se2"])
    est = shrink(history["observed"], history["se2"], prior)
    merged = history[["PLAYER_ID"]].assign(raw=est["raw"].values, shrunk=est["shrunk"].values) \
        .merge(future[["PLAYER_ID", "observed", "shots"]], on="PLAYER_ID", suffixes=("", "_next"))
    w = merged["shots"].to_numpy(float)

    def wmse(pred):
        return float(np.sum(w * (merged["observed"] - pred) ** 2) / w.sum())

    return {
        "players": int(len(merged)),
        "prior_mu": prior.mu,
        "true_talent_sd_per_shot": prior.tau,
        "mse_league_average": wmse(prior.mu),
        "mse_raw": wmse(merged["raw"]),
        "mse_shrunk": wmse(merged["shrunk"]),
        "corr_raw_vs_next": float(np.corrcoef(merged["raw"], merged["observed"])[0, 1]),
        "corr_shrunk_vs_next": float(np.corrcoef(merged["shrunk"], merged["observed"])[0, 1]),
        "merged": merged,
    }
