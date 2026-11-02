"""End-to-end run: validate data -> tune -> out-of-time evaluation -> cross-fitted xPTS -> shot-making study -> Jazz tables/figures.

Everything downstream of the raw CSVs is deterministic (fixed seeds), so `make pipeline` reproduces reports/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from shotquality import config, data, features, model, shrinkage  # noqa: E402

NAVY, GOLD, GREY, GREEN = "#002B5C", "#F9A01B", "#8A8F98", "#00471B"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 150})
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def save_fig(fig, name: str) -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- 1. data
con, validation = data.build_database()
log(f"data validated: {sum(v['rows'] for v in validation.values()):,} shots")
df = features.add_features(con.sql("SELECT * FROM shots").df())
train = df[df.season.isin(config.TRAIN_SEASONS)]
test = df[df.season == config.TEST_SEASON]
y_test = test["SHOT_MADE_FLAG"].to_numpy()

# ------------------------------------------------------------------------------- 2. tune on a later season
tune_train, tune_valid = df[df.season.isin([2021, 2022])], df[df.season == 2023]
best, tuning = {}, {}
for name in ["gbm_geometry", "gbm_full"]:
    best[name], tuning[name] = model.tune_gbm(name, tune_train, tune_valid)
    log(f"{name}: best params {best[name]} (valid log loss {tuning[name].loc[0, 'valid_log_loss']:.4f})")

# --------------------------------------------------------------- 3. final fit + out-of-time test (2024-25)
models = {
    "zone_baseline": model.ShotModel("zone_baseline"),
    "logit_geometry": model.ShotModel("logit_geometry"),
    "gbm_geometry": model.ShotModel("gbm_geometry", best["gbm_geometry"]),
    "gbm_full": model.ShotModel("gbm_full", best["gbm_full"]),
}
test_pred, metrics, boot = {}, {}, {}
for name, m in models.items():
    m.fit(train)
    test_pred[name] = m.predict(test)
    metrics[name] = model.score(y_test, test_pred[name])
    log(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics[name].items()))
base_loss = model.pointwise_log_loss(y_test, test_pred["zone_baseline"])
for name in models:
    if name != "zone_baseline":
        boot[name] = model.game_cluster_bootstrap(
            test["GAME_ID"], base_loss, model.pointwise_log_loss(y_test, test_pred[name]))
# also: does the full model beat geometry-only? (isolates what the action-type label buys)
boot["gbm_full_vs_gbm_geometry"] = model.game_cluster_bootstrap(
    test["GAME_ID"], model.pointwise_log_loss(y_test, test_pred["gbm_geometry"]),
    model.pointwise_log_loss(y_test, test_pred["gbm_full"]))

meta = {"train_seasons": config.TRAIN_SEASONS, "test_season": config.TEST_SEASON, "metrics": metrics,
        "params": best, "n_train": int(len(train))}
for name in ["gbm_full", "gbm_geometry"]:
    model.save(models[name], {**meta, "model": name, "features": model.FEATURE_SETS[name]})

# ------------------------------------------ 4. cross-fitted expected make probability for every season
# Seasons 2021-23 are scored by a model trained on the *other two* training seasons, season 2024 by the
# final model. No season's own shots (and no 2024 data for earlier seasons) influence its expectation.
xfg = {n: pd.Series(index=df.index, dtype=float) for n in ["gbm_full", "gbm_geometry"]}
for name in xfg:
    for s in config.TRAIN_SEASONS:
        fit_on = df[df.season.isin([t for t in config.TRAIN_SEASONS if t != s])]
        m = model.ShotModel(name, best[name]).fit(fit_on)
        mask = df.season == s
        xfg[name][mask] = m.predict(df[mask])
    xfg[name][df.season == config.TEST_SEASON] = test_pred[name]
    log(f"cross-fitted {name}")

scored = df.copy()
for name, series in xfg.items():
    tag = "full" if name == "gbm_full" else "geom"
    scored[f"p_{tag}"] = series
scored["pts"] = scored["SHOT_MADE_FLAG"] * scored["pt_value"]
scored["xpts"] = scored["p_full"] * scored["pt_value"]
scored["var_pts"] = scored["pt_value"] ** 2 * scored["p_full"] * (1 - scored["p_full"])
scored["xpts_geom"] = scored["p_geom"] * scored["pt_value"]
scored["var_pts_geom"] = scored["pt_value"] ** 2 * scored["p_geom"] * (1 - scored["p_geom"])
con.register("scored", scored)

# --------------------------------------------------------------------------------- 5. SQL aggregates
season, team = config.TEST_SEASON, config.TEAM
team_tbl = con.execute((config.SQL_DIR / "team_shot_quality.sql").read_text(), [season] * 4).df()
zone_tbl = con.execute((config.SQL_DIR / "zone_profile.sql").read_text(),
                       [season, season, team, season, team, season, season]).df()
team_tbl.to_csv(config.REPORT_DIR / "team_shot_quality_2024.csv", index=False)
zone_tbl.to_csv(config.REPORT_DIR / "jazz_zone_profile_2024.csv", index=False)
jazz_team = team_tbl[team_tbl.team == team].iloc[0].to_dict()
log("SQL aggregates done")

# Does a team's shot quality / shot-making (and what it allows) carry over to the next season?
team_by_season = {s_: con.execute((config.SQL_DIR / "team_shot_quality.sql").read_text(), [s_] * 4).df()
                  .set_index("team") for s_ in config.SEASONS}
team_persistence = {}
for metric in ["xpps", "shotmaking", "opp_xpps", "opp_shotmaking"]:
    pairs = pd.concat([pd.DataFrame({"a": team_by_season[s_ - 1][metric], "b": team_by_season[s_][metric]}).dropna()
                       for s_ in config.SEASONS[1:]])
    team_persistence[metric] = {"r": float(pairs["a"].corr(pairs["b"])), "n_pairs": int(len(pairs)),
                                "league_sd": float(pairs["b"].std())}
log("team year-over-year correlations: " + ", ".join(f"{k}={v['r']:.2f}" for k, v in team_persistence.items()))

# ------------------------------------------------- 6. how much of shot-making is signal? (talent spread)
def tag_shots(frame, kind):
    if kind == "all":
        return frame
    return frame[frame["is_three"] == (1 if kind == "3PT" else 0)]

talent_rows = []
for s in config.SEASONS:
    for kind, min_n in [("all", 200), ("3PT", 100), ("2PT", 150)]:
        tbl = shrinkage.player_table(tag_shots(scored[scored.season == s], kind), ["PLAYER_ID"], min_n)
        prior = shrinkage.fit_prior(tbl["observed"], tbl["se2"])
        talent_rows.append({"season": s, "shots": kind, "players": len(tbl),
                            "talent_sd_per100": prior.tau * 100,
                            "noise_sd_per100_median": float(np.sqrt(tbl["se2"].median()) * 100),
                            "median_shots": int(tbl["shots"].median())})
talent = pd.DataFrame(talent_rows)
talent.to_csv(config.REPORT_DIR / "shotmaking_talent_spread.csv", index=False)
log("talent spread:\n" + talent.to_string(index=False))

# ------------------------------------------- 7. does shrinkage forecast next season better than raw?
def forecast(kind: str, hist_min: int, fut_min: int, col_x="xpts", col_v="var_pts") -> dict:
    d = scored.assign(xpts=scored[col_x], var_pts=scored[col_v])
    hist = shrinkage.player_table(tag_shots(d[d.season.isin(config.TRAIN_SEASONS)], kind), ["PLAYER_ID"], hist_min)
    fut = shrinkage.player_table(tag_shots(d[d.season == config.TEST_SEASON], kind), ["PLAYER_ID"], fut_min)
    return shrinkage.forecast_skill(hist, fut)

fc, fc_merged = {}, {}
for kind, hmin, fmin in [("all", 500, 200), ("3PT", 200, 100), ("2PT", 300, 150)]:
    r = forecast(kind, hmin, fmin)
    fc_merged[kind] = r.pop("merged")
    fc[kind] = r
    log(f"forecast {kind}: n={r['players']}, mse league={r['mse_league_average']:.5f} raw={r['mse_raw']:.5f} "
        f"shrunk={r['mse_shrunk']:.5f}, corr raw={r['corr_raw_vs_next']:.3f} shrunk={r['corr_shrunk_vs_next']:.3f}")
fc_geom = forecast("all", 500, 200, "xpts_geom", "var_pts_geom")
fc_geom.pop("merged")

# bootstrap CI on MSE gain of shrunk over raw (resample players)
def mse_gain_ci(merged, n_boot=2000):
    rng = np.random.default_rng(config.SEED)
    w = merged["shots"].to_numpy(float)
    e_raw = (merged["observed"] - merged["raw"]).to_numpy() ** 2
    e_shr = (merged["observed"] - merged["shrunk"]).to_numpy() ** 2
    idx = rng.integers(0, len(merged), size=(n_boot, len(merged)))
    gains = [(np.sum(w[i] * e_raw[i]) - np.sum(w[i] * e_shr[i])) / w[i].sum() for i in idx]
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]

for kind in fc:
    fc[kind]["mse_gain_shrunk_vs_raw_ci95"] = mse_gain_ci(fc_merged[kind])

# --------------------------------------------------------------- 8. Jazz 2024-25 player shot-making
league24 = shrinkage.player_table(scored[scored.season == season], ["PLAYER_ID"], 100)
prior24 = shrinkage.fit_prior(league24["observed"], league24["se2"])
jz = scored[(scored.season == season) & (scored.TEAM_NAME == team)]
jz_tbl = shrinkage.player_table(jz, ["PLAYER_ID", "PLAYER_NAME"], 150)
jz_est = shrinkage.shrink(jz_tbl["observed"], jz_tbl["se2"], prior24)
jazz_players = pd.concat([jz_tbl[["PLAYER_NAME", "shots", "xpps", "pps"]], jz_est], axis=1)
# robustness: same estimate from the geometry-only model
jz_geom = shrinkage.player_table(jz.assign(xpts=jz["xpts_geom"], var_pts=jz["var_pts_geom"]),
                                 ["PLAYER_ID", "PLAYER_NAME"], 150)
jazz_players["raw_geometry_model"] = jz_geom["observed"].values
rho = stats.spearmanr(jazz_players["raw"], jazz_players["raw_geometry_model"]).correlation
for c in ["raw", "shrunk", "lo", "hi", "post_sd", "raw_geometry_model"]:
    jazz_players[c + "_per100"] = jazz_players[c] * 100
jazz_players = jazz_players.sort_values("shrunk", ascending=False).reset_index(drop=True)
jazz_players.to_csv(config.REPORT_DIR / "jazz_players_2024.csv", index=False)
log(f"Jazz players: {len(jazz_players)}; rank corr full-vs-geometry model raw SMOE = {rho:.2f}")

# ----------------------------------------------------------------------------------------- 9. figures
# Fig 1: calibration + improvement over baseline
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
ax[0].plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
for name, color, label in [("zone_baseline", GREY, "Zone baseline"), ("gbm_full", NAVY, "Gradient boosting")]:
    cal = model.calibration_table(y_test, test_pred[name])
    ax[0].plot(cal.predicted, cal.observed, marker="o", ms=4, color=color, label=label)
ax[0].set(xlim=(0.2, 0.8), ylim=(0.2, 0.8), xlabel="Predicted make probability (decile mean)",
          ylabel="Observed make rate", title="Calibration on held-out 2024-25")
ax[0].legend(frameon=False, loc="upper left")
names = ["logit_geometry", "gbm_geometry", "gbm_full"]
labels = ["Logistic\n(geometry)", "Boosting\n(geometry)", "Boosting\n(+ shot type)"]
vals = [boot[n]["mean"] * 1000 for n in names]
err = [[(boot[n]["mean"] - boot[n]["lo"]) * 1000 for n in names], [(boot[n]["hi"] - boot[n]["mean"]) * 1000 for n in names]]
ax[1].bar(labels, vals, yerr=err, color=[GREY, GREY, NAVY], capsize=4)
ax[1].set(ylabel="Log-loss improvement vs zone baseline (x1000)", title="Out-of-time gain, 95% CI (game bootstrap)")
save_fig(fig, "fig1_model_validation.png")

# Fig 2: shot-making signal
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
m = fc_merged["all"]
for col, color, label in [("raw", GREY, "Raw 2021-23 estimate"), ("shrunk", NAVY, "Shrunk 2021-23 estimate")]:
    q = pd.qcut(m[col].rank(method="first"), 5, labels=False)
    ys = [np.average(m.loc[q == k, "observed"], weights=m.loc[q == k, "shots"]) * 100 for k in range(5)]
    ax[0].plot(range(1, 6), ys, marker="o", color=color, label=label)
ax[0].axhline(0, color=GREY, lw=0.8, ls="--")
ax[0].set(xlabel="Quintile of past shot-making (1 = worst)", ylabel="2024-25 points over expected per 100 shots",
          title="Does past shot-making persist?")
ax[0].legend(frameon=False)
piv = talent.pivot(index="season", columns="shots", values="talent_sd_per100")
for kind, color in [("3PT", NAVY), ("2PT", GOLD)]:
    ax[1].plot(piv.index, piv[kind], marker="o", color=color, label=f"{kind} attempts")
ax[1].set(xlabel="Season start", ylabel="Between-player SD beyond noise (pts / 100 shots)", ylim=(0, None),
          title="Spread of persistent shot-making differences", xticks=piv.index)
ax[1].legend(frameon=False)
save_fig(fig, "fig2_shotmaking_signal.png")

# Fig 3: Jazz player caterpillar
jp = jazz_players.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.2, 0.38 * len(jp) + 1.6))
ax.axvline(0, color=GREY, lw=0.8, ls="--")
ax.hlines(jp.index, jp.lo_per100, jp.hi_per100, color=NAVY, lw=2)
ax.scatter(jp.shrunk_per100, jp.index, color=NAVY, zorder=3, label="Shrunk estimate (90% interval)")
ax.scatter(jp.raw_per100, jp.index, facecolors="none", edgecolors=GOLD, zorder=3, label="Raw")
ax.set_yticks(jp.index)
ax.set_yticklabels([f"{n} ({s})" for n, s in zip(jp.PLAYER_NAME, jp.shots)], fontsize=8)
ax.set(xlabel="Points over expected per 100 shots", title="Jazz 2024-25 shot-making (shots in parentheses)")
ax.legend(frameon=False, loc="lower right", fontsize=8)
save_fig(fig, "fig3_jazz_shotmaking.png")

# Fig 4: Jazz shot mix and shot-making by zone
z = zone_tbl.pivot(index="zone_group", columns="side", values="share")
order = ["Rim", "Paint (non-RA)", "Mid-range", "Corner 3", "Above-break 3"]
z = z.loc[order]
sm = zone_tbl.assign(smoe=(zone_tbl["pps"] - zone_tbl["xpps"]) * 100).pivot(index="zone_group", columns="side", values="smoe").loc[order]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
xs = np.arange(len(order))
ax[0].bar(xs - 0.27, z["own"] * 100, 0.27, color=NAVY, label="Jazz shots")
ax[0].bar(xs, z["allowed"] * 100, 0.27, color=GOLD, label="Jazz opponents")
ax[0].bar(xs + 0.27, z["league"] * 100, 0.27, color=GREY, label="League")
ax[0].set_xticks(xs); ax[0].set_xticklabels(order, rotation=20, ha="right")
ax[0].set(ylabel="% of field-goal attempts", title="Shot mix, 2024-25"); ax[0].legend(frameon=False, fontsize=8)
ax[1].bar(xs - 0.2, sm["own"], 0.4, color=NAVY, label="Jazz shots")
ax[1].bar(xs + 0.2, sm["allowed"], 0.4, color=GOLD, label="Jazz opponents")
ax[1].axhline(0, color=GREY, lw=0.8)
ax[1].set_xticks(xs); ax[1].set_xticklabels(order, rotation=20, ha="right")
ax[1].set(ylabel="Points over expected per 100 shots", title="Shot-making by zone: Jazz vs. opponents")
ax[1].legend(frameon=False, fontsize=8)
save_fig(fig, "fig4_jazz_zones.png")
log("figures written")

# ---------------------------------------------------------------------------------------- 10. results
results = {
    "validation": validation, "tuning": {k: v.to_dict("records") for k, v in tuning.items()},
    "best_params": best, "test_metrics": metrics, "bootstrap_vs_baseline_logloss": boot,
    "forecast": fc, "forecast_geometry_model": fc_geom,
    "league_prior_2024": {"mu_per100": prior24.mu * 100, "tau_per100": prior24.tau * 100, "players": len(league24)},
    "jazz_team": jazz_team, "team_persistence": team_persistence, "jazz_rank_corr_full_vs_geometry": float(rho),
    "n_test_shots": int(len(test)), "n_train_shots": int(len(train)),
}
(config.REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float))
log("done")
