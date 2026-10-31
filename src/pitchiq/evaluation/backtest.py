"""Run every model through walk-forward validation and score it against the market.

All models are scored on the SAME set of matches: those with a result and
with both pre-match and closing odds. Otherwise a model could look better
simply because it was scored on easier games.
"""
from __future__ import annotations

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitchiq import config
from pitchiq.evaluation import metrics as M
from pitchiq.models import ml

log = logging.getLogger(__name__)
P = ["pH", "pD", "pA"]
MODEL_ORDER = ["Always home", "Base rates", "Elo-logit", "Poisson (rho=0)", "Dixon-Coles",
               "Logistic regression", "LightGBM", "LightGBM + temperature",
               "Market pre-match (prop.)", "Market closing (prop.)", "Market closing (Shin)"]
MARKET_REF = "Market closing (Shin)"


def _frame(match_ids, seasons, p, name):
    return pd.DataFrame({"match_id": match_ids, "season": seasons, "pH": p[:, 0],
                         "pD": p[:, 1], "pA": p[:, 2], "model": name})


def market_predictions(matches: pd.DataFrame, seasons) -> pd.DataFrame:
    m = matches[matches.season.isin(seasons)]
    out = []
    for kind, label, fn in [("pre", "Market pre-match (prop.)", M.remove_margin_proportional),
                            ("close", "Market closing (prop.)", M.remove_margin_proportional),
                            ("close", "Market closing (Shin)", M.remove_margin_shin)]:
        cols = [f"odds_{kind}_{k}" for k in "HDA"]
        sub = m.dropna(subset=cols)
        out.append(_frame(sub.match_id.to_numpy(), sub.season.to_numpy(), fn(sub[cols].to_numpy()), label))
    return pd.concat(out, ignore_index=True)


def baseline_predictions(table: pd.DataFrame, seasons) -> pd.DataFrame:
    out = []
    for s in seasons:
        hist = table[(table.season < s) & (table.season >= ml.MODEL_START) & table.result.notna()]
        rates = hist.result.value_counts(normalize=True).reindex(config.OUTCOMES).to_numpy()
        test = table[table.season == s]
        out.append(_frame(test.match_id.to_numpy(), s, np.tile(rates, (len(test), 1)), "Base rates"))
        out.append(_frame(test.match_id.to_numpy(), s, np.tile([1.0, 0, 0], (len(test), 1)), "Always home"))
    return pd.concat(out, ignore_index=True)


def run(table: pd.DataFrame, matches: pd.DataFrame, dc_preds: pd.DataFrame,
        lr_params: dict, lgbm_params: dict) -> dict:
    seasons = config.VALIDATION_SEASONS + config.TEST_SEASONS
    preds = [
        baseline_predictions(table, seasons),
        ml.walk_forward(table, ml.make_elo_logit, seasons, features=["elo_diff"], name="Elo-logit"),
        ml.walk_forward(table, lambda: ml.make_logreg(**lr_params), seasons, name="Logistic regression"),
        ml.walk_forward(table, lambda: ml.make_lgbm(**lgbm_params), seasons, name="LightGBM"),
        ml.walk_forward(table, lambda: ml.make_lgbm(**lgbm_params), seasons, temperature=True,
                        name="LightGBM + temperature"),
        market_predictions(matches, seasons),
    ]
    season_of = table.set_index("match_id")["season"]
    dc = dc_preds[dc_preds.match_id.isin(table[table.season.isin(seasons)].match_id)].copy()
    dc["season"] = dc.match_id.map(season_of)
    preds.append(dc.assign(model="Dixon-Coles")[["match_id", "season"] + P + ["model"]])
    preds.append(dc.assign(model="Poisson (rho=0)", pH=dc.poisson_pH, pD=dc.poisson_pD, pA=dc.poisson_pA)
                 [["match_id", "season"] + P + ["model"]])
    allp = pd.concat(preds, ignore_index=True)

    # Common evaluation set: played + both markets available + every model predicted it.
    result = table.set_index("match_id")["result"]
    allp["result"] = allp.match_id.map(result)
    allp = allp.dropna(subset=["result"])
    counts = allp.groupby("match_id").model.nunique()
    common = counts[counts == allp.model.nunique()].index
    allp = allp[allp.match_id.isin(common)].copy()
    allp["split"] = np.where(allp.season.isin(config.TEST_SEASONS), "test", "validation")

    rows = []
    for (model, split), g in allp.groupby(["model", "split"]):
        y = M.onehot(g.result)
        p = g[P].to_numpy()
        sc = M.score_all(p, y)
        if model == "Always home":
            sc.update(log_loss=np.nan, brier=np.nan, rps=np.nan)
        sc["ece"] = np.nan if model == "Always home" else M.ece(p, y)
        rows.append(dict(model=model, split=split, **sc))
    summary = pd.DataFrame(rows)
    summary["model"] = pd.Categorical(summary.model, MODEL_ORDER, ordered=True)
    summary = summary.sort_values(["split", "model"]).reset_index(drop=True)

    by_season = (allp[allp.model != "Always home"].groupby(["model", "season"])
                 .apply(lambda g: M.log_loss(g[P].to_numpy(), M.onehot(g.result)), include_groups=False)
                 .unstack("season"))

    # Paired bootstrap on TEST: each model minus the closing market (negative = model better).
    test = allp[allp.split == "test"]
    ref = test[test.model == MARKET_REF].set_index("match_id")
    ref_loss = M.per_match_log_loss(ref[P].to_numpy(), M.onehot(ref.result))
    ref_loss = pd.Series(ref_loss, index=ref.index)
    boot = []
    for model, g in test.groupby("model"):
        if model in (MARKET_REF, "Always home"):
            continue
        g = g.set_index("match_id").loc[ref.index]
        loss = M.per_match_log_loss(g[P].to_numpy(), M.onehot(g.result))
        mean, lo, hi = M.paired_bootstrap(loss, ref_loss.to_numpy())
        boot.append(dict(model=model, diff_vs_market=mean, ci_low=lo, ci_high=hi))
    boot = pd.DataFrame(boot).sort_values("diff_vs_market")

    # Overround of the benchmark books on the test seasons (context for the README).
    tm = matches[matches.match_id.isin(test.match_id.unique())]
    margins = {k: float(np.nanmean(M.overround(tm[[f"odds_{k}_{o}" for o in "HDA"]].to_numpy())))
               for k in ("pre", "close")}

    _reliability_figure(test)
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    allp.to_parquet(config.PROCESSED / "backtest_predictions.parquet", index=False)
    summary.to_csv(config.ARTIFACTS / "backtest_summary.csv", index=False)
    by_season.to_csv(config.ARTIFACTS / "backtest_by_season.csv")
    boot.to_csv(config.ARTIFACTS / "bootstrap_vs_market.csv", index=False)
    meta = dict(n_test_matches=int(test.match_id.nunique()),
                n_validation_matches=int(allp[allp.split == "validation"].match_id.nunique()),
                test_seasons=config.TEST_SEASONS, validation_seasons=config.VALIDATION_SEASONS,
                mean_overround=margins, lr_params=lr_params, lgbm_params=lgbm_params)
    (config.ARTIFACTS / "backtest_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return dict(summary=summary, by_season=by_season, bootstrap=boot, meta=meta, predictions=allp)


def _reliability_figure(test: pd.DataFrame) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    models = ["Dixon-Coles", "Logistic regression", "LightGBM + temperature", MARKET_REF]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for k, (ax, name) in enumerate(zip(axes, ["Home win", "Draw", "Away win"])):
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Perfect calibration")
        for model in models:
            g = test[test.model == model]
            if g.empty:
                continue
            t = M.reliability_table(g[P].to_numpy(), M.onehot(g.result), n_bins=8)
            t = t[(t.outcome == config.OUTCOMES[k]) & (t.n >= 15)]
            ax.plot(t.mean_pred, t.observed, marker="o", ms=4, label=model)
        ax.set_title(name)
        ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, 0.9)
        ax.set_ylim(0, 0.9)
    axes[0].set_ylabel("Observed frequency")
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Reliability on test seasons {config.TEST_SEASONS} (bins with n >= 15)")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "reliability_test.png", dpi=130)
    plt.close(fig)
