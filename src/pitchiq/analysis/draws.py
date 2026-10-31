"""Why are draws hard to predict, and does anything help?

The core reason, which this module shows with data:
  A draw is rarely the single most likely outcome. Even in a perfectly even
  game the probabilities look like 37 / 26 / 37, so the draw is never the
  argmax. Accuracy-style "pick the winner" models therefore predict almost
  no draws, even when their draw PROBABILITIES are well calibrated.

What we test:
  1. Calibration of P(draw): is a 30% draw call right about 30% of the time?
  2. Where draws happen: evenly matched games (small |Elo diff|) and
     low-scoring games (low expected total goals).
  3. Dixon-Coles rho vs plain Poisson: does modelling low-score dependence
     improve the draw probability (Brier score on the draw/no-draw event)?
  4. A draw decision rule: call "draw" when P(draw) >= t, with t tuned on
     validation. This trades overall accuracy for draw recall. It's a
     decision-layer change, not a better probability.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitchiq import config

P = ["pH", "pD", "pA"]
FOCUS = ["Dixon-Coles", "Poisson (rho=0)", "Logistic regression", "LightGBM + temperature",
         "Market closing (Shin)"]


def _draw_rule(df: pd.DataFrame, t: float) -> np.ndarray:
    pred = np.array(config.OUTCOMES)[df[P].to_numpy().argmax(axis=1)]
    return np.where(df.pD >= t, "D", pred)


def _rule_scores(df, t):
    pred = _draw_rule(df, t)
    y = df.result.to_numpy()
    tp = np.sum((pred == "D") & (y == "D"))
    rec = tp / max(np.sum(y == "D"), 1)
    prec = tp / max(np.sum(pred == "D"), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(threshold=t, accuracy=float(np.mean(pred == y)), draw_recall=float(rec),
                draw_precision=float(prec), draw_f1=float(f1), draws_called=int(np.sum(pred == "D")))


def run(backtest_preds: pd.DataFrame, table: pd.DataFrame, dc_preds: pd.DataFrame) -> dict:
    preds = backtest_preds[backtest_preds.model.isin(FOCUS)].copy()
    preds["is_draw"] = (preds.result == "D").astype(int)

    # 1 + 3: draw calibration and draw Brier per model.
    model_rows = []
    for (model, split), g in preds.groupby(["model", "split"]):
        model_rows.append(dict(model=model, split=split,
                               actual_draw_rate=g.is_draw.mean(), mean_pred_draw=g.pD.mean(),
                               max_pred_draw=g.pD.max(),
                               draw_is_argmax_pct=float(np.mean(g[P].to_numpy().argmax(1) == 1)),
                               draw_brier=float(np.mean((g.pD - g.is_draw) ** 2))))
    by_model = pd.DataFrame(model_rows)

    # 2: where draws happen (all played matches in the modelling window).
    t = table[table.result.notna() & (table.season >= 2015)].copy()
    t["is_draw"] = (t.result == "D").astype(int)
    t["abs_elo_diff"] = t.elo_diff.abs()
    t["elo_bin"] = pd.qcut(t.abs_elo_diff, 5, duplicates="drop")
    by_elo = t.groupby("elo_bin", observed=True).agg(draw_rate=("is_draw", "mean"), n=("is_draw", "size"))
    dc = dc_preds.assign(exp_total=dc_preds.exp_hg + dc_preds.exp_ag).merge(t[["match_id", "is_draw"]], on="match_id")
    dc["goals_bin"] = pd.qcut(dc.exp_total, 5, duplicates="drop")
    by_goals = dc.groupby("goals_bin", observed=True).agg(draw_rate=("is_draw", "mean"), n=("is_draw", "size"))
    by_season = t.groupby("season").is_draw.mean()

    # 4: draw decision rule tuned on validation, reported on test.
    rule_rows = []
    for model in ["LightGBM + temperature", "Dixon-Coles", "Market closing (Shin)"]:
        g = preds[preds.model == model]
        val, test = g[g.split == "validation"], g[g.split == "test"]
        if val.empty or test.empty:
            continue
        grid = [_rule_scores(val, th) for th in np.arange(0.24, 0.36, 0.005)]
        best_t = max(grid, key=lambda r: r["draw_f1"] + r["accuracy"])["threshold"]
        rule_rows.append(dict(model=model, rule="argmax (default)", **_rule_scores(test, 1.01)))
        rule_rows.append(dict(model=model, rule=f"draw if P(D) >= {best_t:.3f}", **_rule_scores(test, best_t)))
    rules = pd.DataFrame(rule_rows)

    _figure(by_elo, by_goals, preds)
    out = dict(by_model=by_model, by_elo=by_elo, by_goals=by_goals, by_season=by_season, rules=rules)
    (config.ARTIFACTS / "draws_summary.json").write_text(json.dumps({
        "by_model": by_model.to_dict("records"), "rules": rules.to_dict("records"),
        "draw_rate_by_elo_quintile": by_elo.draw_rate.round(4).tolist(),
        "draw_rate_by_goals_quintile": by_goals.draw_rate.round(4).tolist()}, indent=2, default=float))
    return out


def _figure(by_elo, by_goals, preds):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].bar(range(len(by_elo)), by_elo.draw_rate, color="#4C78A8")
    ax[0].set_xticks(range(len(by_elo)), [f"Q{i+1}" for i in range(len(by_elo))])
    ax[0].set_title("Draw rate by |Elo difference| quintile\n(Q1 = most even)")
    ax[1].bar(range(len(by_goals)), by_goals.draw_rate, color="#F58518")
    ax[1].set_xticks(range(len(by_goals)), [f"Q{i+1}" for i in range(len(by_goals))])
    ax[1].set_title("Draw rate by expected total goals quintile\n(Q1 = lowest scoring)")
    for model in ["Dixon-Coles", "LightGBM + temperature", "Market closing (Shin)"]:
        g = preds[(preds.model == model) & (preds.split == "test")]
        if len(g):
            ax[2].hist(g.pD, bins=30, histtype="step", lw=1.6, label=model, density=True)
    ax[2].axvline(1 / 3, ls="--", c="grey", lw=1)
    ax[2].set_title("Distribution of P(draw) on test seasons\n(dashed: 1/3)")
    ax[2].legend(fontsize=8)
    for a in ax[:2]:
        a.set_ylabel("Draw rate")
    fig.tight_layout()
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / "draws.png", dpi=130)
    plt.close(fig)
