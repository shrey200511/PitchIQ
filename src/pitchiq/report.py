"""Write reports/results.md and reports/resume_metrics.json from pipeline outputs.

Everything in the results report is computed. Nothing is typed in by hand, so
the README, blog post and resume bullets can only quote numbers the code
actually produced.
"""
from __future__ import annotations

import json

import pandas as pd

from pitchiq import config


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "—" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def _read(name, **kw):
    p = config.ARTIFACTS / name
    return pd.read_csv(p, **kw) if p.exists() else None


def write_report() -> None:
    A, R = config.ARTIFACTS, config.REPORTS
    meta = json.loads((A / "meta.json").read_text()) if (A / "meta.json").exists() else {}
    bmeta = json.loads((A / "backtest_meta.json").read_text()) if (A / "backtest_meta.json").exists() else {}
    tuned = json.loads((A / "tuned_params.json").read_text()) if (A / "tuned_params.json").exists() else {}
    summary, boot = _read("backtest_summary.csv"), _read("bootstrap_vs_market.csv")
    by_season = _read("backtest_by_season.csv", index_col=0)
    sim, shap_g = _read("season_sim.csv"), _read("shap_global.csv", index_col=0)
    leak = pd.read_csv(R / "leakage_check.csv") if (R / "leakage_check.csv").exists() else None

    L = ["# PitchIQ — results", ""]
    if config.SYNTHETIC or meta.get("synthetic"):
        L += ["> ⚠️ **SYNTHETIC DATA.** These numbers come from generated test data and are",
              "> NOT real results. Do not quote them anywhere.", ""]
    L += [f"_Generated {meta.get('generated_at', '?')} · data through {meta.get('data_through', '?')}_", ""]

    if tuned:
        e = tuned["elo"]
        L += ["## Tuned parameters", "",
              f"- Elo: K = {e['k']:g}, home advantage = {e['hfa']:g} pts, summer regression = {e['season_regress']:g}",
              f"- Logistic regression: {tuned['logreg']}",
              f"- LightGBM: {tuned['lgbm']}", ""]
    if leak is not None:
        L += ["## Leakage check", "",
              f"Perturbation test passed on {len(leak)} dates ({int(leak.matches_checked.sum())} matches): "
              "scrambling every result from a date onward changed none of that date's features.", ""]
    if summary is not None:
        cols = ["model", "log_loss", "brier", "rps", "accuracy", "ece", "n"]
        for split, seasons in [("test", bmeta.get("test_seasons")), ("validation", bmeta.get("validation_seasons"))]:
            L += [f"## Backtest — {split} seasons {seasons}", "",
                  "Walk-forward: each season predicted by models trained only on earlier seasons. "
                  "All models scored on the same matches. Lower is better except accuracy.", "",
                  _md(summary[summary.split == split][cols]), ""]
        if bmeta.get("mean_overround"):
            o = bmeta["mean_overround"]
            L += [f"Average bookmaker margin (overround) removed: pre-match {o['pre']:.2%}, closing {o['close']:.2%}.", ""]
    if boot is not None:
        L += ["## Gap to the closing market (test seasons, paired bootstrap 95% CI)", "",
              "Mean per-match log-loss difference vs Market closing (Shin). Negative = beats the market; "
              "a CI that includes 0 means no detectable difference.", "",
              _md(boot), ""]
    if by_season is not None:
        L += ["## Log loss by season", "", _md(by_season.reset_index()), ""]

    draws_p = A / "draws_summary.json"
    if draws_p.exists():
        d = json.loads(draws_p.read_text())
        L += ["## Draws", "", "![draws](figures/draws.png)", "",
              _md(pd.DataFrame(d["by_model"])), "", "Draw decision rule (threshold tuned on validation, scored on test):", "",
              _md(pd.DataFrame(d["rules"])), ""]
    if shap_g is not None:
        L += ["## Explainability (LightGBM, SHAP)", "", "![shap](figures/shap_global.png)", "",
              _md(shap_g.reset_index().rename(columns={"index": "feature"})[["label", "H", "D", "A", "total"]].head(10)), ""]
    if sim is not None:
        s = sim.copy()
        for c in ["p_title", "p_top4", "p_relegation"]:
            s[c] = s[c].map("{:.1%}".format)
        L += [f"## Season simulation ({meta.get('n_sims', '?')} runs, {meta.get('remaining_fixtures', '?')} fixtures left)", "",
              _md(s[["team", "played", "points", "exp_points", "p_title", "p_top4", "p_relegation"]], "{:.1f}"), ""]
    L += ["![reliability](figures/reliability_test.png)", ""]
    R.mkdir(parents=True, exist_ok=True)
    (R / "results.md").write_text("\n".join(L), encoding="utf-8")

    # Facts for resume bullets / blog. Only computed values.
    if summary is not None:
        t = summary[summary.split == "test"].set_index("model")
        facts = {
            "synthetic": bool(config.SYNTHETIC or meta.get("synthetic")),
            "n_test_matches": bmeta.get("n_test_matches"),
            "n_validation_matches": bmeta.get("n_validation_matches"),
            "test_log_loss": t["log_loss"].dropna().round(4).to_dict(),
            "test_accuracy": t["accuracy"].round(4).to_dict(),
            "test_brier": t["brier"].dropna().round(4).to_dict(),
            "test_ece": t["ece"].dropna().round(4).to_dict(),
            "bootstrap_vs_market": boot.round(4).to_dict("records") if boot is not None else None,
            "leakage_dates_checked": None if leak is None else int(len(leak)),
            "n_simulations": meta.get("n_sims"),
        }
        (R / "resume_metrics.json").write_text(json.dumps(facts, indent=2))
