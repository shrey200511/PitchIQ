"""PitchIQ command-line pipeline.

    python -m pitchiq.pipeline all        # full run: fetch -> ... -> report  (first time)
    python -m pitchiq.pipeline refresh    # weekly: new data, live predictions, simulation
    python -m pitchiq.pipeline <step>     # fetch | build | tune | backtest | draws | live | report
    python -m pitchiq.pipeline synthetic  # generate fake raw data for offline testing

Set PITCHIQ_ROOT to send all data/outputs to another folder (tests use this).
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pitchiq import config

log = logging.getLogger("pitchiq")
TUNED = lambda: config.ARTIFACTS / "tuned_params.json"   # noqa: E731


def _load():
    m = pd.read_parquet(config.PROCESSED / "matches.parquet")
    f = pd.read_parquet(config.PROCESSED / "fixtures.parquet")
    return m, f


def _params():
    from pitchiq.ratings.elo import EloParams
    p = json.loads(TUNED().read_text())
    return EloParams(**p["elo"]), p["logreg"], p["lgbm"]


# ------------------------------------------------------------- steps
def step_fetch(force=False):
    from pitchiq.data import fetch
    fetch.fetch_all(force=force)


def step_build():
    from pitchiq.data import merge
    merge.build_match_table()


def step_tune():
    """Tune Elo on pre-validation seasons; tune ML on validation seasons."""
    from pitchiq.features.build import build_features
    from pitchiq.models import ml
    from pitchiq.ratings.elo import tune_elo
    m, f = _load()
    elo_seasons = range(config.ELO_WARMUP_START + 3, config.VALIDATION_SEASONS[0])
    elo_p, elo_grid = tune_elo(m, elo_seasons)
    if elo_p.k in (5, 40) or elo_p.hfa in (30, 90):
        log.warning("Best Elo params sit on the grid edge (%s) - widen the grid.", elo_p)
    table = build_features(m, f, elo_p)
    table.to_parquet(config.PROCESSED / "features.parquet", index=False)
    lr_best, lr_grid = ml.tune(table, "logreg")
    gb_best, gb_grid = ml.tune(table, "lgbm")
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    TUNED().write_text(json.dumps({"elo": dict(k=elo_p.k, hfa=elo_p.hfa, season_regress=elo_p.season_regress),
                                   "logreg": lr_best, "lgbm": gb_best}, indent=2))
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    elo_grid.head(15).to_csv(config.REPORTS / "tuning_elo.csv", index=False)
    lr_grid.to_csv(config.REPORTS / "tuning_logreg.csv", index=False)
    gb_grid.to_csv(config.REPORTS / "tuning_lgbm.csv", index=False)
    log.info("Tuned: elo=%s logreg=%s lgbm=%s", elo_p, lr_best, gb_best)


def step_features():
    from pitchiq.features.build import build_features
    from pitchiq.features import leakage
    m, f = _load()
    elo_p, _, _ = _params()
    table = build_features(m, f, elo_p)
    table.to_parquet(config.PROCESSED / "features.parquet", index=False)
    leakage.check_feature_names()
    rep = leakage.perturbation_test(m, f, elo_p, leakage.default_test_dates(m))
    rep.to_csv(config.REPORTS / "leakage_check.csv", index=False)
    log.info("Leakage check passed on %d dates", len(rep))
    return table


def step_dc():
    from pitchiq.models import dixon_coles as dc
    m, f = _load()
    fx = f.assign(match_id=f.season.astype(str) + "_" + f.home + "_" + f.away)
    games = pd.concat([m, fx[~fx.match_id.isin(m.match_id)]], ignore_index=True)
    start = m[m.season == config.VALIDATION_SEASONS[0]].date.min()
    preds = dc.walk_forward(games, start)
    preds.to_parquet(config.PROCESSED / "dc_predictions.parquet", index=False)
    return preds


def step_backtest():
    from pitchiq.evaluation import backtest
    m, _ = _load()
    table = pd.read_parquet(config.PROCESSED / "features.parquet")
    dc_preds = pd.read_parquet(config.PROCESSED / "dc_predictions.parquet")
    _, lr_p, gb_p = _params()
    res = backtest.run(table, m, dc_preds, lr_p, gb_p)
    log.info("\n%s", res["summary"].to_string())
    return res


def step_draws():
    from pitchiq.analysis import draws
    preds = pd.read_parquet(config.PROCESSED / "backtest_predictions.parquet")
    table = pd.read_parquet(config.PROCESSED / "features.parquet")
    dc_preds = pd.read_parquet(config.PROCESSED / "dc_predictions.parquet")
    return draws.run(preds, table, dc_preds)


def step_live():
    """Final models -> upcoming fixture predictions, SHAP, season simulation."""
    from pitchiq.explain import shap_explain
    from pitchiq.features.build import FEATURES
    from pitchiq.models import ml
    from pitchiq.models import dixon_coles as dc
    from pitchiq.ratings.elo import final_ratings
    from pitchiq.simulate.season import simulate

    m, f = _load()
    elo_p, _, gb_p = _params()
    table = pd.read_parquet(config.PROCESSED / "features.parquet")
    model = ml.fit_final(table, lambda: ml.make_lgbm(**gb_p))

    # Temperature: train on seasons < last complete season, calibrate on that season.
    last = config.LAST_COMPLETE
    inner = ml.fit_final(table[table.season < last], lambda: ml.make_lgbm(**gb_p))
    Xv, yv = ml.xy(table[table.season == last], FEATURES)
    T = ml.fit_temperature(ml.predict(inner, Xv), yv) if len(Xv) else 1.0

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    upcoming = table[
        table.result.isna()
        & (table.season == config.CURRENT_SEASON)
        & (table.date >= today)
    ].sort_values("date")

    next_dates = upcoming.date.drop_duplicates().head(4)
    upcoming = upcoming[upcoming.date.isin(next_dates)]
    if len(upcoming):
        p = ml.apply_temperature(ml.predict(model, upcoming[FEATURES]), T)
        played = m.dropna(subset=["hg", "ag"])
        dcp = dc.fit(played, as_of=played.date.max() + pd.Timedelta(days=1))
        dc_probs = np.vstack([dc.outcome_probs(dcp, r.home, r.away) for r in upcoming.itertuples()])
        up = upcoming[["match_id", "date", "home", "away", "elo_home", "elo_away"]].copy()
        up[["pH", "pD", "pA"]] = p
        up[["dc_pH", "dc_pD", "dc_pA"]] = dc_probs
        up["date"] = up.date.dt.strftime("%Y-%m-%d")
        up.to_csv(config.ARTIFACTS / "upcoming.csv", index=False)
        expl = shap_explain.explain_fixtures(model, upcoming, FEATURES)
        (config.ARTIFACTS / "explanations.json").write_text(json.dumps(expl, indent=2))
    else:
        pd.DataFrame(columns=["match_id", "date", "home", "away", "pH", "pD", "pA"]).to_csv(
            config.ARTIFACTS / "upcoming.csv", index=False)
        (config.ARTIFACTS / "explanations.json").write_text("[]")

    # Global SHAP on the two most recent complete seasons.
    recent = table[table.season.isin(config.TEST_SEASONS) & table.result.notna()]
    imp = shap_explain.global_importance(model, recent[FEATURES])
    imp.to_csv(config.ARTIFACTS / "shap_global.csv")

    try:
        sim = simulate(m, config.CURRENT_SEASON, fixtures=f)
    except ValueError as err:          # e.g. pre-season, fixture list not out yet
        log.warning("Season simulation skipped: %s", err)
        sim = dict(summary=pd.DataFrame(columns=["team"]), position_probs=pd.DataFrame(),
                   n_sims=0, param_uncertainty=False, remaining=0)
    sim["summary"].to_csv(config.ARTIFACTS / "season_sim.csv", index=False)
    sim["position_probs"].to_csv(config.ARTIFACTS / "position_probs.csv")
    elo_now = final_ratings(m, elo_p)
    elo_now = elo_now[elo_now.index.isin(set(sim["summary"].team))]
    elo_now.rename_axis("team").rename("elo").reset_index().to_csv(config.ARTIFACTS / "elo_current.csv", index=False)

    played = m.dropna(subset=["hg", "ag"])
    meta = dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="minutes"),
                data_through=str(played.date.max().date()), current_season=config.CURRENT_SEASON,
                synthetic=config.SYNTHETIC, temperature=T, n_sims=sim["n_sims"],
                param_uncertainty=sim["param_uncertainty"], remaining_fixtures=sim["remaining"])
    (config.ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Live artifacts written (%d upcoming fixtures)", len(upcoming))


def step_report():
    from pitchiq.report import write_report
    write_report()


def step_synthetic():
    from pitchiq.data import synthetic
    if not config.SYNTHETIC:
        raise SystemExit("Refusing to write synthetic data without PITCHIQ_SYNTHETIC=1 "
                         "(protects your real data folder).")
    synthetic.generate()


STEPS = {"fetch": step_fetch, "build": step_build, "tune": step_tune, "features": step_features,
         "dc": step_dc, "backtest": step_backtest, "draws": step_draws, "live": step_live,
         "report": step_report, "synthetic": step_synthetic}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=list(STEPS) + ["all", "refresh"])
    ap.add_argument("--force", action="store_true", help="re-download completed seasons")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for d in (config.PROCESSED, config.ARTIFACTS, config.FIGURES):
        d.mkdir(parents=True, exist_ok=True)

    if a.step == "all":
        if not config.SYNTHETIC:
            step_fetch(a.force)
        for s in ["build", "tune", "features", "dc", "backtest", "draws", "live", "report"]:
            STEPS[s]()
    elif a.step == "refresh":
        if not config.SYNTHETIC:
            step_fetch()
        for s in ["build", "features", "live", "report"]:
            STEPS[s]()
    elif a.step == "fetch":
        step_fetch(a.force)
    else:
        STEPS[a.step]()


if __name__ == "__main__":
    main()
