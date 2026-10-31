"""Leakage checks: prove that features only use pre-kickoff information.

Check 1, the perturbation test (the important one):
    Pick a cut-off date d. Scramble EVERY outcome-related number (goals, xG,
    shots) for all matches on or after d, rebuild the whole feature table from
    scratch, and compare the features of the matches played ON d. If any
    feature changed, it must have read data from day d or later. That is
    leakage, and the check fails. Repeat for several dates, including
    season openers and the current season.

    This tests behaviour, not code style, so it also catches leakage that a
    future edit might add (e.g. someone forgets .shift(1) in a new rolling feature).

Check 2, the forbidden-columns guard:
    No feature name may be a raw post-match field (goals, result, the match's
    own xG or shots) or a bookmaker price (odds are a benchmark, not an input).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitchiq.features.build import FEATURES, build_features
from pitchiq.ratings.elo import EloParams

FORBIDDEN = {"hg", "ag", "result", "home_xg", "away_xg", "hs", "as_", "hst", "ast"}
OUTCOME_COLS = ["hg", "ag", "home_xg", "away_xg", "hs", "as_", "hst", "ast"]


def check_feature_names(features=FEATURES) -> None:
    bad = [f for f in features if f in FORBIDDEN or "odds" in f]
    if bad:
        raise AssertionError(f"Forbidden columns used as features: {bad}")


def perturbation_test(matches: pd.DataFrame, fixtures: pd.DataFrame | None,
                      elo_params: EloParams, dates, seed: int = 0) -> pd.DataFrame:
    """Return one row per tested date; raises AssertionError on any leak."""
    rng = np.random.default_rng(seed)
    base = build_features(matches, fixtures, elo_params).set_index("match_id")
    report = []
    for d in pd.to_datetime(list(dates)):
        scrambled = matches.copy()
        future = scrambled["date"] >= d
        for c in OUTCOME_COLS:
            if c in scrambled:
                vals = scrambled.loc[future, c]
                scrambled.loc[future, c] = rng.permutation(vals.to_numpy()) + rng.integers(0, 3, len(vals))
        scrambled.loc[future, "result"] = np.select(
            [scrambled.loc[future, "hg"] > scrambled.loc[future, "ag"],
             scrambled.loc[future, "hg"] < scrambled.loc[future, "ag"]], ["H", "A"], "D")
        rebuilt = build_features(scrambled, fixtures, elo_params).set_index("match_id")
        on_day = base.index[base["date"] == d]
        if len(on_day) == 0:
            continue
        a = base.loc[on_day, FEATURES].astype(float)
        b = rebuilt.loc[on_day, FEATURES].astype(float)
        diff = ~np.isclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
        leaked = sorted({FEATURES[j] for j in np.where(diff)[1]})
        # Sanity: the scramble must actually change LATER features, or the test proves nothing.
        later = base.index[base["date"] > d + pd.Timedelta(days=7)]
        later_changed = bool(len(later)) and not np.allclose(
            base.loc[later, FEATURES].astype(float).to_numpy(),
            rebuilt.loc[later, FEATURES].astype(float).to_numpy(), equal_nan=True)
        report.append(dict(date=d.date(), matches_checked=len(on_day),
                           leaked_features=", ".join(leaked) or "none",
                           later_features_changed=later_changed))
        if leaked:
            raise AssertionError(f"Leakage on {d.date()}: {leaked}")
    return pd.DataFrame(report)


def default_test_dates(matches: pd.DataFrame, n_random: int = 4, seed: int = 0) -> list:
    """Season openers (where cross-season logic lives) + random mid-season dates."""
    rng = np.random.default_rng(seed)
    openers = matches.groupby("season")["date"].min().tail(3).tolist()
    dates = sorted(matches["date"].unique())
    picks = rng.choice(len(dates), size=min(n_random, len(dates)), replace=False)
    return sorted(set(openers) | {pd.Timestamp(dates[i]) for i in picks})
