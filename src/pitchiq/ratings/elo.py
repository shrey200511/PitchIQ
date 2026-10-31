"""Elo ratings built from scratch, plus a tuner for K, home advantage and
between-season regression.

How it works, in one paragraph:
  Every team has a rating (start 1500). Before a match, the home side's
  expected score is E = 1 / (1 + 10^(-(R_home + HFA - R_away) / 400)), where a
  win scores 1, a draw 0.5 and a loss 0. After the match both ratings move by
  K * G * (actual - expected); G grows with the goal margin, so a 4-0 says more
  than a 1-0. What one team gains, the other loses (zero-sum).

The function records each team's rating BEFORE kickoff. That pre-match
rating is the only thing the feature table ever sees, which is what keeps Elo
leakage-free.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0                # update speed: bigger = reacts faster, noisier
    hfa: float = 60.0              # home-field advantage in rating points
    season_regress: float = 0.25   # share of distance to the mean removed each summer
    use_margin: bool = True        # scale updates by goal difference
    init: float = 1500.0


def margin_multiplier(gd: np.ndarray | int) -> np.ndarray:
    """World Football Elo convention: 1, 1.5, then (11 + gd) / 8 for gd >= 3."""
    gd = np.abs(gd)
    return np.where(gd <= 1, 1.0, np.where(gd == 2, 1.5, (11 + gd) / 8))


def expected_home(r_home, r_away, hfa):
    return 1.0 / (1.0 + 10 ** (-(r_home + hfa - r_away) / 400.0))


def run_elo(matches: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """Return one row per match with pre-match ratings and the expected home score.

    `matches` needs: match_id, season, date, home, away, hg, ag (sorted by date).
    Rows without a score (future fixtures) receive ratings but trigger no update.
    """
    m = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    current_season = None
    relegated_avg = params.init
    out_h, out_a = np.empty(len(m)), np.empty(len(m))

    seasons = m["season"].to_numpy()
    homes, aways = m["home"].to_numpy(), m["away"].to_numpy()
    hg, ag = m["hg"].to_numpy(dtype=float), m["ag"].to_numpy(dtype=float)

    for i in range(len(m)):
        s = seasons[i]
        if s != current_season:
            if current_season is not None:
                prev = set(homes[seasons == current_season]) | set(aways[seasons == current_season])
                now = set(homes[seasons == s]) | set(aways[seasons == s])
                leaving = prev - now
                if leaving:              # promoted sides inherit the relegated sides' level
                    relegated_avg = float(np.mean([ratings[t] for t in leaving]))
                mean = float(np.mean([ratings[t] for t in prev]))
                for t in prev & now:     # summer regression toward the league mean
                    ratings[t] = mean + (1 - params.season_regress) * (ratings[t] - mean)
                for t in now - prev:     # promoted (even if they were here years ago)
                    ratings[t] = relegated_avg
            current_season = s
        h, a = homes[i], aways[i]
        rh = ratings.setdefault(h, params.init)
        ra = ratings.setdefault(a, params.init)
        out_h[i], out_a[i] = rh, ra
        if np.isnan(hg[i]) or np.isnan(ag[i]):
            continue                      # fixture not played yet: no update
        e = expected_home(rh, ra, params.hfa)
        actual = 1.0 if hg[i] > ag[i] else 0.5 if hg[i] == ag[i] else 0.0
        g = float(margin_multiplier(hg[i] - ag[i])) if params.use_margin else 1.0
        delta = params.k * g * (actual - e)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

    return pd.DataFrame({
        "match_id": m["match_id"],
        "elo_home": out_h,
        "elo_away": out_a,
        "elo_diff": out_h - out_a,
        "elo_exp_home": expected_home(out_h, out_a, params.hfa),
    })


def final_ratings(matches: pd.DataFrame, params: EloParams) -> pd.Series:
    """Ratings after the last played match (used for the live table display)."""
    played = matches.dropna(subset=["hg", "ag"])
    pre = run_elo(played, params).merge(played[["match_id", "home", "away", "hg", "ag"]], on="match_id")
    ratings: dict[str, float] = {}
    for r in pre.itertuples():
        e = r.elo_exp_home
        actual = 1.0 if r.hg > r.ag else 0.5 if r.hg == r.ag else 0.0
        g = float(margin_multiplier(r.hg - r.ag)) if params.use_margin else 1.0
        ratings[r.home] = r.elo_home + params.k * g * (actual - e)
        ratings[r.away] = r.elo_away - params.k * g * (actual - e)
    return pd.Series(ratings).sort_values(ascending=False)


def score_loss(matches: pd.DataFrame, elo: pd.DataFrame, seasons) -> float:
    """Mean squared error between expected score and actual score (draw = 0.5).

    We tune on this, not on H/D/A log loss, because Elo predicts an expected
    SCORE; turning it into three outcome probabilities is the models' job.
    """
    df = matches[["match_id", "season", "hg", "ag"]].merge(elo, on="match_id")
    df = df[df["season"].isin(seasons)].dropna(subset=["hg", "ag"])
    actual = np.where(df.hg > df.ag, 1.0, np.where(df.hg == df.ag, 0.5, 0.0))
    return float(np.mean((actual - df["elo_exp_home"]) ** 2))


def tune_elo(matches: pd.DataFrame, tune_seasons,
             ks=(5, 8, 10, 15, 20, 25, 30, 40), hfas=(30, 45, 60, 75, 90),
             regs=(0.1, 0.2, 0.33, 0.5)) -> tuple[EloParams, pd.DataFrame]:
    """Grid search. Tune ONLY on seasons before validation/test, so the choice
    of K cannot be fitted to the seasons we later report results on."""
    rows = []
    for k, hfa, reg in itertools.product(ks, hfas, regs):
        p = EloParams(k=k, hfa=hfa, season_regress=reg)
        rows.append(dict(k=k, hfa=hfa, season_regress=reg,
                         mse=score_loss(matches, run_elo(matches, p), tune_seasons)))
    grid = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    best = grid.iloc[0]
    return EloParams(k=float(best.k), hfa=float(best.hfa), season_regress=float(best.season_regress)), grid
