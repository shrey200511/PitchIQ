"""Build the match-level feature table.

THE RULE: a feature for a match may only use information available before
kickoff. This module enforces it by construction:

  1. Per-team rolling stats are computed on a "long" table (one row per team
     per played match). Each row's value INCLUDES that match.
  2. Every game (played or upcoming) then looks up each team's latest row
     STRICTLY BEFORE its own date, via merge_asof(allow_exact_matches=False).
     A team never plays twice on one day, so "strictly earlier date" means
     "already finished before this kickoff".
  3. Elo comes from run_elo(), which stores ratings before the update.
  4. Head-to-head reads its history before the match is appended to it.

features/leakage.py then checks all of this with a perturbation test.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from pitchiq.ratings.elo import EloParams, run_elo

EWM_HALFLIFE = 6          # matches; recent games count more, old ones fade
FORM_WINDOW = 5           # last-5 points form
STINT_GAP_DAYS = 200      # longer gap = team was out of the league (relegated)
REST_CAP_DAYS = 14        # after ~2 weeks extra rest stops mattering; also caps summer
H2H_MEETINGS = 6

TEAM_STATS = ["pts5", "gf_ewm", "ga_ewm", "xgf_ewm", "xga_ewm", "sotf_ewm", "sota_ewm"]

FEATURES = (
    ["elo_home", "elo_away", "elo_diff", "elo_exp_home"]
    + [f"h_{c}" for c in TEAM_STATS] + [f"a_{c}" for c in TEAM_STATS]
    + ["h_venue_pts5", "a_venue_pts5",
       "h_rest_days", "a_rest_days", "rest_diff",
       "h_promoted", "a_promoted", "h_games_played", "a_games_played",
       "xg_balance_diff", "goal_balance_diff", "form_diff",
       "h2h_ppg_home", "h2h_gd_home", "h2h_n"]
)
ID_COLS = ["match_id", "season", "date", "home", "away"]


def _long(played: pd.DataFrame) -> pd.DataFrame:
    """One row per team per played match, from that team's point of view."""
    base = dict(match_id=played.match_id, season=played.season, date=played.date)
    home = pd.DataFrame({**base, "team": played.home, "opp": played.away, "is_home": 1,
                         "gf": played.hg, "ga": played.ag, "xgf": played.home_xg,
                         "xga": played.away_xg, "sotf": played.hst, "sota": played.ast})
    away = pd.DataFrame({**base, "team": played.away, "opp": played.home, "is_home": 0,
                         "gf": played.ag, "ga": played.hg, "xgf": played.away_xg,
                         "xga": played.home_xg, "sotf": played.ast, "sota": played.hst})
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0)
    gap = long.groupby("team")["date"].diff().dt.days
    long["stint"] = (gap.isna() | (gap > STINT_GAP_DAYS)).groupby(long["team"]).cumsum()
    return long.reset_index(drop=True)


def _team_rolling(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["team", "stint"], sort=False)
    out = long[["team", "date", "season", "is_home"]].copy()
    out["pts5"] = g["pts"].transform(lambda s: s.rolling(FORM_WINDOW, min_periods=1).mean())
    for src, dst in [("gf", "gf_ewm"), ("ga", "ga_ewm"), ("xgf", "xgf_ewm"),
                     ("xga", "xga_ewm"), ("sotf", "sotf_ewm"), ("sota", "sota_ewm")]:
        out[dst] = g[src].transform(lambda s: s.ewm(halflife=EWM_HALFLIFE, ignore_na=True).mean())
    out["games_played"] = long.groupby(["team", "season"]).cumcount() + 1
    out["last_date"] = long["date"]
    out["last_season"] = long["season"]
    venue = long.groupby(["team", "stint", "is_home"], sort=False)["pts"]
    out["venue_pts5"] = venue.transform(lambda s: s.rolling(FORM_WINDOW, min_periods=1).mean())
    return out


def _asof(games: pd.DataFrame, table: pd.DataFrame, side_col: str, cols: list[str],
          by_extra: list[str] | None = None) -> pd.DataFrame:
    """Latest team row strictly before each game's date."""
    by_extra = by_extra or []
    left = games[["match_id", "date", side_col] + by_extra].rename(columns={side_col: "team"})
    right = table[["team", "date"] + by_extra + cols].sort_values("date")
    res = pd.merge_asof(left.sort_values("date"), right, on="date", by=["team"] + by_extra,
                        allow_exact_matches=False, direction="backward")
    return res.set_index("match_id")[cols]


def _h2h(games: pd.DataFrame) -> pd.DataFrame:
    hist: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=H2H_MEETINGS))
    rows = []
    for r in games.sort_values(["date", "match_id"]).itertuples():
        key = tuple(sorted((r.home, r.away)))
        past = hist[key]                                     # read BEFORE appending
        if past:
            pts = [3 if (w == r.home) else 1 if w is None else 0 for w, _ in past]
            gd = [d if h == r.home else -d for (_, (h, d)) in past]
            rows.append((r.match_id, np.mean(pts), np.mean(gd), len(past)))
        else:
            rows.append((r.match_id, np.nan, np.nan, 0))
        if not (np.isnan(r.hg) or np.isnan(r.ag)):
            winner = r.home if r.hg > r.ag else r.away if r.ag > r.hg else None
            past.append((winner, (r.home, r.hg - r.ag)))
    return pd.DataFrame(rows, columns=["match_id", "h2h_ppg_home", "h2h_gd_home", "h2h_n"]).set_index("match_id")


def build_features(matches: pd.DataFrame, fixtures: pd.DataFrame | None = None,
                   elo_params: EloParams = EloParams()) -> pd.DataFrame:
    """Return ID_COLS + FEATURES + target columns for every played match and fixture."""
    games = matches.copy()
    if fixtures is not None and len(fixtures):
        fx = fixtures.copy()
        fx["match_id"] = fx["season"].astype(str) + "_" + fx["home"] + "_" + fx["away"]
        fx = fx[~fx["match_id"].isin(games["match_id"])]
        games = pd.concat([games, fx], ignore_index=True)
    for c in ["hg", "ag", "home_xg", "away_xg", "hst", "ast"]:
        games[c] = pd.to_numeric(games.get(c), errors="coerce").astype(float)
    games = games.sort_values(["date", "match_id"]).reset_index(drop=True)

    played = games.dropna(subset=["hg", "ag"])
    roll = _team_rolling(_long(played))
    stat_cols = TEAM_STATS + ["games_played", "last_date", "last_season"]

    feats = games[ID_COLS + ["hg", "ag"]].set_index("match_id")
    feats = feats.join(run_elo(games, elo_params).set_index("match_id"))
    for side, col in [("h", "home"), ("a", "away")]:
        f = _asof(games, roll, col, stat_cols).add_prefix(f"{side}_")
        feats = feats.join(f)
        venue_flag = games[["match_id"]].assign(is_home=1 if side == "h" else 0)
        g2 = games.merge(venue_flag, on="match_id")
        v = _asof(g2, roll, col, ["venue_pts5"], by_extra=["is_home"]).add_prefix(f"{side}_")
        feats = feats.join(v)

    # Previous-season membership -> promoted flag; long gaps -> wipe stale stats.
    teams_by_season = {s: set(g.home) | set(g.away) for s, g in games.groupby("season")}
    for side, col in [("h", "home"), ("a", "away")]:
        prev = feats["season"].map(lambda s: teams_by_season.get(s - 1, set()))
        feats[f"{side}_promoted"] = [int(t not in p) if p else 0
                                     for t, p in zip(feats[col], prev)]
        rest = (feats["date"] - feats[f"{side}_last_date"]).dt.days
        stale = rest.isna() | (rest > STINT_GAP_DAYS)
        feats.loc[stale, [f"{side}_{c}" for c in TEAM_STATS] + [f"{side}_venue_pts5"]] = np.nan
        feats[f"{side}_rest_days"] = rest.clip(upper=REST_CAP_DAYS).fillna(REST_CAP_DAYS)
        same_season = feats[f"{side}_last_season"] == feats["season"]
        feats[f"{side}_games_played"] = feats[f"{side}_games_played"].where(same_season, 0).fillna(0)
        feats = feats.drop(columns=[f"{side}_last_date", f"{side}_last_season"])

    feats["rest_diff"] = feats["h_rest_days"] - feats["a_rest_days"]
    feats["xg_balance_diff"] = (feats.h_xgf_ewm - feats.h_xga_ewm) - (feats.a_xgf_ewm - feats.a_xga_ewm)
    feats["goal_balance_diff"] = (feats.h_gf_ewm - feats.h_ga_ewm) - (feats.a_gf_ewm - feats.a_ga_ewm)
    feats["form_diff"] = feats.h_pts5 - feats.a_pts5
    feats = feats.join(_h2h(games))

    feats["result"] = np.where(feats.hg.isna(), None,
                               np.where(feats.hg > feats.ag, "H", np.where(feats.hg < feats.ag, "A", "D")))
    feats = feats.reset_index()
    return feats[ID_COLS + FEATURES + ["hg", "ag", "result"]].sort_values(["date", "match_id"]).reset_index(drop=True)
