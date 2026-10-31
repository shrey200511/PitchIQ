"""Monte Carlo simulation of the rest of the season.

Steps:
  1. Current table from the matches already played this season.
  2. Remaining fixtures = every (home, away) pairing among this season's 20
     teams that hasn't been played. A double round-robin means we don't need
     the official calendar to know who still plays whom.
  3. Fit Dixon-Coles on data up to today. It generates SCORELINES, not just
     H/D/A, and goal difference decides ties on points.
  4. Parameter uncertainty: we don't know team strengths exactly. We draw
     K parameter sets from the fitted model's approximate sampling
     distribution (normal, covariance = inverse Hessian). Each set runs
     N/K seasons. Skip this and title odds come out overconfident, because
     every simulation assumes the same, perfectly known strengths.
  5. Rank each simulated final table by points, then goal difference, then
     goals scored (the Premier League's order), and count outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitchiq import config
from pitchiq.models import dixon_coles as dc

PARAM_DRAWS = 200


def current_table(played: pd.DataFrame, teams) -> pd.DataFrame:
    t = pd.DataFrame(0, index=sorted(teams), columns=["played", "points", "gf", "ga"])
    for r in played.itertuples():
        t.loc[r.home, ["played", "gf", "ga"]] += [1, r.hg, r.ag]
        t.loc[r.away, ["played", "gf", "ga"]] += [1, r.ag, r.hg]
        t.loc[r.home, "points"] += 3 if r.hg > r.ag else 1 if r.hg == r.ag else 0
        t.loc[r.away, "points"] += 3 if r.ag > r.hg else 1 if r.hg == r.ag else 0
    t["gd"] = t.gf - t.ga
    return t


def remaining_fixtures(played: pd.DataFrame, teams) -> pd.DataFrame:
    done = set(zip(played.home, played.away))
    return pd.DataFrame([(h, a) for h in teams for a in teams if h != a and (h, a) not in done],
                        columns=["home", "away"])


def _psd(cov: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((cov + cov.T) / 2)
    return (v * np.clip(w, 1e-10, None)) @ v.T


def simulate(all_matches: pd.DataFrame, season: int, fixtures: pd.DataFrame | None = None,
             n_sims: int = config.N_SIMULATIONS, param_uncertainty: bool = True,
             seed: int = config.RANDOM_SEED) -> dict:
    rng = np.random.default_rng(seed)
    played_all = all_matches.dropna(subset=["hg", "ag"])
    this = played_all[played_all.season == season]
    teams = set(this.home) | set(this.away)
    if fixtures is not None and len(fixtures):
        teams |= set(fixtures.home) | set(fixtures.away)
    teams = sorted(teams)
    if len(teams) != 20:
        raise ValueError(f"Expected 20 teams in {season}, found {len(teams)}: {teams}")

    table = current_table(this, teams)
    rem = remaining_fixtures(this, teams)
    as_of = (this.date.max() if len(this) else played_all.date.max()) + pd.Timedelta(days=1)
    params = dc.fit(played_all, as_of=as_of, with_cov=param_uncertainty)

    n_team = len(teams)
    tid = {t: i for i, t in enumerate(teams)}
    hi, ai = rem.home.map(tid).to_numpy(), rem.away.map(tid).to_numpy()
    pts = np.tile(table.points.to_numpy(), (n_sims, 1)).astype(float)
    gf = np.tile(table.gf.to_numpy(), (n_sims, 1)).astype(float)
    ga = np.tile(table.ga.to_numpy(), (n_sims, 1)).astype(float)

    k_draws = PARAM_DRAWS if param_uncertainty else 1
    per = int(np.ceil(n_sims / k_draws))
    thetas = (rng.multivariate_normal(params._theta, _psd(params.cov), size=k_draws)
              if param_uncertainty else params._theta[None, :])
    n_par = len(params.teams)
    g = config.MAX_GOALS + 1
    expected = []
    for k, theta in enumerate(thetas):
        lo, hi_ = k * per, min((k + 1) * per, n_sims)
        if lo >= hi_:
            break
        att, dfn, mu, home, rho = dc._unpack(theta, n_par)
        p_k = dc.DCParams(params.teams, att, dfn, float(home), float(mu), float(np.clip(rho, *dc.RHO_BOUNDS)))
        mats = np.empty((len(rem), g * g))
        for j, r in enumerate(rem.itertuples()):
            lam, nu = p_k.rates(r.home, r.away)
            mats[j] = dc.score_matrix(lam, nu, p_k.rho).ravel()
            if k == 0:
                expected.append((lam, nu))
        cum = np.cumsum(mats, axis=1)
        u = rng.random((len(rem), hi_ - lo))
        idx = (cum[:, None, :] < u[:, :, None]).sum(axis=2)          # inverse-CDF sampling
        hg, ag = idx // g, idx % g                                     # (fixtures, sims)
        hp = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        ap = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        block = slice(lo, hi_)
        for j in range(len(rem)):
            pts[block, hi[j]] += hp[j]
            pts[block, ai[j]] += ap[j]
            gf[block, hi[j]] += hg[j]
            ga[block, hi[j]] += ag[j]
            gf[block, ai[j]] += ag[j]
            ga[block, ai[j]] += hg[j]

    gd = gf - ga
    key = pts * 1e6 + (gd + 500) * 1e3 + gf + rng.random(pts.shape) * 1e-3   # points, GD, GF, coin-flip
    order = np.argsort(-key, axis=1)
    position = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    position[rows, order] = np.arange(1, n_team + 1)[None, :]

    pos_probs = pd.DataFrame([(position == p).mean(axis=0) for p in range(1, n_team + 1)],
                             index=range(1, n_team + 1), columns=teams).T
    summary = pd.DataFrame({
        "team": teams,
        "played": table.played.to_numpy(),
        "points": table.points.to_numpy(),
        "gd": table.gd.to_numpy(),
        "exp_points": pts.mean(axis=0),
        "p_title": (position == 1).mean(axis=0),
        "p_top4": (position <= 4).mean(axis=0),
        "p_relegation": (position >= n_team - 2).mean(axis=0),
        "exp_position": position.mean(axis=0),
    }).sort_values(["exp_points", "p_title"], ascending=False).reset_index(drop=True)
    return dict(summary=summary, position_probs=pos_probs.loc[summary.team],
                remaining=len(rem), as_of=str(as_of.date()), n_sims=n_sims,
                param_uncertainty=param_uncertainty, params=params)
