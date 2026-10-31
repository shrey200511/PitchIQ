"""Poisson / Dixon-Coles goals model, turned into H/D/A probabilities.

Model (Dixon & Coles, 1997):
    home goals ~ Poisson(lam),  lam = exp(mu + home + att[h] - dfn[a])
    away goals ~ Poisson(nu),   nu  = exp(mu        + att[a] - dfn[h])
plus a correction tau(x, y; rho) for the four low scores 0-0, 1-0, 0-1, 1-1.
Independent Poissons get those wrong (real football has slightly more 0-0
and 1-1). rho = 0 gives back the plain independent-Poisson model.

A weak Gaussian prior (sd 0.3) on each team's attack/defence makes this a
MAP fit: it barely matters for teams with seasons of data, but stops teams
with a handful of games (newly promoted) from getting extreme strengths and
huge uncertainty.

Older matches count less: weight = exp(-xi * days_ago). xi = 0.0019/day gives
a half-life of about one year, which balances "use lots of data" against
"teams change".

Outputs:
  * fit(...)            -> DCParams (attack/defence per team, home, mu, rho)
  * score_matrix(...)   -> P(home goals = i, away goals = j)
  * outcome_probs(...)  -> P(H), P(D), P(A)
  * walk_forward(...)   -> refit every 7 days on data strictly before the
                           refit date, predict the following 7 days.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from pitchiq import config

XI_PER_DAY = 0.0019
WINDOW_DAYS = 3 * 365
RHO_BOUNDS = (-0.2, 0.2)
PRIOR_SD = 0.3            # Gaussian prior on attack/defence (team strengths rarely exceed +-0.6)
NEW_TEAM_MIN_MATCHES = 10 # fewer matches in the window = "new" (usually newly promoted)


@dataclass
class DCParams:
    teams: list
    att: np.ndarray
    dfn: np.ndarray
    home: float
    mu: float
    rho: float
    cov: np.ndarray | None = field(default=None, repr=False)   # parameter covariance (for simulation)
    fit_date: pd.Timestamp | None = None

    def index(self, team: str) -> int | None:
        try:
            return self.teams.index(team)
        except ValueError:
            return None

    def rates(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals. Unknown (e.g. newly promoted, no history in window)
        teams get the average strength of the 3 weakest known teams."""
        weak = np.argsort(self.att - self.dfn)[:3]
        ih, ia = self.index(home), self.index(away)
        ah, dh = (self.att[ih], self.dfn[ih]) if ih is not None else (self.att[weak].mean(), self.dfn[weak].mean())
        aa, da = (self.att[ia], self.dfn[ia]) if ia is not None else (self.att[weak].mean(), self.dfn[weak].mean())
        return float(np.exp(self.mu + self.home + ah - da)), float(np.exp(self.mu + aa - dh))


def _tau(x, y, lam, nu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * nu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + nu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def _unpack(theta, n):
    att = np.append(theta[: n - 1], -theta[: n - 1].sum())        # sum(att) = 0
    dfn = np.append(theta[n - 1: 2 * n - 2], -theta[n - 1: 2 * n - 2].sum())
    mu, home, rho = theta[2 * n - 2:]
    return att, dfn, mu, home, rho


def _neg_loglik(theta, hi, ai, x, y, w, n, prior_att=None, prior_dfn=None):
    att, dfn, mu, home, rho = _unpack(theta, n)
    penalty = 0.0
    if prior_att is not None:   # MAP estimate: shrink each team toward its prior mean
        penalty = (np.sum((att - prior_att) ** 2) + np.sum((dfn - prior_dfn) ** 2)) / (2 * PRIOR_SD ** 2)
    lam = np.exp(mu + home + att[hi] - dfn[ai])
    nu = np.exp(mu + att[ai] - dfn[hi])
    tau = np.clip(_tau(x, y, lam, nu, rho), 1e-10, None)
    ll = np.log(tau) + x * np.log(lam) - lam - gammaln(x + 1) + y * np.log(nu) - nu - gammaln(y + 1)
    return -np.sum(w * ll) + penalty


def fit(played: pd.DataFrame, as_of: pd.Timestamp, xi: float = XI_PER_DAY,
        window_days: int = WINDOW_DAYS, init: DCParams | None = None,
        with_cov: bool = False) -> DCParams:
    """Fit on matches strictly before `as_of`, weighted by recency."""
    df = played[(played["date"] < as_of) & (played["date"] >= as_of - pd.Timedelta(days=window_days))]
    df = df.dropna(subset=["hg", "ag"])
    teams = sorted(set(df["home"]) | set(df["away"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    hi = df["home"].map(idx).to_numpy()
    ai = df["away"].map(idx).to_numpy()
    x, y = df["hg"].to_numpy(float), df["ag"].to_numpy(float)
    w = np.exp(-xi * (as_of - df["date"]).dt.days.to_numpy())

    theta0 = np.zeros(2 * n + 1)
    theta0[-3:] = [0.1, 0.25, -0.05]
    if init is not None:                       # warm start from last week's fit
        for t, i in idx.items():
            j = init.index(t)
            if j is not None and i < n - 1:
                theta0[i], theta0[n - 1 + i] = init.att[j], init.dfn[j]
        theta0[-3:] = [init.mu, init.home, init.rho]
    bounds = [(None, None)] * (2 * n) + [RHO_BOUNDS]

    # Prior: every team shrinks toward average (0), except teams with little
    # data in the window (mostly newly promoted). Those shrink toward the
    # level of the three weakest established teams. A first pass without the
    # new teams' prior finds that level. Same idea as the Elo promoted-team rule.
    counts = pd.concat([df["home"], df["away"]]).value_counts().reindex(teams).to_numpy()
    prior_att, prior_dfn = np.zeros(n), np.zeros(n)
    args = (hi, ai, x, y, w, n, prior_att, prior_dfn)
    res = minimize(_neg_loglik, theta0, args=args, method="L-BFGS-B", bounds=bounds)
    new = counts < NEW_TEAM_MIN_MATCHES
    if new.any() and (~new).sum() >= 3:
        att0, dfn0, *_ = _unpack(res.x, n)
        est = np.where(~new)[0]
        weak = est[np.argsort((att0 - dfn0)[est])[:3]]
        prior_att[new], prior_dfn[new] = att0[weak].mean(), dfn0[weak].mean()
        args = (hi, ai, x, y, w, n, prior_att, prior_dfn)
        res = minimize(_neg_loglik, res.x, args=args, method="L-BFGS-B", bounds=bounds)
    att, dfn, mu, home, rho = _unpack(res.x, n)

    cov = None
    if with_cov:
        from statsmodels.tools.numdiff import approx_hess3
        H = approx_hess3(res.x, _neg_loglik, args=args)
        cov = np.linalg.pinv(H)
    params = DCParams(teams, att, dfn, float(home), float(mu), float(rho), cov=cov, fit_date=as_of)
    params._theta = res.x                     # kept for sampling in the simulator
    return params


def score_matrix(lam: float, nu: float, rho: float, max_goals: int = config.MAX_GOALS) -> np.ndarray:
    g = np.arange(max_goals + 1)
    ph = np.exp(g * np.log(lam) - lam - gammaln(g + 1))
    pa = np.exp(g * np.log(nu) - nu - gammaln(g + 1))
    m = np.outer(ph, pa)
    m[0, 0] *= 1 - lam * nu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + nu * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()                          # renormalise the truncated grid


def outcome_probs(params: DCParams, home: str, away: str, rho: float | None = None) -> np.ndarray:
    lam, nu = params.rates(home, away)
    m = score_matrix(lam, nu, params.rho if rho is None else rho)
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def walk_forward(games: pd.DataFrame, start: pd.Timestamp, refit_every_days: int = 7) -> pd.DataFrame:
    """Predict every game on/after `start` using a model fitted strictly before
    the start of that game's 7-day block. Also returns the plain-Poisson
    (rho = 0) probabilities and expected goals, which the draws analysis uses."""
    played = games.dropna(subset=["hg", "ag"])
    todo = games[games["date"] >= start].sort_values("date")
    out, params = [], None
    block_start = start
    last_played = played["date"].max()
    future = todo[todo["date"] > last_played]          # unplayed fixtures: one fit "as of today"
    todo = todo[todo["date"] <= last_played]
    while len(todo) and block_start <= todo["date"].max():
        block_end = block_start + pd.Timedelta(days=refit_every_days)
        block = todo[(todo["date"] >= block_start) & (todo["date"] < block_end)]
        if len(block):
            params = fit(played, as_of=block_start, init=params)
            for r in block.itertuples():
                lam, nu = params.rates(r.home, r.away)
                p = outcome_probs(params, r.home, r.away)
                p0 = outcome_probs(params, r.home, r.away, rho=0.0)
                out.append((r.match_id, *p, *p0, lam, nu, params.rho))
        block_start = block_end
    if len(future):
        params = fit(played, as_of=last_played + pd.Timedelta(days=1), init=params)
        for r in future.itertuples():
            lam, nu = params.rates(r.home, r.away)
            p = outcome_probs(params, r.home, r.away)
            p0 = outcome_probs(params, r.home, r.away, rho=0.0)
            out.append((r.match_id, *p, *p0, lam, nu, params.rho))
    cols = ["match_id", "pH", "pD", "pA", "poisson_pH", "poisson_pD", "poisson_pA", "exp_hg", "exp_ag", "rho"]
    return pd.DataFrame(out, columns=cols)
