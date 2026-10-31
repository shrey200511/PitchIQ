"""Scoring rules, bookmaker-margin removal and calibration.

Why several metrics?
  * Log loss: punishes confident mistakes hard. The main metric for
    probability quality, and the one we select models on.
  * Brier score: mean squared error of the probability vector. More forgiving
    of rare confident misses, and easy to explain.
  * RPS (ranked probability score): respects the ORDER H > D > A, so calling
    a home win when the result is a draw counts as "less wrong" than when the
    result is an away win. The standard score in football forecasting papers.
  * Accuracy: easy to understand but blind to probability quality. The market
    is often "right" only about 50% of the time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from pitchiq.config import OUTCOMES

EPS = 1e-12


def onehot(results) -> np.ndarray:
    y = pd.Series(results).map({k: i for i, k in enumerate(OUTCOMES)}).to_numpy()
    out = np.zeros((len(y), 3))
    out[np.arange(len(y)), y] = 1
    return out


def log_loss(p, y1h):
    return float(-np.mean(np.sum(y1h * np.log(np.clip(p, EPS, 1)), axis=1)))


def brier(p, y1h):
    return float(np.mean(np.sum((p - y1h) ** 2, axis=1)))


def rps(p, y1h):
    cp, cy = np.cumsum(p, axis=1)[:, :2], np.cumsum(y1h, axis=1)[:, :2]
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / 2))


def accuracy(p, y1h):
    return float(np.mean(p.argmax(axis=1) == y1h.argmax(axis=1)))


def score_all(p, y1h) -> dict:
    return dict(log_loss=log_loss(p, y1h), brier=brier(p, y1h), rps=rps(p, y1h),
                accuracy=accuracy(p, y1h), n=len(p))


# ------------------------------------------------------------- bookmaker margin
def implied_raw(odds: np.ndarray) -> np.ndarray:
    """1 / decimal odds. These sum to MORE than 1; the excess is the overround."""
    return 1.0 / odds


def remove_margin_proportional(odds: np.ndarray) -> np.ndarray:
    """Divide each 1/odds by their sum. Simple and standard; assumes the margin
    is spread across outcomes in proportion to their probability."""
    q = implied_raw(odds)
    return q / q.sum(axis=1, keepdims=True)


def _shin_row(q: np.ndarray) -> np.ndarray:
    B = q.sum()

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / B) - z) / (2 * (1 - z))

    try:
        z = brentq(lambda z: probs(z).sum() - 1, 0.0, 0.4)
    except ValueError:
        return q / B
    p = probs(z)
    return p / p.sum()


def remove_margin_shin(odds: np.ndarray) -> np.ndarray:
    """Shin (1993): models the margin as protection against insiders, which
    puts proportionally MORE margin on longshots. That is the well-documented
    favourite-longshot bias, which is why Shin usually beats proportional
    normalisation as a 'true probability' estimate."""
    q = implied_raw(odds)
    return np.vstack([_shin_row(row) for row in q])


def overround(odds: np.ndarray) -> np.ndarray:
    return implied_raw(odds).sum(axis=1) - 1


# ------------------------------------------------------------- calibration
def reliability_table(p: np.ndarray, y1h: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per outcome: bin predicted probabilities, compare with observed frequency."""
    rows = []
    edges = np.linspace(0, 1, n_bins + 1)
    for k, name in enumerate(OUTCOMES):
        b = np.clip(np.digitize(p[:, k], edges) - 1, 0, n_bins - 1)
        for i in range(n_bins):
            m = b == i
            if m.sum():
                rows.append(dict(outcome=name, bin=i, mean_pred=p[m, k].mean(),
                                 observed=y1h[m, k].mean(), n=int(m.sum())))
    return pd.DataFrame(rows)


def ece(p: np.ndarray, y1h: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, averaged over the three outcomes."""
    t = reliability_table(p, y1h, n_bins)
    t["gap"] = (t.mean_pred - t.observed).abs() * t.n
    return float(t.groupby("outcome").gap.sum().sum() / (3 * len(p)))


def paired_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """CI for mean(loss_a - loss_b) by resampling MATCHES (paired). A CI that
    contains 0 means we can't tell the two apart on this sample."""
    rng = np.random.default_rng(seed)
    d = loss_a - loss_b
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_match_log_loss(p, y1h):
    return -np.sum(y1h * np.log(np.clip(p, EPS, 1)), axis=1)
