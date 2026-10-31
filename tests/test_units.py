"""Fast unit tests for the building blocks."""
import numpy as np
import pandas as pd
import pytest

from pitchiq.data.clean import parse_fd_dates, pick_odds, PRE_ODDS_CHAIN
from pitchiq.data.fetch import _parse_understat_html, season_code
from pitchiq.data.team_names import UnmappedTeamError, understat_to_fd
from pitchiq.evaluation import metrics as M
from pitchiq.models import dixon_coles as dc
from pitchiq.ratings.elo import EloParams, expected_home, margin_multiplier, run_elo


# ---------------------------------------------------------------- data
def test_season_code():
    assert season_code(2024) == "2425"
    assert season_code(2009) == "0910"
    assert season_code(1999) == "9900"


def test_dates_both_formats_day_first():
    s = parse_fd_dates(pd.Series(["03/02/15", "03/02/2024"]))
    assert list(s.dt.month) == [2, 2] and list(s.dt.day) == [3, 3]
    assert list(s.dt.year) == [2015, 2024]


def test_odds_fallback_chain():
    raw = pd.DataFrame({"PSH": [2.0, np.nan], "PSD": [3.5, np.nan], "PSA": [4.0, np.nan],
                        "AvgH": [1.9, 2.1], "AvgD": [3.4, 3.3], "AvgA": [3.9, 3.5]})
    odds, src = pick_odds(raw, PRE_ODDS_CHAIN)
    assert list(src) == ["PS", "Avg"]
    assert odds.loc[1, "H"] == 2.1


def test_understat_html_parse():
    html = r"""<script>var datesData = JSON.parse('[{\x22id\x22:\x221\x22}]');</script>"""
    assert _parse_understat_html(html) == [{"id": "1"}]


def test_team_mapping_and_loud_failure():
    known = {"Man United", "Wolves", "Arsenal"}
    assert understat_to_fd("Manchester United", known) == "Man United"
    assert understat_to_fd("Arsenal", known) == "Arsenal"
    with pytest.raises(UnmappedTeamError, match="Add it to UNDERSTAT_TO_FD"):
        understat_to_fd("Some New Club FC", known)


# ---------------------------------------------------------------- Elo
def test_elo_expected_is_symmetric_and_hfa_helps_home():
    assert expected_home(1500, 1500, 0) == pytest.approx(0.5)
    assert expected_home(1500, 1500, 60) > 0.5
    assert expected_home(1600, 1500, 0) + expected_home(1500, 1600, 0) == pytest.approx(1.0)


def test_margin_multiplier():
    assert list(margin_multiplier(np.array([0, 1, 2, 3, 5]))) == [1, 1, 1.5, 14 / 8, 16 / 8]


def test_elo_uses_pre_match_ratings_and_is_zero_sum(tiny_matches):
    p = EloParams(k=20, hfa=0, season_regress=0, use_margin=False)
    out = run_elo(tiny_matches, p).set_index("match_id")
    # Round 1: everyone starts at 1500 and the stored ratings are PRE-match.
    assert out.loc["2020_A_B", "elo_home"] == 1500 and out.loc["2020_A_B", "elo_away"] == 1500
    # A beat B (+10 / -10 with K=20, E=0.5). Then B hosts C: B is 1490 before kickoff.
    assert out.loc["2020_B_C", "elo_home"] == pytest.approx(1490)
    # C drew D (no change), so C is still 1500 before round 2.
    assert out.loc["2020_B_C", "elo_away"] == pytest.approx(1500)


# ---------------------------------------------------------------- metrics
def test_margin_removal_sums_to_one_and_shin_shades_longshots():
    odds = np.array([[1.50, 4.20, 7.00], [2.60, 3.30, 2.90]])
    for fn in (M.remove_margin_proportional, M.remove_margin_shin):
        assert np.allclose(fn(odds).sum(axis=1), 1)
    prop, shin = M.remove_margin_proportional(odds), M.remove_margin_shin(odds)
    assert shin[0, 2] < prop[0, 2]          # longshot gets LESS probability under Shin
    assert shin[0, 0] > prop[0, 0]          # favourite gets more
    assert (M.overround(odds) > 0).all()


def test_scores_known_values():
    y = M.onehot(["H", "D"])
    p = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    assert M.log_loss(p, y) == pytest.approx(-(np.log(0.5) + np.log(0.3)) / 2)
    assert M.accuracy(p, y) == 0.5
    # RPS: predicting H when the result is D is less bad than when it is A.
    assert M.rps(p, M.onehot(["D"])) < M.rps(p, M.onehot(["A"]))


def test_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 500)
    mean, lo, hi = M.paired_bootstrap(a, a - 0.05)
    assert lo > 0 and mean == pytest.approx(0.05)


# ---------------------------------------------------------------- Dixon-Coles
def test_score_matrix_rho_zero_is_independent_poisson():
    from scipy.stats import poisson
    m = dc.score_matrix(1.4, 1.1, 0.0)
    assert m.sum() == pytest.approx(1)
    assert m[2, 1] == pytest.approx(poisson.pmf(2, 1.4) * poisson.pmf(1, 1.1) / m.sum() * m.sum(), rel=1e-3)


def test_negative_rho_adds_draws():
    lo = dc.score_matrix(1.3, 1.1, 0.0)
    hi = dc.score_matrix(1.3, 1.1, -0.1)
    assert np.trace(hi) > np.trace(lo)


def test_dc_recovers_home_advantage():
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(10)]
    rows, day = [], pd.Timestamp("2020-01-01")
    for rep in range(6):
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append((day, h, a, rng.poisson(np.exp(0.2 + 0.3)), rng.poisson(np.exp(0.2))))
        day += pd.Timedelta(days=60)
    df = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag"])
    p = dc.fit(df, as_of=day, xi=0.0, window_days=10_000)
    assert p.home == pytest.approx(0.3, abs=0.08)
    probs = dc.outcome_probs(p, "T1", "T2")
    assert probs.sum() == pytest.approx(1) and probs[0] > probs[2]
