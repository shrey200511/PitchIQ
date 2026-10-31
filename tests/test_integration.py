"""Integration tests on synthetic data: merge, leakage, simulator, full pipeline."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pitchiq.features import build, leakage
from pitchiq.ratings.elo import EloParams

ELO = EloParams(k=15, hfa=60, season_regress=0.25)


def test_merge_shapes_and_xg_window(synth_data):
    matches, fixtures = synth_data
    complete = matches[matches.season < 2026].groupby("season").size()
    assert (complete == 380).all()
    assert matches.match_id.is_unique
    assert matches[matches.season < 2014].home_xg.isna().all()        # no xG before Understat
    assert matches[matches.season >= 2014].home_xg.notna().mean() > 0.99
    assert "Nottingham Forest" not in set(matches.home)                # alias normalised
    assert len(fixtures) > 0 and fixtures.season.eq(2026).all()


def test_leakage_perturbation_passes(synth_data):
    matches, fixtures = synth_data
    leakage.check_feature_names()
    rep = leakage.perturbation_test(matches, fixtures, ELO, leakage.default_test_dates(matches, n_random=2))
    assert (rep.leaked_features == "none").all()
    assert rep.later_features_changed.all()       # the scramble really did change later data


def test_leakage_test_catches_a_real_leak(synth_data, monkeypatch):
    """Inject a classic bug (including the same day's match in rolling stats)
    and check the perturbation test catches it."""
    matches, fixtures = synth_data

    def leaky_asof(games, table, side_col, cols, by_extra=None):
        by_extra = by_extra or []
        left = games[["match_id", "date", side_col] + by_extra].rename(columns={side_col: "team"})
        right = table[["team", "date"] + by_extra + cols].sort_values("date")
        res = pd.merge_asof(left.sort_values("date"), right, on="date", by=["team"] + by_extra,
                            allow_exact_matches=True)
        return res.set_index("match_id")[cols]

    monkeypatch.setattr(build, "_asof", leaky_asof)
    with pytest.raises(AssertionError, match="Leakage"):
        leakage.perturbation_test(matches, fixtures, ELO, leakage.default_test_dates(matches, n_random=1))


def test_simulator_probabilities_are_coherent(synth_data):
    from pitchiq.simulate.season import simulate
    matches, fixtures = synth_data
    res = simulate(matches, 2026, fixtures, n_sims=2000, param_uncertainty=True)
    s = res["summary"]
    assert len(s) == 20
    assert s.p_title.sum() == pytest.approx(1.0)
    assert s.p_top4.sum() == pytest.approx(4.0)
    assert s.p_relegation.sum() == pytest.approx(3.0)
    assert (s.exp_points >= s.points).all()
    # every remaining match hands out 2 or 3 points
    gained = (s.exp_points - s.points).sum()
    assert 2 * res["remaining"] <= gained <= 3 * res["remaining"]
    assert np.allclose(res["position_probs"].sum(axis=1), 1)


@pytest.mark.slow
def test_full_pipeline_end_to_end(tmp_path):
    env = {**os.environ, "PITCHIQ_ROOT": str(tmp_path), "PITCHIQ_SYNTHETIC": "1",
           "PITCHIQ_CURRENT_SEASON": "2026"}
    for step in ("synthetic", "all"):
        subprocess.run([sys.executable, "-m", "pitchiq.pipeline", step], env=env, check=True)
    for f in ["upcoming.csv", "season_sim.csv", "backtest_summary.csv", "explanations.json", "meta.json"]:
        assert (tmp_path / "artifacts" / f).exists(), f
    report = (tmp_path / "reports" / "results.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in report
