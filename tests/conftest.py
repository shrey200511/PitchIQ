"""Shared fixtures. Synthetic data is generated into a temp folder by
re-pointing the config paths, so tests never touch real data or websites."""
import pandas as pd
import pytest

from pitchiq import config


def _point_config_at(root, monkeypatch):
    paths = dict(ROOT=root, DATA=root / "data", RAW=root / "data/raw", RAW_FD=root / "data/raw/football_data",
                 RAW_US=root / "data/raw/understat", INTERIM=root / "data/interim",
                 PROCESSED=root / "data/processed", ARTIFACTS=root / "artifacts",
                 REPORTS=root / "reports", FIGURES=root / "reports/figures")
    for k, v in paths.items():
        monkeypatch.setattr(config, k, v)
    monkeypatch.setattr(config, "SYNTHETIC", True)
    monkeypatch.setattr(config, "CURRENT_SEASON", 2026)
    monkeypatch.setattr(config, "LAST_COMPLETE", 2025)


@pytest.fixture(scope="session")
def synth_data(tmp_path_factory):
    """(matches, fixtures) built from synthetic raw files, generated once per session."""
    root = tmp_path_factory.mktemp("synth")
    mp = pytest.MonkeyPatch()
    _point_config_at(root, mp)
    from pitchiq.data import merge, synthetic
    synthetic.generate()
    matches, fixtures = merge.build_match_table()
    yield matches, fixtures
    mp.undo()


@pytest.fixture
def tiny_matches():
    rows = [("2020_A_B", 2020, "2020-08-15", "A", "B", 2, 0),
            ("2020_C_D", 2020, "2020-08-15", "C", "D", 1, 1),
            ("2020_B_C", 2020, "2020-08-22", "B", "C", 0, 3),
            ("2020_D_A", 2020, "2020-08-22", "D", "A", 1, 2)]
    df = pd.DataFrame(rows, columns=["match_id", "season", "date", "home", "away", "hg", "ag"])
    df["date"] = pd.to_datetime(df["date"])
    return df
