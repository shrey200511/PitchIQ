"""Central configuration: paths, season windows and modelling constants.

Every "magic number" in the project lives here so it can be found, changed
and defended in one place. The reasoning for each choice is in DECISIONS.md.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(os.environ.get("PITCHIQ_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_FD = RAW / "football_data"
RAW_US = RAW / "understat"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"          # small outputs the app reads (committed)
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Set PITCHIQ_SYNTHETIC=1 to run the whole pipeline on generated data
# (used for tests and CI; results from synthetic data are NOT real results).
SYNTHETIC = os.environ.get("PITCHIQ_SYNTHETIC", "0") == "1"

# ---------------------------------------------------------------- seasons
# A season is identified by its starting year: 2024 == 2024/25.
ELO_WARMUP_START = 2005       # results-only seasons used to warm up Elo
XG_START = 2014               # Understat xG starts in 2014/15


def _season_now() -> int:
    """Seasons start in August; from July we treat the new season as current."""
    from datetime import date
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


CURRENT_SEASON = int(os.environ.get("PITCHIQ_CURRENT_SEASON", _season_now()))
LAST_COMPLETE = CURRENT_SEASON - 1

# Walk-forward evaluation: for each season S in these lists, train on every
# complete season < S and predict S. Model selection only looks at VALIDATION.
VALIDATION_SEASONS = [2020, 2021, 2022, 2023]
TEST_SEASONS = [2024, 2025]   # fixed hold-out; touched once, at the end

# ---------------------------------------------------------------- sources
FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
US_LEAGUE_URL = "https://understat.com/league/EPL/{season}"
US_JSON_URL = "https://understat.com/getLeagueData/EPL/{season}"
REQUEST_DELAY_S = 3.0         # polite gap between Understat requests
USER_AGENT = "PitchIQ portfolio project (non-commercial; github.com/shrey200511)"

# ---------------------------------------------------------------- modelling
OUTCOMES = ["H", "D", "A"]
RANDOM_SEED = 42
N_SIMULATIONS = 10_000
MAX_GOALS = 10                # scoreline grid 0..10 for Poisson models
