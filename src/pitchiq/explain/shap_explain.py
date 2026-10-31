"""SHAP explanations for the LightGBM model.

SHAP splits one prediction into additive feature contributions:
    model score for class k = base value + sum of SHAP values
For a multiclass LightGBM, the "score" is the raw (pre-softmax) logit of
each class. A positive SHAP value for class H pushes the model toward a
home win. We report:
  * Global: mean |SHAP| per feature (which features matter overall).
  * Local: the top drivers for one fixture ("why does the model favour the
    home side here?").
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from pitchiq import config

LABELS = {
    "elo_home": "Home Elo rating", "elo_away": "Away Elo rating", "elo_diff": "Elo gap (home - away)",
    "elo_exp_home": "Elo expected score (home)",
    "h_pts5": "Home team form (pts/game, last 5)", "a_pts5": "Away team form (pts/game, last 5)",
    "h_gf_ewm": "Home goals scored (recent avg)", "h_ga_ewm": "Home goals conceded (recent avg)",
    "a_gf_ewm": "Away goals scored (recent avg)", "a_ga_ewm": "Away goals conceded (recent avg)",
    "h_xgf_ewm": "Home xG created (recent avg)", "h_xga_ewm": "Home xG allowed (recent avg)",
    "a_xgf_ewm": "Away xG created (recent avg)", "a_xga_ewm": "Away xG allowed (recent avg)",
    "h_sotf_ewm": "Home shots on target (recent avg)", "h_sota_ewm": "Home shots on target allowed",
    "a_sotf_ewm": "Away shots on target (recent avg)", "a_sota_ewm": "Away shots on target allowed",
    "h_venue_pts5": "Home team's home form", "a_venue_pts5": "Away team's away form",
    "h_rest_days": "Home rest days", "a_rest_days": "Away rest days", "rest_diff": "Rest-day gap",
    "h_promoted": "Home side newly promoted", "a_promoted": "Away side newly promoted",
    "h_games_played": "Home games played this season", "a_games_played": "Away games played this season",
    "xg_balance_diff": "xG balance gap (home - away)", "goal_balance_diff": "Goal balance gap (home - away)",
    "form_diff": "Form gap (home - away)", "h2h_ppg_home": "Head-to-head pts/game (home side)",
    "h2h_gd_home": "Head-to-head goal diff (home side)", "h2h_n": "Head-to-head meetings counted",
}


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return SHAP array shaped (n_rows, n_features, n_classes) + base values."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                         # older shap: list of (n, f) per class
        sv = np.stack(sv, axis=-1)
    base = np.atleast_1d(explainer.expected_value)
    return sv, base


def global_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    sv, _ = shap_values(model, X)
    imp = pd.DataFrame(np.abs(sv).mean(axis=0), index=X.columns, columns=config.OUTCOMES)
    imp["total"] = imp.sum(axis=1)
    imp = imp.sort_values("total", ascending=False)
    imp.insert(0, "label", [LABELS.get(f, f) for f in imp.index])

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    left = np.zeros(len(top))
    for k, color in zip(config.OUTCOMES, ["#4C78A8", "#9D9D9D", "#E45756"]):
        ax.barh(top.label, top[k], left=left, color=color, label={"H": "Home win", "D": "Draw", "A": "Away win"}[k])
        left += top[k].to_numpy()
    ax.set_xlabel("Mean |SHAP| (logit units), stacked by outcome")
    ax.set_title("LightGBM global feature importance (SHAP)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / "shap_global.png", dpi=130)
    plt.close(fig)
    return imp


def explain_fixtures(model, rows: pd.DataFrame, features, top_n: int = 5) -> list[dict]:
    """Per-fixture top drivers toward / away from a HOME WIN."""
    X = rows[features]
    sv, base = shap_values(model, X)
    out = []
    for i, r in enumerate(rows.itertuples()):
        contrib = pd.Series(sv[i, :, 0], index=features)
        order = contrib.abs().sort_values(ascending=False).index[:top_n]
        out.append({
            "match_id": r.match_id,
            "base_home_logit": float(base[0]),
            "drivers": [{"feature": f, "label": LABELS.get(f, f),
                         "value": None if pd.isna(X.iloc[i][f]) else float(X.iloc[i][f]),
                         "shap_home": float(contrib[f]),
                         "direction": "towards home win" if contrib[f] > 0 else "away from home win"}
                        for f in order],
        })
    return out
