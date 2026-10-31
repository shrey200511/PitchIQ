"""Machine-learning models + time-based (walk-forward) validation.

Walk-forward rule: to predict season S, train ONLY on seasons < S.
A random split would put future matches into training: the model would learn
late-season team strength and then "predict" earlier matches. That inflates
every metric. Walk-forward copies how the model is really used.

Models
  * Elo-logit: multinomial logistic regression on Elo difference only. A
    one-feature yardstick that shows how much the other features add.
  * Logistic regression: multinomial, all features, median imputation +
    missing-indicators + standardisation. Linear and easy to explain.
  * LightGBM: gradient-boosted trees (multiclass). Handles NaNs and
    non-linear interactions natively. Optional temperature scaling fixes
    over- or under-confidence, fitted on a season the model never trained on.
"""
from __future__ import annotations

import itertools
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pitchiq import config
from pitchiq.features.build import FEATURES

log = logging.getLogger(__name__)
LABELS = config.OUTCOMES                     # column order of every probability array
Y_MAP = {k: i for i, k in enumerate(LABELS)}
MODEL_START = 2015                           # first season with a full season of prior xG


# ------------------------------------------------------------- factories
def make_elo_logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def make_logreg(C: float = 0.1):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def make_lgbm(n_estimators=300, num_leaves=15, min_child_samples=60, learning_rate=0.03):
    return lgb.LGBMClassifier(objective="multiclass", n_estimators=n_estimators,
                              num_leaves=num_leaves, min_child_samples=min_child_samples,
                              learning_rate=learning_rate, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=config.RANDOM_SEED, verbose=-1)


# ------------------------------------------------------------- helpers
def xy(df: pd.DataFrame, features):
    d = df.dropna(subset=["result"])
    return d[features], d["result"].map(Y_MAP).to_numpy()


def predict(model, X) -> np.ndarray:
    """Probabilities in H/D/A order regardless of the model's internal class order."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3))
    out[:, model.classes_] = p
    return out


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1)) / T
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """T > 1 softens over-confident probabilities, T < 1 sharpens timid ones."""
    res = minimize_scalar(lambda T: log_loss(y, apply_temperature(p, T), labels=[0, 1, 2]),
                          bounds=(0.5, 3.0), method="bounded")
    return float(res.x)


# ------------------------------------------------------------- walk-forward
def walk_forward(table: pd.DataFrame, factory, seasons, features=FEATURES,
                 temperature: bool = False, name: str = "model") -> pd.DataFrame:
    out = []
    for s in seasons:
        train = table[(table.season >= MODEL_START) & (table.season < s)]
        test = table[table.season == s]
        X, y = xy(train, features)
        model = factory().fit(X, y)
        p = predict(model, test[features])
        T = 1.0
        if temperature:
            inner_tr = train[train.season < s - 1]
            inner_va = train[train.season == s - 1]
            Xi, yi = xy(inner_tr, features)
            Xv, yv = xy(inner_va, features)
            T = fit_temperature(predict(factory().fit(Xi, yi), Xv), yv)
            p = apply_temperature(p, T)
        out.append(pd.DataFrame({"match_id": test.match_id.to_numpy(), "season": s,
                                 "pH": p[:, 0], "pD": p[:, 1], "pA": p[:, 2],
                                 "model": name, "temperature": T}))
    return pd.concat(out, ignore_index=True)


def tune(table: pd.DataFrame, kind: str) -> tuple[dict, pd.DataFrame]:
    """Small grid search scored by mean log loss over VALIDATION seasons only."""
    if kind == "logreg":
        grid = [dict(C=c) for c in (0.01, 0.03, 0.1, 0.3, 1.0)]
        factory = lambda **kw: (lambda: make_logreg(**kw))
    elif kind == "lgbm":
        grid = [dict(n_estimators=n, num_leaves=l, min_child_samples=m)
                for n, l, m in itertools.product((150, 300, 600), (7, 15), (40, 100))]
        factory = lambda **kw: (lambda: make_lgbm(**kw))
    else:
        raise ValueError(kind)
    rows = []
    for params in grid:
        pred = walk_forward(table, factory(**params), config.VALIDATION_SEASONS)
        y = pred.match_id.map(table.set_index("match_id")["result"]).map(Y_MAP)
        mask = y.notna()
        ll = log_loss(y[mask].astype(int), pred.loc[mask, ["pH", "pD", "pA"]].to_numpy(), labels=[0, 1, 2])
        rows.append({**params, "val_log_loss": ll})
        log.info("%s %s -> %.4f", kind, params, ll)
    res = pd.DataFrame(rows).sort_values("val_log_loss").reset_index(drop=True)
    best = {k: (int(v) if k != "C" else float(v)) for k, v in res.iloc[0].drop("val_log_loss").items()}
    return best, res


def fit_final(table: pd.DataFrame, factory, features=FEATURES):
    """Train on every completed season (used for live predictions + SHAP)."""
    train = table[(table.season >= MODEL_START) & table.result.notna()]
    X, y = xy(train, features)
    return factory().fit(X, y)
