from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SEED, THRESHOLD_UGM3
from .features import CATEGORICAL, NUMERIC


class PersistenceBaseline(BaseEstimator, ClassifierMixin):
    """Baseline ingenuo (persistencia): el nivel futuro será igual al actual."""

    def __init__(self, threshold: float = THRESHOLD_UGM3):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (X["pm25"].to_numpy() > self.threshold).astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.predict_proba(X)[:, 1].astype(int)


def _preprocess(scale: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])


def build_models() -> dict:
    return {
        "dummy": Pipeline([("prep", _preprocess(False)), ("model", DummyClassifier(strategy="most_frequent"))]),
        "persistencia": PersistenceBaseline(),
        "logistica": Pipeline([("prep", _preprocess(True)),
                               ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))]),
        "gradient_boosting": Pipeline([("prep", _preprocess(False)),
                                       ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                                class_weight="balanced", random_state=SEED))]),
    }


def best_threshold(y_true, proba, grid=None) -> float:
    """Umbral de probabilidad que maximiza el F1 macro (se elige en VALIDACIÓN, nunca en prueba)."""
    from sklearn.metrics import f1_score
    grid = np.round(np.arange(0.20, 0.81, 0.05), 2) if grid is None else grid
    scores = [f1_score(y_true, (proba >= t).astype(int), average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])
