import numpy as np

from calidad_aire.features import modeling_table, temporal_split, xy
from calidad_aire.models import PersistenceBaseline, best_threshold, build_models


def test_persistence_predicts_current_level(raw_df):
    X, _ = xy(modeling_table(raw_df, horizon=12))
    pred = PersistenceBaseline(threshold=75).fit(X).predict(X)
    assert np.array_equal(pred, (X["pm25"] > 75).astype(int).to_numpy())


def test_models_return_valid_probabilities(raw_df):
    train, _, test = temporal_split(modeling_table(raw_df, horizon=12))
    X_tr, y_tr = xy(train)
    X_te, _ = xy(test)
    for name, model in build_models().items():
        proba = model.fit(X_tr, y_tr).predict_proba(X_te)
        assert proba.shape == (len(X_te), 2), name
        assert np.all((proba >= 0) & (proba <= 1)), name
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_best_threshold_is_chosen_from_grid():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.3, 0.6, 0.9])
    thr = best_threshold(y, proba, grid=np.array([0.2, 0.5, 0.8]))
    assert thr == 0.5
