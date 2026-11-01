import numpy as np
import pandas as pd

from calidad_aire.features import TARGET, build_features, make_target, modeling_table, temporal_split


def test_target_uses_value_h_hours_later():
    pm = pd.Series([10, 80, 20, 100, 30])
    target = make_target(pm, horizon=2, threshold=75)
    assert target.iloc[0] == 0 and target.iloc[1] == 1      # 20 ≤ 75 ; 100 > 75
    assert target.iloc[-2:].isna().all()                     # no existe el futuro


def test_features_do_not_use_future_information(raw_df):
    original = build_features(raw_df, horizon=12)
    altered = raw_df.copy()
    altered.iloc[200:, altered.columns.get_loc("pm25")] = 999
    changed = build_features(altered, horizon=12)
    cols = [c for c in original.columns if c not in ("pm25_future", TARGET)]
    pd.testing.assert_frame_equal(original.iloc[:200][cols], changed.iloc[:200][cols])


def test_temporal_split_is_ordered_and_disjoint(raw_df):
    train, valid, test = temporal_split(modeling_table(raw_df, horizon=12))
    assert train.index.max() < valid.index.min() < valid.index.max() < test.index.min()
    assert set(train.index.year) == {2010, 2011, 2012}
    assert set(test.index.year) == {2014}


def test_modeling_table_has_no_missing_target(raw_df):
    table = modeling_table(raw_df, horizon=12)
    assert table[TARGET].notna().all() and table["pm25"].notna().all()
    assert set(np.unique(table[TARGET])) <= {0.0, 1.0}
