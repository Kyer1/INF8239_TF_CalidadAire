import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    """Serie horaria sintética con las mismas columnas que el dataset de UCI (ya cargada)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-01", "2014-12-31 23:00", freq="7h")
    n = len(idx)
    df = pd.DataFrame({
        "pm25": rng.gamma(2, 50, n).round(),
        "DEWP": rng.normal(0, 10, n), "TEMP": rng.normal(12, 10, n), "PRES": rng.normal(1015, 8, n),
        "Iws": rng.gamma(1, 20, n), "Is": 0, "Ir": 0,
        "cbwd": rng.choice(["NW", "NE", "SE", "cv"], n),
    }, index=idx)
    df.loc[df.index[5], "pm25"] = np.nan
    return df
