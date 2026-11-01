from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZON_HOURS, LAGS, TEST_YEARS, THRESHOLD_UGM3, TRAIN_YEARS, VALID_YEARS

NUMERIC = ["pm25", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "pm25_roll24", "pm25_diff1",
           "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek"] + [f"pm25_lag{l}" for l in LAGS] + \
          [f"{c}_d{h}" for c in ("PRES", "DEWP", "TEMP") for h in (3, 12, 24)]
CATEGORICAL = ["cbwd"]
TARGET = "high_future"


def make_target(pm25: pd.Series, horizon: int = HORIZON_HOURS, threshold: float = THRESHOLD_UGM3) -> pd.Series:
    """1 si el PM2.5 dentro de `horizon` horas supera el umbral; NaN si ese valor no existe."""
    future = pm25.shift(-horizon)
    target = (future > threshold).astype("float")
    return target.where(future.notna())


def build_features(df: pd.DataFrame, horizon: int = HORIZON_HOURS) -> pd.DataFrame:
    """Solo usa información disponible en la hora t (valores actuales y pasados)."""
    out = pd.DataFrame(index=df.index)
    for col in ["pm25", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "cbwd"]:
        out[col] = df[col]
    for lag in LAGS:
        out[f"pm25_lag{lag}"] = df["pm25"].shift(lag)
    out["pm25_roll24"] = df["pm25"].rolling(24, min_periods=12).mean()   # ventana t-23 … t
    out["pm25_diff1"] = df["pm25"] - df["pm25"].shift(1)
    hour, month = df.index.hour, df.index.month
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["month_sin"], out["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    out["dayofweek"] = df.index.dayofweek
    for col in ("PRES", "DEWP", "TEMP"):                  # tendencia meteorológica (cambio en las últimas h horas)
        for h in (3, 12, 24):
            out[f"{col}_d{h}"] = df[col] - df[col].shift(h)
    out["pm25_future"] = df["pm25"].shift(-horizon)        # solo para análisis de errores, nunca como feature
    out[TARGET] = make_target(df["pm25"], horizon=horizon)
    return out


def modeling_table(df: pd.DataFrame, horizon: int = HORIZON_HOURS) -> pd.DataFrame:
    """Filas con objetivo conocido y PM2.5 actual medido."""
    table = build_features(df, horizon)
    return table[table[TARGET].notna() & table["pm25"].notna()].copy()


def temporal_split(table: pd.DataFrame):
    years = table.index.year
    parts = [table[years.isin(y)] for y in (TRAIN_YEARS, VALID_YEARS, TEST_YEARS)]
    return parts


def xy(part: pd.DataFrame):
    return part[NUMERIC + CATEGORICAL], part[TARGET].astype(int)
