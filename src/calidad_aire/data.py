from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_URL, RAW_CSV, THRESHOLD_UGM3

EXPECTED_COLUMNS = ["No", "year", "month", "day", "hour", "pm2.5",
                    "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str = DATA_URL, destination: Path = RAW_CSV) -> Path:
    """Descarga el ZIP oficial de UCI y guarda el CSV que contiene."""
    if not url.startswith(("https://", "http://")):
        raise ValueError("La fuente debe ser una URL HTTP(S)")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    zip_path = destination.with_suffix(".zip")
    zip_path.write_bytes(response.content)
    frame = pd.read_csv(zip_path)
    validate_raw(frame)
    frame.to_csv(destination, index=False, lineterminator="\n")
    return destination


def validate_raw(df: pd.DataFrame) -> None:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    if df.empty:
        raise ValueError("El dataset está vacío")


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecute: uv run python scripts/download_data.py")
    df = pd.read_csv(path)
    validate_raw(df)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("datetime").set_index("datetime")
    return df.rename(columns={"pm2.5": "pm25"})


def audit(df: pd.DataFrame) -> dict:
    expected = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return {
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "missing_hours": int(len(expected.difference(df.index))),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "nulls": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "pm25_missing_pct": round(float(df["pm25"].isna().mean() * 100), 2),
        "pm25_describe": {k: round(float(v), 1) for k, v in df["pm25"].describe().items()},
        "pct_hours_above_threshold": round(float((df["pm25"].dropna() > THRESHOLD_UGM3).mean() * 100), 1),
    }
