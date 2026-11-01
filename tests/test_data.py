import pandas as pd
import pytest

from calidad_aire.data import EXPECTED_COLUMNS, audit, download, validate_raw


def test_validate_raw_rejects_missing_columns():
    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_raw(pd.DataFrame({"pm2.5": [1]}))


def test_validate_raw_accepts_expected_columns():
    validate_raw(pd.DataFrame({c: [0] for c in EXPECTED_COLUMNS}))


def test_download_rejects_non_http_source(tmp_path):
    with pytest.raises(ValueError):
        download("C:/Users/datos.csv", tmp_path / "x.csv")


def test_audit_detects_missing_hours():
    idx = pd.to_datetime(["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 03:00"])
    df = pd.DataFrame({"pm25": [10, None, 200]}, index=idx)
    result = audit(df)
    assert result["missing_hours"] == 1
    assert result["nulls"] == {"pm25": 1}
