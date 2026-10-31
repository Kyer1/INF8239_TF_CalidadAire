from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_URL = "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip"
RAW_CSV = ROOT / "data" / "raw" / "beijing_pm25.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

THRESHOLD_UGM3 = 75      # µg/m³: límite de 24 h de la norma china GB 3095-2012 (grado II)
HORIZON_HOURS = 12       # se predice si habrá PM2.5 alto 12 horas después (ver comparación de horizontes)
LAGS = [1, 3, 6, 12, 24]
TRAIN_YEARS = [2010, 2011, 2012]
VALID_YEARS = [2013]
TEST_YEARS = [2014]
SEED = 42
