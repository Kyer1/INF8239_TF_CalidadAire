"""Tablero de alerta temprana de PM2.5 (12 horas). Ejecutar: uv run streamlit run app/streamlit_app.py"""
import datetime as dt

import joblib
import pandas as pd
import streamlit as st

from calidad_aire.config import HORIZON_HOURS, MODELS, RAW_CSV, REPORTS, THRESHOLD_UGM3
from calidad_aire.data import load_raw
from calidad_aire.features import modeling_table, temporal_split, xy

st.set_page_config(page_title="Alerta PM2.5", layout="wide")
st.title("Alerta temprana de PM2.5 alto · Beijing")
st.caption(f"Predice si dentro de {HORIZON_HOURS} horas el PM2.5 superará {THRESHOLD_UGM3} µg/m³. "
           "Demostración académica: no sustituye a los avisos oficiales de calidad del aire.")

if not RAW_CSV.exists() or not (MODELS / "selected_model.joblib").exists():
    st.error("Faltan datos o modelo. Ejecute: uv run python scripts/download_data.py y uv run python scripts/train.py")
    st.stop()


@st.cache_resource
def load():
    bundle = joblib.load(MODELS / "selected_model.joblib")
    _, _, test = temporal_split(modeling_table(load_raw()))
    return bundle, test


bundle, test = load()
model, thr = bundle["model"], bundle["threshold"]
tab1, tab2, tab3 = st.tabs(["Escenario", "Historial 2014", "Desempeño"])

with tab1:
    st.subheader("1. Elegir una hora de 2014 y ajustar las condiciones")
    c1, c2 = st.columns(2)
    day = c1.date_input("Fecha", dt.date(2014, 1, 15), min_value=test.index.min().date(), max_value=test.index.max().date())
    hour = c2.slider("Hora", 0, 23, 8)
    stamp = pd.Timestamp(day) + pd.Timedelta(hours=int(hour))
    if stamp not in test.index:
        st.warning("Esa hora no tiene medición de PM2.5; elija otra.")
        st.stop()
    X_row, y_row = xy(test.loc[[stamp]])
    X_row = X_row.copy()
    c1, c2, c3, c4 = st.columns(4)
    X_row["pm25"] = c1.number_input("PM2.5 actual (µg/m³)", 0.0, 1000.0, float(X_row["pm25"].iloc[0]))
    X_row["PRES"] = c2.number_input("Presión (hPa)", 990.0, 1050.0, float(X_row["PRES"].iloc[0]))
    X_row["Iws"] = c3.number_input("Viento acumulado (m/s)", 0.0, 600.0, float(X_row["Iws"].iloc[0]))
    winds = ["NW", "NE", "SE", "cv"]
    X_row["cbwd"] = c4.selectbox("Dirección del viento", winds, index=winds.index(X_row["cbwd"].iloc[0]))
    proba = float(model.predict_proba(X_row)[:, 1][0])
    st.subheader("2. Resultado")
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de PM2.5 alto", f"{proba:.0%}")
    c2.metric("Decisión (umbral {:.2f})".format(thr), "ALERTA" if proba >= thr else "Sin alerta")
    real = test.loc[stamp, "pm25_future"]
    c3.metric(f"PM2.5 real {HORIZON_HOURS} h después", f"{real:.0f} µg/m³")
    st.info("La probabilidad depende de los datos de 2010–2013 de una sola estación. Revise siempre la medición real.")

with tab2:
    st.subheader("Probabilidad predicha y PM2.5 real en el periodo de prueba")
    month = st.selectbox("Mes de 2014", list(range(1, 13)), index=0)
    pred = pd.read_csv(REPORTS / "predictions_test.csv", index_col=0, parse_dates=True)
    sel = pred[pred.index.month == month]
    st.line_chart(sel[["pm25_future"]].rename(columns={"pm25_future": f"PM2.5 real en t+{HORIZON_HOURS} h"}))
    st.line_chart(sel[["proba"]].rename(columns={"proba": "Probabilidad de alerta"}))
    hits = (sel["pred"] == sel["real"]).mean()
    st.write(f"Aciertos en el mes: **{hits:.1%}** · horas con alerta: **{int(sel['pred'].sum())}** de {len(sel)}")

with tab3:
    st.subheader("Comparación de modelos en prueba (2014)")
    metrics = pd.read_csv(REPORTS / "metrics.csv")
    st.dataframe(metrics[["model", "f1_macro", "recall_alto", "precision_alto", "roc_auc", "fit_median_s", "size_kb"]].round(3))
    st.image(str(REPORTS / "confusion_test.png"))
