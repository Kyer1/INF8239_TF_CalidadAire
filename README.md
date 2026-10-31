# Alerta temprana de PM2.5 alto en Beijing (12 horas)

**Trabajo final · INF-8239 Ciencia de Datos II · Modalidad A (proyecto aplicado)**
**Autora:** Kimberly Pereyra

Predice si dentro de **12 horas** la concentración de PM2.5 superará **75 µg/m³** usando la medición actual, sus valores pasados y la meteorología. Compara dos baselines (dummy y persistencia) con dos modelos candidatos (regresión logística y gradient boosting), analiza los errores, mide el costo computacional e incluye un tablero interactivo.

## Datos
*Beijing PM2.5* — UCI ML Repository, id 381 (Chen, S., 2015). DOI 10.24432/C5JS49. Licencia **CC BY 4.0**.
43,824 registros horarios (2010–2014) de la Embajada de EE. UU. en Beijing, con meteorología del aeropuerto. Ficha: `docs/DATASET_CARD.md`.
Los datos **no** se guardan en el repositorio; se descargan con el script.

## Instalación
```bash
uv python install 3.12
uv sync
```

## Ejecución completa
```bash
uv run python scripts/download_data.py   # descarga el ZIP oficial de UCI e imprime SHA-256
uv run python scripts/audit_data.py      # auditoría → reports/audit.json y audit_pm25.png
uv run python scripts/train.py           # baselines, modelos, umbral, errores, costo → reports/ y models/
uv run pytest -q                         # 11 pruebas automáticas
uv run streamlit run app/streamlit_app.py
```
Notebook ejecutado: `notebooks/trabajo_final_calidad_aire.ipynb` (también: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/trabajo_final_calidad_aire.ipynb`).

## Diseño
- **Objetivo:** `high_future = 1` si el PM2.5 de la hora t+12 es mayor a 75 µg/m³.
- **Variables (solo hora t y anteriores):** PM2.5 actual, rezagos de 1, 3, 6, 12 y 24 h, media de 24 h, cambio de 1 h, temperatura, punto de rocío, presión y sus cambios en 3, 12 y 24 h, viento, lluvia, nieve, dirección del viento y codificación cíclica de hora y mes.
- **Partición temporal:** entrenamiento 2010–2012, validación 2013 (umbral y selección), prueba 2014 (evaluación única tras reentrenar con 2010–2013).
- **Prevención de fuga:** sin partición aleatoria; imputación y escalado dentro del `Pipeline`; prueba automática que verifica que cambiar el futuro no altera las variables.

## Resultados (prueba 2014)
| Modelo | F1 macro | Recall alto | ROC-AUC | Ajuste (s) | Tamaño (KB) |
|---|---|---|---|---|---|
| dummy | 0.342 | 0.000 | — | 0.10 | 4.2 |
| persistencia | 0.702 | 0.691 | — | 0.00 | 0.4 |
| regresión logística | 0.722 | 0.776 | 0.793 | 0.24 | 5.8 |
| **gradient boosting** (seleccionado) | **0.747** | **0.820** | **0.823** | 4.71 | 1,090.5 |

Detalle en `reports/metrics.csv`; comparación de horizontes (6, 12 y 24 h) en `reports/horizon_comparison.csv`; ficha del modelo en `MODEL_CARD.md`; ética y Green AI en `docs/ETICA_Y_GREEN_AI.md`.

## Estructura
```
app/               tablero Streamlit
data/raw/          datos descargados (no se versionan)
docs/              ficha del dataset, ética y Green AI
models/            modelos entrenados (.joblib)
notebooks/         notebook ejecutado
reports/           métricas, figuras y análisis de errores
scripts/           descarga, auditoría y entrenamiento
src/calidad_aire/  configuración, datos, variables y modelos
tests/             pruebas automáticas (pytest)
```
