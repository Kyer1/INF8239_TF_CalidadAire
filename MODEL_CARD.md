# Model Card · Alerta de PM2.5 alto a 12 horas (gradient boosting)

## Modelo
- **Archivo:** `models/selected_model.joblib` (modelo + umbral). Seleccionado por F1 macro en validación (2013).
- **Algoritmo:** `HistGradientBoostingClassifier` (300 iteraciones, tasa 0.05, `class_weight="balanced"`, semilla 42) dentro de un `Pipeline` con imputación por mediana y one-hot de la dirección del viento.
- **Umbral de decisión:** 0.50 (elegido en validación entre 0.20 y 0.80).
- **Comparado con:** dummy, persistencia y regresión logística.

## Uso previsto
Proyecto académico: explorar cómo anticipar episodios de PM2.5 alto y comunicar su incertidumbre en un tablero.

## Usos fuera de alcance
Alertas oficiales de salud pública, decisiones médicas o laborales, otras ciudades, otras estaciones o periodos sin reentrenar y evaluar.

## Datos y partición
Beijing PM2.5 (UCI 381, CC BY 4.0). Entrenamiento 2010–2012, validación 2013, prueba 2014. El modelo final se reentrena con 2010–2013 y se evalúa una sola vez en 2014 (8,556 horas, 48.1 % en clase *alto*).

## Métricas (prueba 2014)
| Modelo | F1 macro | Recall alto | Precisión alto | ROC-AUC |
|---|---|---|---|---|
| persistencia | 0.702 | 0.691 | 0.691 | — |
| regresión logística | 0.722 | 0.776 | 0.687 | 0.793 |
| **gradient boosting** | **0.747** | **0.820** | **0.703** | **0.823** |

Matriz de confusión del modelo: 3,374 horas *alto* detectadas, 740 no detectadas (falsos negativos) y 1,424 alertas innecesarias (falsos positivos).

## Costo (CPU x86_64, Linux, Python 3.12, scikit-learn 1.9.1)
| Modelo | Ajuste (mediana de 3, s) | Inferencia (8,556 filas, ms) | Tamaño (KB) |
|---|---|---|---|
| regresión logística | 0.24 | 13.4 | 5.8 |
| gradient boosting | 4.71 | 84.1 | 1,090.5 |

## Limitaciones
- Los errores se concentran en cambios bruscos: no anticipa bien el inicio de un episodio (falsos negativos cuando el PM2.5 sube 50–150 µg/m³) ni la limpieza por frentes de viento (falsos positivos).
- No usa pronóstico meteorológico; con 24 horas de horizonte casi no supera a la persistencia.
- Una sola estación y datos de 2010–2014.

## Supervisión
Mostrar siempre la probabilidad, la medición real y la advertencia de uso académico. Reentrenar con datos recientes y vigilar el F1 macro y el recall por estación del año antes de cualquier uso real.
