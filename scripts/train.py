import json
import platform
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, average_precision_score,
                             f1_score, precision_score, recall_score, roc_auc_score)

from calidad_aire.config import HORIZON_HOURS, MODELS, REPORTS, SEED
from calidad_aire.data import load_raw
from calidad_aire.features import modeling_table, temporal_split, xy
from calidad_aire.models import best_threshold, build_models


def scores(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    has_proba = len(np.unique(proba)) > 2
    return {"f1_alto": f1_score(y, pred, zero_division=0), "recall_alto": recall_score(y, pred, zero_division=0),
            "precision_alto": precision_score(y, pred, zero_division=0),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y, proba) if has_proba else np.nan,
            "pr_auc": average_precision_score(y, proba) if has_proba else np.nan}


def main() -> None:
    REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    table = modeling_table(load_raw())
    train, valid, test = temporal_split(table)
    X_tr, y_tr = xy(train); X_va, y_va = xy(valid); X_te, y_te = xy(test)
    X_trva, y_trva = xy(pd.concat([train, valid]))
    print(f"Entrenamiento {len(train)} · Validación {len(valid)} · Prueba {len(test)}")

    rows, final_models, test_proba = [], {}, {}
    for name, model in build_models().items():
        # 1) Ajuste con entrenamiento y elección del umbral con VALIDACIÓN
        m = clone(model).fit(X_tr, y_tr)
        p_va = m.predict_proba(X_va)[:, 1]
        thr = best_threshold(y_va, p_va) if name in ("logistica", "gradient_boosting") else 0.5
        valid_scores = scores(y_va, p_va, thr)
        # 2) Modelo final reentrenado con entrenamiento + validación; evaluación única en PRUEBA (2014)
        fit_times = []
        for _ in range(3):
            final = clone(model)
            start = perf_counter(); final.fit(X_trva, y_trva); fit_times.append(perf_counter() - start)
        start = perf_counter(); p_te = final.predict_proba(X_te)[:, 1]; pred_ms = (perf_counter() - start) * 1000
        path = MODELS / f"{name}.joblib"; joblib.dump({"model": final, "threshold": thr}, path)
        final_models[name], test_proba[name] = (final, thr), p_te
        row = {"model": name, "threshold": thr, **{f"valid_{k}": v for k, v in valid_scores.items()},
               **scores(y_te, p_te, thr), "fit_median_s": float(np.median(fit_times)),
               "predict_ms": pred_ms, "size_kb": path.stat().st_size / 1024}
        rows.append(row)
        print(f"{name}: umbral={thr:.2f} · F1 alto prueba={row['f1_alto']:.3f} · recall={row['recall_alto']:.3f}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORTS / "metrics.csv", index=False)
    candidates = metrics[metrics["model"].isin(["logistica", "gradient_boosting"])]
    selected = candidates.sort_values("valid_f1_macro", ascending=False).iloc[0]["model"]   # selección por validación
    model, thr = final_models[selected]
    joblib.dump({"model": model, "threshold": thr, "name": selected}, MODELS / "selected_model.joblib")

    # Matrices de confusión: persistencia vs modelo seleccionado
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ["persistencia", selected]):
        t = final_models[name][1]
        ConfusionMatrixDisplay.from_predictions(y_te, (test_proba[name] >= t).astype(int), display_labels=["Normal", "Alto"],
                                                cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{name} (umbral {t:.2f})")
    fig.tight_layout(); fig.savefig(REPORTS / "confusion_test.png", dpi=160)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["logistica", "gradient_boosting"]:
        PrecisionRecallDisplay.from_predictions(y_te, test_proba[name], name=name, ax=ax)
    ax.set_title("Curva precisión–recall en prueba (2014)")
    fig.tight_layout(); fig.savefig(REPORTS / "pr_curve.png", dpi=160)

    # Análisis de errores del modelo seleccionado
    pred = (test_proba[selected] >= thr).astype(int)
    errors = test.assign(proba=test_proba[selected], pred=pred, real=y_te)
    errors["tipo"] = np.select([(errors.real == 1) & (errors.pred == 0), (errors.real == 0) & (errors.pred == 1)],
                               ["falso_negativo", "falso_positivo"], "acierto")
    errors["mes"] = errors.index.month
    by_month = errors.groupby("mes")["tipo"].value_counts().unstack(fill_value=0)
    by_month.to_csv(REPORTS / "errors_by_month.csv")
    errors["cambio_h"] = errors["pm25_future"] - errors["pm25"]
    bins = [-1e9, -150, -50, 50, 150, 1e9]
    labels = ["baja >150", "baja 50–150", "estable ±50", "sube 50–150", "sube >150"]
    errors["cambio"] = pd.cut(errors["cambio_h"], bins=bins, labels=labels)
    by_change = errors.groupby("cambio", observed=False)["tipo"].value_counts().unstack(fill_value=0)
    by_change.to_csv(REPORTS / "errors_by_change.csv")
    keep = ["pm25", "pm25_roll24", "pm25_future", "cambio_h", "DEWP", "TEMP", "PRES", "Iws", "cbwd", "proba", "tipo"]
    errors[errors.tipo != "acierto"][keep].round(3).to_csv(REPORTS / "errors_test.csv")
    errors[["pm25", "pm25_future", "proba", "pred", "real"]].round(4).to_csv(REPORTS / "predictions_test.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    by_month[["falso_negativo", "falso_positivo"]].plot.bar(ax=axes[0], rot=0)
    axes[0].set(title=f"Errores de {selected} por mes (2014)", xlabel="Mes", ylabel="Horas")
    by_change[["falso_negativo", "falso_positivo"]].plot.bar(ax=axes[1], rot=20)
    axes[1].set(title=f"Errores según el cambio real de PM2.5 en {HORIZON_HOURS} h", xlabel="Cambio (µg/m³)", ylabel="Horas")
    fig.tight_layout(); fig.savefig(REPORTS / "errors_analysis.png", dpi=160)

    # Importancia por permutación (sobre validación, modelo ajustado solo con entrenamiento)
    m_tr = clone(build_models()[selected]).fit(X_tr, y_tr)
    imp = permutation_importance(m_tr, X_va, y_va, scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=1)
    importances = pd.Series(imp.importances_mean, index=X_va.columns).sort_values()
    importances.to_csv(REPORTS / "feature_importance.csv", header=["importance"])
    fig, ax = plt.subplots(figsize=(7, 6))
    importances.tail(12).plot.barh(ax=ax)
    ax.set(title=f"Importancia por permutación ({selected}, validación)", xlabel="Caída de PR-AUC")
    fig.tight_layout(); fig.savefig(REPORTS / "feature_importance.png", dpi=160)

    week = errors.loc["2014-01-10":"2014-02-10"]
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(week.index, week["pm25_future"], label=f"PM2.5 real en t+{HORIZON_HOURS} h", lw=1)
    ax1.axhline(75, color="red", ls="--", lw=1, label="Umbral 75 µg/m³")
    ax1.set_ylabel("µg/m³")
    ax2 = ax1.twinx(); ax2.plot(week.index, week["proba"], color="orange", lw=1, label="Probabilidad de alerta")
    ax2.axhline(thr, color="orange", ls=":", lw=1); ax2.set_ylabel("Probabilidad"); ax2.set_ylim(0, 1)
    ax1.set_title(f"Ejemplo de alerta temprana ({HORIZON_HOURS} h): enero–febrero 2014")
    fig.legend(loc="upper left", bbox_to_anchor=(0.07, 0.95), fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "timeline_example.png", dpi=160)

    # Comparación de horizontes: ¿cuánto se puede anticipar? (gradient boosting vs persistencia)
    rows_h = []
    raw = load_raw()
    for h in (6, 12, 24):
        tr_h, va_h, te_h = temporal_split(modeling_table(raw, h))
        (Xa, ya), (Xb, yb), (Xc, yc) = xy(tr_h), xy(va_h), xy(te_h)
        m_h = clone(build_models()["gradient_boosting"]).fit(Xa, ya)
        t_h = best_threshold(yb, m_h.predict_proba(Xb)[:, 1])
        p_h = m_h.predict_proba(Xc)[:, 1]
        rows_h.append({"horizon_h": h, "persistencia_f1_macro": f1_score(yc, (Xc["pm25"] > 75).astype(int), average="macro"),
                       "gb_f1_macro": f1_score(yc, (p_h >= t_h).astype(int), average="macro"),
                       "gb_roc_auc": roc_auc_score(yc, p_h)})
    horizons = pd.DataFrame(rows_h); horizons["mejora"] = horizons.gb_f1_macro - horizons.persistencia_f1_macro
    horizons.round(4).to_csv(REPORTS / "horizon_comparison.csv", index=False)
    print(horizons.round(3).to_string())

    summary = {"selected_model": selected, "threshold": thr, "horizon_hours": HORIZON_HOURS,
               "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
               "test_high_rate": float(y_te.mean()),
               "environment": {"python": platform.python_version(), "platform": platform.platform(),
                               "processor": platform.processor(), "scikit_learn": sklearn.__version__}}
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.round(4).to_string())
    print("Modelo seleccionado por validación:", selected)


if __name__ == "__main__":
    main()
