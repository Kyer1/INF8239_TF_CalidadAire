import json

import matplotlib.pyplot as plt

from calidad_aire.config import REPORTS, THRESHOLD_UGM3
from calidad_aire.data import audit, load_raw
from calidad_aire.features import modeling_table, temporal_split


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load_raw()
    result = audit(df)
    table = modeling_table(df)
    parts = dict(zip(["train", "valid", "test"], temporal_split(table)))
    result["modeling_rows"] = {k: int(len(v)) for k, v in parts.items()}
    result["high_rate"] = {k: round(float(v["high_future"].mean()), 3) for k, v in parts.items()}
    result["missing_pm25_by_year"] = df["pm25"].isna().groupby(df.index.year).sum().astype(int).to_dict()
    (REPORTS / "audit.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))

    daily = df["pm25"].resample("D").mean()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(daily.index, daily.values, lw=0.7)
    axes[0].axhline(THRESHOLD_UGM3, color="red", ls="--", label=f"Umbral {THRESHOLD_UGM3} µg/m³")
    for year in (2013, 2014):
        axes[0].axvline(df.index[df.index.year == year][0], color="gray", ls=":")
    axes[0].set(title="PM2.5 promedio diario (líneas grises: inicio de validación y prueba)", ylabel="µg/m³")
    axes[0].legend()
    monthly = df.groupby(df.index.month)["pm25"]
    axes[1].boxplot([g.dropna() for _, g in monthly], showfliers=False)
    axes[1].axhline(THRESHOLD_UGM3, color="red", ls="--")
    axes[1].set(title="Distribución horaria de PM2.5 por mes (sin valores extremos)", xlabel="Mes", ylabel="µg/m³")
    fig.tight_layout()
    fig.savefig(REPORTS / "audit_pm25.png", dpi=160)
    print("Guardado: reports/audit.json y reports/audit_pm25.png")


if __name__ == "__main__":
    main()
