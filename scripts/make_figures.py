"""Figures for README / slides from committed results (no retraining).

Usage: python scripts/make_figures.py
Writes: docs/img/*.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hackalem.config import load_config
from hackalem.data.turbines import load_hourly

cfg = load_config()
DOCS = cfg["paths"]["outputs_dir"]
OUT = DOCS / "img"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def fig_feb():
    f = pd.read_csv(DOCS / "agent" / "agent_forecasts.csv", parse_dates=["time_local", "issue_date"])
    p = f[(f["unit"] == "plant")].sort_values("issue_date").groupby("time_local", as_index=False).last()
    p = p[p["time_local"].between(cfg["periods"]["test_start"], cfg["periods"]["test_end"])]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(p["time_local"], p["p10"], p["p90"], alpha=0.25, color="#1f77b4", label="интервал P10–P90")
    ax.plot(p["time_local"], p["p50"], color="#1f77b4", lw=1.3, label="прогноз P50")
    ax.set_ylim(0, 1.02); ax.set_ylabel("мощность, доля номинала")
    ax.set_title("Прогноз агента на февраль 2026 (станция, самый свежий выпуск для каждого часа)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "feb_forecast.png", dpi=140); plt.close(fig)


def fig_jan():
    y = load_hourly()["power_plant"]
    a = pd.read_csv(DOCS / "agent_rehearsal" / "agent_forecasts.csv", parse_dates=["time_local"])
    b = pd.read_csv(DOCS / "forecast" / "rehearsal_jan2026.csv", parse_dates=["time_local"])
    a = a[(a["unit"] == "plant") & (a["day_ahead"] == 1)].set_index("time_local").sort_index()
    b = b[(b["unit"] == "plant") & (b["day_ahead"] == 1)].set_index("time_local").sort_index()
    w = slice("2026-01-08", "2026-01-22")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(a.loc[w].index, a.loc[w, "p10"], a.loc[w, "p90"], alpha=0.2, color="#1f77b4", label="агент P10–P90")
    ax.plot(y.loc[w], color="black", lw=1.6, label="факт SCADA")
    ax.plot(a.loc[w, "p50"], color="#1f77b4", lw=1.3, label="агент P50")
    ax.plot(b.loc[w, "p50"], color="#ff7f0e", lw=1, ls="--", label="без агента P50")
    ax.set_ylim(0, 1.02); ax.set_ylabel("мощность, доля номинала")
    ax.set_title("Репетиция на январе 2026: прогноз на сутки вперёд против факта (модели не видели дек–янв)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "jan_rehearsal.png", dpi=140); plt.close(fig)


def fig_bars():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    items = [("Точность (MAE, holdout дек–янв)",
              ["Климатология", "NWP + кривая\nмощности", "Наша модель"], [0.321, 0.192, 0.175], "{:.3f}"),
             ("Стресс-тест: GFS выдаёт мусор (MAE)",
              ["Конвейер\nбез агента", "Агент\n(правила)", "Агент\n(LLM)"], [0.307, 0.161, 0.160], "{:.3f}"),
             ("Небалансы БРЭ, тыс. тг/МВт·мес",
              ["Климатология", "Агент,\nмедиана", "Агент,\nквантиль P31"], [6240, 3642, 2680], "{:,.0f}")]
    for ax, (title, labels, vals, fmt) in zip(axes, items):
        bars = ax.bar(labels, vals, color=["#bbbbbb", "#9ecae1", "#1f77b4"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, fmt.format(v).replace(",", " "), ha="center", va="bottom")
        ax.set_title(title); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "key_results.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    fig_feb(); fig_jan(); fig_bars()
    print("saved to", OUT)
