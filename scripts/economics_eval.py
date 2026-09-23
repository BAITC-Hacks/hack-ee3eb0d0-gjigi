"""Economic value of the forecast on the January 2026 rehearsal (actual SCADA known).

Day-ahead plan (D+1) submitted to the balancing market; imbalance cost with the
prices from config.yaml, per 1 MW of installed capacity over the month.

Usage: python scripts/economics_eval.py
Writes: docs/economics/ECONOMICS.md, economics.csv
"""

import pandas as pd

from hackalem.config import load_config
from hackalem.data.turbines import load_hourly
from hackalem.economics import critical_quantile, imbalance_cost_tg, plan_from_quantiles
from hackalem.evaluation import holdout_split
from hackalem.features.build import load_features
from hackalem.models.baselines import Climatology

cfg = load_config()
DOCS = cfg["paths"]["outputs_dir"]
OUT = DOCS / "economics"
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    f = pd.read_csv(path, parse_dates=["time_local", "issue_date"])
    return f[(f["unit"] == "plant") & (f["day_ahead"] == 1)].sort_values("time_local").reset_index(drop=True)


def main():
    q = critical_quantile()
    y = load_hourly()["power_plant"]
    agent = load(DOCS / "agent_rehearsal" / "agent_forecasts.csv")
    base = load(DOCS / "forecast" / "rehearsal_jan2026.csv")
    feats = load_features()
    train, _ = holdout_split(feats)
    clim = Climatology().fit(train)

    rows = []
    for name, fc in [("без агента", base), ("агент", agent)]:
        actual = y.reindex(fc["time_local"]).to_numpy()
        rows.append({"система": name, "план": "медиана P50",
                     "cost": imbalance_cost_tg(fc["p50"], actual)})
        rows.append({"система": name, "план": f"оптимальный квантиль P{q * 100:.0f}",
                     "cost": imbalance_cost_tg(plan_from_quantiles(fc, q), actual)})
    actual = y.reindex(agent["time_local"]).to_numpy()
    rows.insert(0, {"система": "климатология (без прогноза погоды)", "план": "среднее месяц×час",
                    "cost": imbalance_cost_tg(clim.predict(agent), actual)})
    t = pd.DataFrame(rows)
    ref = t.loc[0, "cost"]
    t["экономия к климатологии"] = 1 - t["cost"] / ref
    t.to_csv(OUT / "economics.csv", index=False)
    e = cfg["economics"]
    best = t["cost"].min()
    lines = ["# Экономический эффект прогноза (репетиция на январе 2026)", "",
             f"Суточный план подачи на БРЭ (D+1), цены из `config.yaml`: недовыработка "
             f"**{e['shortfall_cost_tg_per_kwh']} тг/кВт·ч**, перевыработка **{e['surplus_loss_tg_per_kwh']} тг/кВт·ч**, "
             f"поэтому оптимальный план — квантиль **q* = {q:.2f}** прогноза, а не медиана.",
             "Стоимость небалансов — на **1 МВт установленной мощности за месяц**, факт — SCADA.", "",
             "| Система | План | Небалансы, тыс. тг / МВт·мес | Экономия к климатологии |", "|---|---|---|---|"]
    for r in t.itertuples():
        lines.append(f"| {r[1]} | {r[2]} | {r.cost / 1000:,.0f}{' **←**' if r.cost == best else ''} | {r[4]:.0%} |")
    lines += ["", "Цены иллюстративные (порядок величин с балансирующего рынка Казахстана); "
              "с реальными тарифами меняются только числа в конфиге."]
    (OUT / "ECONOMICS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
