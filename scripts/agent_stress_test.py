"""Fault-injection test: does the agent notice broken inputs that a plain pipeline passes through?

Scenarios (applied to the weather the agent has just fetched):
  none          clean inputs
  gfs_garbage   GFS 100 m wind is corrupted (x2.5 + 4 m/s) - a broken source
  icon_outage   ICON returns nothing - a missing source
Compared on January 2026 issues (models trained before 2025-12-01) against actual SCADA:
  pipeline  - run_issue on the same corrupted inputs, no agent
  rules     - rule agent
  llm       - OpenAI agent (if OPENAI_API_KEY is set)

Usage: python scripts/agent_stress_test.py [--issues 2026-01-10:2026-01-19] [--no-llm]
Writes: docs/agent_stress/stress_results.csv, STRESS.md
"""

import argparse
import os

import numpy as np
import pandas as pd

from hackalem.agent import tools as T
from hackalem.agent.agent import make_agent
from hackalem.agent.tools import ForecastSession
from hackalem.config import ROOT, load_config
from hackalem.data.turbines import load_hourly
from hackalem.features.build import load_features
from hackalem.forecast import run_issue
from hackalem.models.gbm import QuantileGBM
from hackalem.weather.fetch import fetch_for_issue
from hackalem.weather.store import load_store

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

cfg = load_config()
OUT = cfg["paths"]["outputs_dir"] / "agent_stress"
OUT.mkdir(parents=True, exist_ok=True)


def corrupt(rows: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = rows.copy()
    if scenario == "gfs_garbage":
        m = rows["model"] == "gfs_seamless"
        rows.loc[m, "wind_speed_100m"] = rows.loc[m, "wind_speed_100m"] * 2.5 + 4
    elif scenario == "icon_outage":
        rows = rows[rows["model"] != "icon_seamless"]
    return rows


def patched_fetch(scenario):
    def f(*a, **kw):
        rows, rep = fetch_for_issue(*a, **kw)     # always the original fetch
        return corrupt(rows, scenario), rep
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issues", default="2026-01-10:2026-01-19")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    start, end = args.issues.split(":")
    issues = pd.date_range(start, end)

    feats = load_features()
    train = feats[feats["time_local"] < cfg["validation"]["holdout_start"]]
    models = {"plant": QuantileGBM(target="y_plant").fit(train)}
    archive = load_store()
    y = load_hourly()["power_plant"]
    modes = ["rules"] + ([] if args.no_llm or not os.getenv("OPENAI_API_KEY") else ["llm"])

    rows = []
    for scenario in ["none", "gfs_garbage", "icon_outage"]:
        T.fetch_for_issue = patched_fetch(scenario)
        for d in issues:
            fetched, _ = fetch_for_issue(d, pd.Timestamp(d) + pd.Timedelta(hours=4), archive=archive)
            store = corrupt(fetched, scenario)
            base = run_issue(d, models=models, store=store).forecast
            res = {"pipeline": (base, [])}
            for mode in modes:
                s = ForecastSession(d, models=models, archive=archive)
                ag = make_agent(mode, s)
                ag.run()
                res[mode] = (s.final, s.excluded)
            for name, (fc, excl) in res.items():
                p = fc[fc["unit"] == "plant"].set_index("time_local")["p50"]
                yy = y.reindex(p.index)
                ok = yy.notna()
                rows.append({"scenario": scenario, "issue": d.date(), "system": name,
                             "mae": float((p[ok] - yy[ok]).abs().mean()), "excluded": ",".join(excl)})
            print(f"  {scenario} {d.date()} done", flush=True)
    T.fetch_for_issue = fetch_for_issue

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "stress_results.csv", index=False)
    tab = r.pivot_table(index="scenario", columns="system", values="mae", aggfunc="mean") \
        .reindex(["none", "gfs_garbage", "icon_outage"])
    tab = tab[[c for c in ["pipeline", "rules", "llm"] if c in tab.columns]]
    det = r[r["system"] != "pipeline"].groupby(["scenario", "system"])["excluded"] \
        .apply(lambda s: f"{(s != '').sum()}/{len(s)}").unstack()
    lines = ["# Стресс-тест агента: намеренно испорченные входные данные", "",
             f"Выпуски {start} … {end}, модели обучены до 01.12.2025, MAE P50 станции против факта SCADA.",
             "Конвейер — выпуск в 10:00 без проверок; агент получает те же испорченные данные "
             "(и, как обычно, может пересчитать прогноз после публикации свежего прогона).", "",
             "| Сценарий | " + " | ".join(f"MAE {c}" for c in tab.columns) + " |",
             "|---|" + "---|" * len(tab.columns)]
    names = {"none": "данные исправны", "gfs_garbage": "GFS выдаёт мусор (×2.5 + 4 м/с)",
             "icon_outage": "ICON недоступен"}
    for sc, row in tab.iterrows():
        lines.append(f"| {names[sc]} | " + " | ".join(f"{v:.3f}" for v in row.values) + " |")
    lines += ["", "Сколько выпусков агент исключил хотя бы один источник:", "",
              det.to_markdown() if hasattr(det, "to_markdown") else det.to_string()]
    (OUT / "STRESS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
