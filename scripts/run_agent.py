"""Stage 5: run the forecasting agent issue by issue.

Usage:
    python scripts/run_agent.py                         # Feb 2026 test period, mode auto
    python scripts/run_agent.py --mode rules            # no LLM
    python scripts/run_agent.py --mode llm --issues 2026-02-10:2026-02-12
    python scripts/run_agent.py --rehearsal             # Jan 2026 with models trained before Dec,
                                                        # scored vs actual SCADA

Mode auto = LLM (OpenAI) if OPENAI_API_KEY is set (e.g. in .env), otherwise rules.
Writes docs/agent[_rehearsal]/ (reports) and outputs/ (deliverable):
    logs/<issue>.json            full decision log (tool, args, reasoning, result)
    agent_forecasts.csv          final forecast of every issue (plant, T1, T2)
    outputs/forecast_feb2026.csv THE DELIVERABLE: every issue x 48 h, plant power forecast
    AGENT_REPORT.md              human-readable digest of decisions and notes
"""

import argparse
import json
import os

import pandas as pd

from hackalem.agent.agent import make_agent
from hackalem.agent.tools import ForecastSession
from hackalem.config import ROOT, load_config
from hackalem.evaluation import metrics
from hackalem.forecast import load_models
from hackalem.models.gbm import interval_metrics
from hackalem.timeutils import utc_to_local
from hackalem.weather.store import load_store

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

cfg = load_config()


def run(issues, models, mode, out_dir):
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    store = load_store()
    prev, finals, digest = None, [], []
    for d in issues:
        s = ForecastSession(d, models=models, store=store)
        agent = make_agent(mode, s, prev)
        try:
            agent.run()
        except Exception as e:  # noqa: BLE001 - LLM/network failure -> rules fallback
            print(f"  {d.date()}: {agent.mode} agent failed ({e}); falling back to rules")
            s = ForecastSession(d, models=models, store=store)
            agent = make_agent("rules", s, prev)
            agent.run()
        with open(out_dir / "logs" / f"{d.date()}.json", "w", encoding="utf-8") as f:
            json.dump({"issue_date": str(d.date()), "mode": agent.mode,
                       "final_issue_time_local": str(utc_to_local(s.as_of_utc)),
                       "excluded_models": s.excluded, "widen_factor": s.widen_factor,
                       "dispatcher_note": s.notes[-1] if s.notes else "",
                       "steps": agent.log}, f, indent=2, ensure_ascii=False, default=str)
        finals.append(s.final)
        prev = s.final
        tools_used = [st["tool"] for st in agent.log]
        digest.append({"issue": str(d.date()), "mode": agent.mode, "steps": len(tools_used),
                       "final_time_local": str(utc_to_local(s.as_of_utc)),
                       "excluded": ",".join(s.excluded), "widen": s.widen_factor,
                       "updated": "rerun_with_update" in tools_used,
                       "note": s.notes[-1] if s.notes else ""})
        print(f"  {d.date()} [{agent.mode}] steps={len(tools_used)} final@{utc_to_local(s.as_of_utc):%m-%d %H:%M}"
              f" excluded={s.excluded or '-'} widen={s.widen_factor}")
    fc = pd.concat(finals, ignore_index=True)
    fc.round(4).to_csv(out_dir / "agent_forecasts.csv", index=False)
    return fc, pd.DataFrame(digest)


def write_submission(fc: pd.DataFrame):
    """The deliverable required by the task: for every daily issue, the hourly plant
    forecast for the next 24-48 h (local days D+1, D+2). One value per hour (P50).
    """
    p = fc[fc["unit"] == "plant"].copy()
    issue_local = utc_to_local(pd.to_datetime(p["final_issue_time_utc"]))
    out = pd.DataFrame({
        "issue_date": pd.to_datetime(p["issue_date"]).dt.date,
        "issue_time": issue_local.dt.strftime("%Y-%m-%d %H:%M"),
        "datetime": pd.to_datetime(p["time_local"]).dt.strftime("%Y-%m-%d %H:%M"),
        "horizon_h": ((pd.to_datetime(p["time_local"]) - issue_local) / pd.Timedelta("1h")).round().astype(int),
        "power_forecast": p["p50"].round(4),
    }).sort_values(["issue_date", "datetime"])
    d = cfg["paths"]["submission_dir"]
    d.mkdir(parents=True, exist_ok=True)
    path = d / "forecast_feb2026.csv"
    out.to_csv(path, index=False)
    assert len(out) == 48 * out["issue_date"].nunique() and out["power_forecast"].between(0, 1).all()
    return path


def write_report(out_dir, digest, title, scores=None):
    lines = [f"# {title}", "",
             f"Выпусков: {len(digest)}. Режим: {', '.join(sorted(digest['mode'].unique()))}. "
             f"Пересчётов на свежих прогонах: {int(digest['updated'].sum())}. "
             f"Исключений источников: {int((digest['excluded'] != '').sum())}. "
             f"Расширений интервала: {int((digest['widen'] > 1).sum())}.", ""]
    if scores:
        lines += ["## Качество (факт SCADA)", "", "| | MAE P50 | RMSE mean | покрытие 80% |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['mae_p50']:.4f} | {v['rmse_mean']:.4f} | {v['coverage_80']:.1%} |"
                  for k, v in scores.items()]
        lines += [""]
    lines += ["## Решения по выпускам", "", "| Выпуск | Итоговое время | Исключено | Интервал × | Пересчёт |",
              "|---|---|---|---|---|"]
    lines += [f"| {r.issue} | {r.final_time_local[5:16]} | {r.excluded or '—'} | {r.widen:.2f} | "
              f"{'да' if r.updated else 'нет'} |" for r in digest.itertuples()]
    lines += ["", "## Сводки для диспетчера", ""]
    lines += [f"**{r.issue}.** {r.note}\n" for r in digest.itertuples()]
    (out_dir / "AGENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto", choices=["auto", "llm", "rules"])
    ap.add_argument("--issues", default=None, help="start:end issue dates")
    ap.add_argument("--rehearsal", action="store_true")
    args = ap.parse_args()
    if args.mode == "llm" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (put it in .env)")

    if args.rehearsal:
        from hackalem.features.build import load_features
        from hackalem.models.gbm import QuantileGBM
        feats = load_features()
        train = feats[feats["time_local"] < cfg["validation"]["holdout_start"]]
        models = {u: QuantileGBM(target=t).fit(train) for u, t in
                  [("plant", "y_plant"), ("T1", "y_T1"), ("T2", "y_T2")]}
        default = ("2025-12-31", "2026-01-30")
        out_dir = cfg["paths"]["outputs_dir"] / "agent_rehearsal"
    else:
        models = load_models()
        default = (cfg["forecast"]["first_issue"], cfg["forecast"]["last_issue"])
        out_dir = cfg["paths"]["outputs_dir"] / "agent"
    start, end = args.issues.split(":") if args.issues else default
    fc, digest = run(pd.date_range(start, end), models, args.mode, out_dir)

    scores = None
    if args.rehearsal:
        from hackalem.data.turbines import load_hourly
        y = load_hourly()["power_plant"]
        p = fc[fc["unit"] == "plant"].join(y.rename("y"), on="time_local")
        scores = {}
        for d in [1, 2]:
            g = p[p["day_ahead"] == d]
            scores[f"D+{d}"] = {"mae_p50": metrics(g["y"], g["p50"])["mae"],
                                "rmse_mean": metrics(g["y"], g["mean"])["rmse"],
                                **interval_metrics(g["y"].to_numpy(), g["p10"].to_numpy(), g["p90"].to_numpy())}
        print(json.dumps(scores, indent=2))
    else:
        path = write_submission(fc)
        print(f"Submission: {path}")
    write_report(out_dir, digest, "Отчёт агента" + (" — репетиция на январе 2026" if args.rehearsal else " — февраль 2026"),
                 scores)
    digest.to_csv(out_dir / "decisions.csv", index=False)


if __name__ == "__main__":
    main()
