"""Live mode: the agent issues a real forecast for the next two local days, right now.

Same agent, same tools, same models as the backtest - only `as_of` is the current
time, so fetch_weather downloads the freshest published NWP runs from Open-Meteo.

Usage: python scripts/forecast_live.py [--mode auto|rules|llm]
Writes: docs/live/forecast_<issue date>.csv and the decision log next to it.
"""

import argparse
import json

import pandas as pd

from hackalem.agent.agent import make_agent
from hackalem.agent.tools import ForecastSession
from hackalem.config import ROOT, load_config
from hackalem.forecast import load_models
from hackalem.timeutils import utc_to_local

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto", choices=["auto", "rules", "llm"])
    args = ap.parse_args()
    cfg = load_config()
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    issue_date = utc_to_local(now_utc).normalize()

    s = ForecastSession(issue_date, models=load_models(), as_of_utc=now_utc)
    agent = make_agent(args.mode, s)
    # live: never look past "now" - only runs actually published by now can be used
    agent.cfg = {**agent.cfg, "update_window_h": 0}
    agent.run()

    out = cfg["paths"]["outputs_dir"] / "live"
    out.mkdir(parents=True, exist_ok=True)
    p = s.final[s.final["unit"] == "plant"][["time_local", "p10", "p50", "p90"]]
    p.round(3).to_csv(out / f"forecast_{issue_date.date()}.csv", index=False)
    with open(out / f"log_{issue_date.date()}.json", "w", encoding="utf-8") as f:
        json.dump({"issued_local": str(utc_to_local(s.as_of_utc)), "mode": agent.mode,
                   "note": s.notes[-1] if s.notes else "", "steps": agent.log},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f"Issued {utc_to_local(s.as_of_utc):%Y-%m-%d %H:%M} (plant time), mode={agent.mode}")
    for st in agent.log:
        print(f"  {st['step']}. {st['tool']:22s} {st['reason'][:90]}")
    print("\n" + (s.notes[-1] if s.notes else ""))
    daily = p.assign(day=p["time_local"].dt.date).groupby("day")["p50"].agg(["mean", "max"])
    print("\nDaily plant forecast (P50, share of capacity):\n", daily.round(2).to_string())


if __name__ == "__main__":
    main()
