"""Download archived NWP forecasts and build data/weather/forecast_store.parquet.

Usage:
    python scripts/build_weather_archive.py              # download (resumable, cached)
    python scripts/build_weather_archive.py --offline    # rebuild from HTTP cache only
    python scripts/build_weather_archive.py --start 2026-01-20 --end 2026-03-02

Every HTTP response is cached in data/cache/http, so re-runs are free and an
interrupted download continues where it stopped.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from hackalem.config import load_config
from hackalem.weather.openmeteo import OpenMeteoClient
from hackalem.weather.sources import fetch_previous_runs, fetch_single_run, month_chunks
from hackalem.weather.store import merge_into_store, store_path


def main():
    cfg = load_config()
    w = cfg["weather"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=w["archive_start"])
    ap.add_argument("--end", default=w["archive_end"])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    client = OpenMeteoClient(cfg["paths"]["http_cache_dir"], w["request_interval_s"])
    frames = []

    # 1. Exact runs (Single Runs API)
    inits = [t for t in pd.date_range(args.start, f"{args.end} 23:00", freq="6h")
             if t.hour in w["single_run_hours"]]
    jobs = [(m, t) for m in w["single_run_models"] for t in inits]
    missing, done = [], 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_single_run, client, cfg, m, t, args.offline): (m, t)
                for m, t in jobs}
        for f in as_completed(futs):
            m, t = futs[f]
            try:
                df = f.result()
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"  ! {m} {t}: {e}", file=sys.stderr)
                df = None
            if df is None:
                missing.append(f"{m} {t:%Y-%m-%d %H}z")
            else:
                frames.append(df)
            done += 1
            if done % 200 == 0:
                print(f"  single runs: {done}/{len(jobs)}", flush=True)
    print(f"Single runs: {len(jobs) - len(missing)}/{len(jobs)} available")
    if missing:
        print(f"  missing runs ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    # 2. Multi-model previous runs
    for m in w["prev_run_models"]:
        for s, e in month_chunks(args.start, args.end, months=3):
            df = fetch_previous_runs(client, cfg, m, s, e, offline=args.offline)
            frames.append(df)
        print(f"Previous runs {m}: done")

    new = pd.concat(frames, ignore_index=True)
    path = store_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(path) if path.exists() else None
    store = merge_into_store(existing, new)
    store.to_parquet(path, index=False)
    print(f"Store: {len(store):,} rows -> {path}")
    print(store.groupby("model").agg(
        runs=("init_time", "nunique"), first=("valid_time", "min"), last=("valid_time", "max")))


if __name__ == "__main__":
    main()
