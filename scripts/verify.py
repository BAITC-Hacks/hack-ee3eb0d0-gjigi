"""Reproducibility check for the jury: re-run the whole February process and compare.

Runs the agent in deterministic rule mode (no LLM, no randomness) over all 28
issues - fetch weather, forecast, validate, update, finalize - and compares the
result with the committed reference docs/reference/forecast_feb2026_rules.csv.
(The LLM agent's output in outputs/ may differ slightly run to run: LLM decisions
are not bit-for-bit deterministic.)

Usage: python scripts/verify.py            # exit code 0 = reproduced
       python scripts/verify.py --update   # regenerate the reference
"""

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd

from hackalem.config import ROOT, load_config

cfg = load_config()
REF = cfg["paths"]["outputs_dir"] / "reference" / "forecast_feb2026_rules.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("run_agent", ROOT / "scripts" / "run_agent.py")
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)
    from hackalem.forecast import load_models

    with tempfile.TemporaryDirectory() as tmp:
        issues = pd.date_range(cfg["forecast"]["first_issue"], cfg["forecast"]["last_issue"])
        fc, _ = ra.run(issues, load_models(), "rules", Path(tmp))
        new_path = Path(tmp) / "forecast.csv"
        ra.write_submission(fc, new_path)
        new = pd.read_csv(new_path)
    if args.update:
        REF.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(REF, index=False)
        print(f"reference written: {REF}")
        return 0
    ref = pd.read_csv(REF)
    same_keys = new[["issue_date", "issue_time", "datetime"]].equals(ref[["issue_date", "issue_time", "datetime"]])
    diff = (new["power_forecast"] - ref["power_forecast"]).abs().max()
    ok = same_keys and diff < 1e-3
    print(f"rows {len(new)} vs {len(ref)}, same issues/hours: {same_keys}, max |diff| = {diff:.5f}")
    print("REPRODUCED" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
