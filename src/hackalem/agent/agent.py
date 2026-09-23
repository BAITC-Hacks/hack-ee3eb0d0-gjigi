"""Forecasting agent: observe -> decide -> act -> verify, for one issue.

Two interchangeable brains over the SAME tools (agent/tools.py):
  * LLMAgent  - OpenAI chat model with function calling decides which tool to
                call next and writes the dispatcher note.
  * RuleAgent - deterministic policy, used when no OPENAI_API_KEY is set or the
                LLM call fails, so the pipeline always completes.
Every step (tool, arguments, result, reasoning) is recorded in a decision log.
"""

import json
import os
import time

import pandas as pd

from hackalem.agent import tools as T
from hackalem.config import load_config

SYSTEM_PROMPT = """You are the forecasting agent of a wind farm (2 turbines, Kazakhstan).
Every day at 10:00 plant time you issue an hourly power forecast (normalised 0..1,
P10/P50/P90) for the next two local days (D+1, D+2), using ONLY weather forecasts
already published at issue time. You operate through tools; the ML model and all
checks run in code, you decide what to do with the facts they return.

Workflow (adapt when facts require it):
1. check_weather_inputs - coverage, freshness and agreement of the 4 NWP sources.
   If a source is an outlier (listed in outlier_candidates) or has missing hours,
   consider excluding it in run_forecast. Never exclude ECMWF 9 km unless it is broken.
2. run_forecast.
3. validate_forecast. If there are issues, fix them (e.g. re-run without a bad source).
4. compare_with_previous - large revisions vs yesterday's forecast deserve an explanation.
5. The interval is already calibrated. Only if the ensemble spread is very high (>= 24 hours above
   threshold) widen_intervals by 1.1-1.25.
6. check_for_new_runs; if a newer NWP run is published within the update window,
   call rerun_with_update to re-forecast with it and report what changed.
7. finalize with a short dispatcher note in Russian (3-5 sentences): expected output
   for D+1 and D+2, windy/calm periods, uncertainty, what you changed and why.
Before EVERY tool call write one short sentence (in Russian) explaining why you call it,
based on the facts so far - this reasoning is logged for auditors.
Be efficient: do not call the same tool twice without a reason. Always finish with finalize."""

TOOL_SPECS = [
    {"name": "check_weather_inputs", "description": "Coverage, freshness and agreement of NWP sources available at the current issue time.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "run_forecast", "description": "Run the ML forecast (plant, T1, T2). Optionally exclude NWP sources.",
     "parameters": {"type": "object", "properties": {"exclude_models": {"type": "array", "items": {"type": "string"},
                    "description": "NWP models to exclude, e.g. ['gfs_seamless']"}}}},
    {"name": "validate_forecast", "description": "Sanity checks of the current forecast.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "compare_with_previous", "description": "Revision vs the previous day's forecast for the overlapping day.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "widen_intervals", "description": "Widen the P10-P90 interval by a factor in [1, 1.5].",
     "parameters": {"type": "object", "properties": {"factor": {"type": "number"}}, "required": ["factor"]}},
    {"name": "check_for_new_runs", "description": "Check whether newer NWP runs get published within the update window.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "rerun_with_update", "description": "Re-forecast at the moment newer NWP runs are published.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "finalize", "description": "Accept the current forecast and attach the dispatcher note (Russian).",
     "parameters": {"type": "object", "properties": {"dispatcher_note": {"type": "string"}},
                    "required": ["dispatcher_note"]}},
]


# every tool requires a short justification that goes into the decision log
for _t in TOOL_SPECS:
    _t["parameters"]["properties"]["reason"] = {
        "type": "string", "description": "One sentence in Russian: why this call, based on the facts so far."}
    _t["parameters"]["required"] = sorted(set(_t["parameters"].get("required", [])) | {"reason"})


class BaseAgent:
    mode = "base"

    def __init__(self, session: T.ForecastSession, prev_forecast: pd.DataFrame | None = None):
        self.s = session
        self.prev = prev_forecast
        self.cfg = load_config()["agent"]
        self.log: list[dict] = []

    def call(self, name: str, args: dict | None = None, reason: str = "") -> dict:
        args = args or {}
        t0 = time.time()
        try:
            if name == "compare_with_previous":
                res = T.compare_with_previous(self.s, self.prev)
            elif name == "check_for_new_runs":
                res = T.check_for_new_runs(self.s, self.cfg["update_window_h"])
            elif name == "rerun_with_update":
                res = T.rerun_with_update(self.s, self.cfg["update_window_h"])
            else:
                res = getattr(T, name)(self.s, **args)
        except Exception as e:  # noqa: BLE001 - tool errors are facts for the agent
            res = {"error": f"{type(e).__name__}: {e}"}
        self.log.append({"step": len(self.log) + 1, "tool": name, "args": args, "reason": reason,
                         "result": res, "seconds": round(time.time() - t0, 2)})
        return res

    def ensure_final(self):
        """Safety net: never leave an issue without a forecast."""
        if self.s.final is None:
            if self.s.result is None:
                self.call("run_forecast", reason="safety net: no forecast produced")
            self.call("finalize", {"dispatcher_note": RuleAgent.note(self.s, [])},
                      reason="safety net: agent did not finalize")


class RuleAgent(BaseAgent):
    mode = "rules"

    def run(self):
        chk = self.call("check_weather_inputs", reason="observe inputs")
        exclude = [m for m in chk["outlier_candidates"] if m != T.MAIN_MODEL]
        exclude += [m for m, v in chk["models"].items() if v["hours"] < 48 and m != T.MAIN_MODEL]
        exclude = sorted(set(exclude))
        self.call("run_forecast", {"exclude_models": exclude},
                  reason=f"exclude {exclude}" if exclude else "all sources healthy")
        val = self.call("validate_forecast", reason="verify")
        if not val.get("ok", True) and exclude:
            self.call("run_forecast", {"exclude_models": []}, reason="validation failed, retry with all sources")
            self.call("validate_forecast", reason="verify retry")
        actions = []
        hi = chk["ensemble_spread_ms"]["hours_above_threshold"]
        if hi >= 24:   # the CQR interval is already calibrated; widen only for strong disagreement
            f = 1.0 + min(0.25, hi / 192)
            self.call("widen_intervals", {"factor": round(f, 2)}, reason=f"{hi} uncertain hours")
            actions.append(f"интервалы расширены ×{f:.2f} из-за расхождения моделей ({hi} ч)")
        rev = self.call("compare_with_previous", reason="consistency with yesterday")
        upd = self.call("check_for_new_runs", reason="look for fresher NWP")
        if upd.get("new_runs"):
            r = self.call("rerun_with_update", reason="fresher run published")
            runs = sorted(set(upd["new_runs"].values()))
            actions.append(f"пересчёт после публикации прогонов {', '.join(runs)} "
                           f"(итоговый выпуск {r.get('new_as_of_local', '')[11:16]})")
            rv = r.get("revision_vs_before_update", {})
            if rv.get("available"):
                actions.append(f"средняя правка P50 {rv['mean_abs_revision']:.2f}")
            self.call("validate_forecast", reason="verify after update")
        if exclude:
            actions.append(f"исключены источники: {', '.join(exclude)}")
        if rev.get("large_revision"):
            actions.append(f"крупная правка к вчерашнему прогнозу ({rev['mean_abs_revision']:.2f})")
        self.call("finalize", {"dispatcher_note": self.note(self.s, actions)}, reason="done")
        self.ensure_final()

    @staticmethod
    def note(s: T.ForecastSession, actions: list[str]) -> str:
        summ = T.summarize(s)["plant"]
        parts = []
        for k, v in summ.items():
            level = "высокая" if v["mean_p50"] > 0.55 else "средняя" if v["mean_p50"] > 0.25 else "низкая"
            parts.append(f"{k} ({v['date']}): {level} выработка, средняя {v['mean_p50']:.0%} от номинала, "
                         f"часов >80%: {v['hours_above_0.8']}, штиль (<5%): {v['hours_below_0.05']} ч, "
                         f"ширина интервала {v['mean_interval_width']:.2f}.")
        if actions:
            parts.append("Действия агента: " + "; ".join(actions) + ".")
        return " ".join(parts)


class LLMAgent(BaseAgent):
    mode = "llm"

    def __init__(self, *a, client=None, model: str | None = None, **kw):
        super().__init__(*a, **kw)
        from openai import OpenAI
        self.client = client or OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL") or self.cfg["llm_model"]

    def run(self):
        tools = [{"type": "function", "function": t} for t in TOOL_SPECS]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Issue the forecast for issue day {self.s.issue_date.date()}. "
                                            f"Previous day's forecast available: {self.prev is not None}."}]
        for _ in range(self.cfg["max_steps"]):
            resp = self.client.chat.completions.create(model=self.model, messages=msgs, tools=tools,
                                                       temperature=0)
            msg = resp.choices[0].message
            msgs.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                if self.s.final is not None:
                    break
                msgs.append({"role": "user", "content": "Finish by calling finalize."})
                continue
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                reason = args.pop("reason", "") or (msg.content or "").strip()
                res = self.call(tc.function.name, args, reason=reason)
                msgs.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(res, ensure_ascii=False, default=str)})
            if self.s.final is not None:
                break
        self.ensure_final()


def make_agent(mode: str, session, prev_forecast=None) -> BaseAgent:
    """mode: 'llm', 'rules' or 'auto' (LLM if OPENAI_API_KEY is set)."""
    if mode == "auto":
        mode = "llm" if os.getenv("OPENAI_API_KEY") else "rules"
    return (LLMAgent if mode == "llm" else RuleAgent)(session, prev_forecast)
