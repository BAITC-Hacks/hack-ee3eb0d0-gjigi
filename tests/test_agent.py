import pytest

from hackalem.config import load_config

pytestmark = pytest.mark.skipif(
    not (load_config()["paths"]["models_dir"] / "gbm_y_plant.pkl").exists(),
    reason="final models not trained")


def test_rule_agent_full_cycle_without_llm():
    from hackalem.agent.agent import RuleAgent
    from hackalem.agent.tools import ForecastSession
    from hackalem.forecast import load_models
    from hackalem.weather.store import load_store

    s = ForecastSession("2026-02-11", models=load_models(), archive=load_store())
    agent = RuleAgent(s)
    agent.run()
    tools = [st["tool"] for st in agent.log]
    assert tools[:2] == ["fetch_weather", "check_weather_inputs"] and tools[-1] == "finalize"
    assert "validate_forecast" in tools
    assert s.final is not None and len(s.final[s.final["unit"] == "plant"]) == 48
    # any re-run still only uses runs published before the final issue time
    w = s.result.weather
    assert (w.loc[w["is_target"], "available_at"] <= s.as_of_utc).all()
    assert s.as_of_utc.date() == s.issue_date.date()  # update stays on the issue day


def test_fetch_weather_offline_falls_back_to_repo_archive(tmp_path, monkeypatch):
    """Without network/cache the agent still gets exactly the archived inputs."""
    from hackalem.config import load_config
    from hackalem.timeutils import issue_time_utc
    from hackalem.weather.fetch import fetch_for_issue
    from hackalem.weather.issued import issued_forecasts
    from hackalem.weather.store import load_store

    cfg = load_config()
    monkeypatch.setitem(cfg["paths"], "http_cache_dir", tmp_path)   # empty cache
    archive = load_store()
    rows, report = fetch_for_issue("2026-02-11", issue_time_utc("2026-02-11"), archive=archive, offline=True)
    assert all("archive" in v["source"] for v in report.values())
    a = issued_forecasts(["2026-02-11"], archive)
    b = issued_forecasts(["2026-02-11"], rows)
    assert len(a) == len(b) and set(a["init_time"]) == set(b["init_time"])


@pytest.mark.parametrize("missing", ["ecmwf_ifs", "gfs_seamless", "icon_seamless", "ecmwf_ifs025"])
def test_forecast_survives_any_missing_source(missing):
    from hackalem.forecast import load_models, run_issue
    from hackalem.weather.store import load_store

    st = load_store()
    fc = run_issue("2026-01-15", models=load_models(), store=st[st["model"] != missing]).forecast
    assert fc[["p10", "p50", "p90"]].notna().all().all() and len(fc) == 3 * 48


def test_guardrail_keeps_healthy_ecmwf():
    """Even if the LLM asks to drop the healthy main source, code refuses."""
    from hackalem.agent import tools as T
    from hackalem.forecast import load_models
    from hackalem.weather.store import load_store

    s = T.ForecastSession("2026-01-10", models=load_models(), archive=load_store())
    T.fetch_weather(s)
    res = T.run_forecast(s, exclude_models=["gfs_seamless", "ecmwf_ifs"])
    assert "ecmwf_ifs" not in s.excluded and "gfs_seamless" in s.excluded
    assert "guardrail" in res
