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

    s = ForecastSession("2026-02-11", models=load_models(), store=load_store())
    agent = RuleAgent(s)
    agent.run()
    tools = [st["tool"] for st in agent.log]
    assert tools[0] == "check_weather_inputs" and tools[-1] == "finalize"
    assert "validate_forecast" in tools
    assert s.final is not None and len(s.final[s.final["unit"] == "plant"]) == 48
    # any re-run still only uses runs published before the final issue time
    w = s.result.weather
    assert (w.loc[w["is_target"], "available_at"] <= s.as_of_utc).all()
    assert s.as_of_utc.date() == s.issue_date.date()  # update stays on the issue day
