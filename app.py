"""Demo dashboard: streamlit run app.py

Tabs:
  1. Прогноз на февраль — freshest agent forecast for every hour + P10-P90
  2. Выпуск и решения агента — one issue: 48 h forecast, decision log, dispatcher note
  3. Репетиция: январь vs факт — agent vs no-agent vs actual SCADA
  4. Запуск агента — run the agent live for any issue date (rules or LLM)
"""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hackalem.config import ROOT, load_config

cfg = load_config()
OUT = cfg["paths"]["outputs_dir"]
st.set_page_config(page_title="ВЭС: агент прогноза", layout="wide")
st.title("Agentic AI: прогноз выработки ВЭС на 24–48 ч")
st.caption("Прогнозы погоды — только опубликованные к моменту выпуска. "
           "Мощность нормализована (доля номинала).")


def band_chart(df, x, title, actual=None, extra=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df[x], y=df["p10"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(31,119,180,0.2)", name="P10–P90"))
    fig.add_trace(go.Scatter(x=df[x], y=df["p50"], name="P50", line=dict(color="#1f77b4")))
    if extra is not None:
        fig.add_trace(go.Scatter(x=extra[0], y=extra[1], name=extra[2], line=dict(dash="dot", color="#ff7f0e")))
    if actual is not None:
        fig.add_trace(go.Scatter(x=actual[0], y=actual[1], name="факт SCADA", line=dict(color="black")))
    fig.update_layout(title=title, yaxis=dict(range=[0, 1.02], title="мощность"), height=420,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=["time_local", "issue_date"])


tab1, tab2, tab3, tab4 = st.tabs(["Прогноз на февраль", "Выпуск и решения агента",
                                  "Репетиция: январь vs факт", "Запуск агента"])

with tab1:
    allf = load_csv(OUT / "agent" / "agent_forecasts.csv")
    unit = st.radio("Объект", ["plant", "T1", "T2"], horizontal=True)
    d = (allf[allf["unit"] == unit].sort_values("issue_date")
         .groupby("time_local", as_index=False).last())
    d = d[d["time_local"].between(cfg["periods"]["test_start"], cfg["periods"]["test_end"])]
    st.plotly_chart(band_chart(d, "time_local", f"Февраль 2026, {unit}: самый свежий прогноз агента"),
                    use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Часов в прогнозе", len(d))
    c2.metric("Средняя ожидаемая мощность", f"{d['mean'].mean():.0%}")
    c3.metric("Средняя ширина P10–P90", f"{(d['p90'] - d['p10']).mean():.2f}")
    sub_path = cfg["paths"]["submission_dir"] / "forecast_feb2026.csv"
    st.download_button("Скачать файл сдачи (forecast_feb2026.csv)", sub_path.read_bytes(), sub_path.name)

with tab2:
    fc = load_csv(OUT / "agent" / "agent_forecasts.csv")
    issue = st.selectbox("Дата выпуска", sorted(fc["issue_date"].dt.date.unique()), index=10)
    f = fc[(fc["issue_date"].dt.date == issue) & (fc["unit"] == "plant")]
    log = json.loads((OUT / "agent" / "logs" / f"{issue}.json").read_text(encoding="utf-8"))
    st.plotly_chart(band_chart(f, "time_local", f"Выпуск {issue}: станция, D+1 и D+2"), use_container_width=True)
    st.info(f"**Сводка для диспетчера** ({log['mode']}, итоговый выпуск {log['final_issue_time_local'][:16]}): "
            f"{log['dispatcher_note']}")
    st.subheader("Журнал решений")
    st.dataframe(pd.DataFrame([{"шаг": s["step"], "инструмент": s["tool"],
                                "аргументы": json.dumps(s["args"], ensure_ascii=False),
                                "почему": s["reason"]} for s in log["steps"]]),
                 use_container_width=True, hide_index=True)
    with st.expander("Полные результаты инструментов (JSON)"):
        st.json(log["steps"])

with tab3:
    from hackalem.data.turbines import load_hourly
    y = load_hourly()["power_plant"]
    ag = load_csv(OUT / "agent_rehearsal" / "agent_forecasts.csv")
    base = load_csv(OUT / "forecast" / "rehearsal_jan2026.csv")
    day = st.radio("Горизонт", [1, 2], format_func=lambda d: f"D+{d}", horizontal=True)
    a = ag[(ag["unit"] == "plant") & (ag["day_ahead"] == day)].sort_values("time_local")
    b = base[(base["unit"] == "plant") & (base["day_ahead"] == day)].sort_values("time_local")
    yy = y.reindex(a["time_local"])
    st.plotly_chart(band_chart(a, "time_local", f"Январь 2026, D+{day}: агент (LLM) против факта",
                               actual=(a["time_local"], yy.values),
                               extra=(b["time_local"], b["p50"], "без агента, P50")), use_container_width=True)
    ok = yy.notna().values
    mae_a = (a["p50"].values[ok] - yy.values[ok]).__abs__().mean()
    yb = y.reindex(b["time_local"]).values
    okb = ~pd.isna(yb)
    mae_b = abs(b["p50"].values[okb] - yb[okb]).mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE агент (LLM)", f"{mae_a:.3f}", f"{(mae_a / mae_b - 1):+.0%} к без агента", delta_color="inverse")
    c2.metric("MAE без агента", f"{mae_b:.3f}")
    c3.metric("Покрытие P10–P90", f"{((yy.values >= a['p10'].values) & (yy.values <= a['p90'].values))[ok].mean():.0%}")
    st.caption("Модели для репетиции обучены только на данных до 01.12.2025.")

with tab4:
    st.write("Агент выполняет полный цикл для выбранной даты выпуска: проверка погоды → прогноз → валидация → "
             "сравнение со вчерашним → поиск свежих прогонов → пересчёт → сводка.")
    c1, c2 = st.columns(2)
    issue = c1.date_input("Дата выпуска", pd.Timestamp("2026-02-15"),
                          min_value=pd.Timestamp("2024-03-10"), max_value=pd.Timestamp("2026-02-27"))
    mode = c2.radio("Режим", ["auto", "rules", "llm"], horizontal=True,
                    help="auto = LLM, если в .env есть OPENAI_API_KEY")
    if st.button("Запустить агента", type="primary"):
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        from hackalem.agent.agent import make_agent
        from hackalem.agent.tools import ForecastSession
        from hackalem.forecast import load_models
        from hackalem.weather.store import load_store
        with st.spinner("Агент работает…"):
            s = ForecastSession(pd.Timestamp(issue), models=load_models(), store=load_store())
            agent = make_agent(mode, s)
            agent.run()
        p = s.final[s.final["unit"] == "plant"]
        st.success(f"Готово ({agent.mode}): {len(agent.log)} шагов, итоговый выпуск "
                   f"{(s.as_of_utc + pd.Timedelta(hours=cfg['scada_utc_offset_h'])):%Y-%m-%d %H:%M}")
        st.plotly_chart(band_chart(p, "time_local", f"Выпуск {issue}"), use_container_width=True)
        st.info(s.notes[-1] if s.notes else "")
        st.dataframe(pd.DataFrame([{"шаг": x["step"], "инструмент": x["tool"], "почему": x["reason"]}
                                   for x in agent.log]), use_container_width=True, hide_index=True)
