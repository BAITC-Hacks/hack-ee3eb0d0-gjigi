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
st.caption("Каждый день в 10:00 агент сам скачивает прогнозы погоды, считает почасовой прогноз выработки ВЭС "
           "на следующие двое суток, проверяет его и при выходе свежего прогноза погоды пересчитывает. "
           "Используются только прогнозы погоды, опубликованные к моменту выпуска.")

with st.expander("📖 Обозначения — что значит T1, P50, D+1 и т.д.", expanded=True):
    st.markdown("""
| Обозначение | Что значит |
|---|---|
| **Станция (plant)** | Вся ВЭС: средняя мощность двух турбин. Это основной прогноз |
| **T1, T2** | Турбина 1 и турбина 2 — две ветротурбины станции (стоят в ~350 м друг от друга). Для каждой — свой прогноз |
| **Мощность 0…1** | Нормализованная активная мощность — доля от номинальной: 0 — турбина стоит, 1 — работает на полную мощность, 0.4 — 40% |
| **P50** | Основной прогноз — самое вероятное значение (медиана) |
| **P10 – P90** | Интервал неопределённости: с вероятностью ~80% фактическая мощность будет внутри этой полосы |
| **Выпуск, день D** | День, когда агент делает прогноз (в 10:00 по времени станции) |
| **D+1 / D+2** | Прогноз на завтра / на послезавтра относительно дня выпуска (по 24 часа, всего 48) |
| **Пересчёт в 14:00** | В 14:00 публикуется свежий прогноз погоды ECMWF; если он заметно меняет ветер, агент пересчитывает прогноз |
| **MAE** | Средняя ошибка прогноза в долях номинала: 0.15 = в среднем ошибка 15% мощности. Меньше — лучше |
| **Факт SCADA** | Реально измеренная мощность турбин (данные организаторов) |
| **ECMWF, GFS, ICON** | Погодные модели (европейская, американская, немецкая), из которых агент берёт прогноз ветра |
| **Агент (LLM / правила)** | LLM — решения принимает нейросеть OpenAI; правила — тот же агент без нейросети, по заданным порогам |
""")


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


tab1, tab2, tab3, tab4 = st.tabs(["📈 Прогноз на февраль", "🤖 Выпуск и решения агента",
                                  "✅ Репетиция: январь против факта", "▶️ Запустить агента"])

with tab1:
    allf = load_csv(OUT / "agent" / "agent_forecasts.csv")
    UNIT_NAMES = {"plant": "Станция целиком (среднее двух турбин)", "T1": "Турбина 1 (T1)", "T2": "Турбина 2 (T2)"}
    unit = st.radio("Объект прогноза", list(UNIT_NAMES), format_func=UNIT_NAMES.get, horizontal=True)
    d = (allf[allf["unit"] == unit].sort_values("issue_date")
         .groupby("time_local", as_index=False).last())
    d = d[d["time_local"].between(cfg["periods"]["test_start"], cfg["periods"]["test_end"])]
    st.caption("Для каждого часа февраля показан самый свежий прогноз агента (выпуск накануне, D+1). "
               "Факта за февраль в данных нет — его проверяют организаторы; качество на реальном факте — во вкладке «Репетиция».")
    st.plotly_chart(band_chart(d, "time_local", f"Февраль 2026 — {UNIT_NAMES[unit]}"),
                    use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Часов в прогнозе", len(d), help="Все 672 часа февраля 2026")
    c2.metric("Средняя ожидаемая мощность", f"{d['mean'].mean():.0%}", help="Доля от номинальной мощности")
    c3.metric("Средняя ширина интервала P10–P90", f"{(d['p90'] - d['p10']).mean():.2f}",
              help="Чем шире — тем менее уверен прогноз")
    sub_path = cfg["paths"]["submission_dir"] / "forecast_feb2026.csv"
    st.download_button("Скачать файл сдачи (forecast_feb2026.csv)", sub_path.read_bytes(), sub_path.name)

with tab2:
    fc = load_csv(OUT / "agent" / "agent_forecasts.csv")
    st.caption("Выберите день выпуска: график — прогноз на следующие двое суток, ниже — что и почему делал агент.")
    issue = st.selectbox("Дата выпуска (день D)", sorted(fc["issue_date"].dt.date.unique()), index=10)
    f = fc[(fc["issue_date"].dt.date == issue) & (fc["unit"] == "plant")]
    log = json.loads((OUT / "agent" / "logs" / f"{issue}.json").read_text(encoding="utf-8"))
    st.plotly_chart(band_chart(f, "time_local", f"Выпуск {issue}: прогноз станции на D+1 (завтра) и D+2 (послезавтра)"),
                    use_container_width=True)
    st.info(f"**Сводка для диспетчера** ({log['mode']}, итоговый выпуск {log['final_issue_time_local'][:16]}): "
            f"{log['dispatcher_note']}")
    st.subheader("Журнал решений агента")
    st.caption("Каждая строка — одно действие агента (вызов инструмента) и его объяснение на основе цифр.")
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
    st.caption("Факта за февраль в данных нет, поэтому та же процедура прогнана на январе 2026 с моделями, "
               "обученными только до 01.12.2025, и сравнена с реально измеренной мощностью (SCADA).")
    day = st.radio("Горизонт", [1, 2], format_func=lambda d: "D+1 — прогноз на завтра" if d == 1 else "D+2 — на послезавтра",
                   horizontal=True)
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
    c1.metric("MAE агента (LLM)", f"{mae_a:.3f}", f"{(mae_a / mae_b - 1):+.0%} к без агента", delta_color="inverse")
    c2.metric("MAE без агента", f"{mae_b:.3f}", help="Та же модель, выпуск строго в 10:00, без проверок и пересчёта")
    c3.metric("Факт внутри P10–P90 (цель ~80%)", f"{((yy.values >= a['p10'].values) & (yy.values <= a['p90'].values))[ok].mean():.0%}")
    st.caption("Модели для репетиции обучены только на данных до 01.12.2025.")

with tab4:
    st.write("Агент выполняет полный цикл для выбранной даты выпуска: проверка погоды → прогноз → валидация → "
             "сравнение со вчерашним → поиск свежих прогонов → пересчёт → сводка.")
    c1, c2 = st.columns(2)
    issue = c1.date_input("Дата выпуска", pd.Timestamp("2026-02-15"),
                          min_value=pd.Timestamp("2024-03-10"), max_value=pd.Timestamp("2026-02-27"))
    MODES = {"auto": "авто (LLM, если есть ключ)", "rules": "правила (без нейросети)", "llm": "LLM (OpenAI)"}
    mode = c2.radio("Режим агента", list(MODES), format_func=MODES.get, horizontal=True,
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
            s = ForecastSession(pd.Timestamp(issue), models=load_models(), archive=load_store())
            agent = make_agent(mode, s)
            agent.run()
        p = s.final[s.final["unit"] == "plant"]
        st.success(f"Готово ({agent.mode}): {len(agent.log)} шагов, итоговый выпуск "
                   f"{(s.as_of_utc + pd.Timedelta(hours=cfg['scada_utc_offset_h'])):%Y-%m-%d %H:%M}")
        st.plotly_chart(band_chart(p, "time_local", f"Выпуск {issue}"), use_container_width=True)
        st.info(s.notes[-1] if s.notes else "")
        st.dataframe(pd.DataFrame([{"шаг": x["step"], "инструмент": x["tool"], "почему": x["reason"]}
                                   for x in agent.log]), use_container_width=True, hide_index=True)
