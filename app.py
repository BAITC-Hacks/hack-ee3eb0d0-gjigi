"""Demo dashboard: streamlit run app.py

Sections (sidebar):
  1. Прогноз на февраль — freshest agent forecast for every hour + P10-P90
  2. Выпуск и решения агента — one issue: 48 h forecast, decision log, dispatcher note
  3. Проверка на факте — agent vs no-agent vs actual SCADA (January 2026 rehearsal)
  4. Запуск агента — run the agent live for any issue date (rules or LLM)
"""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hackalem.config import ROOT, load_config

cfg = load_config()
OUT = cfg["paths"]["outputs_dir"]
st.set_page_config(page_title="Wind48 — прогноз выработки ВЭС", page_icon="🌬️", layout="wide")

GREEN, GREEN_DARK, SAGE, INK, MUTED, LINE = "#4f7a5e", "#2f5d46", "#8aa38f", "#1f2a24", "#6b776f", "#e3e7dc"

st.markdown(f"""
<style>
  .stApp {{ background: #f6f7f2; }}
  [data-testid="stSidebar"] {{ background: #eef1e8; border-right: 1px solid {LINE}; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
  h1, h2, h3, p, span, div, label {{ color: {INK}; }}
  .brand {{ font-size: 30px; font-weight: 700; color: {GREEN_DARK}; letter-spacing: -0.5px; margin-bottom: 6px; }}
  .brand span {{ color: {SAGE}; }}
  .eyebrow {{ font-size: 11px; font-weight: 600; letter-spacing: 1.6px; text-transform: uppercase; color: {MUTED}; margin: 14px 0 6px; }}
  .crumbs {{ font-size: 13px; color: {MUTED}; border-bottom: 1px solid {LINE}; padding-bottom: 14px; margin-bottom: 18px; }}
  .hero-title {{ font-size: 44px !important; font-weight: 700; line-height: 1.12; margin: 0; color: {INK}; letter-spacing: -0.5px; }}
  .hero-title span {{ color: {SAGE}; }}
  .hero-sub {{ color: {MUTED}; font-size: 15px; margin-top: 10px; }}
  .card {{ background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 18px 20px; height: 100%; }}
  .card.accent {{ background: #e6eddf; }}
  .kpi-label {{ font-size: 13px; color: {MUTED}; margin-bottom: 10px; }}
  .kpi-value {{ font-size: 36px; font-weight: 600; color: {INK}; line-height: 1.1; }}
  .kpi-unit {{ font-size: 15px; color: {MUTED}; margin-left: 4px; font-weight: 400; }}
  .kpi-sub {{ font-size: 12px; color: #8a958c; margin-top: 8px; }}
  .side-card {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 16px; margin-top: 10px; }}
  .side-title {{ font-size: 22px; font-weight: 700; color: {INK}; }}
  .side-muted {{ font-size: 12px; color: {MUTED}; line-height: 1.6; }}
  .badge {{ display: inline-block; background: #dfe9d8; color: {GREEN_DARK}; font-size: 11px; font-weight: 700;
            letter-spacing: 1px; padding: 4px 10px; border-radius: 8px; text-transform: uppercase; }}
  .note {{ background: #fff; border-left: 4px solid {GREEN}; border-radius: 10px; padding: 14px 18px; color: {INK}; }}
  .step {{ display: flex; gap: 12px; padding: 10px 0; border-bottom: 1px solid #f0f2ec; }}
  .step-dot {{ min-width: 26px; height: 26px; border-radius: 50%; background: #e6eddf; color: {GREEN_DARK};
               font-size: 12px; font-weight: 700; display: flex; align-items: center; justify-content: center; }}
  .step-name {{ font-weight: 600; font-size: 14px; }}
  .step-why {{ font-size: 12.5px; color: {MUTED}; }}
  div[data-testid="stRadio"] label p {{ font-size: 14px; }}
  .stDownloadButton button, .stButton button {{ border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)

TURBINE_SVG = f"""
<svg viewBox="0 0 220 170" width="210" xmlns="http://www.w3.org/2000/svg">
  <circle cx="150" cy="70" r="46" fill="#e6eddf"/>
  <path d="M20 150 Q110 118 210 150" stroke="{LINE}" stroke-width="2" fill="none"/>
  <line x1="18" y1="52" x2="78" y2="52" stroke="{SAGE}" stroke-width="2" stroke-linecap="round"/>
  <line x1="30" y1="66" x2="92" y2="66" stroke="{SAGE}" stroke-width="2" stroke-linecap="round"/>
  <line x1="40" y1="80" x2="84" y2="80" stroke="{SAGE}" stroke-width="2" stroke-linecap="round"/>
  <rect x="146" y="72" width="6" height="80" rx="2" fill="{GREEN}"/>
  <g transform="translate(149 72)" fill="{GREEN_DARK}">
    <path d="M0 0 L-4 -58 Q0 -64 4 -58 Z"/>
    <path d="M0 0 L-4 -58 Q0 -64 4 -58 Z" transform="rotate(120)"/>
    <path d="M0 0 L-4 -58 Q0 -64 4 -58 Z" transform="rotate(240)"/>
    <circle r="6"/>
  </g>
</svg>"""

TOOL_RU = {
    "fetch_weather": "Получение прогнозов погоды", "check_weather_inputs": "Проверка источников погоды",
    "run_forecast": "Запуск модели прогноза", "validate_forecast": "Проверка результата",
    "compare_with_previous": "Сравнение со вчерашним прогнозом", "widen_intervals": "Расширение интервала",
    "check_for_new_runs": "Поиск свежих прогнозов погоды", "rerun_with_update": "Пересчёт на свежих данных",
    "recommend_bid": "План для балансирующего рынка", "finalize": "Утверждение и записка диспетчеру",
}
UNIT_NAMES = {"plant": "Станция целиком (среднее двух турбин)", "T1": "Турбина 1 (T1)", "T2": "Турбина 2 (T2)"}


def band_chart(df, x, title, actual=None, extra=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df[x], y=df["p10"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(79,122,94,0.18)", name="P10–P90"))
    fig.add_trace(go.Scatter(x=df[x], y=df["p50"], name="P50", line=dict(color=GREEN_DARK, width=2.2)))
    if extra is not None:
        fig.add_trace(go.Scatter(x=extra[0], y=extra[1], name=extra[2], line=dict(dash="dot", color="#c08a3e")))
    if actual is not None:
        fig.add_trace(go.Scatter(x=actual[0], y=actual[1], name="факт SCADA", line=dict(color=INK, width=1.6)))
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=INK)), height=420,
                      paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK),
                      yaxis=dict(range=[0, 1.02], title="мощность, доля номинала", gridcolor="#eef1e8"),
                      xaxis=dict(gridcolor="#f3f5ef"), margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.12))
    return fig


def kpi(label, value, unit="", sub="", accent=False):
    return (f'<div class="card{" accent" if accent else ""}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def steps_html(steps):
    rows = []
    for s in steps:
        rows.append(f'<div class="step"><div class="step-dot">{s["step"]}</div><div>'
                    f'<div class="step-name">{TOOL_RU.get(s["tool"], s["tool"])} '
                    f'<span style="color:#9aa59c;font-weight:400;font-size:12px">· {s["tool"]}</span></div>'
                    f'<div class="step-why">{s["reason"]}</div></div></div>')
    return '<div class="card">' + "".join(rows) + "</div>"


@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=["time_local", "issue_date"])


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">Wind<span>48</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Панель управления</div>', unsafe_allow_html=True)
    PAGES = ["📈  Прогноз на февраль", "🤖  Выпуск и решения агента", "✅  Проверка на факте", "▶️  Запуск агента"]
    page = st.radio("Раздел", PAGES, label_visibility="collapsed")
    t1, t2 = cfg["turbines"]["T1"], cfg["turbines"]["T2"]
    st.markdown(f"""
<div class="side-card">
  <div class="eyebrow" style="margin-top:0">Объект прогноза</div>
  <div class="side-title">ВЭС · 2 турбины</div>
  <div class="side-muted">T1: {t1['lat']:.5f}° N, {t1['lon']:.5f}° E<br>T2: {t2['lat']:.5f}° N, {t2['lon']:.5f}° E</div>
  <div class="eyebrow">Горизонт</div>
  <div class="side-title">48 <span class="side-muted">часов · шаг 1 час</span></div>
  <div class="eyebrow">Выпуск</div>
  <div class="side-muted">ежедневно в 10:00, пересчёт при выходе свежего прогноза погоды</div>
</div>
<div class="side-muted" style="margin-top:14px">HACKALEM AI · Agentic AI</div>""", unsafe_allow_html=True)

st.markdown('<div class="crumbs">Ветроэнергетика &nbsp;/&nbsp; Прогнозирование выработки</div>',
            unsafe_allow_html=True)

# ── 1. February forecast ─────────────────────────────────────────────────────
if page == PAGES[0]:
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown('<div class="eyebrow">Планируйте на два дня вперёд</div>'
                    '<div class="hero-title">Ветер меняется.<br><span>Агент видит прогноз.</span></div>'
                    '<div class="hero-sub">Каждый день в 10:00 агент сам скачивает прогнозы погоды, считает почасовой '
                    'прогноз выработки ВЭС на следующие двое суток, проверяет его и при выходе свежего прогноза '
                    'пересчитывает. Используются только прогнозы, опубликованные к моменту выпуска.</div>',
                    unsafe_allow_html=True)
    with h2:
        st.markdown(TURBINE_SVG, unsafe_allow_html=True)

    allf = load_csv(OUT / "agent" / "agent_forecasts.csv")
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        unit = c1.selectbox("Объект прогноза", list(UNIT_NAMES), format_func=UNIT_NAMES.get)
        sub_path = cfg["paths"]["submission_dir"] / "forecast_feb2026.csv"
        c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        c2.download_button("↓ Файл сдачи (CSV)", sub_path.read_bytes(), sub_path.name, use_container_width=True)
    d = (allf[allf["unit"] == unit].sort_values("issue_date").groupby("time_local", as_index=False).last())
    d = d[d["time_local"].between(cfg["periods"]["test_start"], cfg["periods"]["test_end"])]

    st.markdown(f'<span class="badge">Прогноз агента</span> &nbsp; <span class="side-muted">'
                f'{d["time_local"].min():%d.%m.%Y %H:%M} — {d["time_local"].max():%d.%m.%Y %H:%M} · '
                f'28 выпусков, для каждого часа — самый свежий (D+1)</span>', unsafe_allow_html=True)
    st.write("")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Средняя ожидаемая мощность", f"{d['mean'].mean() * 100:.0f}", "%", "доля от номинальной"),
                unsafe_allow_html=True)
    k2.markdown(kpi("Часов в прогнозе", f"{len(d)}", "ч", "все часы февраля 2026"), unsafe_allow_html=True)
    k3.markdown(kpi("Средняя ширина P10–P90", f"{(d['p90'] - d['p10']).mean():.2f}", "",
                    "чем шире — тем менее уверен прогноз"), unsafe_allow_html=True)
    k4.markdown(kpi("Горизонт прогноза", "48", "часов", "почасовой шаг · 2 суток", accent=True),
                unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        st.plotly_chart(band_chart(d, "time_local", f"Февраль 2026 — {UNIT_NAMES[unit]}"), use_container_width=True)
        st.caption("Факта за февраль в данных нет — его проверяют организаторы. "
                   "Качество на реальном факте — в разделе «Проверка на факте».")

# ── 2. One issue and the agent's decisions ───────────────────────────────────
elif page == PAGES[1]:
    st.markdown('<div class="eyebrow">Как работал агент</div><div class="hero-title" style="font-size:34px !important">'
                'Выпуск и решения агента</div><div class="hero-sub">Выберите день выпуска: слева — прогноз на следующие '
                'двое суток, справа — что и почему делал агент.</div>', unsafe_allow_html=True)
    fc = load_csv(OUT / "agent" / "agent_forecasts.csv")
    st.write("")
    issue = st.selectbox("Дата выпуска (день D)", sorted(fc["issue_date"].dt.date.unique()), index=10)
    f = fc[(fc["issue_date"].dt.date == issue) & (fc["unit"] == "plant")]
    log = json.loads((OUT / "agent" / "logs" / f"{issue}.json").read_text(encoding="utf-8"))
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.plotly_chart(band_chart(f, "time_local", f"Выпуск {issue}: D+1 (завтра) и D+2 (послезавтра)"),
                            use_container_width=True)
        st.markdown(f'<div class="note"><b>Записка диспетчеру</b> · {log["mode"]}, итоговый выпуск '
                    f'{log["final_issue_time_local"][:16]}<br>{log["dispatcher_note"]}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="eyebrow" style="margin-top:0">Процесс · от данных к прогнозу</div>',
                    unsafe_allow_html=True)
        st.markdown(steps_html(log["steps"]), unsafe_allow_html=True)
    with st.expander("Полные результаты инструментов (JSON)"):
        st.json(log["steps"])

# ── 3. January rehearsal vs actual ───────────────────────────────────────────
elif page == PAGES[2]:
    from hackalem.data.turbines import load_hourly
    st.markdown('<div class="eyebrow">Проверка на реальном факте</div><div class="hero-title" style="font-size:34px !important">'
                'Январь 2026: прогноз против факта</div><div class="hero-sub">Факта за февраль в данных нет, поэтому та же '
                'процедура прогнана на январе 2026 с моделями, обученными только до 01.12.2025, и сравнена с реально '
                'измеренной мощностью (SCADA).</div>', unsafe_allow_html=True)
    st.write("")
    y = load_hourly()["power_plant"]
    ag = load_csv(OUT / "agent_rehearsal" / "agent_forecasts.csv")
    base = load_csv(OUT / "forecast" / "rehearsal_jan2026.csv")
    day = st.radio("Горизонт", [1, 2], horizontal=True,
                   format_func=lambda d: "D+1 — прогноз на завтра" if d == 1 else "D+2 — на послезавтра")
    a = ag[(ag["unit"] == "plant") & (ag["day_ahead"] == day)].sort_values("time_local")
    b = base[(base["unit"] == "plant") & (base["day_ahead"] == day)].sort_values("time_local")
    yy = y.reindex(a["time_local"])
    ok = yy.notna().values
    mae_a = abs(a["p50"].values[ok] - yy.values[ok]).mean()
    yb = y.reindex(b["time_local"]).values
    okb = ~pd.isna(yb)
    mae_b = abs(b["p50"].values[okb] - yb[okb]).mean()
    cover = ((yy.values >= a["p10"].values) & (yy.values <= a["p90"].values))[ok].mean()
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi("Ошибка агента (MAE)", f"{mae_a:.3f}", "", f"{(mae_a / mae_b - 1):+.0%} к прогнозу без агента",
                    accent=True), unsafe_allow_html=True)
    k2.markdown(kpi("Ошибка без агента (MAE)", f"{mae_b:.3f}", "",
                    "та же модель, выпуск в 10:00 без проверок и пересчёта"), unsafe_allow_html=True)
    k3.markdown(kpi("Факт внутри P10–P90", f"{cover * 100:.0f}", "%", "цель ≈ 80%"), unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        st.plotly_chart(band_chart(a, "time_local", f"Январь 2026, D+{day}: агент против факта",
                                   actual=(a["time_local"], yy.values),
                                   extra=(b["time_local"], b["p50"], "без агента, P50")), use_container_width=True)
        st.caption("MAE — средняя ошибка в долях номинала: 0.15 = в среднем 15% мощности. Меньше — лучше.")

# ── 4. Live agent run ────────────────────────────────────────────────────────
else:
    st.markdown('<div class="eyebrow">Агент вживую</div><div class="hero-title" style="font-size:34px !important">'
                'Запуск агента</div><div class="hero-sub">Агент выполняет полный цикл для выбранной даты выпуска: '
                'получение погоды → проверка → прогноз → проверка результата → сравнение со вчерашним → '
                'поиск свежих прогнозов → пересчёт → записка диспетчеру.</div>', unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 2, 1])
        issue = c1.date_input("Дата выпуска", pd.Timestamp("2026-02-15"),
                              min_value=pd.Timestamp("2024-03-10"), max_value=pd.Timestamp("2026-02-27"))
        MODES = {"auto": "авто (LLM, если есть ключ)", "rules": "правила (без нейросети)", "llm": "LLM (OpenAI)"}
        mode = c2.radio("Режим агента", list(MODES), format_func=MODES.get, horizontal=True,
                        help="auto = LLM, если в .env есть OPENAI_API_KEY")
        c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run = c3.button("↗ Запустить агента", type="primary", use_container_width=True)
    if run:
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
        left, right = st.columns([3, 2])
        with left:
            with st.container(border=True):
                st.plotly_chart(band_chart(p, "time_local", f"Выпуск {issue}"), use_container_width=True)
            st.markdown(f'<div class="note"><b>Записка диспетчеру</b><br>{s.notes[-1] if s.notes else ""}</div>',
                        unsafe_allow_html=True)
        with right:
            st.markdown('<div class="eyebrow" style="margin-top:0">Процесс · от данных к прогнозу</div>',
                        unsafe_allow_html=True)
            st.markdown(steps_html(agent.log), unsafe_allow_html=True)

        # hourly forecast table (48 h)
        fin = s.final.copy()
        wide = fin.pivot_table(index="time_local", columns="unit", values="p50")
        plant = fin[fin["unit"] == "plant"].set_index("time_local").sort_index()
        tbl = pd.DataFrame({
            "Час": plant.index.strftime("%d.%m %H:%M"),
            "Сутки": plant["day_ahead"].map({1: "D+1 · завтра", 2: "D+2 · послезавтра"}).values,
            "Прогноз станции, %": (plant["p50"] * 100).round(1).values,
            "Интервал P10–P90, %": [f"{a * 100:.0f} – {b * 100:.0f}" for a, b in zip(plant["p10"], plant["p90"])],
            "Турбина 1, %": (wide.reindex(plant.index)["T1"] * 100).round(1).values if "T1" in wide else None,
            "Турбина 2, %": (wide.reindex(plant.index)["T2"] * 100).round(1).values if "T2" in wide else None,
        })
        if "plan_bid" in plant and plant["plan_bid"].notna().any():
            tbl["План для рынка, %"] = (plant["plan_bid"] * 100).round(1).values
        st.markdown('<div class="eyebrow">Почасовой прогноз · 48 часов</div>', unsafe_allow_html=True)
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=460, column_config={
            "Прогноз станции, %": st.column_config.ProgressColumn(
                "Прогноз станции (P50)", min_value=0, max_value=100, format="%.1f%%"),
            "Турбина 1, %": st.column_config.NumberColumn("Турбина 1 (T1)", format="%.1f%%"),
            "Турбина 2, %": st.column_config.NumberColumn("Турбина 2 (T2)", format="%.1f%%"),
            "План для рынка, %": st.column_config.NumberColumn(
                "План для рынка", format="%.1f%%", help="Выгодный план подачи на балансирующий рынок (квантиль P31)"),
        })
        st.download_button("↓ Скачать прогноз выпуска (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                           f"forecast_{issue}.csv")
