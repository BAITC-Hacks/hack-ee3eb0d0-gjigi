# One-command entry points. The weather archive and trained models are in the repo,
# so `make forecast` / `make agent` work offline in ~1-5 minutes.
PY ?= python3

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

eda:              ## stage 0
	$(PY) scripts/run_eda.py
weather:          ## stage 1 (downloads ~15 min on first run, then cached)
	$(PY) scripts/build_weather_archive.py && $(PY) scripts/check_weather.py
baselines:        ## stage 2
	$(PY) scripts/run_baselines.py --refresh
train:            ## stage 3 (holdout + 12-month CV + final models, ~20 min)
	$(PY) scripts/train_model.py
forecast:         ## stage 4: Feb 2026 rolling forecast, no agent
	$(PY) scripts/run_backtest.py
agent:            ## stage 5: agent over Feb 2026 (LLM if OPENAI_API_KEY in .env, else rules)
	$(PY) scripts/run_agent.py --mode auto
rehearsal:        ## agent on Jan 2026 with models that never saw Dec-Jan, scored vs SCADA
	$(PY) scripts/run_agent.py --mode auto --rehearsal
demo:             ## dashboard on http://localhost:8501
	$(PY) -m streamlit run app.py
test:
	$(PY) -m pytest -q
all: eda weather baselines train forecast agent test
