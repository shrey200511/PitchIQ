# Convenience targets (Windows: run the python commands directly, see README)
.PHONY: install all refresh test app synthetic docker

install:
	pip install -r requirements.txt && pip install -e . --no-deps

all:            ## first full run: download -> tune -> backtest -> live -> report
	python -m pitchiq.pipeline all

refresh:        ## weekly: new results, predictions, simulation
	python -m pitchiq.pipeline refresh

test:
	pytest -q

app:
	streamlit run app/streamlit_app.py

synthetic:      ## full run on generated data in ./synthetic_run (offline demo)
	PITCHIQ_ROOT=./synthetic_run PITCHIQ_SYNTHETIC=1 python -m pitchiq.pipeline synthetic
	PITCHIQ_ROOT=./synthetic_run PITCHIQ_SYNTHETIC=1 python -m pitchiq.pipeline all

docker:
	docker build -t pitchiq-app . && docker run --rm -p 8501:8501 pitchiq-app
