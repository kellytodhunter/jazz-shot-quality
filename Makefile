PY ?= python3

.PHONY: all data test pipeline clean

all: data test pipeline

data:
	$(PY) scripts/download_data.py

test:
	$(PY) -m pytest -q

pipeline:
	$(PY) scripts/run_pipeline.py

clean:
	rm -rf data/processed models/*.joblib reports/figures/*.png reports/results.json
