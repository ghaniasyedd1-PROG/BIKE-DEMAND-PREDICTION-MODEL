
.PHONY: train test run

train:
	python -m src.pipeline.train_pipeline

test:
	python -m pytest -q

run:
	python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
