.PHONY: train test run
train:
	python -m src.pipeline.train_pipeline

test:
	pytest -q

run:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload
