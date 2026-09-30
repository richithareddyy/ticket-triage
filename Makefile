PY := .venv/bin/python
N ?= 20000

.PHONY: help setup data train train-spark spark test api ui up down space demo deploy teardown clean

help:
	@echo "setup        create .venv and install dependencies"
	@echo "data         generate synthetic tickets (N=$(N))"
	@echo "train        build features with pandas and train/compare models"
	@echo "spark        build features with PySpark (in Docker, bundles Java)"
	@echo "train-spark  train from the Spark-built feature table"
	@echo "test         run the test suite (+ Spark parity test in Docker)"
	@echo "api / ui     run the Flask API (:5050) / Streamlit UI (:8501) locally"
	@echo "up / down    start / stop the Dockerized API + UI"
	@echo "space        run API + UI in a single container (:7860)"
	@echo "demo         run the single-process Streamlit Cloud demo locally (:8501)"
	@echo "deploy       push images to ECR and deploy to ECS Fargate (needs AWS_REGION)"
	@echo "teardown     delete the AWS stack and ECR repos"

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) -m triage.data --n $(N)

train:
	$(PY) -m triage.train

spark:
	docker compose --profile batch run --rm spark

train-spark: spark
	$(PY) -m triage.train --features data/features.parquet

test:
	$(PY) -m pytest -q
	docker compose --profile batch build spark
	docker compose --profile batch run --rm --entrypoint pytest spark -q -p no:cacheprovider tests/test_spark_parity.py

api:
	$(PY) -m api.app

ui:
	API_URL=http://localhost:5050 .venv/bin/streamlit run ui/streamlit_app.py

up:
	docker compose up -d --build
	@echo "UI:  http://localhost:8501"
	@echo "API: http://localhost:5050"

down:
	docker compose down

space:
	docker build --target space -t ticket-triage-space .
	docker run --rm -p 7860:7860 ticket-triage-space

demo:
	.venv/bin/streamlit run demo/streamlit_app.py

deploy:
	./deploy/aws/deploy.sh

teardown:
	./deploy/aws/teardown.sh

clean:
	rm -rf data/features.parquet .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
