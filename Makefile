.PHONY: install data test lint train api docker-up docker-down mlflow clean

install:
	pip install -r requirements.txt

# Download the public dataset and write the standard customer-record parquet files.
data:
	python src/data/download_opendata.py

test:
	pytest tests/ -v --tb=short

lint:
	python -m py_compile src/data/download_opendata.py
	python -m py_compile src/features/actuarial_features.py
	python -m py_compile src/features/build_features.py
	python -m py_compile src/models/train.py
	python -m py_compile src/models/evaluate.py
	python -m py_compile src/models/predict.py
	python -m py_compile src/api/main.py
	python -m py_compile src/api/schemas.py
	python -m py_compile src/api/utils.py

# Train the model. MLFLOW_TRACKING_URI defaults to http://127.0.0.1:5000;
# start `make mlflow` first or override the env var to point elsewhere.
train:
	python -m src.models.train --experiment-name churn_v1 \
		--data-path data/processed/insurance_churn_train.parquet

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow server --host 127.0.0.1 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db --serve-artifacts

docker-up:
	docker-compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
