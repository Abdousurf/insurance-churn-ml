.PHONY: install test lint train api docker-up docker-down mlflow clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	python -m py_compile src/features/actuarial_features.py
	python -m py_compile src/features/build_features.py
	python -m py_compile src/models/train.py
	python -m py_compile src/models/evaluate.py
	python -m py_compile src/models/predict.py
	python -m py_compile src/api/main.py
	python -m py_compile src/api/schemas.py

train:
	python src/models/train.py --experiment-name churn_v1

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow ui --port 5000

docker-up:
	docker-compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
