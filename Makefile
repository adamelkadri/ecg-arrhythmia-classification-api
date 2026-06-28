# Convenience commands. Run e.g. `make test`.
# These assume your virtualenv is active (python/pip resolve to it).

.PHONY: install test train evaluate serve docker-build docker-run clean

install:        ## Install all dependencies
	pip install -r requirements.txt

test:           ## Run the pytest suite (fast, no training needed)
	pytest

train:          ## Train the 1D CNN and save the best model
	python -m src.train

evaluate:       ## Evaluate the saved model, write report + confusion matrix
	python -m src.evaluate

serve:          ## Run the FastAPI app with auto-reload
	uvicorn app.main:app --reload

docker-build:   ## Build the Docker image (run AFTER training)
	docker build -t ecg-api .

docker-run:     ## Run the container, exposing the API on localhost:8000
	docker run -p 8000:8000 ecg-api

clean:          ## Remove caches and generated Python artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
