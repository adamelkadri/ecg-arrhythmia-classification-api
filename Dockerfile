# Slim Python base — CPU-only inference, no CUDA needed in the container.
FROM python:3.11-slim

# Don't write .pyc files; flush stdout/stderr immediately (nice for logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model.
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

# Serve the API. Use 0.0.0.0 so it's reachable from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
