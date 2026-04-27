FROM python:3.11-slim AS builder
WORKDIR /app
RUN pip install poetry
COPY pyproject.toml .
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# MODEL_PATH can be overridden to a mounted local path, e.g. /models/colpali-v1.2
ENV MODEL_PATH=vidore/colpali-v1.2
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
