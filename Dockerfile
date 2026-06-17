# syntax=docker/dockerfile:1
# Single image: build the frontend (same-origin `docker` target), then serve it and the API
# from one FastAPI process. No second server, no CORS pair - `docker compose up` is the whole app.

# --- stage 1: build the UI --------------------------------------------------------
FROM node:20-alpine AS frontend
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
# VITE_DATA_SOURCE=app (from .env.docker) + no VITE_API_BASE => calls the backend on the same origin.
RUN npm run build:docker

# --- stage 2: the backend that also serves the built UI ---------------------------
FROM python:3.11-slim AS app
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .
COPY --from=frontend /web/dist ./static
ENV ALPHALINEAGE_STATIC_DIR=/app/static
ENV ALPHALINEAGE_DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "alphalineage.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
