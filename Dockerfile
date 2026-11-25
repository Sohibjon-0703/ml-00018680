# Reproducible container for the Streamlit app (dependencies pinned by uv.lock)
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY models ./models
COPY data/processed ./data/processed
EXPOSE 8501
CMD ["uv", "run", "--frozen", "--no-dev", "streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
