FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY app/ app/
COPY main.py .

# Expose port
EXPOSE 8000

# Run FastAPI in development mode with auto-reload
CMD ["uv", "run", "fastapi", "dev", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
