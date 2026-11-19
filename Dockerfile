FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[anthropic,openai,bedrock]"

COPY config.yaml ./
COPY data ./data

# non-root user; audit DB lives on a volume
RUN useradd -m app && mkdir -p /app/runs && chown -R app /app
USER app
VOLUME ["/app/runs"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "gapagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
