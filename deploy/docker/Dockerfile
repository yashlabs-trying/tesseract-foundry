FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" && rm -rf ~/.cache/pip

COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev]" && rm -rf ~/.cache/pip

COPY fixtures/ fixtures/
COPY schemas/ schemas/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8080/healthz || exit 1

ENTRYPOINT ["foundry"]
CMD ["serve"]
