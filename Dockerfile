# syntax=docker/dockerfile:1.7
#
# CPU variant — default. Smaller image (~2.5 GB total: 2.2 GB model +
# torch CPU + python deps). For NVIDIA GPU acceleration use
# Dockerfile.gpu instead.
#
# Build: docker build -t orion-bge-m3:cpu .

FROM python:3.12-slim AS builder

# PIP_INDEX_URL points pip at PyTorch's CPU index for torch resolution;
# PIP_EXTRA_INDEX_URL falls back to PyPI for everything else. With
# torch+cpu pinned at the source, pip never pulls the nvidia-cuda-* /
# triton wheels (~5 GB) that the default torch metadata depends on.
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential zlib1g-dev libffi-dev libssl-dev libxml2-dev \
    libxslt-dev libjpeg-dev libpng-dev \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --upgrade pip && pip wheel --wheel-dir=/wheels .


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --home /app orion \
 && mkdir -p /opt/orion && chown orion:orion /opt/orion

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels orion-bge-m3 && rm -rf /wheels

COPY --chown=orion:orion bge-m3/ /opt/orion/bge-m3/

USER orion
EXPOSE 9301

HEALTHCHECK --interval=10s --timeout=2s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:9301/healthz || exit 1

CMD ["uvicorn", "orion_bge_m3.asgi:app", "--host", "0.0.0.0", "--port", "9301", "--loop", "asyncio"]
