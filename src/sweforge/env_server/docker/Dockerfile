# sweforge-base — clean, non-root execution environment for task/verify containers.
# Build: docker build -t sweforge-base -f src/sweforge/env_server/docker/Dockerfile src/sweforge/env_server/docker
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 1000 --create-home sweforge \
    && mkdir -p /workspace \
    && chown 1000:1000 /workspace

RUN pip install --no-cache-dir pytest

USER 1000:1000
WORKDIR /workspace
