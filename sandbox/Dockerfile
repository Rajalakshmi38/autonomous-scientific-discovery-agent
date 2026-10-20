# Image used to run untrusted, model-written experiment scripts.
# Build once with:  python -m src.main sandbox-build
# Contains only Python and numpy. The run-time restrictions (no network, read-only filesystem,
# memory/CPU/process limits, timeout) are applied by src/execution/sandbox.py when the container starts.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install numpy==2.5.4 \
 && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox

WORKDIR /work
USER 10001:10001
CMD ["python", "-"]
