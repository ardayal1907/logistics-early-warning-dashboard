# syntax=docker/dockerfile:1.7
#
# Multi-stage build: builder -> runtime-base -> {api, ui, trainer}
#
# Base image is PATCH-PINNED, not `python:3.12-slim`. The production artefact is
# a pickle: the interpreter and scikit-learn versions used to load it must match
# the ones recorded in models/production_risk_model_metadata.json
# (python 3.12.10, scikit-learn 1.9.0). A floating tag silently moves the
# interpreter underneath a pickle, which is exactly the drift requirements.txt
# exists to prevent.
ARG PYTHON_VERSION=3.12.10
ARG BASE=python:${PYTHON_VERSION}-slim-bookworm

# ---------------------------------------------------------------------------
# builder - compiles wheels once, so the runtime images carry no toolchain
# ---------------------------------------------------------------------------
FROM ${BASE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencies are copied and installed BEFORE the source, so editing a .py file
# does not invalidate the layer that took minutes to build.
COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# The package itself. Only what the build needs - see .dockerignore for the rest.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN /opt/venv/bin/pip install --no-deps .

# ---------------------------------------------------------------------------
# runtime-base - the venv, a non-root user, no build tools
# ---------------------------------------------------------------------------
FROM ${BASE} AS runtime-base

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Paths are configuration. Nothing resolves a directory from __file__.
    LOGISTICS_MODEL_PATH=/app/models/production_risk_model.pkl \
    LOGISTICS_PROCESSED_DIR=/app/data/processed \
    LOGISTICS_RAW_DIR=/app/data/raw

# Running as root in a container that deserialises a pickle is a needless
# escalation path: joblib.load executes code from the file.
RUN useradd --create-home --uid 10001 logistics \
 && mkdir -p /app/models /app/data/processed /app/data/raw \
 && chown -R logistics:logistics /app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER logistics

# ---------------------------------------------------------------------------
# api
# ---------------------------------------------------------------------------
FROM runtime-base AS api

USER root
RUN /opt/venv/bin/pip install --no-cache-dir "fastapi==0.141.1" "uvicorn[standard]==0.52.1"
USER logistics

# THE ARTEFACT IS COPIED LAST, on purpose. It is 8.7 MB and changes on every
# retrain; copying it earlier would invalidate the dependency layers above it
# and turn a model refresh into a full rebuild.
COPY --chown=logistics:logistics models/production_risk_model.pkl \
     models/production_risk_model_metadata.json /app/models/

# Build-time smoke test. Loading the bundle here means a pickle/scikit-learn
# mismatch fails the IMAGE BUILD rather than the first production request.
RUN python -c "\
from logistics.infrastructure.model_repository import load_bundle; \
b = load_bundle('/app/models/production_risk_model.pkl', verify_checksum=False); \
print('artefact loads:', b.version)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "\
import sys, urllib.request, json; \
r = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); \
sys.exit(0 if r.get('status') == 'ok' and r.get('model_version') else 1)"

CMD ["uvicorn", "logistics.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------
FROM runtime-base AS ui

USER root
RUN /opt/venv/bin/pip install --no-cache-dir "streamlit==1.60.0"
USER logistics

COPY --chown=logistics:logistics models/production_risk_model.pkl \
     models/production_risk_model_metadata.json /app/models/

EXPOSE 8501

CMD ["streamlit", "run", "/opt/venv/lib/python3.12/site-packages/logistics/ui/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]

# ---------------------------------------------------------------------------
# trainer - batch stages. Carries NO web framework and NO artefact: it
# produces one.
# ---------------------------------------------------------------------------
FROM runtime-base AS trainer

USER root
RUN /opt/venv/bin/pip install --no-cache-dir "pandera==0.32.1"
USER logistics

# Written to at runtime, so they are volumes rather than image content.
VOLUME ["/app/data", "/app/models"]

ENTRYPOINT ["logistics-train"]
CMD ["--stage", "all"]
