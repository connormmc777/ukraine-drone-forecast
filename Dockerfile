# syntax=docker/dockerfile:1.7

# ---- Stage 1: build wheel deps in a slim builder ----
FROM python:3.12-slim-bookworm AS builder

# system deps for geopandas + matplotlib fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgeos-dev \
    libgdal-dev \
    libproj-dev \
    gdal-bin \
    proj-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY drone_app/requirements.txt* ./
# Fallback: derive requirements from install.sh if requirements.txt absent
RUN if [ ! -f requirements.txt ]; then \
      printf "streamlit\npandas\nnumpy\nmatplotlib\nscipy\nscikit-learn\ngeopandas\nrequests\nstreamlit-autorefresh\n" > requirements.txt ; \
    fi

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: slim runtime ----
FROM python:3.12-slim-bookworm

# Runtime-only geo libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-c1v5 \
    libgdal32 \
    libproj25 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r app && useradd -r -g app -u 1000 app

COPY --from=builder /install /usr/local

WORKDIR /app
COPY drone_app/ /app/

# Make sure the data directory exists (mounted volume in prod)
RUN mkdir -p /app/data && chown -R app:app /app

USER app:app

# Streamlit config: no CORS, no telemetry, headless
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    PYTHONUNBUFFERED=1

EXPOSE 8501

# Healthcheck against Streamlit's built-in endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        r=urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3); \
        sys.exit(0 if r.read().startswith(b'ok') else 1)"

ENTRYPOINT ["streamlit", "run", "app.py"]
