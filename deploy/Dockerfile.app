# ClaimSleuth app: FastAPI, the ADK agents, the dashboard and the MCP server at /mcp.
# Build from the project folder:  docker build -f deploy/Dockerfile.app -t claimsleuth-app .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000 DATA_DIR=/data \
    HF_HOME=/opt/hf CLAIMSLEUTH_RAG_INDEX_DIR=/data/rag_index

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Litestream (SQLite replication to a bucket); used only when LITESTREAM_BUCKET is set.
ARG LITESTREAM_VERSION=0.3.13
RUN curl -fsSL "https://github.com/benbjohnson/litestream/releases/download/v${LITESTREAM_VERSION}/litestream-v${LITESTREAM_VERSION}-linux-amd64.tar.gz" \
    | tar -xz -C /usr/local/bin litestream

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch first, so sentence-transformers does not pull the CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
# The embedding model is downloaded at build time; at run time it is loaded offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY data/sample_claims ./data/sample_claims
COPY data/rag/policies ./data/rag/policies
COPY data/past_claims.json ./data/past_claims.json
COPY eval/claims ./eval/claims
COPY eval/personas ./eval/personas
COPY deploy/litestream.yml /etc/litestream.yml
COPY deploy/entrypoint-app.sh /entrypoint.sh

# Editable install: the code stays in /app/src, where it finds configs/ and data/.
RUN pip install --no-deps -e . && chmod +x /entrypoint.sh \
    && useradd --create-home --uid 10001 claimsleuth && mkdir -p /data && chown -R claimsleuth /data
USER claimsleuth
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=4)"
ENTRYPOINT ["/entrypoint.sh"]
