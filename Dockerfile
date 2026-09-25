FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch FIRST, from PyTorch's own CPU wheel index. Without
# this, sentence-transformers (a requirements.txt dependency) pulls the
# default GPU build of torch, which drags in several GB of unused NVIDIA
# CUDA libraries (cuDNN, cuBLAS, cuSOLVER, etc.) even though this image has
# no GPU access — turning a ~1 minute build into a 20+ minute one for
# nothing. Installing the CPU wheel first satisfies sentence-transformers'
# torch requirement without ever considering the CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model (app/engine/embedding.py _MODEL_NAME) into the
# image. Otherwise every container downloads it from HuggingFace on boot —
# ~70s with two replicas starting together, which blew past the healthcheck
# window and left compose reporting the replica as failed.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy full application codebase
COPY . ./

# Create directories for persistent FAISS store
RUN mkdir -p faiss_store

EXPOSE 8000

# Health check at container level (start-period covers model + FAISS loading)
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Launch uvicorn server in production mode
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "asyncio"]
