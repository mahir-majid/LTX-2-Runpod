# Use NVIDIA CUDA base image for GPU support (CUDA 12.8)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    wget \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager and set PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create working directory
WORKDIR /workspace

# Copy workspace configuration and packages
COPY pyproject.toml uv.lock ./
COPY packages/ ./packages/

# Install workspace dependencies with xformers
# This will install PyTorch 2.7 from cu129 index (specified in ltx-core/pyproject.toml)
RUN uv sync --frozen --extra xformers && \
    uv cache clean

# Install additional dependencies for RunPod
RUN uv pip install --no-cache \
    runpod>=1.5.0 \
    requests>=2.31.0 \
    huggingface_hub>=0.30.0

# Create model directories
RUN mkdir -p /workspace/models/checkpoints /workspace/models/gemma

# Copy application files
COPY download_models.py ./
COPY api_server.py ./

# Download models at build time
RUN /workspace/.venv/bin/python download_models.py

# Expose port for API (not used for serverless, but kept for consistency)
EXPOSE 8000

# Health check: Verify CUDA availability
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD /workspace/.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}')" || exit 1

# Start the API server using venv python
CMD ["/workspace/.venv/bin/python", "api_server.py"]
