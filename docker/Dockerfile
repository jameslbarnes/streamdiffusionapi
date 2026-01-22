# StreamDiffusion API Dockerfile
# For local development and general deployment

FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04

# Prevent timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install MediaMTX
ARG MEDIAMTX_VERSION=1.5.1
RUN wget -q https://github.com/bluenviron/mediamtx/releases/download/v${MEDIAMTX_VERSION}/mediamtx_v${MEDIAMTX_VERSION}_linux_amd64.tar.gz \
    && tar -xzf mediamtx_v${MEDIAMTX_VERSION}_linux_amd64.tar.gz \
    && mv mediamtx /usr/local/bin/ \
    && rm -f mediamtx_v${MEDIAMTX_VERSION}_linux_amd64.tar.gz mediamtx.yml

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install StreamDiffusion with all extras
RUN pip install --no-cache-dir \
    "git+https://github.com/daydreamlive/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt,controlnet,ipadapter]"

# Install TensorRT tools
RUN python -m streamdiffusion.tools.install-tensorrt || echo "TensorRT install warning (may be ok)"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p engines models

# Expose ports
# 8080 - API
# 8554 - RTSP
# 1935 - RTMP
# 8888 - HLS
# 8889 - WebRTC WHIP
# 8890 - WebRTC WHEP
# 9997 - MediaMTX API
EXPOSE 8080 8554 1935 8888 8889 8890 9997

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start command
CMD ["python", "-m", "src.api.app"]
