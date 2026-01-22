# StreamDiffusion API Docker Image
# Pre-built with all dependencies for fast pod startup

FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MediaMTX
ARG MEDIAMTX_VERSION=v1.5.1
RUN curl -sL https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/mediamtx_${MEDIAMTX_VERSION}_linux_amd64.tar.gz | \
    tar xz -C /usr/local/bin

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install StreamDiffusion with all extras
RUN pip install --no-cache-dir \
    'git+https://github.com/daydreamlive/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt,controlnet,ipadapter]'

# Install TensorRT tools (may show warnings, that's ok)
RUN python -m streamdiffusion.tools.install-tensorrt || echo "TensorRT install completed with warnings"

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create directories for caches
RUN mkdir -p /app/engines /app/models /root/.cache/huggingface

# Pre-download SDXL-turbo model (optional - makes first stream faster)
# Uncomment to include model in image (adds ~7GB to image size)
# RUN python -c "from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained('stabilityai/sdxl-turbo', variant='fp16')"

# Expose ports
# 8080 - API
# 8554 - RTSP (internal)
# 8889 - WebRTC (WHIP/WHEP)
# 8888 - HLS
# 1935 - RTMP
EXPOSE 8080 8554 8889 8888 1935

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_MODULE_LOADING=LAZY

# Start script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
